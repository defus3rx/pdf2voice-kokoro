"""Kokoro-82M text-to-speech wrapper: device selection, offline cache handling, synthesis."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np

from .textclean import chunk_text

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000
MODEL_FILES = ("config.json", "kokoro-v1_0.pth")

LANGUAGES = {
    "a": "American English", "b": "British English", "e": "Spanish", "f": "French",
    "h": "Hindi", "i": "Italian", "j": "Japanese", "p": "Brazilian Portuguese", "z": "Mandarin Chinese",
}

VOICES = {
    "a": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova",
          "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
          "am_michael", "am_onyx", "am_puck", "am_santa"],
    "b": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "i": ["if_sara", "im_nicola"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "p": ["pf_dora", "pm_alex", "pm_santa"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
}


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def cached_file(name: str) -> Path | None:
    snaps = hf_cache_dir() / f"models--{REPO_ID.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (snap / name).exists():
            return snap / name
    return None


def cached_voices() -> set[str]:
    snaps = hf_cache_dir() / f"models--{REPO_ID.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return set()
    return {p.stem for p in snaps.glob("*/voices/*.pt")}


def go_offline_if_cached(voice: str) -> bool:
    """Skip network checks entirely once the model and voice are in the HF cache."""
    needed = list(MODEL_FILES) + [f"voices/{voice}.pt"]
    if all(cached_file(n) for n in needed):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return True
    return False


def default_lang_for(voice: str) -> str:
    return voice[0] if voice and voice[0] in LANGUAGES else "a"


def _quiet() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for name in ("torch.distributed.elastic.multiprocessing.redirects", "phonemizer", "kokoro"):
        logging.getLogger(name).setLevel(logging.ERROR)


def pick_device(requested: str = "auto") -> str:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "cpu":
        return "cpu"
    if requested in available:
        if available[requested]:
            return requested
        print(f"{requested.upper()} requested but not available; using CPU.")
        return "cpu"
    return next((dev for dev, ok in available.items() if ok), "cpu")


class Narrator:
    def __init__(self, voice: str = "af_heart", lang: str = "a", speed: float = 1.0, device: str = "auto"):
        _quiet()
        self.voice, self.lang, self.speed = voice, lang, speed
        self.offline = go_offline_if_cached(voice)
        self.device = pick_device(device)
        self._load(self.device)

    def _load(self, device: str) -> None:
        from kokoro import KPipeline

        self.device = device
        self.pipeline = KPipeline(lang_code=self.lang, repo_id=REPO_ID, device=device)
        self.pipeline.load_voice(self.voice)

    def _run(self, text: str) -> np.ndarray:
        parts = []
        for result in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=None):
            audio = result.audio
            if audio is None:
                continue
            parts.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        try:
            return self._run(text)
        except Exception as exc:  # noqa: BLE001 - any GPU failure should fall back, not abort a long run
            if self.device == "cpu":
                raise
            print(f"\nGPU synthesis failed ({type(exc).__name__}: {exc}); switching to CPU.")
            self._load("cpu")
            return self._run(text)

    @staticmethod
    def chunks(paragraph: str, max_chars: int = 350) -> list[str]:
        return chunk_text(paragraph, max_chars)


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)
