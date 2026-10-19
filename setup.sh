#!/usr/bin/env bash
# One-shot installer for pdf2voice.
# - checks system packages (espeak-ng, ffmpeg) and uv, and asks before installing any
# - creates .venv with Python 3.12 and installs pdf2voice into it
# - downloads the Kokoro-82M weights into the Hugging Face cache (one time)
# - optionally links `pdf2voice` into ~/.local/bin so it runs from any terminal
set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

ask() {  # ask "question" -> returns 0 for yes
  local reply
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

pkg_install_cmd() {  # prints the install command for this system's package manager
  if command -v brew >/dev/null 2>&1; then echo "brew install"
  elif command -v apt-get >/dev/null 2>&1; then echo "sudo apt-get install -y"
  elif command -v dnf >/dev/null 2>&1; then echo "sudo dnf install -y"
  elif command -v pacman >/dev/null 2>&1; then echo "sudo pacman -S --noconfirm"
  fi
}

echo "==> Checking system packages"
missing=()
for pkg in espeak-ng ffmpeg; do
  command -v "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if ((${#missing[@]})); then
  echo "Missing: ${missing[*]}"
  installer="$(pkg_install_cmd)"
  if [[ -z "$installer" ]]; then
    echo "Install ${missing[*]} with your package manager, then re-run this script." >&2
    exit 1
  fi
  if ask "Run '$installer ${missing[*]}' now?"; then
    $installer "${missing[@]}"
  else
    echo "Cannot continue without: ${missing[*]}" >&2
    exit 1
  fi
else
  echo "    espeak-ng and ffmpeg are installed."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv (Python package manager) is missing."
  if command -v brew >/dev/null 2>&1 && ask "Run 'brew install uv' now?"; then
    brew install uv
  elif ask "Install uv with its official installer (curl -LsSf https://astral.sh/uv/install.sh | sh)?"; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "Cannot continue without uv: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
fi

echo "==> Creating virtual environment (.venv, Python 3.12)"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
echo "==> Installing pdf2voice and dependencies"
uv pip install --python .venv/bin/python -e ".[dev]"

echo "==> Downloading Kokoro-82M weights and the default voice (first time only)"
.venv/bin/python - <<'EOF'
import time
from pdf2voice.tts import Narrator
t = time.time()
n = Narrator("af_heart", "a", 1.0)
audio = n.synthesize("Hello. pdf2voice is ready.")
print(f"    Model ready on {n.device} ({len(audio) / 24000:.1f}s test clip in {time.time() - t:.1f}s).")
EOF

echo "==> Running tests"
.venv/bin/python -m pytest -q

SHIM_DIR="$HOME/.local/bin"
echo
if ask "Link pdf2voice into $SHIM_DIR so it runs from any terminal?"; then
  mkdir -p "$SHIM_DIR"
  ln -sf "$PROJECT_DIR/.venv/bin/pdf2voice" "$SHIM_DIR/pdf2voice"
  echo "    Linked $SHIM_DIR/pdf2voice"
  case ":$PATH:" in
    *":$SHIM_DIR:"*) ;;
    *) echo "    Add this line to your shell profile (~/.zshrc or ~/.bashrc) so the shell finds it:"
       echo "      export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
  esac
else
  echo "Alternatively add this alias to your shell profile:"
  echo "  alias pdf2voice='$PROJECT_DIR/.venv/bin/pdf2voice'"
fi
echo
echo "Done. Try:  pdf2voice --list-voices"
