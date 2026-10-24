#!/usr/bin/env bash
# Start SentinelAI with the local LLM on the GPU and open the dashboard.
#
#   ./start.sh                 # GPU if available (CPU fallback), opens the dashboard
#   ./start.sh --cpu           # force CPU inference
#   ./start.sh --no-browser    # don't open a browser
#
# Environment overrides: SENTINEL_MODEL_NAME (default qwen2.5:1.5b), API_PORT (8000), UI_PORT (8501),
# OLLAMA_URL (http://127.0.0.1:11434). Everything binds to 127.0.0.1. Ctrl+C stops what this script started.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

MODEL="${SENTINEL_MODEL_NAME:-qwen2.5:1.5b}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
DEVICE="gpu"
OPEN_BROWSER=1
LOG_DIR="var/logs"

for arg in "$@"; do
    case "$arg" in
        --cpu) DEVICE="cpu" ;;
        --no-browser) OPEN_BROWSER=0 ;;
        -h | --help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
    esac
done

info() { printf '\033[1;34m[sentinelai]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[sentinelai]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[sentinelai]\033[0m %s\n' "$*" >&2; exit 1; }

# Wait until a URL answers, or fail after N half-second tries.
wait_for() {
    local url="$1" tries="${2:-60}"
    for _ in $(seq "$tries"); do
        curl -sf -o /dev/null "$url" && return 0
        sleep 0.5
    done
    return 1
}

PIDS=()
cleanup() {
    trap - INT TERM EXIT
    if ((${#PIDS[@]})); then
        info "stopping SentinelAI"
        # Each server runs in its own process group: signal the whole group (uv, python, streamlit).
        for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; done
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap 'echo; cleanup; exit 0' INT TERM
trap cleanup EXIT

# --- prerequisites ------------------------------------------------------------------------
command -v uv >/dev/null || die "uv is not installed: https://docs.astral.sh/uv/"
command -v curl >/dev/null || die "curl is required"
mkdir -p "$LOG_DIR"
info "syncing the Python environment (uv sync)"
uv sync --frozen --quiet || die "uv sync failed"

open_dashboard() {
    ((OPEN_BROWSER)) || return 0
    if command -v xdg-open >/dev/null; then
        xdg-open "$1" >/dev/null 2>&1 &
    elif command -v open >/dev/null; then
        open "$1"
    fi
}

# Already running? Then just open the dashboard instead of starting a second copy.
if curl -sf "http://127.0.0.1:$API_PORT/health" 2>/dev/null | grep -q '"status":"ok"' &&
    curl -sf -o /dev/null "http://127.0.0.1:$UI_PORT/_stcore/health"; then
    model=$(curl -s "http://127.0.0.1:$API_PORT/api/v1/status" | sed -n 's/.*"model_id":"\([^"]*\)".*"device":\("\{0,1\}[^,"]*\).*/\1 on \2/p' | tr -d '"')
    info "SentinelAI is already running (${model:-model unknown}); opening the dashboard"
    echo "    Dashboard : http://127.0.0.1:$UI_PORT/dashboard"
    open_dashboard "http://127.0.0.1:$UI_PORT/dashboard"
    exit 0
fi
for port in "$API_PORT" "$UI_PORT"; do
    if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
        die "port $port is in use by another program. Free it or set API_PORT/UI_PORT."
    fi
done

# --- GPU ----------------------------------------------------------------------------------
if [[ "$DEVICE" == "gpu" ]]; then
    if command -v nvidia-smi >/dev/null && gpu=$(nvidia-smi --query-gpu=name,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null); then
        info "GPU: $gpu"
    else
        warn "no usable NVIDIA GPU ($(nvidia-smi -L 2>&1 | head -1 || echo 'nvidia-smi not found')); using CPU"
        warn "after a driver update, a reboot usually fixes 'Driver/library version mismatch'"
        DEVICE="cpu"
    fi
fi

# --- Ollama -------------------------------------------------------------------------------
if ! curl -sf -o /dev/null "$OLLAMA_URL/api/version"; then
    command -v ollama >/dev/null || die "Ollama is not running and not installed: https://ollama.com"
    info "starting ollama serve (log: $LOG_DIR/ollama.log)"
    setsid ollama serve >"$LOG_DIR/ollama.log" 2>&1 &
    PIDS+=($!)
    wait_for "$OLLAMA_URL/api/version" 40 || die "Ollama did not start; see $LOG_DIR/ollama.log"
fi
info "Ollama $(curl -s "$OLLAMA_URL/api/version" | sed 's/.*"version":"\([^"]*\)".*/\1/') at $OLLAMA_URL"

# Models are never downloaded automatically; ask first.
if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$MODEL\""; then
    warn "model $MODEL is not installed (~1 GB download)"
    read -r -p "Pull it now with 'ollama pull $MODEL'? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || die "aborted; install it with: ollama pull $MODEL"
    ollama pull "$MODEL"
fi

# Load the model now with the requested placement, so the first chat is fast. Unload any existing
# instance first: Ollama would otherwise reuse a copy previously loaded on the other device.
num_gpu=999
[[ "$DEVICE" == "cpu" ]] && num_gpu=0
info "loading $MODEL on $DEVICE"
curl -sf -o /dev/null "$OLLAMA_URL/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" || true
curl -sf -o /dev/null --max-time 180 "$OLLAMA_URL/api/generate" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"30m\",\"options\":{\"num_gpu\":$num_gpu,\"num_predict\":1}}" ||
    die "could not load $MODEL; check 'ollama ps' and $LOG_DIR/ollama.log"
placement=$(curl -s "$OLLAMA_URL/api/ps" | uv run --quiet python -c '
import json, sys
for m in json.load(sys.stdin).get("models", []):
    if m["name"] == sys.argv[1]:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        print("gpu" if vram and vram >= size else ("cpu" if not vram else f"gpu+cpu ({vram / size:.0%} in VRAM)"))
' "$MODEL")
info "model placement: ${placement:-unknown}"
if [[ "$DEVICE" == "gpu" && "$placement" != "gpu" ]]; then
    warn "the model is not fully on the GPU (not enough free VRAM?); inference will be slower"
fi

# --- admin token (session only, unless .env already defines one) --------------------------
if [[ -z "${SENTINEL_ADMIN_TOKEN:-}" ]] && ! grep -qs '^SENTINEL_ADMIN_TOKEN=.\+' .env; then
    SENTINEL_ADMIN_TOKEN="$(openssl rand -hex 16 2>/dev/null || uv run --quiet python -c 'import secrets; print(secrets.token_hex(16))')"
    export SENTINEL_ADMIN_TOKEN
    GENERATED_TOKEN=1
fi

# --- gateway + UI -------------------------------------------------------------------------
export SENTINEL_MODEL_BACKEND=ollama SENTINEL_MODEL_NAME="$MODEL" SENTINEL_MODEL_BASE_URL="$OLLAMA_URL"
export SENTINEL_INFERENCE_DEVICE="$DEVICE" SENTINEL_API_URL="http://127.0.0.1:$API_PORT"

info "starting the API on http://127.0.0.1:$API_PORT (log: $LOG_DIR/api.log)"
setsid uv run sentinelai serve --port "$API_PORT" >"$LOG_DIR/api.log" 2>&1 &
PIDS+=($!)
wait_for "http://127.0.0.1:$API_PORT/health" 90 || die "API did not start; see $LOG_DIR/api.log"

info "starting the UI on http://127.0.0.1:$UI_PORT (log: $LOG_DIR/ui.log)"
setsid uv run sentinelai ui --port "$UI_PORT" >"$LOG_DIR/ui.log" 2>&1 &
PIDS+=($!)
wait_for "http://127.0.0.1:$UI_PORT/_stcore/health" 60 || die "UI did not start; see $LOG_DIR/ui.log"

DASHBOARD="http://127.0.0.1:$UI_PORT/dashboard"
echo
info "SentinelAI is running"
echo "    Dashboard : $DASHBOARD"
echo "    Chat      : http://127.0.0.1:$UI_PORT/chat"
echo "    API docs  : http://127.0.0.1:$API_PORT/docs"
echo "    Model     : $MODEL on $DEVICE (${placement:-unknown})"
if [[ "${GENERATED_TOKEN:-0}" == 1 ]]; then
    echo "    Admin     : $SENTINEL_ADMIN_TOKEN  (session token; set SENTINEL_ADMIN_TOKEN in .env to keep one)"
fi
echo "    Logs      : $LOG_DIR/"
echo
open_dashboard "$DASHBOARD"
info "press Ctrl+C to stop"

# Exit (and clean up) as soon as either server dies.
wait -n "${PIDS[@]}" || true
warn "a server exited unexpectedly; see $LOG_DIR/"
exit 1
