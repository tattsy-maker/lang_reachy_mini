#!/usr/bin/env bash
# Restart just the voice agent by hand. Since T14.3 the booth loop swaps
# visitors by itself (a new face, a walk-away), so this is only for
# recovering a wedged agent. serve and the vendor daemon stay up, so it
# takes ~6 s. Stand in front of the camera while it starts: it looks for
# a face once, in the first ~3 s.
#
#   booth/next_visitor.sh            # cloud voice, booth persona
#   booth/next_visitor.sh --say "…"  # extra agent flags pass through
set -uo pipefail
cd "$(dirname "$0")/.."
pkill -INT -f "voice/agent.py" 2>/dev/null && for i in $(seq 1 25); do
    pgrep -f "voice/agent.py" >/dev/null || break; sleep 1; done
OFF=$(wc -l < voice/run.log 2>/dev/null || echo 0)
setsid nohup voice/.venv/bin/python voice/agent.py --broker zenoh://127.0.0.1:7447 \
    --speech cloud --face-source "${BOOTH_FACE_SOURCE:-0}" \
    --audio-device "${BOOTH_AUDIO_DEVICE:-Reachy Mini Audio}" \
    --persona "${BOOTH_PERSONA:-booth}" "$@" >> voice/run.log 2>&1 < /dev/null &
for i in $(seq 1 120); do
    tail -n +$((OFF+1)) voice/run.log | grep -q "ready -- say something" && break
    pgrep -f "voice/agent.py" >/dev/null || { echo "agent died; tail voice/run.log"; exit 1; }
    sleep 1
done
tail -n +$((OFF+1)) voice/run.log | grep -E "agent: face:|voice id|ready --" | cut -c1-120
