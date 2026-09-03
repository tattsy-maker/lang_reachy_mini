#!/usr/bin/env bash
# One-command booth startup (T11): robot serve + voice agent in session
# mode, pinned zenoh, preflight checklist, clean SIGINT shutdown with the
# end-of-day guest wipe.
#
#   ./start_booth.sh                 # the booth: session mode, Haiku
#   BOOTH_MODEL=claude-opus-5 ./start_booth.sh    # home mode: Opus
#   BOOTH_KEEP_GUESTS=1 ./start_booth.sh          # skip the wipe on exit
#
# Overridable knobs (env vars):
#   BOOTH_MODEL         LLM for the tutor (default claude-haiku-4-5-20251001;
#                       spec 9: booth runs Haiku for pace, home runs Opus)
#   BOOTH_AUDIO_DEVICE  substring of the mic/speaker device (default the
#                       robot's own; T11 mic test decides the handheld name)
#   BOOTH_FACE_SOURCE   camera for face recognition (default 0 = /dev/video0)
#   BOOTH_ABSENT_SECS   walk-away timer (default 60; "still there?" at 2/3)
#   BOOTH_ATTRACT_SECS  idle attractor: nobody in frame this long -> a short
#                       dance every few minutes (default 120; 0 = off)
#   BOOTH_PERSONA       booth (default: gentle quips + wishlist question)
#                       or plain
#   BOOTH_EXTRA_AGENT   extra flags appended to the agent command
#
# Startup takes ~40s (robot connect ~15s, model warmup ~10s) -- start it
# before the doors open. A cold start can hit the documented serve/daemon
# race; this script already retries once.
set -uo pipefail
cd "$(dirname "$0")"

MODEL="${BOOTH_MODEL:-claude-haiku-4-5-20251001}"
AUDIO_DEVICE="${BOOTH_AUDIO_DEVICE:-Reachy Mini Audio}"
FACE_SOURCE="${BOOTH_FACE_SOURCE:-0}"
ABSENT_SECS="${BOOTH_ABSENT_SECS:-60}"
ATTRACT_SECS="${BOOTH_ATTRACT_SECS:-120}"
PERSONA="${BOOTH_PERSONA:-booth}"
ZENOH_LISTEN="tcp/0.0.0.0:7447"
BROKER="zenoh://127.0.0.1:7447"
SERVE_LOG="serve.log"
AGENT_LOG="voice/run.log"

say() { printf '%s\n' "$*"; }
ok()  { printf '  [ok]   %s\n' "$*"; }
warn(){ printf '  [WARN] %s\n' "$*"; }
die() { printf '  [FAIL] %s\n' "$*"; exit 1; }

say "== Reachy language tutor: booth startup =="
say "-- preflight --"

[ -e /dev/ttyACM0 ] && ok "robot serial present (/dev/ttyACM0)" \
    || die "no /dev/ttyACM0 -- is the robot's USB plugged in?"
[ -x .venv/bin/python ] && ok "robot venv" || die "./.venv missing (CLAUDE.md setup)"
[ -x voice/.venv/bin/python ] && ok "voice venv" || die "voice/.venv missing"
grep -qs "ANTHROPIC_API_KEY=" voice/.env && ok "Anthropic key in voice/.env" \
    || die "no ANTHROPIC_API_KEY in voice/.env"
grep -qs . /proc/asound/cards && ok "sound card visible" \
    || die "no sound card (audio group membership? see CLAUDE.md)"

SESSION_FLAGS=(--session --face-source "$FACE_SOURCE"
               --absent-secs "$ABSENT_SECS" --attract-secs "$ATTRACT_SECS")
if [ -r "/dev/video$FACE_SOURCE" ] 2>/dev/null || [ -r "$FACE_SOURCE" ]; then
    ok "camera readable (face source $FACE_SOURCE)"
else
    warn "camera not readable (video group? sudo usermod -aG video \$USER)"
    warn "starting WITHOUT face recognition -- no greeting by name"
    SESSION_FLAGS=()
fi
if [ -f voice/piper_voices/ru_RU-irina-medium.onnx ]; then
    ok "Piper voices present (Russian/Mandarin available)"
else
    warn "no Piper voices -- ru/zh disabled (voice/piper_tts.py has the fetch)"
fi
# Recorded moves (dances, emotions) come from two HuggingFace datasets.
# Fetch them now, while there is time, never at the moment someone says
# "can you dance?" -- and never let venue internet block the start.
if .venv/bin/python moves.py --cached >/dev/null 2>&1; then
    ok "recorded moves cached (dances + emotions)"
else
    warn "recorded moves not cached; fetching (60s cap) ..."
    if timeout 60 .venv/bin/python moves.py --preload >/dev/null 2>&1 \
       && .venv/bin/python moves.py --cached >/dev/null 2>&1; then
        ok "recorded moves fetched"
    else
        warn "recorded moves unavailable -- 'perform' will fail except spin/wiggle"
    fi
fi

say "-- starting robot serve (zenoh $ZENOH_LISTEN) --"
# Both logs are append-only (runbook), so a grep over the whole file would
# match a PREVIOUS run's readiness lines. Only read lines from this run.
SERVE_OFFSET=$(wc -l < "$SERVE_LOG" 2>/dev/null || echo 0)
AGENT_OFFSET=$(wc -l < "$AGENT_LOG" 2>/dev/null || echo 0)
fresh() { tail -n "+$(($2 + 1))" "$1" 2>/dev/null; }

start_serve() {
    .venv/bin/python controller.py serve --zenoh-listen "$ZENOH_LISTEN" \
        >> "$SERVE_LOG" 2>&1 &
    SERVE_PID=$!
}
start_serve
for i in $(seq 1 30); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        # The documented cold-start race: serve gave up before the vendor
        # daemon finished its ~15s motor configuration. Run it again.
        warn "serve exited early (daemon race) -- retrying once"
        sleep 3
        start_serve
    fi
    fresh "$SERVE_LOG" "$SERVE_OFFSET" | grep -q "\[reachy\] serving" && break
    sleep 1
done
kill -0 "$SERVE_PID" 2>/dev/null || die "serve did not stay up; tail $SERVE_LOG"
ok "serve up (pid $SERVE_PID)"

say "-- starting voice agent (model $MODEL) --"
say "   warmup is ~40s cold, seconds when the daemon is warm"
# Directly exec python (no subshell): the shutdown trap must SIGINT the
# real agent process, not a wrapper that would swallow the signal.
voice/.venv/bin/python voice/agent.py \
    --broker "$BROKER" \
    --model "$MODEL" \
    --audio-device "$AUDIO_DEVICE" \
    --persona "$PERSONA" \
    "${SESSION_FLAGS[@]}" \
    ${BOOTH_EXTRA_AGENT:-} \
    >> "$AGENT_LOG" 2>&1 &
AGENT_PID=$!

# SIGINT then wait, with a bounded patience -- never SIGKILL first
# (cleanup returns the robot to neutral and hands media back).
stop() { # pid, name, seconds
    kill -INT "$1" 2>/dev/null || return 0
    for i in $(seq 1 "$3"); do
        kill -0 "$1" 2>/dev/null || return 0
        sleep 1
    done
    warn "$2 ignored SIGINT for $3 s; killing it"
    kill -9 "$1" 2>/dev/null
}

cleanup() {
    trap - EXIT INT TERM
    say ""
    say "-- shutting down (SIGINT everywhere; exit code 1 afterwards is normal) --"
    stop "$AGENT_PID" "agent" 25
    stop "$SERVE_PID" "serve" 15
    if [ -z "${BOOTH_KEEP_GUESTS:-}" ]; then
        say "-- end of day: wiping guest profiles (family survives) --"
        python3 tutor/wipe_guests.py || true
    else
        say "-- BOOTH_KEEP_GUESTS set: guests kept --"
    fi
    say "== booth down =="
}
trap cleanup EXIT INT TERM

for i in $(seq 1 90); do
    fresh "$AGENT_LOG" "$AGENT_OFFSET" | grep -q "ready -- say something" && break
    kill -0 "$AGENT_PID" 2>/dev/null || die "agent died; tail $AGENT_LOG"
    sleep 1
done
fresh "$AGENT_LOG" "$AGENT_OFFSET" | grep -q "ready -- say something" \
    && ok "agent ready -- the booth is live" \
    || die "agent never reached ready; tail $AGENT_LOG"

say ""
say "Booth checklist:"
say "  * robot at neutral, antennas up?"
say "  * walk up: greeted (by name if enrolled) within ~3s of a stable face?"
say "  * step left/right: head and body follow? 'can you dance?' -> a dance?"
say "  * walk-away: 'still there?' at ${ABSENT_SECS}*2/3 s, goodbye + notes at ${ABSENT_SECS}s"
say "  * demo insurance: kill and rerun with BOOTH_EXTRA_AGENT='--say \"...\"'"
say "  * signage up (booth/SIGNAGE.md), one chair, one mic"
say "Ctrl-C stops everything and wipes guest profiles."
wait "$AGENT_PID"
