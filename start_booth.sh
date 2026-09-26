#!/usr/bin/env bash
# One-command booth startup (T11): robot serve + voice agent in session
# mode, pinned zenoh, preflight checklist, clean SIGINT shutdown with the
# end-of-day guest wipe.
#
#   ./start_booth.sh                 # the booth: cloud voice, session mode
#   BOOTH_SPEECH=local BOOTH_MODEL=claude-opus-5 ./start_booth.sh  # local
#   BOOTH_KEEP_GUESTS=1 ./start_booth.sh          # skip the wipe on exit
#
# Overridable knobs (env vars):
#   BOOTH_SPEECH        cloud (default since the family chose the Gemini
#                       voice, T14.3) or local (Whisper + Claude + Kokoro)
#   BOOTH_MODEL         LLM for the tutor in local mode (default
#                       claude-haiku-4-5-20251001; home runs Opus)
#   BOOTH_MIC_DEVICE    substring of the preferred USB microphone (default
#                       "USB Composite Device", the Jieli desk mic). When no
#                       such device is plugged in, the robot's built-in mic
#                       is used; the agent log says which one it opened.
#   BOOTH_AUDIO_DEVICE  substring of the robot's speaker device (default
#                       "Reachy Mini Audio"); also the fallback mic
#   BOOTH_FACE_SOURCE   camera for face recognition (default 0 = /dev/video0)
#   BOOTH_ABSENT_SECS   walk-away timer (default 20 since 2026-09-25, was 60;
#                       "still there?" at 2/3)
#   BOOTH_ATTRACT_SECS  idle attractor: nobody in frame this long -> a
#                       dance (default 5; 0 = off), then the next one
#   BOOTH_ATTRACT_EVERY seconds after the last one started, or as soon as
#                       it ends (default 12: mostly dancing when idle)
#   BOOTH_IDLE_GLANCE_SECS  between dances, a small random head glance
#                       about this often (default 4; 0 = off)
#   BOOTH_BARGE_IN      cloud: 1 (default) = a visitor can interrupt the
#                       robot by talking over it; 0 = the robot always
#                       finishes (the pre-2026-09-25 mute)
#   BOOTH_BARGE_IN_MARGIN_DB  how far above the robot's own echo a voice
#                       must be to interrupt (default 10; grep 'barge-in:')
#   BOOTH_ONBOARDING    quick (default: "which language, what level?" and
#                       the lesson starts, nothing stored unless they ask
#                       to be remembered) or full (the T17.4 interview)
#   BOOTH_PERSONA       booth (default: gentle quips + wishlist question)
#                       or plain
#   BOOTH_NATIVE_LANGUAGE  local speech only: the language the robot
#                       explains in by default (the voice for untagged
#                       text), e.g. ru; unset = en. Cloud mode reads each
#                       learner's own language from their profile.
#   BOOTH_TURN_PATIENCE_MS  cloud: silence before Gemini takes the turn
#                       (default 1200 since 2026-09-25, was 1800 from T17.1;
#                       0 = Gemini's default). Every reply starts that much
#                       later; a thinking pause is not the end of a
#                       sentence, but at the Faire visitors waited "a few
#                       seconds too long" after saying a word back.
#   BOOTH_TURN_ONSET_MS cloud: speech needed before Gemini notices the
#                       visitor started (default 100, was 300: a quick
#                       "hola" went unheard; raise it if hall noise
#                       starts turns)
#   BOOTH_CALL_OUT_SECS nobody at the robot but someone watching from a
#                       few steps away: invite them over out loud, at
#                       most this often (default 40; 0 = off)
#   BOOTH_SPEECH_RATE   play the robot's speech at this speed, pitch kept
#                       (default 1.0 = off; 0.85 = fifteen percent slower,
#                       T17.10 -- try the prompt first, read 'pace:' lines)
#   BOOTH_VOLUME        speaker volume at startup, percent (default 100,
#                       the mixer's top). set_volume changes it during a
#                       visit; every start puts it back here.
#   BOOTH_LOUDNESS_DB   louder than the mixer can go: soft-clip drive on
#                       the robot's speech (default 0 = off since
#                       2026-09-25, evening: a bigger speaker does the
#                       volume; 12 = about 8 dB louder on the robot's own
#                       speaker, 9 = about 6.5; voice/loudness.py).
#   BOOTH_SPEAKER_DEVICE  where the robot's voice plays: auto (default) =
#                       any other USB sound card that can play, e.g. a
#                       USB-to-jack adapter for a bigger speaker, else the
#                       robot's own; or name substrings, comma-separated;
#                       '' = always the robot's. Only the robot's own
#                       speaker has its echo cancelled at the robot's mic,
#                       so barge-in mostly stops working on another one.
#   BOOTH_EXTRA_AGENT   extra flags appended to the agent command
#   BOOTH_SERVICE       1 = unattended (systemd, booth/reachy-booth.service):
#                       wait for the robot, camera and network instead of
#                       failing; once up, exit on any fault (USB unplugged,
#                       serve or agent gone, agent heartbeat stale, Gemini
#                       lost) so systemd restarts the stack from cold; stop
#                       the vendor daemon too; wipe guests only on a real
#                       stop, never on a fault restart.
#
# Startup takes ~40s (robot connect ~15s, model warmup ~10s) -- start it
# before the doors open. A cold start can hit the documented serve/daemon
# race; this script already retries once.
set -uo pipefail
cd "$(dirname "$0")"
# serve's "[reachy] serving" line is the readiness signal below; block
# buffering into the log hid it until exit.
export PYTHONUNBUFFERED=1

SPEECH="${BOOTH_SPEECH:-cloud}"
MODEL="${BOOTH_MODEL:-claude-haiku-4-5-20251001}"
AUDIO_DEVICE="${BOOTH_AUDIO_DEVICE:-Reachy Mini Audio}"
MIC_DEVICE="${BOOTH_MIC_DEVICE:-USB Composite Device}"
FACE_SOURCE="${BOOTH_FACE_SOURCE:-0}"
ABSENT_SECS="${BOOTH_ABSENT_SECS:-20}"
ATTRACT_SECS="${BOOTH_ATTRACT_SECS:-5}"
ATTRACT_EVERY="${BOOTH_ATTRACT_EVERY:-12}"
IDLE_GLANCE_SECS="${BOOTH_IDLE_GLANCE_SECS:-4}"
BARGE_IN="${BOOTH_BARGE_IN:-1}"
BARGE_IN_MARGIN_DB="${BOOTH_BARGE_IN_MARGIN_DB:-10}"
ONBOARDING="${BOOTH_ONBOARDING:-quick}"
PERSONA="${BOOTH_PERSONA:-booth}"
NATIVE_LANGUAGE="${BOOTH_NATIVE_LANGUAGE:-}"
TURN_PATIENCE_MS="${BOOTH_TURN_PATIENCE_MS:-1200}"
TURN_ONSET_MS="${BOOTH_TURN_ONSET_MS:-100}"
CALL_OUT_SECS="${BOOTH_CALL_OUT_SECS:-40}"
SPEECH_RATE="${BOOTH_SPEECH_RATE:-1.0}"
LOUDNESS_DB="${BOOTH_LOUDNESS_DB:-0}"
SPEAKER_DEVICE="${BOOTH_SPEAKER_DEVICE-auto}"
ZENOH_LISTEN="tcp/0.0.0.0:7447"
BROKER="zenoh://127.0.0.1:7447"
SERVICE="${BOOTH_SERVICE:-}"
HEARTBEAT="voice/.heartbeat"
HEARTBEAT_STALE=60        # seconds; the agent touches it every 5
ROBOT_SERIAL="/dev/serial/by-id/usb-1a86_USB_Single_Serial_*"
ROBOT_CAMERA="/dev/v4l/by-id/usb-SunplusIT_Inc_Reachy_Mini_Camera_*-video-index0"
SERVE_LOG="serve.log"
AGENT_LOG="voice/run.log"

say() { printf '%s\n' "$*"; }
ok()  { printf '  [ok]   %s\n' "$*"; }
warn(){ printf '  [WARN] %s\n' "$*"; }
die() { printf '  [FAIL] %s\n' "$*"; exit 1; }

say "== Reachy language tutor: booth startup$( [ -n "$SERVICE" ] && echo " (service mode)") =="

robot_present() { # the robot's serial bus and its sound card, both on its USB
    compgen -G "$ROBOT_SERIAL" >/dev/null \
        && grep -qs -- "$AUDIO_DEVICE" /proc/asound/cards
}
gemini_reachable() {
    timeout 5 bash -c '</dev/tcp/generativelanguage.googleapis.com/443' 2>/dev/null
}
# Service mode waits for what can come back by itself (a replugged robot,
# the venue wifi) instead of failing and being restarted every 5 s.
wait_for() { # description, test command...
    local what="$1"; shift
    "$@" && return 0
    say "  [wait] $what"
    until "$@"; do sleep 3; done
    ok "$what: back"
}
if [ -n "$SERVICE" ]; then
    wait_for "robot USB (serial + Reachy Mini Audio)" robot_present
    # Give the camera a moment to enumerate after a replug; a replug can
    # also move it, so follow the by-id link to its current index.
    for i in $(seq 1 10); do compgen -G "$ROBOT_CAMERA" >/dev/null && break; sleep 1; done
    cam=$(readlink -f $ROBOT_CAMERA 2>/dev/null | head -n1)
    [ -z "${BOOTH_FACE_SOURCE:-}" ] && [ -n "$cam" ] && FACE_SOURCE="${cam#/dev/video}"
    [ "$SPEECH" = cloud ] && wait_for "network (Gemini reachable)" gemini_reachable
fi
say "-- preflight --"

compgen -G "$ROBOT_SERIAL" >/dev/null \
    && ok "robot serial present ($(readlink -f $ROBOT_SERIAL))" \
    || die "no robot serial -- is the robot's USB plugged in?"
[ -x .venv/bin/python ] && ok "robot venv" || die "./.venv missing (CLAUDE.md setup)"
[ -x voice/.venv/bin/python ] && ok "voice venv" || die "voice/.venv missing"
if [ "$SPEECH" = cloud ]; then
    grep -qs -E "(GEMINI|GOOGLE)_API_KEY=" voice/.env && ok "Gemini key in voice/.env" \
        || die "no GEMINI_API_KEY in voice/.env (BOOTH_SPEECH=local for Claude)"
else
    grep -qs "ANTHROPIC_API_KEY=" voice/.env && ok "Anthropic key in voice/.env" \
        || die "no ANTHROPIC_API_KEY in voice/.env"
fi
grep -qs . /proc/asound/cards && ok "sound card visible" \
    || die "no sound card (audio group membership? see CLAUDE.md)"
# The robot's speaker at a known level (2026-09-23: one visitor's "louder"
# or "quieter" used to carry over to every visitor after them).
VOLUME="${BOOTH_VOLUME:-100}"
# The card the agent will play through (voice/audio_devices.py makes the
# same pick): BOOTH_SPEAKER_DEVICE, else the robot's.
speaker_card() {
    local name
    if [ "$SPEAKER_DEVICE" = auto ]; then
        awk -F'[][]' '/USB-Audio/ {print $1}' /proc/asound/cards 2>/dev/null \
            | while read -r n; do
                line=$(grep -E "^ *$n \[" /proc/asound/cards)
                case "$line" in *"$AUDIO_DEVICE"*) continue ;; esac
                [ -n "$MIC_DEVICE" ] && case "$line" in *"$MIC_DEVICE"*) continue ;; esac
                ls /proc/asound/card"$n"/pcm*p >/dev/null 2>&1 && { echo "$n"; break; }
            done
    elif [ -n "$SPEAKER_DEVICE" ]; then
        echo "$SPEAKER_DEVICE" | tr ',' '\n' | while read -r name; do
            [ -n "$name" ] || continue
            n=$(awk -v d="$name" 'index($0, d) && /^ *[0-9]/ {print $1; exit}' /proc/asound/cards)
            [ -n "$n" ] && { echo "$n"; break; }
        done
    fi
}
card=$(speaker_card | head -n1)
if [ -n "$card" ]; then
    ok "speaker: $(awk -v n="$card" '$1 == n {sub(/.* - /, ""); print; exit}' /proc/asound/cards) (card $card) -- not the robot's, barge-in weaker"
else
    card=$(awk -v d="$AUDIO_DEVICE" 'index($0, d) {print $1; exit}' /proc/asound/cards)
fi
set_any=
for control in PCM,0 PCM,1 Speaker Headphone Master; do
    [ -n "$card" ] && amixer -c "$card" sset "$control" "$VOLUME%" >/dev/null 2>&1 \
        && set_any=1
done
if [ -n "$set_any" ]; then
    ok "speaker volume $VOLUME% (card $card)"
else
    warn "could not set the speaker volume"
fi
# USB mic if present, else the robot's own. /proc/asound/cards names the
# cards the same way PyAudio does, so this preview matches the agent's pick.
if [ -n "$MIC_DEVICE" ] && grep -qsi -- "$MIC_DEVICE" /proc/asound/cards; then
    ok "USB mic present ($MIC_DEVICE) -- using it"
else
    warn "no USB mic matching '$MIC_DEVICE' -- using the robot's built-in mic"
fi

SESSION_FLAGS=(--session --face-source "$FACE_SOURCE"
               --absent-secs "$ABSENT_SECS" --attract-secs "$ATTRACT_SECS"
               --attract-every "$ATTRACT_EVERY" --onboarding "$ONBOARDING"
               --idle-glance-secs "$IDLE_GLANCE_SECS"
               --call-out-secs "$CALL_OUT_SECS")
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

# The vendor daemon first, and only then serve. Left to serve, a cold
# start spawns the daemon and connects before it listens; the connect
# error is swallowed and serve hangs with no robot (2026-09-23; it used
# to exit with ConnectionRefused). Started here, serve finds it running.
DAEMON_STATUS="http://127.0.0.1:8000/api/daemon/status"
daemon_ready() {
    curl -s -m 3 "$DAEMON_STATUS" 2>/dev/null \
        | grep -q '"backend_status":{"ready":true.*"error":null},"error":null'
}
if daemon_ready; then
    ok "reachy daemon already running (warm start)"
else
    say "-- starting reachy daemon (~15 s cold) --"
    # --no-wake-up-on-start (2026-09-25): the vendor's wake-up ends with a
    # 20-degree head snap in 0.4 s. The agent wakes the robot slowly
    # itself (wake_gently in voice/agent.py).
    if [ -n "$SERVICE" ]; then
        .venv/bin/reachy-mini-daemon --no-wake-up-on-start >> "$SERVE_LOG" 2>&1 &
    else
        # Hand-started: the daemon outlives the booth, as the runbook says.
        setsid .venv/bin/reachy-mini-daemon --no-wake-up-on-start \
            >> "$SERVE_LOG" 2>&1 < /dev/null &
    fi
    for i in $(seq 1 60); do daemon_ready && break; sleep 1; done
    daemon_ready || die "reachy daemon not ready after 60 s; tail $SERVE_LOG"
    ok "reachy daemon ready"
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
# A cold daemon (first start, or every service-mode restart) spends ~15 s
# on the motors before serve can even connect; the agent only looks for
# the robot for 20 s, so do not start it until the device is on the bus.
# ("[reachy] serving" is printed before the driver has even connected.)
serving() { fresh "$SERVE_LOG" "$SERVE_OFFSET" | grep -aq "Subscribed to commands on .*\.cmd"; }
for i in $(seq 1 90); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        # The documented cold-start race: serve gave up before the vendor
        # daemon finished its ~15s motor configuration. Run it again.
        warn "serve exited early (daemon race) -- retrying once"
        sleep 3
        start_serve
    fi
    serving && break
    sleep 1
done
kill -0 "$SERVE_PID" 2>/dev/null || die "serve did not stay up; tail $SERVE_LOG"
serving || die "robot not on the bus after 90 s; tail $SERVE_LOG"
ok "serve up (pid $SERVE_PID)"

say "-- starting voice agent (speech $SPEECH$( [ "$SPEECH" = local ] && echo ", model $MODEL")) --"
say "   warmup is ~40s cold, seconds when the daemon is warm"
MODEL_FLAGS=(--speech "$SPEECH")
[ "$SPEECH" = local ] && MODEL_FLAGS+=(--model "$MODEL")
[ "$SPEECH" = local ] && [ -n "$NATIVE_LANGUAGE" ] \
    && MODEL_FLAGS+=(--native-language "$NATIVE_LANGUAGE")
MODEL_FLAGS+=(--turn-patience-ms "$TURN_PATIENCE_MS" --turn-onset-ms "$TURN_ONSET_MS"
             --speech-rate "$SPEECH_RATE"
             --loudness-db "$LOUDNESS_DB")
[ "$SPEECH" = cloud ] && [ "$BARGE_IN" = 1 ] \
    && MODEL_FLAGS+=(--barge-in --barge-in-margin-db "$BARGE_IN_MARGIN_DB")
# Directly exec python (no subshell): the shutdown trap must SIGINT the
# real agent process, not a wrapper that would swallow the signal.
voice/.venv/bin/python voice/agent.py \
    --broker "$BROKER" \
    "${MODEL_FLAGS[@]}" \
    --audio-device "$AUDIO_DEVICE" \
    --speaker-device "$SPEAKER_DEVICE" \
    --mic-device "$MIC_DEVICE" \
    --persona "$PERSONA" \
    "${SESSION_FLAGS[@]}" \
    ${SERVICE:+--heartbeat "$HEARTBEAT"} \
    ${BOOTH_EXTRA_AGENT:-} \
    >> "$AGENT_LOG" 2>&1 &
AGENT_PID=$!

# SIGINT then wait, with a bounded patience -- never SIGKILL first
# (cleanup returns the robot to neutral and hands media back).
stop() { # pid, name, seconds[, signal]
    kill -"${4:-INT}" "$1" 2>/dev/null || return 0
    for i in $(seq 1 "$3"); do
        kill -0 "$1" 2>/dev/null || return 0
        sleep 1
    done
    warn "$2 ignored SIGINT for $3 s; killing it"
    kill -9 "$1" 2>/dev/null
}

STOP_REQUESTED=""
cleanup() {
    trap - EXIT INT TERM
    say ""
    say "-- shutting down (SIGINT everywhere; exit code 1 afterwards is normal) --"
    stop "$AGENT_PID" "agent" 25
    stop "$SERVE_PID" "serve" 15
    if [ -n "$SERVICE" ]; then
        # The daemon lives in the unit's cgroup and systemd would SIGKILL
        # it anyway; stop it politely. After an unplug it holds a dead
        # serial handle, so a fresh one is the point of the restart.
        pid=$(pgrep -f "[r]eachy-mini-daemon" | head -n1)
        # Its shutdown lifts the head to init and lowers it to sleep. When
        # serve already laid it down, that is a second bow for nothing, so
        # stop the robot side without it first (2026-09-24). A serve that
        # died without parking leaves the full sleep to the daemon.
        if [ -n "$pid" ] && fresh "$SERVE_LOG" "$SERVE_OFFSET" \
                | grep -aq "asleep at rest, torque off"; then
            curl -s -m 3 -X POST "${DAEMON_STATUS%/status}/stop?goto_sleep=false" \
                > /dev/null 2>&1 || true
            for i in $(seq 1 10); do
                curl -s -m 2 "$DAEMON_STATUS" 2>/dev/null \
                    | grep -q '"state":"stopped"' && break
                sleep 1
            done
        fi
        # TERM: it may have inherited an ignored SIGINT (a background
        # job of this script), unlike serve and the agent it has no
        # handler of its own to override that.
        [ -n "$pid" ] && stop "$pid" "reachy-mini-daemon" 10 TERM
    fi
    if [ -n "$SERVICE" ] && [ -z "$STOP_REQUESTED" ]; then
        say "-- fault restart: guests kept --"
    elif [ -z "${BOOTH_KEEP_GUESTS:-}" ]; then
        say "-- end of day: wiping guest profiles (family survives) --"
        python3 tutor/wipe_guests.py || true
    else
        say "-- BOOTH_KEEP_GUESTS set: guests kept --"
    fi
    say "== booth down =="
}
trap cleanup EXIT
trap 'STOP_REQUESTED=1; exit 0' INT TERM

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
say "  * identity (T15): A enrolls, says bye -> notes + the wish question;"
say "    B takes A's seat mid-lesson without a word -> A's goodbye, B greeted"
say "    as new within ~5 s; A back after B's bye -> 'welcome back' + notes;"
say "    A speaks only the lesson language for a minute -> no voice challenge"
say "  * lag: grep 'turn: first sound' voice/run.log after a few exchanges"
say "  * T17: pause mid-sentence for 2 s -> no reply until you finish;"
say "    a silly voice after enrollment -> no 'you don't sound like yourself';"
say "    someone leans in from the next chair for 5 s -> 'face check: same';"
say "    intake asks name, language, own language + explain-in, level, goal,"
say "    minutes + today, corrections, one per turn (grep 'intake' $AGENT_LOG);"
say "    three mistakes in a row -> three recasts; 'give me a test' -> a spoken"
say "    blank and lettered options (grep 'quiz:'); 'turn your head' x3 ->"
say "    third one refused; goodbye -> closing answer in booth/feedback.md"
say "  * pace: grep 'pace:' $AGENT_LOG (wpm per reply); BOOTH_SPEECH_RATE=0.85 to slow"
say "  * demo insurance: kill and rerun with BOOTH_EXTRA_AGENT='--say \"...\"'"
say "  * signage up (booth/SIGNAGE.md), one chair, one mic (grep 'audio: mic' $AGENT_LOG)"
say "Ctrl-C stops everything and wipes guest profiles."

if [ -z "$SERVICE" ]; then
    wait "$AGENT_PID"
    exit
fi
# Service mode: watch everything and leave on the first fault; systemd
# brings the whole stack back (after the waits at the top).
touch "$HEARTBEAT"
while :; do
    sleep 2
    robot_present \
        || die "robot USB gone (unplugged or power lost) -- restarting cold"
    kill -0 "$SERVE_PID" 2>/dev/null || die "serve exited -- restarting"
    daemon_ready || { sleep 3; daemon_ready; } \
        || die "reachy daemon not ready ($(curl -s -m 3 "$DAEMON_STATUS" | head -c 300)) -- restarting"
    kill -0 "$AGENT_PID" 2>/dev/null \
        || die "agent exited ($(fresh "$AGENT_LOG" "$AGENT_OFFSET" | grep -o "watchdog: .*" | tail -n1)) -- restarting"
    age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0) ))
    [ "$age" -le "$HEARTBEAT_STALE" ] || die "agent heartbeat ${age}s old (hung) -- restarting"
done
