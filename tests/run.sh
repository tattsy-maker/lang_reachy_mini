#!/usr/bin/env bash
# Test entry point for the language-tutor tasks (see TASKS.md).
#
#   tests/run.sh          # run every task's tests
#   tests/run.sh t4       # run one task's tests (tests/t4/)
#   tests/run.sh t0 -k stub   # extra args go straight to pytest
#
# Creates tests/.venv on first use (pytest + OpenCV for fixture handling).
# Tests that need hardware, big models, a sound card, or an API key skip
# with a printed reason when the prerequisite is absent -- see conftest.py.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=tests/.venv
if [ ! -x "$VENV/bin/python" ]; then
    echo "[tests] creating $VENV ..."
    python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c "import pytest, cv2, numpy" >/dev/null 2>&1; then
    echo "[tests] installing test dependencies into $VENV ..."
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q pytest opencv-python-headless numpy
fi

TARGET=tests
if [ $# -ge 1 ] && [[ "$1" != -* ]]; then
    id=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    TARGET="tests/$id"
    if [ ! -d "$TARGET" ]; then
        echo "[tests] no tests for task '$1' (expected directory $TARGET/)" >&2
        exit 2
    fi
    shift
fi

exec "$VENV/bin/python" -m pytest "$TARGET" "$@"
