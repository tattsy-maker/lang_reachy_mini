# Test suite

Integration tests for the language-tutor tasks in [TASKS.md](../TASKS.md).

## Running

```bash
tests/run.sh          # everything
tests/run.sh t0       # one task's tests (directory tests/t0/)
tests/run.sh t0 -k stub -x   # extra args pass through to pytest
```

First run creates `tests/.venv` (pytest + opencv-python-headless + numpy).
The suite must be green on any machine: tests that need something this
machine lacks **skip with a printed reason** rather than fail.

## Layout convention

One directory per task: `tests/t0/`, `tests/t1/`, ... Each task lands its
tests in its own directory so `tests/run.sh <task-id>` selects them.
Shared fixtures live in [conftest.py](conftest.py) (the `paths`,
`stub_robot`, and `run_agent_say` fixtures) and
[fixtures/](fixtures/README.md) (face photos, video clip, sample learner
tree). Measurement scripts (T5/T7 round-trip gates) write their reports to
`tests/reports/`.

## Markers

Declared in [pytest.ini](pytest.ini), enforced in
[conftest.py](conftest.py):

| Marker | Needs | Skip condition |
|---|---|---|
| `robot` | the physical robot | no `/dev/ttyACM*` |
| `models` | big local models (Whisper, Kokoro, ...) | no `voice/.venv` |
| `anthropic` | an Anthropic key | no `ANTHROPIC_API_KEY` in env or `voice/.env` |
| `google` | a Google key | no `GOOGLE_API_KEY` in env or `voice/.env` |
| `audio` | a sound card | `/proc/asound/cards` empty or `/dev/snd` inaccessible |

Unmarked tests must run anywhere. A test may carry several markers; it
skips if any prerequisite is missing.

Two suite-wide conventions from CLAUDE.md are baked into the fixtures:
subprocesses are always stopped with **SIGINT** (exit code 1 afterwards is
normal — never assert on it), and the test zenoh address is
`tcp/127.0.0.1:7557`, not the runbook's 7447, so a live `serve` and the
suite never fight over a port.
