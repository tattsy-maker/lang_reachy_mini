"""T8 addendum: Gemini Live's native Google Search grounding (on by
default in cloud mode, --no-web-search to disable)."""

import re
import subprocess

import pytest

@pytest.mark.google
@pytest.mark.audio
def test_cloud_mode_has_google_search_grounding(run_agent_say):
    """Gemini Live's native Google Search tool is on by default in cloud
    mode: the setup carries it, and a question that needs a live fact
    produces a grounded answer (the SearchLogger line)."""
    log, found = run_agent_say(
        ["Please look this up on the web: what is the weather in Madrid "
         "right now? Answer in one sentence."],
        wait_for=r"web search: (\d+ sources|unavailable on this key)",
        # after a fallback the session must still talk
        also_wait_for=r"Bot started speaking",
        extra_args=["--speech", "cloud"],
        settle=6, timeout=240)
    unavailable = re.search(r"web search: unavailable on this key \((.*?)\)", log)
    if unavailable:
        # The tool is wired correctly but this key cannot use it (free
        # tier); that is a billing fact, not a code failure -- as long
        # as the robot still converses without it.
        assert found, "after the search fallback the session never spoke:\n" + log[-3000:]
        pytest.skip("Google Search grounding not available on this key: "
                    + unavailable.group(1))
    assert "Google Search grounding enabled" in log
    assert re.search(r"Setting tools: .*google_search", log), \
        "google_search never reached the Live setup:\n" + log[-3000:]
    assert found, "no grounded answer came back; log tail:\n" + log[-4000:]


@pytest.mark.google
def test_cloud_mode_web_search_can_be_switched_off(paths):
    out = subprocess.run(
        [str(paths.voice_py), "agent.py", "--speech", "cloud", "--no-robot",
         "--no-web-search", "--help"],
        cwd=paths.voice_dir, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--no-web-search" in out.stdout
