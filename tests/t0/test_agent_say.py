"""T0's end-to-end demo: one `--say` turn through the real voice agent.

Injects a typed utterance (no microphone), lets the full local pipeline run
(Claude reply -> Kokoro synthesis -> speaker), and asserts from the log that
a reply was actually synthesized. Marked with everything it truly needs:
an Anthropic key, the big local models, and a sound card.
"""

import pytest


@pytest.mark.anthropic
@pytest.mark.models
@pytest.mark.audio
def test_say_hello_synthesizes_a_reply(run_agent_say):
    log, found = run_agent_say(
        ["Hello! Please answer with one short sentence."])
    assert "injecting utterance 1/1" in log, (
        "the --say injection never happened; agent log:\n" + log[-4000:])
    assert found, (
        "no 'Generating TTS' line — no reply reached synthesis; log tail:\n"
        + log[-4000:])
