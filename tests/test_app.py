from pathlib import Path

from streamlit.testing.v1 import AppTest
import torch

from minigpt.engine import make_optimizer, save_checkpoint
from minigpt.generate import continuation
from minigpt.model import GPT, GPTConfig


APP = Path(__file__).resolve().parents[1] / "app.py"


def write_fixture(path, context_length):
    config = GPTConfig(context_length=context_length, n_layer=1, n_head=2, n_embd=16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        model = GPT(config)
        save_checkpoint(path, model, make_optimizer(model, 0.001), 1,
                        torch.Generator().manual_seed(4), "fixture", {"lr": 0.001})
    return model.eval()


def test_missing_checkpoint_explains_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIGPT_CHECKPOINT", str(tmp_path / "missing.pt"))
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert "train a checkpoint" in app.info[0].value
    assert "python -m minigpt.train fit" in app.code[0].value
    assert not app.chat_input


def test_submit_seed_controls_and_refresh_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    model = write_fixture(path, 12)
    monkeypatch.setenv("MINIGPT_CHECKPOINT", str(path))
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception
    assert any("incoherent" in paragraph.value for paragraph in app.markdown)

    temperature, top_k, length = app.sidebar.slider
    temperature.set_value(0.5)
    top_k.set_value(4)
    length.set_value(8)
    app.sidebar.number_input[0].set_value(17)
    for prompt in ("One day", "Alice", "One day"):
        app.chat_input[0].set_value(prompt).run()
        assert not app.exception
        expected = continuation(model, prompt, 8, 0.5, 4, 17)
        assert app.session_state["history"][-2:] == [("user", prompt), ("assistant", expected)]
        assert app.chat_message[-1].text[0].value == expected.strip()
    assert len(app.chat_message) == 6
    before = app.session_state["history"][1][1]
    assert before == app.session_state["history"][-1][1]

    write_fixture(path, 16)
    app.run()
    assert not app.exception
    assert any("16 bytes" in caption.value for caption in app.caption)


def local_assistant(monkeypatch, stream, *, online=True):
    from types import SimpleNamespace
    import minigpt

    class LocalChatError(Exception):
        pass

    monkeypatch.delenv("MINIGPT_CHECKPOINT", raising=False)
    backend = SimpleNamespace(ready=lambda: online, stream_reply=stream, LocalChatError=LocalChatError)
    monkeypatch.setattr(minigpt, "local_chat", backend, raising=False)
    return backend, AppTest.from_file(APP, default_timeout=20).run()


def test_general_chat_offline_shows_local_launcher(monkeypatch):
    _, app = local_assistant(monkeypatch, None, online=False)
    assert not app.exception
    assert app.sidebar.radio[0].value == "General chat"
    assert "scripts/start_chat.py" in app.code[0].value
    assert app.chat_input[0].disabled


def test_general_chat_streams_plain_text_and_keeps_turns(monkeypatch):
    calls = []
    picture = "![external](https://example.invalid/track)"
    def stream(messages):
        calls.append(messages)
        yield "Hello "
        yield picture

    _, app = local_assistant(monkeypatch, stream)
    for prompt in ("first turn", "follow up"):
        app.chat_input[0].set_value(prompt).run()
        assert not app.exception
    previous_reply = {"role": "assistant", "content": "Hello " + picture}
    first = {"role": "user", "content": "first turn"}
    second = {"role": "user", "content": "follow up"}
    assert calls == [[first], [first, previous_reply, second]]
    assert app.session_state["chat_messages"] == [first, previous_reply, second, previous_reply]
    assert len(app.chat_message) == 4
    assert app.chat_message[-1].text[0].value == previous_reply["content"]
    assert not app.chat_message[-1].markdown
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert not app.chat_message
    assert not app.session_state["chat_messages"]


def test_partial_stream_failure_can_be_retried_without_duplicate_history(monkeypatch):
    calls = []
    def stream(messages):
        calls.append(messages)
        yield "a partial answer"
        if len(calls) == 1:
            raise backend.LocalChatError("Connection interrupted")
        yield " with the ending"

    backend, app = local_assistant(monkeypatch, stream)
    app.chat_input[0].set_value("Please continue").run()
    assert not app.exception
    assert not app.session_state["chat_messages"]
    assert app.session_state["chat_pending"]["partial"] == "a partial answer"
    assert app.error
    next(button for button in app.button if button.label == "Retry reply").click().run()
    assert not app.exception
    assert calls[0] == calls[1] == [{"role": "user", "content": "Please continue"}]
    assert len(app.session_state["chat_messages"]) == 2
    assert app.chat_message[-1].text[0].value == "a partial answer with the ending"
