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
