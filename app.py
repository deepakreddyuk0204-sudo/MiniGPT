"""Local chat-style playground for a MiniGPT continuation model."""

import os
from pathlib import Path
import pickle

import streamlit as st

from minigpt.engine import load_checkpoint
from minigpt.generate import continuation
from minigpt.tokenizer import ByteTokenizer


@st.cache_resource(max_entries=2, show_spinner="Loading MiniGPT…")
def load_for_app(path: str, revision: tuple[int, int, int]):
    """The file revision in the cache key picks up newly saved weights."""
    model, payload = load_checkpoint(path, "cpu")
    if model.config.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("The checkpoint needs a 256-token UTF-8 byte vocabulary.")
    return model, payload["step"]


st.set_page_config(page_title="MiniGPT", page_icon="📖", layout="centered")
st.title("MiniGPT")
st.write(
    "Write the start of a passage and see how MiniGPT continues it. "
    "This experimental model learned from ten chapters of "
    "*Alice’s Adventures in Wonderland*. Its output may be incoherent."
)

default_path = Path(__file__).parent / "runs/alice/last.pt"
checkpoint = Path(os.environ.get("MINIGPT_CHECKPOINT", default_path)).expanduser()
if not checkpoint.is_file():
    st.info("Prepare the book and train a checkpoint locally to open the playground.")
    st.code("python -m minigpt.data\npython -m minigpt.train fit", language="bash")
    st.caption(f"Expected checkpoint at {checkpoint}")
    st.stop()

try:
    stat = checkpoint.stat()
    model, step = load_for_app(str(checkpoint.resolve()), (stat.st_ino, stat.st_mtime_ns, stat.st_size))
except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as error:
    st.error(f"Could not load the local checkpoint. {error}")
    st.stop()

with st.sidebar:
    st.header("Sampling")
    temperature = st.slider(
        "Temperature", 0.0, 2.0, 0.8, 0.1,
        help="Lower values favour likely bytes. Zero always takes the most likely byte.",
    )
    top_k = st.slider(
        "Top-k", 1, ByteTokenizer.vocab_size, 20,
        help="At each step, choose among this many of the most likely next bytes.",
    )
    length = st.slider("New bytes", 8, 256, 96, 8)
    seed = st.number_input(
        "Seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
        help="Repeat the same prompt and settings with this seed to reproduce a sample.",
    )
    st.caption(f"Checkpoint at training step {step:,} · CPU inference")

st.caption(
    f"Each message starts a fresh continuation. "
    f"At each step MiniGPT sees the most recent {model.config.context_length} bytes."
)

history = st.session_state.setdefault("history", [])
for role, content in history:
    with st.chat_message(role):
        st.text(content)

prompt = st.chat_input("Start a passage, e.g. Alice opened the door")
if prompt:
    with st.chat_message("user"):
        st.text(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating…"):
            reply = continuation(model, prompt, length, temperature, top_k, seed)
        st.text(reply)
    history.extend([("user", prompt), ("assistant", reply)])
