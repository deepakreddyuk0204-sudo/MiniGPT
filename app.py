"""MiniGPT's local assistant and from-scratch language model playground."""

import os
from pathlib import Path
import pickle

import streamlit as st


@st.cache_resource(max_entries=2, show_spinner="Loading MiniGPT…")
def load_for_app(path: str, revision: tuple[int, int, int]):
    """The file revision in the cache key picks up newly saved weights."""
    from minigpt.engine import load_checkpoint
    from minigpt.tokenizer import ByteTokenizer

    model, payload = load_checkpoint(path, "cpu")
    if model.config.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError("The checkpoint needs a 256-token UTF-8 byte vocabulary.")
    return model, payload["step"]


def playground():
    from minigpt.generate import continuation
    from minigpt.tokenizer import ByteTokenizer

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


def general_chat():
    from minigpt import local_chat

    st.title("Local assistant")
    st.caption("Qwen3-8B · Runs on this Mac · Uses recent turns for follow-up questions")

    messages = st.session_state.setdefault("chat_messages", [])
    if st.sidebar.button("New chat"):
        messages.clear()
        st.session_state.pop("chat_pending", None)

    for message in messages:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    connected = local_chat.ready()
    if not connected:
        st.info("Start the local model in the project folder, then check the connection.")
        st.code(".venv/bin/python scripts/start_chat.py", language="bash")
        st.caption("For a fresh checkout, follow the setup instructions in docs/LOCAL_CHAT.md.")
        st.button("Check connection")

    prompt = st.chat_input("Ask a question or follow up", key="assistant_prompt", disabled=not connected)
    if prompt and prompt.strip():
        st.session_state.chat_pending = {"prompt": prompt}

    # Only complete exchanges enter model history. A failed stream can be retried
    # after a server restart or replaced by sending a new prompt.
    pending = st.session_state.get("chat_pending")
    if not pending:
        return
    user_message = {"role": "user", "content": pending["prompt"]}
    with st.chat_message("user"):
        st.text(pending["prompt"])
    with st.chat_message("assistant"):
        if "error" in pending:
            if pending["partial"]:
                st.caption("Partial reply")
                st.text(pending["partial"])
            st.error("The reply could not be completed.")
            st.text(pending["error"])
            if not st.button("Retry reply", disabled=not connected):
                return
        if not connected:
            return
        output = st.empty()
        output.text("Thinking…")
        reply = ""
        try:
            for chunk in local_chat.stream_reply([*messages, user_message]):
                reply += chunk
                # Plain text keeps generated image URLs and HTML inert in the browser.
                output.text(reply)
            if not reply.strip():
                raise local_chat.LocalChatError("The model returned an empty reply.")
        except local_chat.LocalChatError as error:
            pending.update(error=str(error), partial=reply)
            st.rerun()

    messages.extend([user_message, {"role": "assistant", "content": reply}])
    del st.session_state.chat_pending
    st.rerun()


st.set_page_config(page_title="MiniGPT", page_icon="📖", layout="centered")
mode = st.sidebar.radio(
    "Mode",
    ("General chat", "MiniGPT playground"),
    index=1 if "MINIGPT_CHECKPOINT" in os.environ else 0,
)
if mode == "General chat":
    general_chat()
else:
    playground()
