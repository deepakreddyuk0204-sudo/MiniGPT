# Local general chat

The **General chat** mode runs Qwen3-8B on your Mac using llama.cpp with Apple Metal. It streams answers, remembers recent turns and supports follow-up questions. The model can write, summarize, answer questions and explain code. It can also make factual and coding errors. The app gives it the conversation only. It has no web search, file access or code execution.

Qwen published and post-trained these weights. This project supplies the local runtime setup, conversation handling, interface and checks. The **MiniGPT playground** in the sidebar opens the separate model trained from scratch on *Alice’s Adventures in Wonderland*. [Qwen’s model card](https://huggingface.co/Qwen/Qwen3-8B) · [MiniGPT model card](MODEL_CARD.md)

## Start on this Mac

From the project directory:

```sh
.venv/bin/python scripts/start_chat.py
```

Open the local browser address printed by the launcher. Select **General chat**, ask a question, then send a follow-up in the same chat. **New chat** clears the conversation in that browser session. Keep the launcher running while using the assistant. Press Control-C in its terminal to release the services it started.

## Set up a fresh checkout

Follow the Python environment setup in the project README. Then run these commands from the project directory on an Apple Silicon Mac:

```sh
.venv/bin/python scripts/setup_llama.py
.venv/bin/python scripts/download_chat_model.py
.venv/bin/python scripts/start_chat.py
```

The first two commands need internet access. The runtime download is approximately 17 MB. The weights are 5.03 GB (4.68 GiB). Rerun the model download after an interruption to use the bytes already received. The installer verifies the release archive and the downloader verifies the complete weight file before making it available to the launcher. Inference then runs offline. The app and model server listen on the loopback address of this computer. Downloads, native binaries and temporary logs stay outside Git. [Python environment setup](../README.md) · [Pinned weight manifest](../minigpt/qwen_manifest.json)

## Model provenance

| Item | Recorded value |
| --- | --- |
| Publisher and model | Qwen, Qwen3-8B |
| Distribution | Official Qwen3-8B-GGUF, Q4_K_M quantization |
| Exact file | `Qwen3-8B-Q4_K_M.gguf`, 5,027,783,488 bytes |
| Repository revision | `7c41481f57cb95916b40956ab2f0b139b296d974` |
| SHA-256 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| Weights licence | Apache License 2.0 |
| Inference software | llama.cpp build 12503, commit `4577932fa`, macOS arm64 Metal |

The quantized file stores most weights at reduced precision to lower memory and disk use. The app uses the chat template distributed with the weights. Local inference consumes power and unified memory and incurs no per-message API fee. [Official weight file](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf) · [Publisher’s licence](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/7c41481f57cb95916b40956ab2f0b139b296d974/LICENSE) · [Native runtime release](https://github.com/ggml-org/llama.cpp/releases/tag/b12503)

## How the conversation works

The browser sends each message to Streamlit on this Mac. The app retains complete user/assistant exchanges in the browser session’s Streamlit state. The Python client sends the recent conversation to the local inference process. Each answer appears as the model generates it. Interrupted replies can be retried and enter conversation history only after completion. Restarting the app or ending the browser session can discard the transcript.

The runtime is configured for 8,192 tokens, including the formatted conversation and the answer. The client reserves up to 1,024 tokens for the reply and eight tokens of margin. It counts the formatted conversation with the model’s tokenizer. When space runs short it drops the oldest complete exchanges from the model input. They remain visible in the interface. A single oversized message prompts you to shorten it.

The app uses Qwen’s direct-answer mode, temperature 0.7, top-p 0.8 and top-k 20. Repeated prompts can produce different answers. The local verification below tests specific tasks and reports measurements on one machine. It establishes no benchmark comparison with Claude Haiku. [Conversation client](../minigpt/local_chat.py) · [Streamlit app](../app.py)

## Verified on the M4 Pro

On the 24 GiB Mac, the complete 5.03 GB file matched the publisher’s SHA-256 and the Metal runtime loaded it. In the browser, the assistant explained UTF-8 byte tokenization in one sentence. When asked a follow-up about the word `cat`, it correctly returned the byte IDs 99, 97 and 116.

In a separate live check, it recalled the fictional workshop name “Copper Finch” and answered `17 + 25` with `42`. First visible text arrived in 0.10–0.29 seconds across three short prompts with the server already running. The largest server RSS sample during those prompts was 5.59 GiB. RSS measures resident process memory and can differ from total unified memory in use.

A separate 109-token explanation took 2.53 seconds end to end. The server reported 44.4 generated tokens/second. It reused 36 cached prompt tokens. These are individual local timings that vary with the prompt, caching, other apps and sampling. [Exact prompts, outputs and timings](../results/local_chat.json)

To repeat the live check while the launcher is running:

```sh
.venv/bin/python scripts/check_chat.py --output runs/local-chat/check.json
```

The offline suite also exercises the chat UI with a simulated backend, complete-turn trimming, interrupted streams, retry behavior, download recovery and launcher ownership. Run it with `.venv/bin/python -m pytest -q`.
