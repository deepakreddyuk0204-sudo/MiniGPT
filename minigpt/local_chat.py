"""Stream replies from the local Qwen server. All requests go to IPv4 loopback."""

from collections.abc import Iterator
import http.client
import json
import os

MODEL_ID = "qwen3-8b-local"
CONTEXT_TOKENS = 8192
REPLY_TOKENS = 1024
SYSTEM_MESSAGE = (
    "You are a helpful local assistant. Answer the user's request directly. "
    "Use earlier messages when relevant. If you are unsure, say so."
)


class LocalChatError(RuntimeError):
    """A local inference or input error that can be shown in the interface."""


def port() -> int:
    return int(os.environ.get("MINIGPT_CHAT_PORT", "8091"))


def _request(path, payload=None, *, timeout=45):
    conn = http.client.HTTPConnection("127.0.0.1", port(), timeout=timeout)
    try:
        body = json.dumps(payload, allow_nan=False) if payload is not None else None
        conn.request("POST" if body else "GET", path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raw = response.read(4096).decode(errors="replace")
            try:
                detail = json.loads(raw)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = raw
            raise LocalChatError(f"Local inference returned HTTP {response.status}. {detail}")
        return conn, response
    except (OSError, http.client.HTTPException) as err:
        conn.close()
        raise LocalChatError("The local model server is unavailable. Check its terminal and retry.") from err
    except BaseException:
        conn.close()
        raise


def _json(path, payload=None, *, timeout=45):
    conn, response = _request(path, payload, timeout=timeout)
    try:
        result = json.load(response)
        if not isinstance(result, dict):
            raise LocalChatError("The local model server returned an unexpected response.")
        return result
    except (ValueError, OSError, http.client.HTTPException) as err:
        raise LocalChatError("The local model server returned an incomplete response.") from err
    finally:
        conn.close()


def ready() -> bool:
    try:
        if _json("/health", timeout=1).get("status") != "ok":
            return False
        return any(m.get("id") == MODEL_ID for m in _json("/v1/models", timeout=1)["data"])
    except (LocalChatError, KeyError, TypeError, ValueError):
        return False


def _count_tokens(messages):
    rendered = _json("/apply-template", {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}})
    if not isinstance(rendered.get("prompt"), str):
        raise LocalChatError("The local runtime could not format the conversation.")
    encoded = _json("/tokenize", {"content": rendered["prompt"], "add_special": True, "parse_special": True})
    if not isinstance(encoded.get("tokens"), list):
        raise LocalChatError("The local runtime could not measure this conversation.")
    return len(encoded["tokens"])


def _fit_messages(messages):
    if not messages or len(messages) % 2 == 0:
        raise LocalChatError("A conversation must end with a user message.")
    for i, message in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        if message.get("role") != role or not isinstance(message.get("content"), str) or not message["content"].strip():
            raise LocalChatError("Conversation history must contain alternating, nonempty user and assistant messages.")
    system = {"role": "system", "content": SYSTEM_MESSAGE}
    # Keep the largest suffix of complete turns that leaves room for the answer.
    # Check the common case first, then search instead of measuring every turn.
    budget = CONTEXT_TOKENS - REPLY_TOKENS - 8
    full = [system, *messages]
    if _count_tokens(full) <= budget:
        return full
    if len(messages) == 1 or _count_tokens([system, messages[-1]]) > budget:
        raise LocalChatError("This message is too long for the local context. Shorten it and try again.")
    low, high = 1, len(messages) // 2
    while low < high:
        skip = (low + high) // 2
        if _count_tokens([system, *messages[2 * skip:]]) <= budget:
            high = skip
        else:
            low = skip + 1
    return [system, *messages[2 * low:]]


def stream_reply(messages: list[dict[str, str]]) -> Iterator[str]:
    """Send the retained conversation and yield visible answer text as it arrives."""
    conversation = _fit_messages(messages)
    payload = {
        "model": MODEL_ID, "messages": conversation, "stream": True,
        "max_tokens": REPLY_TOKENS, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn, response = _request("/v1/chat/completions", payload, timeout=120)
    try:
        received = False
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                if not received:
                    raise LocalChatError("The model returned an empty answer. Try again.")
                return
            event = json.loads(data)
            if "error" in event:
                raise LocalChatError(f"Local inference failed. {event['error']}")
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    received = True
                    yield content
        raise LocalChatError("The local model disconnected before completing the answer. Retry this message.")
    except (ValueError, OSError, http.client.HTTPException) as err:
        raise LocalChatError("The local reply was interrupted. Retry this message.") from err
    finally:
        conn.close()
