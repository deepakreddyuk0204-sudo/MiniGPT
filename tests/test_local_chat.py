"""Offline coverage of token budgeting and the local llama.cpp streaming protocol."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from minigpt import local_chat as chat


def message(role, text):
    return {"role": role, "content": text}


def test_context_drops_complete_oldest_turns_and_keeps_new_prompt(monkeypatch):
    # An intentionally simple tokenizer stand-in makes the exact boundary visible.
    monkeypatch.setattr(chat, "CONTEXT_TOKENS", 13)
    monkeypatch.setattr(chat, "REPLY_TOKENS", 1)  # 4 tokens left after the safety margin
    monkeypatch.setattr(chat, "_count_tokens", lambda turns: sum(len(m["content"]) for m in turns[1:]))
    history = [message("user", "a"), message("assistant", "b"),
               message("user", "c"), message("assistant", "d"), message("user", "ef")]
    retained = chat._fit_messages(history)
    assert retained[0]["role"] == "system"
    assert retained[1:] == history[-3:]
    assert len(history) == 5
    assert chat._fit_messages([*history[:4], message("user", "abcd")])[1:] == [message("user", "abcd")]
    with pytest.raises(chat.LocalChatError, match="too long"):
        chat._fit_messages([message("user", "abcde")])


@pytest.mark.parametrize("turns", [[], [message("user", "hello"), message("assistant", "hi")],
                                  [message("assistant", "hello")], [message("user", "   ")]])
def test_invalid_turns_are_rejected_before_network(turns):
    with pytest.raises(chat.LocalChatError):
        chat._fit_messages(turns)


@pytest.fixture
def local_server(monkeypatch):
    calls = {}
    events = []
    served_models = [chat.MODEL_ID]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, obj):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(obj).encode())

        def do_GET(self):
            if self.path == "/health":
                self.send_json({"status": "ok"})
            elif self.path == "/v1/models":
                self.send_json({"data": [{"id": model} for model in served_models]})
            else:
                self.send_error(404)

        def do_POST(self):
            calls[self.path] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/apply-template":
                self.send_json({"prompt": "formatted prompt"})
            elif self.path == "/tokenize":
                self.send_json({"tokens": [2, 3]})
            elif self.path == "/v1/chat/completions":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(b"data: " + event.encode("utf8") + b"\n\n")
            else:
                self.send_error(404)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        monkeypatch.setenv("MINIGPT_CHAT_PORT", str(server.server_port))
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield calls, events, served_models
        finally:
            server.shutdown()
            thread.join()


def test_real_http_stream_uses_history_and_yields_only_visible_text(local_server):
    calls, events, models = local_server
    for delta in [{"role": "assistant"}, {"reasoning_content": "internal"}, {"content": "Hi"},
                  {"content": " there."}]:
        events.append(json.dumps({"choices": [{"delta": delta}]}))
    events.append("[DONE]")
    assert chat.ready()
    turns = [message("user", "My name is Alex."), message("assistant", "Hello!"),
             message("user", "What is my name?")]
    assert list(chat.stream_reply(turns)) == ["Hi", " there."]
    templated = calls["/apply-template"]["messages"]
    assert templated[1:] == turns
    assert calls["/v1/chat/completions"]["messages"] == templated
    assert calls["/tokenize"]["content"] == "formatted prompt"
    assert calls["/v1/chat/completions"]["chat_template_kwargs"] == {"enable_thinking": False}
    models[:] = ["unrelated-model"]
    assert not chat.ready()


@pytest.mark.parametrize("ending", ["disconnect", "empty", "server_error", "invalid_json"])
def test_incomplete_streams_can_be_retried(local_server, ending):
    _, events, _ = local_server
    if ending == "empty":
        events.append("[DONE]")
    elif ending == "disconnect":
        events.append(json.dumps({"choices": [{"delta": {"content": "partial"}}]}))
    elif ending == "server_error":
        events.append(json.dumps({"error": {"message": "inference failed"}}))
    else:
        events.append("incomplete JSON")
    with pytest.raises(chat.LocalChatError):
        list(chat.stream_reply([message("user", "Hi")]))
