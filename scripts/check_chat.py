"""Check the live local assistant with public synthetic prompts and measure response time."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minigpt import local_chat as chat
from scripts.start_chat import listener


def ask(history, prompt, process):
    turns = [*history, {"role": "user", "content": prompt}]
    start = time.perf_counter()
    parts = []
    first = None
    rss = process.memory_info().rss
    for part in chat.stream_reply(turns):
        if first is None:
            first = time.perf_counter() - start
        parts.append(part)
        rss = max(rss, process.memory_info().rss)
    elapsed = time.perf_counter() - start
    answer = "".join(parts)
    history.extend([turns[-1], {"role": "assistant", "content": answer}])
    return {"prompt": prompt, "answer": answer, "first_visible_seconds": first,
            "total_seconds": elapsed, "max_sampled_server_rss_bytes": rss}


def check():
    if not chat.ready():
        raise SystemExit("Start the local assistant with scripts/start_chat.py before running this check.")
    process = listener(chat.port())
    history = []
    first = ask(history, "My fictional workshop is called Copper Finch. Reply with just OK.", process)
    second = ask(history, "What is my workshop called? Reply with just its name.", process)
    arithmetic = ask([], "What is 17 + 25? Reply with the integer only.", process)
    for result, expected in [(first, "ok"), (second, "copper finch"), (arithmetic, "42")]:
        result["expected"] = expected
        result["passed"] = result["answer"].strip().rstrip(".!").lower() == expected

    # llama.cpp reports token counts and decoder time separately from prompt evaluation.
    prompt = "In one short paragraph, explain why glass can make a rainbow when sunlight passes through it."
    start = time.perf_counter()
    response = chat._json("/v1/chat/completions", {
        "model": chat.MODEL_ID, "messages": [{"role": "system", "content": chat.SYSTEM_MESSAGE},
                                            {"role": "user", "content": prompt}],
        "max_tokens": 160, "temperature": 0.7, "top_k": 20, "top_p": 0.8,
        "seed": 42, "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)
    perf = {"prompt": prompt, "answer": response["choices"][0]["message"]["content"],
            "finish_reason": response["choices"][0]["finish_reason"], "usage": response["usage"],
            "timings": response.get("timings"), "wall_seconds": time.perf_counter() - start}
    return {
        "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "model": chat.MODEL_ID, "file": "Qwen3-8B-Q4_K_M.gguf", "context_tokens": chat.CONTEXT_TOKENS,
        "build": "b12503", "checks": [first, second, arithmetic], "generation_probe": perf,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not all(item["passed"] for item in report["checks"]):
        raise SystemExit("A live answer missed its exact-match target. See the recorded replies.")
