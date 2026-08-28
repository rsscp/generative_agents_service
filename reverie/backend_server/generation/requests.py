import os
import time
from typing import Dict, Optional
from fastapi import HTTPException
from pathlib import Path
from numpy.typing import NDArray

from persona.aid import ChatMessage, Tool, ToolCall
from utils import get_prompt_log_counter, increment_prompt_log_counter, run_log_name, log_json, log_text

import numpy as np
import requests

import json


EmbeddingArray = NDArray[np.float32]


llm_choice = os.getenv("API_CHOICE", "custom-local-gemma")
keys = json.loads(Path("keys.json").read_text(encoding="utf-8"))


llm_choices = {
    "groq-gpt-20b": {
        "model": "openai/gpt-oss-20b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": keys["groq-gpt-20b"]
        
    },
    "groq-gpt-120b": {
        "model": "openai/gpt-oss-120b",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": keys["groq-gpt-120b"]
    },
    "gemini-3.6": {
        "model": "gemini-3.6-flash",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key": keys["gemini-3.6"]
    },
    "custom-local-gemma": {
        "model": "custom_gemma4:e4b",
        "url": "http://localhost:11434/v1/chat/completions",
        "key": keys["custom-local-gemma"]
    }
}


def embedding_request(string: str) -> EmbeddingArray:
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "all-minilm:22m",
            "input": string,
        },
        timeout=5,
    )

    response.raise_for_status()
    data = response.json()

    # /api/embed returns "embeddings", usually a list of embeddings
    return np.array(data["embeddings"][0], dtype=np.float32)


def llm_request(
    messages: list[ChatMessage],
    log_name: str,
    content_format: str,
    tools: Optional[list[Tool]] = None,
    llm_choice: str = llm_choice, #"gemini-3.6-flash" #"custom_gemma4:e4b" #"qwen3.5:4b" #
    reasoning_effort: str = "none"
):
      
    from pathlib import Path
    output_file = Path(f"service_logs/{run_log_name}/{get_prompt_log_counter()}")
    output_file.mkdir(exist_ok=True, parents=True)

    message_counter = 0
    for m in messages:
        if m.content is not None and m.content != "":
            log_text(m.content, f"{message_counter}_{m.role}_content")
        if m.reasoning is not None and m.reasoning != "":
            log_text(m.reasoning, f"{message_counter}_{m.role}_reasoning")
        elif m.tool_calls is not None:
            log_json(m.tool_calls, f"{message_counter}_{m.role}_calls")
        message_counter += 1

    headers={ "Content-Type": "application/json" }

    json_body = {
        "model": llm_choices[llm_choice]['model'],
        "messages": [m.dict(exclude_none=True) for m in messages],
        "stream": False,
        "temperature": 0.2,
        "reasoning_effort": reasoning_effort
    }

    if llm_choices[llm_choice]['key'] is not None:
        headers["Authorization"] = f"Bearer {llm_choices[llm_choice]['key']}"

    if tools is not None:
        json_body["tools"] = [tool.dict(exclude_none=True) for tool in tools]
        json_body["tool_choice"] = "auto"
        log_json([t.dict() for t in tools], "tools")

    try:
        session = requests.Session()
        session.trust_env = False

        start_time = time.perf_counter()

        response = requests.post(
            llm_choices[llm_choice]['url'],
            headers={
                "Authorization": f"Bearer {llm_choices[llm_choice]['key']}",
                "Content-Type": "application/json"
            },
            json=json_body,
            timeout=120
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

    except requests.exceptions.RequestException as exc:
        print("Exception type:", type(exc).__name__, flush=True)
        print("Exception repr:", repr(exc), flush=True)
        raise

    response.raise_for_status()
    data = response.json()

    log_json({
        "prompt_tokens": data["usage"].get("prompt_tokens"),
        "completion_tokens": data["usage"].get("completion_tokens"),
        "total_tokens": data["usage"].get("total_tokens"),
        "latency_ms": latency_ms
    }, "metrics")

    msg_content = None
    msg_tool_calls = None

    if "content" in data["choices"][0]["message"].keys() and data["choices"][0]["message"]["content"] != "":
        msg_content = data["choices"][0]["message"]["content"]
        start = msg_content.find('{')
        end = msg_content.rfind('}') + 1

        print("<CONTENT>\n" + msg_content)

        if content_format == "json":
            clean_string = msg_content[start:end]
            obj = json.loads(clean_string)
            log_json(obj, log_name)
        elif content_format == "text":
            log_text(msg_content, log_name)
        else:
            raise Exception(f"Format '{content_format}' is not valid")

    if "reasoning" in data["choices"][0]["message"].keys() and data["choices"][0]["message"]["reasoning"] != "":
        msg_reasoning = data["choices"][0]["message"]["reasoning"]
        log_text(msg_reasoning, "reasoning")

    if "tool_calls" in data["choices"][0]["message"].keys():
        msg_tool_calls = data["choices"][0]["message"]["tool_calls"]

        log_json(msg_tool_calls, "tool_calls")

    return msg_content, msg_tool_calls