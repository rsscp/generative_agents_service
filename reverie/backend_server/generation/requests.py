from typing import Optional
from fastapi import HTTPException
from numpy.typing import NDArray

from persona.aid import Tool
from utils import get_prompt_log_counter, increment_prompt_log_counter, run_log_name, log_json, log_text

import numpy as np
import requests

import json


EmbeddingArray = NDArray[np.float32]


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
    system_prompt: str,
    messages: list[str],
    log_name: str,
    tools: Optional[list[Tool]] = None,
    model: str = "custom_gemma4:e4b"#"qwen3.5:4b"
):
    
    from pathlib import Path
    output_file = Path(f"service_logs/{run_log_name}/{get_prompt_log_counter()}")
    output_file.mkdir(exist_ok=True, parents=True)

    all_messages = [{
        "role": "system",
        "content": system_prompt
    }]

    log_text(system_prompt, "m_system")

    role = "assistant"
    message_counter = 0
    for m in messages:
        if role == "user": role = "assistant"
        else: role = "user"
        
        all_messages.append({
            "role": role,
            "content": m
        })
        print(f">>> m_{role}_{message_counter}")
        log_text(m, f"m_{role}_{message_counter}")
        message_counter += 1

    json_body = {
            "model": model,
            "messages": all_messages,
            "stream": False,
            "think": False,
        }

    if tools is not None:
        json_body["tools"] = [tool.dict() for tool in tools]
        log_json([t.dict() for t in tools], "tools")

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=json_body,
        timeout=120
    )

    if not response.ok:
        print("Ollama error status:", response.status_code, flush=True)
        print("Ollama error body:", response.text, flush=True)

        increment_prompt_log_counter()

        raise HTTPException(
            status_code = 502,
            detail = {
                "error": "ollama_request_failed",
                "ollama_status": response.status_code,
                "ollama_body": response.text,
            },
        )

    response.raise_for_status()
    data = response.json()

    log_json({
        "total_time": data["total_duration"] / 1e9,
        "load_time": data["load_duration"] / 1e9,
        "prompt_eval_time": data["prompt_eval_duration"] / 1e9,
        "generation_time": data["eval_duration"] / 1e9,
        "input tokens": data["prompt_eval_count"],
        "output tokens": data["eval_count"]
    }, "ollama_stats")

    if "content" in data["message"].keys() and data["message"]["content"] != "":
        start = data["message"]["content"].find('{')
        end = data["message"]["content"].rfind('}') + 1
        if start != -1 and end != 0:
            clean_string = data["message"]["content"][start:end]
            obj = json.loads(clean_string)
            log_json(obj, log_name)
        else:
            log_text(data["message"]["content"], log_name)

    if "tool_calls" in data["message"].keys():
        log_json(data["message"]["tool_calls"], "tool_calls")

    return data