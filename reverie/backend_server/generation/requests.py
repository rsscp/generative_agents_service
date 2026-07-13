from typing import Optional
from fastapi import HTTPException
from numpy.typing import NDArray

from persona.aid import Tool
from utils import prompt_log_counter

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
    user_prompt: str,
    tools: Optional[list[Tool]] = None,
    model: str = "qwen3.5:4b"
):
    global prompt_log_counter

    from pathlib import Path
    output_file = Path(f"service_logs/{prompt_log_counter}")
    output_file.mkdir(exist_ok=True, parents=True)

    with open(f'service_logs/{prompt_log_counter}/system_prompt.txt', 'w') as f:
        f.write(system_prompt)
    with open(f'service_logs/{prompt_log_counter}/user_prompt.txt', 'w') as f:
        f.write(user_prompt)

    with open("example.txt", "w") as file:
        file.write("Hello, world!")

    json_body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": False,
            "think": False,
        }

    if tools is not None:
        json_body["tools"] = [tool.dict() for tool in tools]

        with open(f'service_logs/{prompt_log_counter}/tools.txt', 'w') as f:
            f.write(json.dumps([t.dict() for t in tools], indent=4))

    prompt_log_counter += 1

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=json_body,
        timeout=120
    )

    if not response.ok:
            print("Ollama error status:", response.status_code, flush=True)
            print("Ollama error body:", response.text, flush=True)

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

    print("total:", data["total_duration"] / 1e9)
    print("load:", data["load_duration"] / 1e9)
    print("prompt eval:", data["prompt_eval_duration"] / 1e9)
    print("generation:", data["eval_duration"] / 1e9)

    print("input tokens:", data["prompt_eval_count"])
    print("output tokens:", data["eval_count"])

    return data