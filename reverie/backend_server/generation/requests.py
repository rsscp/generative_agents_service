from typing import Optional
from fastapi import HTTPException
from numpy.typing import NDArray

from persona.aid import Tool

import numpy as np
import requests
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
            "think": False
        }

    if tools is not None:
        json_body["tools"] = [tool.dict() for tool in tools]

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

    return data