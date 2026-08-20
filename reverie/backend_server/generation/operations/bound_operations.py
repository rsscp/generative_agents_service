import json
from typing import Dict, Optional
from persona.memory_structures.memory_blocks.node import MemoryNode
from generation.prompt_building import create_core_nodes_sec, create_instructions_sec, create_mainschema_sec, create_object_sec, create_task_sec
from persona.aid import Schema


def clean_up_node_description(response_string: str) -> str:
    return response_string


def create_node_req_prompt(
    task: str,
    instructions: Optional[list[str]] = None,
    object: Optional[Dict] = None,
    core_nodes: Optional[list[MemoryNode]] = None,
    schema: Optional[str] = None
): #TODO CONTINUE
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    if instructions is not None and len(instructions) > 0:
        system_prompt += create_instructions_sec(instructions)
    if schema is not None:
        system_prompt += create_mainschema_sec(schema)

    ### User prompt

    if core_nodes is not None:
        user_prompt += create_core_nodes_sec(core_nodes)
    if object is not None:
        user_prompt += create_object_sec(object)

    user_prompt += create_task_sec(task)

    return system_prompt, user_prompt


def clean_up_node_reqs(response_string: str) -> tuple[str, int]:
    obj = json.loads(response_string)
    return (obj["node_description"]["node_poignancy"])


def create_node_description_prompt(
    task: str,
    instructions: Optional[list[str]] = None,
    object: Optional[Dict] = None
):
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    if instructions is not None and len(instructions) > 0:
        system_prompt += create_instructions_sec(instructions)

    ### User prompt

    if object is not None:
        user_prompt += create_object_sec(object)

    user_prompt += create_task_sec(task)

    return system_prompt, user_prompt


def create_node_poignancy_prompt(
    task: str,
    instructions: Optional[list[str]] = None,
    core_nodes: Optional[list[MemoryNode]] = None
):
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    if instructions is not None and len(instructions) > 0:
        system_prompt += create_instructions_sec(instructions)

    ### User prompt

    if core_nodes is not None:
        user_prompt += create_core_nodes_sec(core_nodes)

    user_prompt += create_task_sec(task)

    return system_prompt, user_prompt
