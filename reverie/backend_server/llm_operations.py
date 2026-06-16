from numpy.typing import NDArray

from persona.aid import Schema, Tool, Property, SchemaField, ToolCall
from persona.memory_structures.memory_blocks.node import CoreNode, EmbeddingArray, Node, RawNode
from standard import FOCAL_POINT_SCHEMA, FOCAL_POINT_AUX_SCHEMAS, NODE_REQ_SCHEMA, STANDARD_INSTRUCTIONS
from persona.agent import Agent
from typing import Dict, Optional, Any
from fastapi import HTTPException

import numpy as np
import json
import requests


def gen_plan(
    agent: Agent,
    relevant_state: Dict[str, Any],
    relevant_memory: list
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        goal = agent.goal,
        instructions = agent.settings.planning.instructions,
        main_schema = agent.settings.planning.main_schema,
        aux_schemas = agent.settings.planning.aux_schemas,
        task = "Make a plan."
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
    )["message"]["content"]

    return clean_up_plan(response)


def gen_grounding(
    agent: Agent,
    relevant_state: Dict[str, Any],
    relevant_memory: list,
    plan_task: Dict[str, Any],
    actions_taken: list[ToolCall]
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        instructions = agent.settings.grounding.instructions,
        plan_task = plan_task,
        actions_taken = actions_taken,
        task = "Generate a complete sequence of necessary tool calls to resolve the task"
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
        tools = agent.blackboard.tools
    )["message"]["tool_calls"]

    return clean_up_ground(response)


def gen_focal_points(
    agent: Agent,
    relevant_state: Dict[str, Any],
    relevant_memory: list,
    length: int = 3
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        goal = agent.goal,
        main_schema = FOCAL_POINT_SCHEMA,
        aux_schemas = FOCAL_POINT_AUX_SCHEMAS,
        task = f"Respond with a list of {length} focal points that would be useful for retrieval of memories for this agent."
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt
    )["message"]["content"]

    return clean_up_focal_points(response)


def gen_node_description(raw_node: RawNode) -> str: #TODO
    system_prompt, user_prompt = create_node_req_prompt(
        instructions = [
            "Your response will be less than 50 words",
            "Your response will not contain JSON",
            "Your response will be written entirely as natural language"
        ],
        object = raw_node.object,
        task = "Respond with a sentence describing the JSON object presented on Object."
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
    )["message"]["content"]

    return clean_up_node_description(response)


def gen_node_poignancy(core_nodes: list[Node], description: str) -> int: #TODO
    system_prompt, user_prompt = create_node_req_prompt(
        instructions = [
            "Your response will be a single integer, between 0 and 100, which reflects the importance ",
            "Your response will not contain JSON",
        ],
        object = {"description": description},
        task = "Respond with an integer between 0 and 100 representing the overall importance of the object presented as Object."
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
    )["message"]["content"]

    return clean_up_node_poignancy(response)


def gen_node_reqs(core_nodes: list[Node], raw_node: RawNode) -> tuple[str, int]:
    system_prompt, user_prompt = create_node_req_prompt(
        instructions = [
            "Your response will be a single integer, between 0 and 100, which reflects the importance ",
            "Your response will not contain JSON",
        ],
        schema = NODE_REQ_SCHEMA,
        core_nodes = core_nodes,
        object = raw_node.object,
        task = "Respond with an integer between 0 and 100 representing the overall importance of the object presented as Object."
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
    )["message"]["content"]

    return clean_up_node_reqs(response)


def clean_up_plan(response_string: str) -> Dict:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]

    return json.loads(clean_string)


def clean_up_ground(actions_response: list) -> list[ToolCall]:
    actions = [ToolCall(
        key = call["function"]["name"],
        arguments = call["function"]["arguments"])
    for call in actions_response]

    return actions

def clean_up_focal_points(response_string: str) -> list[str]:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    clean_json = json.loads(clean_string)

    return [point["key"] for point in clean_json["focal_points"]]


def clean_up_node_description(response_string: str) -> str:
    return response_string


def clean_up_node_poignancy(response_string: str) -> int:
    return int(response_string)


def clean_up_node_reqs(response_string: str) -> tuple[str, int]:
    obj = json.loads(response_string)
    return (obj["node_description"]["node_poignancy"])


def create_standard_prompt(
    task: str,
    instructions: Optional[list[str]] = None,
    main_schema: Optional[Schema] = None,
    aux_schemas: Optional[Dict[str, Schema]] = None,
    goal: Optional[str] = None,
    state: Optional[Dict[str, Dict]] = None,
    memory: Optional[list] = None,
    plan_task: Optional[Dict] = None,
    actions_taken: Optional[list[ToolCall]] = None,
):
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    if instructions is not None and len(instructions) > 0:
        system_prompt += create_instructions_sec(instructions)
    if main_schema is not None:
        system_prompt += create_mainschema_sec(main_schema)
    if aux_schemas is not None:
        system_prompt += create_auxschemas_sec(aux_schemas)

    ### User prompt

    if goal is not None:
        user_prompt += create_goal_sec(goal)
    if state is not None:
        user_prompt += create_state_sec(state)
    if memory is not None:
        user_prompt += create_memory_sec(memory)

    user_prompt += create_task_sec(task, plan_task, actions_taken)

    return system_prompt, user_prompt


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
    core_nodes: Optional[list[Node]] = None
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


def create_node_req_prompt(
    task: str,
    instructions: Optional[list[str]] = None,
    object: Optional[Dict] = None,
    core_nodes: Optional[list[Node]] = None,
    schema: Optional[Schema] = None
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


def create_instructions_sec(instructions: list[str]) -> str:
    result = "# Instructions\n"
    result += "".join([f"- {i}\n" for i in instructions]) + "\n"
    
    return result


def create_mainschema_sec(schema: Schema) -> str:
    result = "# Main Schema\n"
    fields, schema = get_schema_ready(schema)
    result += create_schema_str("##", fields, schema)
    result += "\n"

    return result


def create_auxschemas_sec(schemas: Dict[str, Schema]) -> str:
    result = "# Auxiliary Schemas\n"

    for k, v in schemas.items():
        result += f"## {k}\n"
        fields, schema = get_schema_ready(v)
        result += create_schema_str("###", fields, schema)
    result += "\n"

    return result


def create_tools_sec(tools: Dict[str, Tool]) -> str:
    result = "# Available Tools\n"

    for name, tool in tools.items():
        result += f"## {name}\n"
        result += f"{tool.function.description}\n\n"
        args, schema = get_tool_schema_ready(name, tool.function.parameters.properties)
        result += create_tool_schema_str("###", args, schema)

    return result


def create_goal_sec(goal: str) -> str:
    result = "# Goal\n"
    result += goal + "\n\n"

    return result


def create_state_sec(state: Dict[str, Dict]) -> str:
    result = "# State\n"
    result += json.dumps(state) + "\n\n"

    return result


def create_memory_sec(memory: list) -> str:
    result = "# Memory\n"
    result += json.dumps(memory) + "\n\n"

    return result

def create_object_sec(object: Dict) -> str:
    result = "# Object"
    result += json.dumps(object) + "\n\n"

    return result


def create_core_nodes_sec(core_nodes: list[Node]) -> str:
    result = "# Core Memories"
    result += json.dumps([node.description for node in core_nodes])

    return result


def create_task_sec(
    task: str,
    plan_task: Optional[Dict] = None,
    actions_taken: Optional[list[ToolCall]] = None
) -> str:
    result = "# Task\n"
    
    if plan_task is not None and actions_taken is not None:
        result += f"The task being deconstruncted is:\n\t{json.dumps(plan_task)}\n"
        #result += f"The tool calls in this list have been executed:\n\t{json.dumps([action.dict() for action in actions_taken])}\n"

    return result + task


def create_schema_str(header: str, fields: list[str], schema: Dict) -> str:
    string = f"{header} Fields\n"
    string += "".join([f"- {f}\n" for f in fields])
    string += f"{header} Schema\n"
    string += json.dumps(schema) + "\n"
    return string


def get_schema_ready(schema: Schema) -> tuple[list[str], Dict]:
    field_list = []
    json_schema = {}

    for k, v in schema.items():
        field_item = f"{k}:\n"
        field_item += f"\t- Field of type {v.field_type}\n"
        field_item += f"\t- {v.description}\n"
        field_item += f"\t- {v.guidelines}\n"
        field_list.append(field_item)

        if (v.field_type == "object"):
            field_list_add, json_schema_add = get_schema_ready(v.sub_fields)
            json_schema[k] = json_schema_add
            field_list += field_list_add
        else:
            json_schema[k] = f"<{v.field_type}>"

    return field_list, json_schema


def create_tool_schema_str(header: str, args: list[str], schema: Dict) -> str:
    string = f"{header} Arguments\n"
    string += "".join([f"- {f}\n" for f in args]) + "\n"
    string += f"{header} Schema\n"
    string += json.dumps(schema) + "\n\n"
    return string


def get_tool_schema_ready(tool_name: str, schema: Dict[str, Property]) -> tuple[list[str], Dict]:
    arg_list = []
    json_schema = {
        "action": tool_name,
        "arguments": {}
    }

    for k, v in schema.items():
        arg_item = f"{k}:\n"
        arg_item += f"\t- type: {v.type}\n"
        arg_item += f"\t- description: {v.description}\n"
        #arg_item += f"\t- restrictions: {v.restrictions}"
        arg_list.append(arg_item)

        json_schema["arguments"][k] = f"<{v.type}>"

    return arg_list, json_schema


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


def embedding_request(string: str) -> EmbeddingArray:
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text:latest",
            "input": string,
        },
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    # /api/embed returns "embeddings", usually a list of embeddings
    return np.array(data["embeddings"][0], dtype=np.float32)
