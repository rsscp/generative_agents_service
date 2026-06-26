from typing import Dict, Optional

from persona.aid import Property, Schema, Tool, ToolCall
from persona.memory_structures.memory_blocks.node import Node

import json


def create_instructions_sec(instructions: list[str]) -> str:
    result = "# Instructions\n"
    result += "".join([f"- {i}\n" for i in instructions]) + "\n"

    return result


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


def create_memory_sec(memory: list[str]) -> str:
    result = "# Memory\n"
    result += json.dumps(memory) + "\n\n"

    return result


def create_entities_sec(entities: list[str]):
    result = "# Entity Instances\n"
    result += json.dumps(entities) + "\n\n"

    return result


def create_object_sec(object: Dict) -> str:
    result = "# Object"
    result += json.dumps(object) + "\n\n"

    return result


def create_core_nodes_sec(core_nodes: list[Node]) -> str:
    result = "# Core Memories"
    result += json.dumps([node.core.description for node in core_nodes])

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