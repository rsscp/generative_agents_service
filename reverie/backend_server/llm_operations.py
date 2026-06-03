from persona.aid import Action, Property, SchemaField, ActionCall, Contract
from utils import merge
from persona.agent import Agent
from typing import Dict, Optional, Any
from api_classes import MemoryList
from fastapi import HTTPException

import json
import requests


def plan_broad_request(
    agent: Agent,
    relevant_state: Dict[str, Any],
    relevant_memory: list
):
    system_prompt, user_prompt = create_prompt(
        state = relevant_state,
        memory = relevant_memory,
        goal = agent.goal,
        instructions = agent.plan_instructions,
        main_schema = agent.plan_main_schema,
        aux_schemas = agent.plan_aux_schemas,
        task = "Make a plan."
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
    )["message"]["content"]

    return clean_up_plan(response)


def plan_grounded_request(
    agent: Agent,
    relevant_state: Dict[str, Any],
    relevant_memory: list,
    plan_task: Dict[str, Any],
    actions_taken: list[ActionCall]
):
    system_prompt, user_prompt = create_prompt(
        state = relevant_state,
        memory = relevant_memory,
        instructions = agent.plan_grounded_instructions,
        plan_task = plan_task,
        actions_taken = actions_taken,
        task = "Generate a sequence of necessary tool calls to resolve the task"
    )

    print("SYSTEM_PROMPT:\n" + system_prompt + "\n")
    print("USER_PROMPT:\n" + user_prompt + "\n")

    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
        tools = agent.blackboard.available_actions
    )["message"]["tool_calls"]

    print(response)

    return clean_up_plan_grounded(response)


def get_relevant_pieces(agent: Agent, contract: Contract):
    common_keys_state = set(contract.state_keys) & set(agent.blackboard.state.keys())
    common_keys_cache = set(contract.memory_keys) & set(agent.blackboard.cache_lists.keys())
    common_keys_memory = set(contract.memory_keys) & set(agent.recall.memory_lists.keys())
    
    relevant_state = {k: agent.blackboard.state[k] for k in common_keys_state}
    relevant_memory = relevance_filter(merge(
        {k: agent.blackboard.cache_lists[k] for k in common_keys_cache},
        {k: agent.recall.memory_lists[k] for k in common_keys_memory},
        lambda a, b : a + b
    ))

    return relevant_state, relevant_memory


def clean_up_plan(response_string: str) -> Dict:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]

    return json.loads(clean_string)


def clean_up_plan_grounded(actions_response: list) -> list[ActionCall]:
    actions = [ActionCall(
        key = call["function"]["name"],
        arguments = call["function"]["arguments"])
    for call in actions_response]

    return actions


def create_prompt(
    instructions: list[str],
    main_schema: Optional[Dict[str, SchemaField]] = None,
    aux_schemas: Optional[Dict[str, Dict[str, SchemaField]]] = None,
    goal: Optional[str] = None,
    state: Optional[Dict[str, Dict]] = None,
    memory: Optional[list] = None,
    plan_task: Optional[Dict] = None,
    actions_taken: Optional[list[ActionCall]] = None,
    task: Optional[str] = None
):
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    system_prompt += create_instructions_sec(instructions)

    if main_schema is not None:
        system_prompt += create_mainschema_sec(main_schema)
    if aux_schemas is not None and aux_schemas.keys():
        system_prompt += create_auxschemas_sec(aux_schemas)
    #if available_actions is not None: TODO uncomment
        #system_prompt += create_tools_sec(available_actions) TODO uncomment

    ### User prompt

    if goal is not None:
        user_prompt += create_goal_sec(goal)
    if state is not None:
        user_prompt += create_state_sec(state)
    if memory is not None:
        user_prompt += create_memory_sec(memory)
    
    
    if plan_task is not None and actions_taken is not None:
        user_prompt += f"The following task is being deconstructed:\n\t{json.dumps(plan_task)}\n"
        user_prompt += f"The tool calls in this list have been executed:\n\t{json.dumps([action.dict() for action in actions_taken])}\n"
    if task is not None:
        user_prompt += f"{task}"

    return system_prompt, user_prompt


def create_instructions_sec(instructions: list[str]) -> str:
    result = "# Instructions\n"
    result += "".join([f"- {i}\n" for i in instructions]) + "\n"
    
    return result


def create_mainschema_sec(schema: Dict[str, SchemaField]) -> str:
    result = "# Main Schema\n"
    fields, schema = get_schema_ready(schema)
    result += create_schema_str("##", fields, schema)
    result += "\n"

    return result


def create_auxschemas_sec(schemas: Dict[str, Dict[str, SchemaField]]) -> str:
    result = "# Auxiliary Schemas\n"

    for k, v in schemas.items():
        result += f"## {k}\n"
        fields, schema = get_schema_ready(v)
        result += create_schema_str("###", fields, schema)
    result += "\n"

    return result


def create_tools_sec(tools: Dict[str, Action]) -> str:
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


def create_schema_str(header: str, fields: list[str], schema: Dict) -> str:
    string = f"{header} Fields\n"
    string += "".join([f"- {f}\n" for f in fields])
    string += f"{header} Schema\n"
    string += json.dumps(schema) + "\n"
    return string


def get_schema_ready(schema: Dict[str, SchemaField]) -> tuple[list[str], Dict]:
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


def relevance_filter(memories: Dict[str, list]):
    memories_list = []
    [memories_list.extend(v) for k, v in memories.items()]

    return memories_list


def llm_request(
    system_prompt: str,
    user_prompt: str,
    tools: Optional[list[Action]] = None,
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








'''
### TEST CODE WORKING


memory_list = MemoryList(memories = [
    {
        "name": "cake",
        "location": "kitchen",
        "state": "good to eat"
    },
    {
        "name": "computer",
        "location": "bedroom",
        "state": "charged and on"
    }
])

curr_ruleset = Ruleset(
    instructions = [
        "Formulate a plan with no more than 5 steps",
        "Each step must be less complex than actions on the level of \" walk to point A and do B; pick up object C, place object D on point E"
    ]
)

field_schemaA = FieldSchema(
    description = "This field describes what actions are taken at this step of the plan",
    field_type = "string",
    sub_fields = {}
)

field_schemaB = FieldSchema(
   description = "This field is a 0-5 rating of the complexity of this step of the plan",
   field_type = "integer",
   sub_fields = {}
)

curr_schema = Schema(
    description = "Reference schema for the production of a plan",
    fields_definitions = {
        "step_description": field_schemaA,
        "complexity_rating": field_schemaB
    }
)

agent = Agent(
    "first_agent",
    "Satisfy one's own needs",
    {
        "hunger": 100,
        "sleepy": 0,
    },
    {
        "objects": memory_list
    },
    {},
    {},
    curr_ruleset,
    curr_schema,
    None,
    None
)

contract = Contract(
    state_keys = ["hunger", "sleepy"],
    memory_keys = ["objects"]    
)


print(plan(agent, contract))
'''