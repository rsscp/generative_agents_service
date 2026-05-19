from persona.aid import Action, SchemaField
from utils import merge
from persona.agent import Agent
from typing import Dict
from api_classes import MemoryList, Contract

import json
import requests


def plan(agent: Agent):

    contract = agent.plan_contract

    common_keys_state = set(contract.state_keys) & set(agent.blackboard.state.keys())
    common_keys_cache = set(contract.memory_keys) & set(agent.blackboard.cache_lists.keys())
    common_keys_memory = set(contract.memory_keys) & set(agent.recall.memory_lists.keys())
    
    relevant_state = {k: agent.blackboard.state[k] for k in common_keys_state}

    relevant_memory = relevance_filter(merge(
        {k: agent.blackboard.cache_lists[k] for k in common_keys_cache},
        {k: agent.recall.memory_lists[k] for k in common_keys_memory},
        lambda a, b : a + b
    ))

    system_prompt, user_prompt = create_plan_prompt(
        state = relevant_state,
        memory = relevant_memory,
        goal = agent.goal,
        plan_instructions = agent.plan_instructions,
        plan_main_schema = agent.plan_main_schema,
        plan_aux_schemas = agent.plan_aux_schemas
    )

    print("SYSTEM_PROMPT:\n\n" + system_prompt + "\n\n")
    print("USER_PROMPT:\n\n" + user_prompt + "\n\n")

    return llm_request(system_prompt, user_prompt)


def create_plan_prompt(
    plan_instructions: list[str],
    plan_main_schema: Dict[str, SchemaField],
    plan_aux_schemas: Dict[str, Dict[str, SchemaField]],
    goal: str,
    state: Dict[str, Dict],
    memory: list,
):
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    system_prompt += "# Instructions\n"
    system_prompt += "".join([f"- {i}\n" for i in plan_instructions]) + "\n\n"

    system_prompt += "# Main Schema\n"
    fields, schema = get_schema_ready(plan_main_schema)
    system_prompt += create_schema_str("##", fields, schema)
    system_prompt += "\n"

    if len(plan_aux_schemas.items()) > 0:
        system_prompt += "# Auxiliary Schemas\n"
        for k, v in plan_aux_schemas.items():
            system_prompt += f"## {k}\n"
            fields, schema = get_schema_ready(v)
            system_prompt += create_schema_str("###", fields, schema)
        system_prompt += "\n"

    ### User prompt

    user_prompt += "# Goal\n"
    user_prompt += goal + "\n\n"

    user_prompt += "# State\n"
    user_prompt += json.dumps(state) + "\n\n"

    user_prompt += "# Memory\n"
    user_prompt += json.dumps(memory) + "\n\n"

    return system_prompt, user_prompt


def create_schema_str(header: str, fields: list[str], schema: Dict) -> str:
    string = ""
    string += f"{header} Fields\n"
    string += "".join([f"- {f}\n" for f in fields])
    string += f"{header} Schema\n"
    string += json.dumps(schema) + "\n"
    return string


def relevance_filter(memories: Dict[str, list]):
    memories_list = []
    [memories_list + v for k, v in memories.items()]
    return memories_list


def llm_request(system_prompt: str, user_prompt: str, model: str = "llama3.2:3b"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
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
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()
    return data["message"]["content"]


def get_schema_ready(schema: Dict[str, SchemaField]) -> tuple[list[str], Dict]:
    field_list = []
    json_schema = {}

    for k, v in schema.items():
        field_item = f"{k} > "
        field_item += f"\t- Field of type {v.field_type}. "
        field_item += f"\t- {v.description}"
        field_item += f"\t- {v.guidelines}"
        field_list.append(field_item)

        if (v.field_type == "object"):
            field_list_add, json_schema_add = get_schema_ready(v.sub_fields)
            json_schema[k] = json_schema_add
            field_list += field_list_add
        else:
            json_schema[k] = f"<{v.field_type}>"

    return field_list, json_schema









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