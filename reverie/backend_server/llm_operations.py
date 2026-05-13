from utils import merge
from persona.agent import Agent
from typing import Dict
from api_classes import MemoryList, ActionDef, Ruleset, Schema, FieldSchema

import json
import requests


class Contract:
    state_keys: list[str]
    memory_keys: list[str]


def plan(agent: Agent, contract: Contract):

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
        plan_ruleset = agent.plan_gen_ruleset,
        plan_schema = agent.plan_gen_schema,
    )

    return llm_request(system_prompt, user_prompt)


def create_plan_prompt(
    # System prompt
    plan_ruleset: Ruleset,
    plan_schema: Schema,
    # User prompt
    goal: str,
    state: Dict[str, Dict],
    memory: list,
):
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    system_prompt += "You are a planner for a video game agent and your task is to create a step by step plan that will be followed by the agent."

    system_prompt += "While creating the plan you must follow these instructions:\n"
    system_prompt += json.dumps(plan_ruleset.dict()) + "\n"

    system_prompt += "Your response should include only a JSON object that describes the plan and each step in the plan is an object following this schema:\n"
    system_prompt += json.dumps(plan_schema.dict()) + "\n"

    ### User prompt

    user_prompt += "Your current goal is "
    user_prompt += goal + "\n"

    user_prompt += "The current state of your agent is described by this object:\n"
    user_prompt += json.dumps(state) + "\n"

    user_prompt += "The current memory of your agent contains the following events/thoughts:\n"
    user_prompt += json.dumps(memory) + "\n"

    return system_prompt, user_prompt


def relevance_filter(memories: Dict[str, MemoryList]):
    memories_list = []
    [memories_list + l.memories for l in [v for k, v in memories.items()]]
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

contract = Contract()
contract.state_keys = ["hunger", "sleepy"]
contract.memory_keys = ["objects"]


print(plan(agent, contract))