from persona.aid import Schema, ToolCall
from generation.prompt_building import create_auxschemas_sec, create_goal_sec, create_instructions_sec, create_mainschema_sec, create_memory_sec, create_state_sec, create_task_sec
from generation.requests import llm_request
from standard import FOCAL_POINT_SCHEMA, FOCAL_POINT_AUX_SCHEMAS, STANDARD_INSTRUCTIONS
from persona.agent import Agent
from typing import Dict, Optional, Any

import json


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
    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt
    )["message"]["content"]

    return clean_up_focal_points(response)


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


