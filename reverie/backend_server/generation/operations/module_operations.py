from persona.aid import Routine, Schema, ToolCall
from generation.prompt_building import create_affordances_sec, create_auxschemas_sec, create_goal_sec, create_instructions_sec, create_mainschema_sec, create_memory_sec, create_routines_sec, create_state_sec, create_task_sec, create_entities_sec
from generation.requests import llm_request
from standard import FOCAL_POINT_SCHEMA, FOCAL_POINT_AUX_SCHEMAS, ROUTINE_SELECTION_SCHEMA
from persona.agent import Agent
from typing import Dict, Optional, Any

import json


done = False

def gen_plan(
    agent: Agent,
    relevant_state: Dict,
    relevant_memory: Dict,
    entities: list[Dict]
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        entities = entities,
        goal = agent.plan.goal,
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
    relevant_state: Dict,
    relevant_memory: Dict,
    entities: list[Dict],
    affordances: list[Dict],
    plan_task: Dict[str, Any],
    actions_taken: list[ToolCall]
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        entities = entities,
        affordances = affordances,
        instructions = agent.settings.grounding.instructions,
        plan_task = plan_task,
        actions_taken = actions_taken,
        task = "Generate the next tool call in a sequence of tool calls to complete the given task"
    )
    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt,
        tools = agent.blackboard.generic_tools
    )["message"]["tool_calls"]

    return clean_up_ground(response)


def gen_routine_selection(
    routines: list[Routine],
    relevant_state: Dict,
    relevant_memory: Dict,
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        routines = routines,
        main_schema = ROUTINE_SELECTION_SCHEMA,
        task = "Generate the next tool call in a sequence of tool calls to complete the given task"
    )
    response = llm_request(
        system_prompt = system_prompt,
        user_prompt = user_prompt
    )["message"]["content"]

    routine_name, goal = clean_up_routine_selection(response) 

    return next(routine for routine in routines if routine.name == routine_name), goal


def gen_focal_points(
    agent: Agent,
    relevant_state: Dict,
    relevant_memory: Dict,
    length: int = 3
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        goal = agent.plan.goal,
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


def clean_up_routine_selection(response_string) -> tuple[str, str]:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    clean_json = json.loads(clean_string)

    return clean_json["routine_choice"], clean_json["goal"]


def create_standard_prompt(
    task: str,
    instructions: Optional[list[str]] = None,
    main_schema: Optional[Schema] = None,
    aux_schemas: Optional[Dict[str, Schema]] = None,
    routines: Optional[list[Routine]] = None,
    goal: Optional[str] = None,
    state: Optional[Dict] = None,
    memory: Optional[Dict] = None,
    entities: Optional[list[Dict]] = None,
    affordances: Optional[list[Dict]] = None,
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
    if routines is not None:
        user_prompt += create_routines_sec(routines)
    if goal is not None:
        user_prompt += create_goal_sec(goal)
    if state is not None:
        user_prompt += create_state_sec(state)
    if memory is not None:
        user_prompt += create_memory_sec(memory)
    if entities is not None:
        user_prompt += create_entities_sec(entities)
    if affordances is not None:
        user_prompt += create_affordances_sec(affordances)

    user_prompt += create_task_sec(task, plan_task, actions_taken)

    return system_prompt, user_prompt


