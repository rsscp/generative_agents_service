from pydantic import BaseModel, Field

from persona.aid import AgentRoutine, ChatMessage, GroundingSequence, Schema, Tool, ToolCall
from generation.prompt_building import create_affordances_sec, create_auxschemas_sec, create_current_task_sec, create_goal_sec, create_instructions_sec, create_mainschema_sec, create_memory_sec, create_routines_sec, create_state_sec, create_task_sec, create_entities_sec, create_tools_sec
from generation.requests import llm_request
from persona.memory_structures.memory_blocks.node import CoreNode
from standard import FOCAL_POINT_SCHEMA, FOCAL_POINT_AUX_SCHEMAS, ROUTINE_SELECTION_SCHEMA, STANDARD_INSTRUCTIONS, STANDARD_ROUTINE_SELECTION_INSTRUCTIONS, FocalPointsSchema, GroundSchema, PlanSchema, ReflectSchema, RoutineSelectionSchema
from persona.agent import Agent
from typing import Dict, Optional, Any
from utils import increment_prompt_log_counter

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
        tools = agent.blackboard.generic_tools,
        instructions = agent.settings.planning.instructions,
        schema = PlanSchema.schema_json(indent=4),
        task = "Examine the available tools and your entity knowledge to formulate a plan aligned with your current goal"
    )
    response = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "plan"
    )["message"]["content"]

    
    increment_prompt_log_counter()

    return clean_up_plan(response)


def gen_grounding(
    agent: Agent,
    relevant_state: Dict,
    relevant_memory: Dict,
    entities: list[Dict],
    plan_task: Dict[str, Any]
    #actions_taken: list[ToolCall]
):
    class CustomSchemaA(BaseModel):
        thoughts: list[str] = Field(description="List of short strings describing the most important conclusions that were drawn from memories and entities")
        actions: list[str] = Field(description="List of short strings representing a batch of specific actions to take on entities in this moment that will contribute to the current task")

    system_prompt = create_instructions_sec([
        "Respond with a single JSON object",
        "Your json response must conform to the schema specified in #Response Schema Definition"
        "The only possible actions are suggested by the tags contained in each entity listed in #Entity Instances"
        "Only produce actions which are suggested by these tags, otherwise default to the action 'Complete this task'"
    ])

    user_prompt = \
        create_state_sec(relevant_state) \
        + create_memory_sec(relevant_memory) \
        + create_entities_sec(entities) \
        + create_goal_sec(agent.plan.goal) \
        + create_mainschema_sec(CustomSchemaA.schema_json(indent=4))

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]
        
    response_msg = llm_request(
        messages = messages,
        log_name = "ground_phase_1"
    )["message"]
    start = response_msg["content"].find('{')
    end = response_msg["content"].rfind('}') + 1
    clean_string = response_msg["content"][start:end]
    reasoning = json.loads(clean_string)

    increment_prompt_log_counter()
    
    system_prompt = create_instructions_sec([
        "Produce tool calls that attempt to execute the actions described in Task"
    ])

    user_prompt = \
        create_entities_sec(entities) \
        + create_memory_sec(relevant_memory) \
        + create_current_task_sec(reasoning) \
        + create_mainschema_sec(CustomSchemaA.schema_json(indent=4))

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]

    valid = False
    errors: list[str] = []
    final = None
    while not valid:
        response_msg = llm_request(
            messages = messages,
            tools = agent.blackboard.generic_tools,
            log_name = "ground_phase_2"
        )["message"]
        messages.append(ChatMessage(**response_msg))

        try:
            final = [ToolCall(name=d["function"]["name"], arguments=d["function"]["arguments"]) for d in response_msg["tool_calls"]]
            valid, errors = validate_ground(final, agent.blackboard.generic_tools)
        except json.JSONDecodeError as ex:
            errors = [f"Your response either didn't respect the schema or was ill-formated"]

        messages.append(ChatMessage(role="user", content= \
            "Your response includes the following errors:\n" + "\n".join([f"\t- {err}" for err in errors]) + "\n\n" + \
            "Provide a new response that corrects those errors"
        ))

    increment_prompt_log_counter()

    return reasoning, final


def gen_routine_selection(
    routines: list[AgentRoutine],
    relevant_state: Dict,
    relevant_memory: Dict,
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        memory = relevant_memory,
        routines = routines,
        instructions = STANDARD_INSTRUCTIONS + STANDARD_ROUTINE_SELECTION_INSTRUCTIONS,
        schema = RoutineSelectionSchema.schema_json(indent=4),
        task = "Select a routine and generate a fitting goal"
    )
    response = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "routine_goal"
    )["message"]["content"]

    
    increment_prompt_log_counter()

    return clean_up_routine_selection(response)


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
        schema = FocalPointsSchema.schema_json(indent=4),
        task = f"Respond with a list of {length} focal points that would be useful for retrieval of memories for this agent."
    )
    response = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "focal_points"
    )["message"]["content"]

    
    increment_prompt_log_counter()

    return clean_up_focal_points(response)


def gen_thought(
    agent: Agent,
    relevant_state: Dict,
    relevant_memory: Dict,
    entities: list[Dict]
):
    system_prompt, user_prompt = create_standard_prompt(
        state = relevant_state,
        goal = agent.plan.goal,
        memory = relevant_memory,
        entities = entities,
        instructions = agent.settings.reflection.instructions,
        schema = ReflectSchema.schema_json(indent=4),
        #aux_schemas = agent.settings.reflection.aux_schemas,
        task = "Make a thought."
    )
    response = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "thought"
    )["message"]["content"]

    
    increment_prompt_log_counter()

    return clean_up_thought(response)


def clean_up_plan(response_string: str) -> PlanSchema:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    obj = json.loads(clean_string)

    return PlanSchema.parse_obj(obj)


def clean_up_ground(response_string: str) -> GroundSchema:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    obj = json.loads(clean_string)

    return GroundSchema.parse_obj(obj)


def clean_up_focal_points(response_string: str) -> FocalPointsSchema:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    obj = json.loads(clean_string)

    return FocalPointsSchema.parse_obj(obj)


def clean_up_routine_selection(response_string) -> RoutineSelectionSchema:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    obj = json.loads(clean_string)

    return RoutineSelectionSchema.parse_obj(obj)


def clean_up_thought(response_string: str) -> ReflectSchema:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    obj = json.loads(clean_string)

    return ReflectSchema.parse_obj(obj)


def validate_ground(final: list[ToolCall], tools: list[Tool]) -> tuple[bool, list[str]]:
    result = True
    errors = []

    for call in final:
        tool_match = next((t for t in tools if t.function.name == call.name), None)
        if tool_match is None:
            result = False
            errors.append(f"You called '{call.name}', this tool does not exist")
        else:
            for key, arg in call.arguments.items():
                arg_match = tool_match.function.parameters.properties.get(key)
                if arg_match is None:
                    result = False
                    errors.append(f"Your call for '{call.name}' contains an argument '{key}', this argument does not exist")
                elif arg_match.enum is not None and arg not in arg_match.enum:
                    result = False
                    errors.append(f"In the call for '{call.name}', the value '{arg}' for the argument '{key}' is not valid, the value should be contained in this set: {set(arg_match.enum)}")

    return result, errors


def create_standard_prompt(
    task: Optional[str] = None,
    instructions: Optional[list[str]] = None,
    schema: Optional[str] = None,
    routines: Optional[list[AgentRoutine]] = None,
    goal: Optional[str] = None,
    state: Optional[Dict] = None,
    memory: Optional[Dict] = None,
    entities: Optional[list[Dict]] = None,
    affordances: Optional[list[Dict]] = None,
    plan_task: Optional[Dict] = None,
    actions_taken: Optional[list[ToolCall]] = None,
    tools: Optional[list[Tool]] = None
):
    system_prompt = ""
    user_prompt = ""

    ### System prompt

    if instructions is not None and len(instructions) > 0:
        system_prompt += create_instructions_sec(instructions)
    if schema is not None:
        system_prompt += create_mainschema_sec(schema)
    if tools is not None:
        system_prompt += create_tools_sec(tools)

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

    #user_prompt += create_task_sec(task, plan_task, actions_taken)

    return system_prompt, user_prompt


