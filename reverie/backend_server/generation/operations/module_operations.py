from pydantic import BaseModel, Field

from persona.aid import AgentRoutine, ChatMessage, GroundingSequence, Schema, Tool, ToolCall, ToolSimplified
from generation.prompt_building import *
from generation.requests import llm_request
from persona.aid import CoreNode
from persona.aid import ChatMessage
from standard import FOCAL_POINT_SCHEMA, FOCAL_POINT_AUX_SCHEMAS, ROUTINE_SELECTION_SCHEMA, STANDARD_INSTRUCTIONS, STANDARD_ROUTINE_SELECTION_INSTRUCTIONS, FocalPointsSchema, GroundSchema, PlanSchema, ThoughtSchema, RoutineSelectionSchema
from persona.agent import Agent
from typing import Dict, Optional, Any
from utils import increment_prompt_log_counter, log_text

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
        tools = [Tool.create(t) for t in agent.blackboard.generic_tools.values()],
        instructions = agent.settings.planning.instructions,
        schema = PlanSchema.schema_json(indent=4),
        task = "Examine the available tools and your entity knowledge to formulate a plan aligned with your current goal"
    )
    content, tool_calls = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "plan",
        content_format = "json"
    )

    if content is not None:
        increment_prompt_log_counter()
        return clean_up_plan(content)
    else:
        raise Exception("Content was null while generating plan")


def gen_grounding(
    agent: Agent,
    relevant_state: Dict,
    relevant_memory: Dict,
    entities: list[Dict],
    plan_task: Dict[str, Any],
    failed_action: Optional[ToolCall] = None
    #actions_taken: list[ToolCall]
):
    # system_prompt = create_instructions_sec([
    #     "Respond with a single JSON object",
    #     "Your json response must conform to the schema specified in #Response Schema Definition"
    #     "The only possible actions are suggested by the tags contained in each entity listed in #Entity Instances"
    #     "Only produce actions which are suggested by these tags, otherwise default to the action 'Complete this task'"
    # ])

    system_prompt = create_instructions_sec([
        "Respond with natural text",
        "Limit your response to less then 50 words"
    ])

    user_prompt = \
        create_state_sec(relevant_state) \
        + create_memory_sec(relevant_memory) \
        + create_entities_sec(entities) \
        + create_goal_sec(agent.plan.goal) \
        + create_recent_thoughts_sec(agent.recall.recent_thoughts) \
        + create_task_sec("Continue the thought sequence in by adding a thought derived from your goals, memories, previous thoughts and the current state of the world. Also finish with a suggestion to what the next immediate action should be, for example in the format: 'Next call: <chosen action>'.")
        #+ create_mainschema_sec(CustomSchemaA.schema_json(indent=4))
        #+ create_current_task_sec(plan_task) \

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]
        
    content, tool_calls = llm_request(
        messages = messages,
        log_name = "ground_phase_1",
        content_format = "text",
        tools = [Tool.create(t) for t in agent.blackboard.generic_tools.values()]
    )

    if content is None:
        log_text("No content was created", "error")
        raise Exception("Content is null at generation of fist phase of grounding")
    else:
        increment_prompt_log_counter()

    agent.recall.add_recent_thought(content)

    user_prompt = "Now produce tool calls that attempt to execute the action you indicated"

    if failed_action is not None:
        user_prompt += create_failed_action_sec(failed_action)

    reasoning_message = ChatMessage(role = "assistant")
    if content is not None:
        reasoning_message.content = content
    if tool_calls is not None:
        reasoning_message.tool_calls = tool_calls

    messages += [
        reasoning_message,
        ChatMessage(role="user", content=user_prompt)
    ]

    valid = False
    errors: list[str] = []
    final = None
    while not valid:
        content, tool_calls = llm_request(
            messages = messages,
            tools = [
                Tool.create(t)
                for t in agent.blackboard.generic_tools.values()
                if t.enabled
                if False not in [
                    a.enum is not None and len(a.enum) > 0
                    for a in t.arguments.values()
                ]
            ],
            content_format = "text",
            log_name = "ground_phase_2",
        )

        new_message = ChatMessage(role = "assistant")
        if content is not None:
            new_message.content = content
        if tool_calls is not None:
            new_message.tool_calls = tool_calls

        messages.append(new_message)

        if tool_calls is None:
            log_text("No tool calls were created", "error")
            raise Exception("Tool_calls is null at generation of second phase of grounding")
        else:
            increment_prompt_log_counter()

        try:
            final = [ToolCall(name=d["function"]["name"], arguments=json.loads(d["function"]["arguments"])) for d in tool_calls]
            valid, errors = validate_ground(final, [t for t in agent.blackboard.generic_tools.values() if t.enabled])
        except json.JSONDecodeError as ex:
            errors = [f"Your response either didn't respect the schema or was ill-formated"]

        messages.append(ChatMessage(role="user", content= \
            "Your response includes the following errors:\n" + "\n".join([f"\t- {err}" for err in errors]) + "\n\n" + \
            "Provide a new response that corrects those errors"
        ))

    increment_prompt_log_counter()

    return final


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
    content, tool_calls = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "routine_goal",
        content_format = "json"
    )

    if content is not None:
        increment_prompt_log_counter()
        return clean_up_routine_selection(content)
    else:
        raise Exception("Content was null while generating routine selection and goal")


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
    content, tool_calls = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "focal_points",
        content_format = "json"
    )

    if content is not None:
        increment_prompt_log_counter()
        return clean_up_focal_points(content)
    else:
        raise Exception("Content was null while generating focal_points")


def gen_node_poignancy(
    agent: Agent,
    relevant_memory: Dict,
    description: str
) -> int:
    system_prompt = create_instructions_sec([
        "Your response will be a single integer, between 0 and 100, which reflects the importance of the information contained in the presented object",
        "Your response will not contain JSON or aditional text"
    ])

    user_prompt = create_goal_sec(agent.plan.goal) \
        + create_memory_sec(relevant_memory) \
        + create_event_sec(description) \
        + create_task_sec("Respond with an integer between 0 and 100 representing the overall importance of the event presented in Event")

    content, tool_calls = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "poignancy",
        content_format = "text"
    )

    if content is not None:
        increment_prompt_log_counter()
        return clean_up_node_poignancy(content)
    else:
        raise Exception("Content was null while generating node poignancy")


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
        schema = ThoughtSchema.schema_json(indent=4),
        #aux_schemas = agent.settings.reflection.aux_schemas,
        task = "Make a thought."
    )
    content, tool_calls = llm_request(
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ],
        log_name = "thought",
        content_format = "json"
    )

    if content is not None:
        increment_prompt_log_counter()
        return clean_up_thought(content)
    else:
        raise Exception("Content was null while generating thought")


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


def clean_up_node_poignancy(response_string: str) -> int:
    return int(response_string)


def clean_up_routine_selection(response_string) -> RoutineSelectionSchema:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    obj = json.loads(clean_string)

    return RoutineSelectionSchema.parse_obj(obj)


def clean_up_thought(response_string: str) -> ThoughtSchema:
    start = response_string.find('{')
    end = response_string.rfind('}') + 1
    clean_string = response_string[start:end]
    obj = json.loads(clean_string)

    return ThoughtSchema.parse_obj(obj)


def validate_ground(final: list[ToolCall], tools: list[ToolSimplified]) -> tuple[bool, list[str]]:
    result = True
    errors = []

    for call in final:
        tool_match = next((t for t in tools if t.name == call.name), None)
        if tool_match is None:
            result = False
            errors.append(f"You called '{call.name}', this tool does not exist")
        else:
            for key, arg in call.arguments.items():
                arg_match = tool_match.arguments.get(key)
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
