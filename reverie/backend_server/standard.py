from pydantic import BaseModel, Field
from typing import Dict, Any
from persona.aid import Property, SchemaField, Tool, Function, Parameters, ToolSimplified, ArgumentSimplified

ENTITY_FENCES = ("*", "*")

#---------------------------

STANDARD_INSTRUCTIONS = [
  "You response will follow the JSON structure specified in Schema, complying with JSON formatting.",
  "Any text should be written in English",
  f"Entity IDs are presented in the format {ENTITY_FENCES[0]}entity{ENTITY_FENCES[1]}",
  "You are only able to reference entity instances that are presented in Entity Instances"
]


#---------------------------

STANDARD_ROUTINE_SELECTION_INSTRUCTIONS = [
  "Routines represent a general guide for behaviour",
  "Choose the single routine that seems more appropriate given most recent memories and general identity",
  "Choose also a specific goal that fits in the description of the chosen routine",
  "Fill the response schema with those choices"
]
ROUTINE_SELECTION_SCHEMA = {
  "routine_choice": SchemaField(
    description = "Name of the chosen routine",
    guidelines = "Has to be presented under Routines",
    field_type = "string"
  ),
  "goal": SchemaField(
    description = "Sentence describing the generated goal",
    guidelines = "Should be in line with the chosen routine",
    field_type = "string"
  )
}

class RoutineSelectionSchema(BaseModel):
    routine_choice: str = Field(description="Name of the chosen routine")
    goal: str = Field(description="Sentence describing the goal generated according to the selected routine")


#---------------------------

STANDARD_PLANNING_INSTRUCTIONS = [
  "Do not make up facts",
  "All information used on planning will be pulled from this message."
]
PLAN_SCHEMA = {
  "plan_steps": SchemaField(
    description = "List of sequencial steps that make up the plan.",
    guidelines = "Should have at most 5 items. Each item follows the Step Schema.",
    field_type = "list"
  )
}
PLAN_AUX_SCHEMAS = {
  "Step": {
    "task": SchemaField(
      description = "Sentence specifying the task for a single step of the plan.",
      guidelines = "Short sentence, less than 30 words",
      field_type = "string"
    ),
    "completion_condition": SchemaField(
      description = "Sentence specifying under what general conditions can this task be considered completed",
      guidelines = "Short sentence, less than 30 words",
      field_type = "string"
    )
  }
}

class PlanStepsSchema(BaseModel):
    broad_task: str = Field(description="Sentence specifying the task for a single step of the plan in less than 30 words")
    positive_guidelines: list[str] = Field(description="Description list of positive guidelines for this task, what should be generaly persued")
    negative_guidelines: list[str] = Field(description="Description list of negative guidelines for this task, what should be generaly avoided")

class PlanSchema(BaseModel):
    plan_steps: list[PlanStepsSchema] = Field(description="List of sequencial logical steps that make up the plan")

class PlanStepLogSchema(BaseModel):
    task: Dict
    actions: list["GroundSchema"] = Field(default_factory=list["GroundSchema"])


#---------------------------

STANDARD_GROUNDING_INSTRUCTIONS = [
  "Tool string properties corresponding to entity IDs can only be filled with the values enumerated for that property",
  "Tool property enumerated values indicate only that a value is possible, not that it is an appropriate choice",
  "If you deduce through the presented memories that the task has been completed already, choose the completed_task tool",
  "You call call a sequence of more than one tool if each call in that sequence doen't rely on feedback from the previous call",
  "Remember to, apart from tool calling, respond using the presented Schema"
  "Pay special attention too enum values in tool arguments, as argument values outside the enum set will result in failure"
]
GROUND_SCHEMA = {
  "thinking": SchemaField(
    description = "Text describing the reasoning behind the possible choice of actions/tools",
    guidelines = "Should be less than 50 words",
    field_type = "string"
  ),
  "tool_calls": SchemaField(
    description = "Sequence of tool calls to be executed next",
    guidelines = "The items in this list should comply with the ToolCall schema",
    field_type = "list"
  ),
  "context_score": SchemaField(
    description = "Sequence of tool calls to be executed next",
    guidelines = "The items in this list should comply with the ToolCall schema",
    field_type = "list"
  )
}
GROUND_AUX_SCHEMAS = {
  "ToolCall": {
    "name": SchemaField(
      description = "Name of the tool being called",
      guidelines = "Must correspond to the name of a valid tool",
      field_type = "string"
    ),
    "arguments": SchemaField(
      description = "Dictionary of arguments for this tool call of which the values comply with the choosen tool",
      guidelines = "Each argument should be filled in the key value format, using the correct parameter names specified by the tool",
      field_type = "object"
    )
  }
}

class ToolCallSchema(BaseModel):
    name: str = Field(description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(description="Dictionary of arguments for this tool call")

class GroundSchema(BaseModel):
    thinking: str = Field(description="Small text describing the reasoning behind the possible choice of tools")
    objectives: list[str] = Field(description="List of small sentences describing real world positive outcomes that are expected from these actions")
    avoidances: list[str] = Field(description="List of small sentences describing real world consequences being avoided")
    tool_calls: list[ToolCallSchema] = Field(description="Sequence of tool calls to be executed next")
    context_score: int = Field(description="Value between 0-100 indicating how appropriate the choosen actions are according to the current context")


#---------------------------

STANDARD_REFLECTION_INSTRUCTIONS = [
  "Generate a single thought object",
  "Entities can be mentioned in the description of the thought if they are contained in Entity Instances"
]
REFLECT_SCHEMA = {
  "thought": SchemaField(
    description = "Object representing a thought for this agent",
    guidelines = "All properties are required",
    field_type = "object",
    sub_fields = {
      "thought_description": SchemaField(
        description = "Description of the thought deduced recent events and other memories",
        guidelines = "Should be the next action to take in a sequence of previous tool calls",
        field_type = "object"
      ),
      "thought_poignancy": SchemaField(
        description = "Poignancy or importance of this thought according to the agent's memory and identity",
        guidelines = "Must be between 0 and 100",
        field_type = "integer"
      ),
      "entities_mentioned": SchemaField(
        description = "List of string entity ids",
        guidelines = "Every entity id in this list must have been mentioned in thought_description",
        field_type = "list"
      )
    }
  )
}

class ThoughtSchema(BaseModel):
    description: str = Field(description="Description of the thought deduced recent events and other memories")
    poignancy: int = Field(description="Value between 0-100 that rates the overall importance of this thought object")
    entities_mentioned: list[str] = Field(description="List of entity string IDs that are mentioned in this thought object")

class ReflectSchema(BaseModel):
    thought: ThoughtSchema = Field(description="Object representing a thought")


#---------------------------

FOCAL_POINT_SCHEMA = {
  "focal_points": SchemaField(
    description = "List of focal points that will be used for memory retrieval.",
    guidelines = "Each focal point should be a short and semantically meaningful phrase.",
    field_type = "list"
  )
}
FOCAL_POINT_AUX_SCHEMAS = {
  "key": SchemaField(
    description = "Generic and semantically meaningful phrase relating to the current goal or task",
    guidelines = "Short phrase, than 10 words",
    field_type = "string"
  )
}

class PointSchema(BaseModel):
    key: str = Field(description="Generic and semantically meaningful phrase relating to the current goal or task")

class FocalPointsSchema(BaseModel):
    focal_points: list[PointSchema]


#---------------------------

DEFAULT_ACTIONS = [
  Tool.create(ToolSimplified(
    name = "completed_task",
    description = "This action is used to end a sequence of actions that already acomplish the described task",
    arguments = {}
  )),
  # Tool.create(ToolSimplified(
  #   name = "execute_affordance",
  #   description = "This action is used to execute an affordance listed under Entity Affordances",
  #   arguments = {
  #     "affordance_id": ArgumentSimplified(
  #       type = "string",
  #       description = "Id of the affordance to execute"
  #     ),
  #     "affordance_arguments": ArgumentSimplified(
  #       type = "list",
  #       description = "List of arguments to be passed for affordance execution"
  #     )
  #   }
  # ))
]


#---------------------------

STANDARD_MEMORY_SECTIONS = ["completed_tasks", "events", "thoughts", "errors"]