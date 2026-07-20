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
    "broad_task": SchemaField(
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


#---------------------------

STANDARD_GROUNDING_INSTRUCTIONS = [
  "Tool string properties corresponding to entity IDs can only be filled with the values enumerated for that property",
  "Tool property enumerated values indicate only that a value is possible, not that it is an appropriate choice",
  "If you deduce through the presented memories that the task has been completed already, choose the completed_task tool",
  "You call call a sequence of more than one tool if each call in that sequence doen't rely on feedback from the previous call",
  "Remember to, apart from tool calling, respond using the presented Schema"
]
GROUND_SCHEMA = {
  "thinking": SchemaField(
    description = "Text describing the reasoning behind the possible choice of tools",
    guidelines = "Should be less than 50 words",
    field_type = "string"
  )
}


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


#---------------------------

FOCAL_POINT_SCHEMA = {
  "focal_points": SchemaField(
    description = "List of focal points that will be used for memory retrieval.",
    guidelines = "Each focal point should be a short and semantically meaningful phrase.",
    field_type = "list"
  )
}
FOCAL_POINT_AUX_SCHEMAS = {
  "Point": {
    "key": SchemaField(
      description = "Semantically meaningful phrase.",
      guidelines = "Short phrase, than 10 words",
      field_type = "string"
    )
  }
}


#---------------------------

NODE_REQ_SCHEMA = {
  "node_description": SchemaField(
    description = "Description of a node",
    guidelines = "Should only contain information available in the raw node JSON object",
    field_type = "string"
  ),
  "node_poignancy": SchemaField(
    description = "Poignancy of a node's content",
    guidelines = "Should be a value between 0 and 100",
    field_type = "integer"
  )
}


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

STANDARD_MEMORY_SECTIONS = ["completed_tasks", "events", "thoughts"]