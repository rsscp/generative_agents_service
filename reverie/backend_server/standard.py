from persona.aid import SchemaField, Tool, Function, Parameters

ENTITY_FENCES = ("*<", ">")

#---------------------------

STANDARD_INSTRUCTIONS = [
  "You response will follow the JSON structure specified in Schema.",
  "Always comply with JSON formating.",
  "Any text should be written in English",
  "Reduce thinking to at most two cycles of reflection",
  f"Entity instances represent places, objects or characters and they are presented in the format {ENTITY_FENCES[0]}entity{ENTITY_FENCES[1]}",
  f"When referencing an entity instance always use their formated id, such as {ENTITY_FENCES[0]}entity{ENTITY_FENCES[1]}",
  "You are only able to reference entity instances that are presented in Entity Instances"
]

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
        )
    }
}

#---------------------------

STANDARD_GROUNDING_INSTRUCTIONS = [
  "Tool calls can only contain primitive values or references to real entity instances",
  "Do not make up values when filling tool call arguments.",
  "The generated tool call sequence must end with a call to \"completed_task\""
]
GROUND_SCHEMA = {
    "sequencial_actions": SchemaField(
        description = "List of sequencial tool calls that aim to complete the task",
        guidelines = "Should have as many tool calls as necessary to complete task",
        field_type = "list"
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
  Tool(
    type = "function",
    function = Function(
      name = "completed_task",
      description = "This action is used to end a sequence of actions that already acomplish the described task",
      parameters = Parameters(
        type = "object",
        required = [],
        properties = {}
      )
    )
  ),
  Tool(
    type = "function",
    function = Function(
      name = "completed_task",
      description = "This action is used to end a sequence of actions that already acomplish the described task",
      parameters = Parameters(
        type = "object",
        required = [],
        properties = {}
      )
    )
  )
]

BLOCKING_ACTIONS = []