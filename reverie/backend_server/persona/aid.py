from pydantic import BaseModel, Field
from typing import Dict, Literal, Any, Optional


class Contract(BaseModel):
    state_keys: list[str]
    memory_keys: list[str]


class ArgumentSimplified(BaseModel):
    type: Literal["string", "integer", "float", "boolean", "object", "list"]
    description: str
    tags: list[str] = []
    enum: Optional[list[str]] = None


class ToolSimplified(BaseModel):
    name: str
    description: str
    arguments: Dict[str, ArgumentSimplified]
    enabled: bool


class Property(BaseModel):
    type: Literal["string", "integer", "float", "boolean", "object", "list"]    
    description: str
    enum: Optional[list[str]] = None
    tags: list[str]


class Parameters(BaseModel):
    type: Literal["object"] = "object"
    required: list[str]
    properties: Dict[str, Property]


class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function

    @staticmethod
    def create(tool: ToolSimplified) -> "Tool":
        return Tool(function = Function(
            name = tool.name,
            description = tool.description,
            parameters = Parameters(
                required = [key for key, arg in tool.arguments.items()],
                properties = {key: Property(
                    type = arg.type,
                    tags = arg.tags,
                    description = arg.description,
                    enum = arg.enum
                ) for key, arg in tool.arguments.items()}
            )
        ))


class Configuration(BaseModel): #TODO complete
    config_1: str
    config_2: int


class SchemaField(BaseModel):
    description: str
    guidelines: str
    field_type: Literal["string", "integer", "float", "boolean", "object", "list"]
    sub_fields: Dict[str, "SchemaField"] = Field(default_factory=dict[str, "SchemaField"])

    
Schema = Dict[str, SchemaField]


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] #TODO specific type instead of Any if possible


class GroundingSequence(BaseModel):
    thinking: str
    tool_calls: list[ToolCall]
    context_score: int


class PlanStep(BaseModel):
    task: Dict
    actions: list[ToolCall] = Field(default_factory=list[ToolCall])
    complete: bool = False


class PlanStepLog(BaseModel):
    task: Dict
    actions: list[GroundingSequence] = Field(default_factory=list[GroundingSequence])


class SimpleSettings(BaseModel):
    instructions: list[str]
    contract: Contract
    main_schema: Schema


class ExtendedSettings(SimpleSettings):
    aux_schemas: Dict[str, Schema]


PlanningSettings = ExtendedSettings
GroundingSettings = ExtendedSettings
ReflectionSettings = ExtendedSettings
InteractionSettings = ExtendedSettings
RoutineGoalSelectionSettings = SimpleSettings

class Affordance(BaseModel):
    name: str
    description: str
    parameters: Property


class CoreNode(BaseModel):
    description: str
    entity_snapshots: Dict[str, str]


class Entity(BaseModel):
    id: str
    tags: list[str]
    description: str
    relations: Dict[str, CoreNode]


class AgentRoutine(BaseModel):
    name: str
    description: str


class SegmentedGround(BaseModel):
    reasoning: list[str]
    calls: list[ToolCall]


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[list[Dict]] = None
