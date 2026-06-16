from pydantic import BaseModel, Field
from typing import Dict, Literal, Any


class Contract(BaseModel):
    state_keys: list[str]
    memory_keys: list[str]


class ActionParam(BaseModel):
    type: Literal["string", "integer", "float", "boolean", "object", "list"]
    description: str
    restrictions: list[str]


class Property(BaseModel):
    type: Literal["string", "integer", "float", "boolean"]
    description: str


class Parameters(BaseModel):
    type: Literal["object"]
    required: list[str]
    properties: Dict[str, Property]


class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters


class Tool(BaseModel):
    type: Literal["function"]
    function: Function


class Configuration(BaseModel): #TODO complete
    config_1: str
    config_2: int


class SchemaField(BaseModel):
    description: str
    guidelines: str
    field_type: Literal["string", "integer", "float", "boolean", "object", "list"]
    sub_fields: Dict[str, "SchemaField"] = Field(default_factory=dict[str, "SchemaField"])

    
Schema = Dict[str, SchemaField]


class ActionCall(BaseModel):
    key: str
    arguments: Dict[str, Any] #TODO specific type instead of Any if possible


class PlanStep(BaseModel):
    task: Dict
    actions: list[ActionCall] = Field(default_factory=list[ActionCall])


class SimpleSettings(BaseModel):
    instructions: list[str]
    contract: Contract
    main_schema: Schema

class ExtendedSettings(SimpleSettings):
    main_schema: Schema
    aux_schemas: Dict[str, Schema]


PlanningSettings = ExtendedSettings
GroundingSettings = SimpleSettings
ReflectionSettings = ExtendedSettings
InteractionSettings = ExtendedSettings