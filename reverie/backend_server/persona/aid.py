from pydantic import BaseModel, Field
from typing import Dict, Literal


class Contract(BaseModel):
    state_keys: list[str]
    memory_keys: list[str]


class ActionParam(BaseModel):
    type: Literal["string", "integer", "float", "boolean", "object", "list"]
    description: str
    restrictions: list[str]


class Action(BaseModel):
    description: str
    parameters: Dict[str, ActionParam]


class Configuration(BaseModel): #TODO complete
    config_1: str
    config_2: int


class SchemaField(BaseModel):
    description: str
    guidelines: str
    field_type: Literal["string", "integer", "float", "boolean", "object", "list"]
    sub_fields: dict[str, "SchemaField"] = Field(default_factory=dict)
