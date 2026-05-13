from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional


class FieldSchema(BaseModel):
    description: str
    field_type: Literal["string", "integer", "float", "boolean", "object", "list"]
    sub_fields: dict[str, "FieldSchema"] = Field(default_factory=dict)


class Schema(BaseModel):
    description: str
    fields_definitions: Dict[str, FieldSchema] = Field(default_factory=dict)


class SchemaSet(BaseModel):
    schema_definitions: Dict[str, Schema] = Field(default_factory=dict)


class CreateSimRequest(BaseModel):
    sim_id: str
    memory_schemas: SchemaSet = Field(default_factory=SchemaSet)
    core_schemas: SchemaSet = Field(default_factory=SchemaSet)
    plan_schemas: SchemaSet = Field(default_factory=SchemaSet)


class CreateSimResponse(BaseModel):
    result_code: str
    registered_sim_id: Optional[str] = None
    errors: Optional[list[str]] = None


class PlanRequest(BaseModel):
    TODOOOO: str #TODO


class PlanResponse(BaseModel):
    TODOOOO: str #TODO

class MemoryList(BaseModel):
    memories: list[Dict]

class MemoryListRuleset(BaseModel):
    memory_parameters: list[str] = Field(default_factory=list)

class Ruleset(BaseModel):
    instructions: list[str]
    state_parameters: list[str] = Field(default_factory=list) #TODO move to contract
    memory_lists: Dict[str, MemoryListRuleset] = Field(default_factory=dict)#TODO move to contract

class ActionParameterDef(BaseModel):
    type: Literal["string", "integer", "float", "boolean", "object", "list"]
    restrictions: list[str]

class ActionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ActionParameterDef]

class CreateAgentRequest(BaseModel):
    agent_id: str
    goal: str
    initial_state: Dict[str, Any]
    initial_memory_lists: Dict[str, MemoryList]
    initial_config: Dict[str, Any]
    available_actions: Dict[str, ActionDef]
    plan_gen_ruleset: Ruleset
    plan_gen_schema: Schema
    thought_gen_ruleset: Ruleset
    thought_gen_schema: Schema
    
class CreateAgentResponse(BaseModel):
    result_code: str
    registered_agent_id: Optional[str] = None
    status: Optional[str] = None
    errors: list[str] = []