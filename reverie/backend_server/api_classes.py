from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from persona.aid import SchemaField


class Contract(BaseModel):
    state_keys: list[str]
    memory_keys: list[str]


class SchemaSet(BaseModel):
    schema_definitions: Dict[str, Dict[str, SchemaField]] = Field(default_factory=dict)


class CreateSimRequest(BaseModel):
    sim_id: str


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


class CreateAgentRequest(BaseModel):
    agent_id: str
    goal: str
    initial_state: Dict[str, Any]

    
class CreateAgentResponse(BaseModel):
    result_code: str
    registered_agent_id: Optional[str] = None
    status: Optional[str] = None
    errors: list[str] = []


class SetAgentContractsRequest(BaseModel):
    plan_contract: Contract
    thought_contract: Contract
    interactions_contracts: list = [] #TODO Worry about interactions later


class SetAgentPlanReqRequest(BaseModel):
    instructions: list[str]
    plan_main_schema: Dict[str, SchemaField]
    plan_aux_schemas: Dict[str, Dict[str, SchemaField]] = Field(default_factory=dict)
















    

