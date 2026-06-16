from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from persona.aid import Schema, SchemaField, Contract
from reverie.backend_server.persona.memory_structures.memory_blocks.node import CoreNode


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


class PlanningSetupRequest(BaseModel):
    instructions: list[str]
    contract: Contract
    aux_schemas: Dict[str, Schema]


class GroundingSetupRequest(BaseModel):
    instructions: list[str]
    contract: Contract


class ReflectionSetupRequest(BaseModel):
    instructions: list[str]
    contract: Contract
    main_schema: Schema
    aux_schemas: Dict[str, Schema]


class InteractionSetupRequest(BaseModel):
    instructions: list[str]
    contract: Contract
    main_schema: Schema
    aux_schemas: Dict[str, Schema]


class FeedEventRequest(BaseModel):
    event: Dict[str, Any]
    weight: float = 1.0


class SetMemoryRequest(BaseModel):
    core_nodes: list[CoreNode]
    node_sections: Dict[str, list[CoreNode]]













    

