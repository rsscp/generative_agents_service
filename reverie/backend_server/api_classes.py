from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from persona.aid import Entity, AgentRoutine, Schema, SchemaField, Contract, ToolCall
from persona.memory_structures.memory_blocks.node import CoreNode


# ---------- Simulation ----------


class CreateSimRequest(BaseModel):
    sim_id: str


class CreateSimResponse(BaseModel):
    result_code: str
    registered_sim_id: Optional[str] = None
    errors: Optional[list[str]] = None


# ---------- Agent Setup ----------


class CreateAgentRequest(BaseModel):
    agent_id: str
    state: Dict[str, Any]
    entities: list[Entity]
    routines: list[AgentRoutine]


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


class SetMemoryRequest(BaseModel):
    core_nodes: list[CoreNode]
    node_sections: Dict[str, list[CoreNode]]


# ---------- Proactive Requests ----------


class ProactiveContext(BaseModel):
    state: Dict
    time: float


class ProactiveRequest(BaseModel):
    context: ProactiveContext
    

class PlanRequest(ProactiveRequest):
    temporary_field: bool = False #TODO DELETE


class GroundRequest(ProactiveRequest):
    pass


class FeedbackRequest(BaseModel):
    state: Optional[Dict[str, Any]] = None
    tools_state: Optional[Dict[str, bool]]
    entities: Optional[list[Entity]] = None
    event: CoreNode


class InteractRequest(ProactiveRequest):
    agents: list[str]


class NextActionRequest(BaseModel):
    correction: bool


# ---------- Debug Requests ----------


class LoadCacheDebugRequest(BaseModel):
    subject: str


# ---------- Debug Requests ----------


class NextActionResponse(BaseModel):
    tool_call: ToolCall
    is_affordance: bool
    target_entity_id: Optional[str] = None