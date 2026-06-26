from selenium import webdriver

from global_methods import *
from persona.aid import Configuration
from persona.cognitive_modules.reflect_ops import feed_event
from generation.requests import embedding_request
from utils import *
from maze import *

from fastapi import FastAPI, BackgroundTasks, HTTPException
from uuid import uuid4

from api_classes import *
from simulation import Simulation
from persona.agent import AgentSetup, MissingAgentRequirements, RepeatedSchemaNames

from typing import Dict
from persona.cognitive_modules.plan_ops import op_plan_full, op_plan, op_ground
from persona.aid import Tool, PlanStep, ToolCall


app = FastAPI()
jobs = {}
sim: Simulation = Simulation("only_sim") #TODO change if later more simultaneous sims are allowed 


# API enpoints

@app.post("/simulation", response_model=CreateSimResponse) #TODO only relevant if multiple simultaneous sims are allowed
def create_sim(request: CreateSimRequest):
    sim = Simulation(request.sim_id)


@app.post("/simulation/agents", response_model=str)
def create_agent(request: CreateAgentRequest):
    sim.add_agent_setup(request.agent_id, AgentSetup(
        request.goal,
        request.initial_state
    ))
    return "ok"


@app.post("/simulation/agents/{agent_id}/setup/config", response_model=str)
def setup_agent_config(agent_id: str, request: Configuration):
    sim.agents_setup[agent_id].set_config(request)
    return "ok"


@app.post("/simulation/agents/{agent_id}/setup/memory", response_model=str)
def setup_agent_memory(agent_id: str, request: SetMemoryRequest):
    sim.agents_setup[agent_id].set_memory(
        request.core_nodes,
        request.node_sections
    )
    return "ok"


@app.post("/simulation/agents/{agent_id}/setup/tools", response_model=str)
def setup_agent_tools(agent_id: str, request: list[Tool]):
    sim.agents_setup[agent_id].set_actions(request)
    return "ok"


@app.post("/simulation/agents/{agent_id}/setup/planning", response_model=str)
def setup_agent_planning(agent_id: str, request: PlanningSetupRequest):
    sim.agents_setup[agent_id].setup_planning(
        request.instructions,
        request.contract,
        request.aux_schemas
    )
    return "ok"


@app.post("/simulation/agents/{agent_id}/setup/grounding", response_model=str)
def setup_agent_grounding(agent_id: str, request: GroundingSetupRequest):
    sim.agents_setup[agent_id].setup_grounding(
        request.instructions,
        request.contract
    )
    return "ok"


@app.post("/simulation/agents/{agent_id}/setup/reflection", response_model=str)
def setup_agent_reflection(agent_id: str, request: ReflectionSetupRequest):
    sim.agents_setup[agent_id].setup_reflection(
        request.instructions,
        request.main_schema,
        request.aux_schemas,
        request.contract
    )
    return "ok" 


@app.post("/simulation/agents/{agent_id}/setup/interaction", response_model=str)
def setup_agent_interaction(agent_id: str, request: InteractionSetupRequest):
    sim.agents_setup[agent_id].setup_interaction()
    return "ok"


@app.post("/simulation/agents/{agent_id}/finalize", response_model=Dict)
def finilaze_agent(agent_id: str):
    response = {}
    try:
        sim.add_agent(agent_id)
        response["result"] = "success"
        return response
    except MissingAgentRequirements as error:
        response["result"] = "error"
        response["reason"] = error.reason
        response["missing_requirements"] = error.missing_requirements
        raise HTTPException(status_code=409, detail=response)
    except RepeatedSchemaNames as error:
        response["result"] = "error"
        response["reason"] = error.reason
        response["repeated_schema_names"] = error.repeated_names
        raise HTTPException(status_code=422, detail=response)


@app.post("/simulation/agents/{agent_id}/plan", response_model=str)
def plan_request(agent_id: str, request: PlanRequest):
    #TODO update time, etc, call the thing on recall
    op_plan(sim.get_agent(agent_id))
    return "ok"


@app.post("/simulation/agents/{agent_id}/ground", response_model=str)
def grounded_request(agent_id: str):
    op_ground(sim.get_agent(agent_id))
    return "ok"


@app.post("/simulation/agents/{agent_id}/feed_event", response_model=str)
def feed_event_request(agent_id: str, request: EventRequest):
    agent = sim.get_agent(agent_id)
    #feed_event(agent, request.event) TODO FIX
    return "ok"


@app.post("/simulation/agents/{agent_id}/next_action", response_model=ToolCall)
def next_action(agent_id: str):
    agent = sim.get_agent(agent_id)
    action = agent.plan.next_action()
    if action is None:
        raise HTTPException(status_code=404, detail="No next action available")
    return action


@app.post("/simulation/agents/{agent_id}/plan_all", response_model=str)
def plan_all_request(agent_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        op_plan_full,
        sim.get_agent(agent_id)
    )

    return "ok"


@app.get("/simulation/agents/{agent_id}/tools", response_model=list[Tool])
def get_tools(agent_id: str):
    return sim.get_agent(agent_id).blackboard.tools


@app.get("/simulation/agents/{agent_id}/plan", response_model=list[PlanStep])
def get_plan(agent_id: str):
    return sim.get_agent(agent_id).plan.steps


@app.get("/simulation/agents/{agent_id}/plan/actions", response_model=list[ToolCall])
def get_actions(agent_id: str):
    for step in reversed(sim.get_agent(agent_id).plan.steps):
        if step.actions and step.actions[-1].key == "completed_task":
            return step.actions[0:-1]
    return []


@app.get("/simulation/agents/{agent_id}/debug/cache", response_model=Dict[str, Dict[str, CoreNode]])
def debug_cache(agent_id: str):
    cache = sim.get_agent(agent_id).recall.cache
    clean = {sec_name: {key: node.core for key, node in sec.items()} for sec_name, sec in cache.sections.items()}
    return clean


@app.get("/simulation/agents/{agent_id}/debug/load_cache", response_model=Dict[str, Any])
def debug_load_cache(agent_id: str, request: LoadCacheDebugRequest):
    return sim.get_agent(agent_id).recall.load_cache_debug(request.subject)
