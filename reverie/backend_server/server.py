from inspect import Parameter

from selenium import webdriver

from global_methods import *
from persona.aid import Configuration, Entity, Function, Parameters, PlanStepLog, Property, ToolSimplified
from persona.cognitive_modules.reflect_ops import op_feed_event
from generation.requests import embedding_request
from persona.memory_structures.memory_blocks.node import MemoryNode
from utils import *
from maze import *

from fastapi import FastAPI, BackgroundTasks, HTTPException
from uuid import uuid4

from api_classes import *
from simulation import Simulation
from persona.agent import AgentSetup, MissingAgentRequirements, RepeatedSchemaNames

from typing import Dict
from persona.cognitive_modules.plan_ops import op_plan_full, op_plan, op_ground, op_select_routine, op_select_routine_setup
from persona.aid import Tool, PlanStep, ToolCall


app = FastAPI()
jobs = {}
sim: Simulation = Simulation("only_sim") #TODO change if later more simultaneous sims are allowed 


import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)


# Setup/Creation requests

@app.post("/simulation", response_model=CreateSimResponse) #TODO only relevant if multiple simultaneous sims are allowed
def create_sim(request: CreateSimRequest):
    sim = Simulation(request.sim_id)


@app.post("/simulation/agents", response_model=str)
def create_agent(request: CreateAgentRequest):
    sim.add_agent_setup(request.agent_id, AgentSetup(
        request.state,
        request.entities,
        request.routines
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
def setup_agent_tools(agent_id: str, request: Dict[str, ToolSimplified]):
    print(json.dumps({k: v.dict() for k, v in request.items()}, indent=4)) #TODO DELETE
    processed = [Tool.create(tool) for tool in request.values()]
    sim.agents_setup[agent_id].set_tools(processed)
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
        op_select_routine_setup(sim.agents_setup[agent_id])
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
    

# Direct Requests


@app.post("/simulation/agents/{agent_id}/plan", response_model=str)
def plan_request(agent_id: str, request: PlanRequest):
    #TODO update time, etc, call the thing on recall
    op_plan(sim.get_agent(agent_id))
    return "ok"


@app.post("/simulation/agents/{agent_id}/ground", response_model=str)
def grounded_request(agent_id: str):
    op_ground(sim.get_agent(agent_id))
    return "ok"


@app.post("/simulation/agents/{agent_id}/plan_all", response_model=str)
def plan_all_request(agent_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        op_plan_full,
        sim.get_agent(agent_id)
    )
    return "ok"


# Feedback requests


@app.post("/simulation/agents/{agent_id}/feedback", response_model=str)
def feeback_request(agent_id: str, request: FeedbackRequest):
    agent = sim.get_agent(agent_id)

    agent.blackboard.state = request.state
    agent.blackboard.attended_entities = {entity.id: entity for entity in request.entities}
    agent.blackboard.update_tool_arguments()

    op_feed_event(agent, RawNode(
        description = request.event.description,
        entity_keys = request.event.entity_keys
    ))
    
    return "ok"


@app.post("/simulation/agents/{agent_id}/next_action", response_model=NextActionResponse)
def next_action(agent_id: str):
    agent = sim.get_agent(agent_id)
    # old_task = agent.plan.current_task()

    # if old_task is None:
    #     op_plan(agent) 
    # if agent.plan.get_action() is None:
    #     op_ground(agent)

    # agent.plan.advance_index()
    # action = agent.plan.get_action()

    signal, action = "", None

    while signal != "ok" and action is None:
        signal, action = agent.plan.dequeue_action()

        if signal == "needs_plan":
            op_plan(agent)
        elif signal == "needs_ground":
            op_ground(agent)

    assert(action is not None)

    is_affordance = action.name == "execute_affordance"
    tool_call = action
    target_entity_id = None

    if is_affordance:
        parts = str(action.arguments["affordance_id"]).split(".")
        target_entity_id = parts[0]
        tool_call = ToolCall(
            name = parts[1],
            arguments = action.arguments
        )

    response = NextActionResponse(
        tool_call = tool_call,
        is_affordance = is_affordance,
        target_entity_id = target_entity_id
    )

    return response


# Getter requests


@app.get("/simulation/agents/{agent_id}/tools", response_model=list[Tool])
def get_tools(agent_id: str):
    return sim.get_agent(agent_id).blackboard.generic_tools


@app.get("/simulation/agents/{agent_id}/plan", response_model=list[PlanStepLog])
def get_plan(agent_id: str):
    return sim.get_agent(agent_id).plan.log_steps


@app.get("/simulation/agents/{agent_id}/plan/actions", response_model=list[ToolCall])
def get_actions(agent_id: str):
    for step in reversed(sim.get_agent(agent_id).plan.steps):
        if step.actions and step.actions[-1].name == "completed_task":
            return step.actions[0:-1]
    return []


@app.get("/simulation/agents/{agent_id}/routine_goal", response_model=Dict)
def get_routine_goal(agent_id: str):
    return {
        "routine": sim.get_agent(agent_id).plan.routine,
        "goal": sim.get_agent(agent_id).plan.goal
    }


# Debug Requests


@app.get("/simulation/agents/{agent_id}/debug/cache", response_model=Dict[str, Dict[str, CoreNode]])
def debug_cache(agent_id: str):
    cache = sim.get_agent(agent_id).recall.cache
    clean = {sec_name: {key: node.core for key, node in sec.items()} for sec_name, sec in cache.sections.items()}
    return clean


@app.get("/simulation/agents/{agent_id}/debug/load_cache", response_model=Dict[str, Any])
def debug_load_cache(agent_id: str, request: LoadCacheDebugRequest):
    return sim.get_agent(agent_id).recall.load_cache_debug(request.subject)


# Test requests


@app.post("/simulation/agents/{agent_id}/test/set_entities", response_model=str)
def test_set_entities(agent_id: str, request: Dict[str, Entity]):
    sim.get_agent(agent_id).blackboard.attended_entities = request
    return "ok"
