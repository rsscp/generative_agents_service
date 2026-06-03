from selenium import webdriver

from global_methods import *
from persona.aid import Configuration
from utils import *
from maze import *

from fastapi import FastAPI

from api_classes import *
from simulation import Simulation
from persona.agent import AgentSetup, MissingAgentRequirements

from typing import Dict
from persona.cognitive_modules.plan_alt import plan_broad, plan_grounded
from persona.aid import Action, PlanStep, ActionCall


app = FastAPI()
sim: Simulation = Simulation("only_sim") #TODO change if later more simultaneous sims are allowed 


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


@app.post("/simulation/agents/{agent_id}/config", response_model=str)
def set_agent_config(agent_id: str, request: Configuration):
    sim.agents_setup[agent_id].set_config(request)
    return "ok"


@app.post("/simulation/agents/{agent_id}/memory", response_model=str)
def set_agent_memory(agent_id: str, request: Dict[str, list]):
    sim.agents_setup[agent_id].set_memory_lists(request)
    return "ok"


@app.post("/simulation/agents/{agent_id}/actions", response_model=str)
def set_agent_actions(agent_id: str, request: list[Action]):
    sim.agents_setup[agent_id].set_actions(request)
    return "ok"


@app.post("/simulation/agents/{agent_id}/contracts", response_model=str)
def set_agent_contracts(agent_id: str, request: SetAgentContractsRequest):
    sim.agents_setup[agent_id].set_plan_contract(request.plan_contract)
    sim.agents_setup[agent_id].set_thought_contract(request.thought_contract)
    sim.agents_setup[agent_id].set_interaction_contracts() #TODO Worry about interactions later
    return "ok"


@app.post("/simulation/agents/{agent_id}/planning_config", response_model=str)
def set_agent_planning_req(agent_id: str, request: SetAgentPlanReqRequest):
    sim.agents_setup[agent_id].set_plan_req(
        request.instructions,
        request.plan_aux_schemas
    )
    return "ok"


@app.post("/simulation/agents/{agent_id}/grounded_planning_config", response_model=str)
def set_agent_grounded_planning_req(agent_id: str, request: SetAgentPlanGroundedReqRequest):
    sim.agents_setup[agent_id].set_plan_grounded_req(
        request.instructions,
    )
    return "ok"


@app.post("/simulation/agents/{agent_id}/finalize", response_model=list[str])
def finilaze_agent(agent_id: str):
    try:
        sim.add_agent(agent_id)
        return []
    except MissingAgentRequirements as error:
        return error.missing_requirements


@app.post("/simulation/agents/{agent_id}/plan", response_model=str)
def plan(agent_id: str):
    plan_broad(sim.get_agent(agent_id))
    return "ok"


@app.post("/simulation/agents/{agent_id}/plan_grounded", response_model=str)
def plan_ground(agent_id: str):
    plan_grounded(sim.get_agent(agent_id))
    return "ok"








@app.get("/simulation/agents/{agent_id}/plan", response_model=list[PlanStep])
def get_plan(agent_id: str):
    return sim.get_agent(agent_id).blackboard.curr_plan
