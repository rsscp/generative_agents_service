from selenium import webdriver

from global_methods import *
from utils import *
from maze import *
from persona.persona import *

from fastapi import FastAPI

from api_classes import CreateAgentRequest, CreateAgentResponse, CreateSimRequest, CreateSimResponse, PlanRequest, PlanResponse
from simulation import Simulation
from persona.agent import Agent


sim = None


@app.post("/simulation", response_model=CreateSimResponse)
def create_sim(request: CreateSimRequest):
  sim = Simulation(
    CreateSimRequest.sim_id,
  )


@app.post("/simulation/agents", response_model=CreateAgentResponse)
def create_agent(request: CreateAgentRequest):
  sim.add_agent(Agent(
    CreateAgentRequest.agent_id,
    CreateAgentRequest.goal,
    # Plan generation
    CreateAgentRequest.plan_gen_ruleset,
    CreateAgentRequest.plan_gen_schemas,
    # Thought generation
    CreateAgentRequest.thought_gen_ruleset,
    CreateAgentRequest.thought_gen_schemas,
    # Scratch content
    CreateAgentRequest.identy_nodes,
    CreateAgentRequest.memory_nodes,
    CreateAgentRequest.available_actions
  ))


@app.post("/simulations/agents/{agent_id}", response_model=PlanResponse)
def advance_sim(request: PlanRequest):
