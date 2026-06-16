from pydantic import BaseModel, Field
from persona.agent import Agent
from llm_operations import gen_plan, gen_grounding
from persona.aid import PlanStep, Contract, ToolCall
from typing import Dict
from reverie.backend_server.persona.agent import ModuleSettings


RELEVANT_RECENCY = 10
RELEVANT_POIGNANCY = 80
RELEVANT_SEMANTIC_DISTANCE = 0.7

def op_plan(agent: Agent):
    agent.recall.load_cache(
        agent.settings.planning.contract.memory_keys,
        agent.goal,
        RELEVANT_RECENCY,
        RELEVANT_POIGNANCY,
        RELEVANT_SEMANTIC_DISTANCE
    )
    state = agent.blackboard.state
    nodes = agent.recall.cache.get_nodes()
    context = [{
        "description": node.description,
        "memory_object": node.object,
        "involved_game_enteties": node.entities_involved
    } for node in nodes]
    
    response = gen_plan(agent, state, context)
    plan = [PlanStep(task=step) for step in response["plan_steps"]]
    
    agent.blackboard.curr_plan = plan
    agent.plan.open_plan()


def op_ground(agent: Agent):
    state = agent.blackboard.state
    nodes = agent.recall.cache.get_nodes()
    context = [{
        "description": node.description,
        "memory_object": node.object,
        "involved_game_enteties": node.entities_involved
    } for node in nodes]

    if agent.plan.ungrounded():
        agent.plan.open_ground()

    for step in agent.plan.steps:
        if len(step.actions) == 0 or step.actions[-1].key != "completed_task":
            response = gen_grounding(agent, state, context, step.task, step.actions) #TODO include in gen the logic for this to addon actions instead of generating an entirely new sequence. Or make the agent plan one action at a time, at the request of the client
            step.actions += response
            break

def op_plan_full(agent: Agent):
    op_plan(agent)
    plan = agent.blackboard.curr_plan
    
    while not plan:
        op_plan(agent)
    while not plan[-1].actions or plan[-1].actions[-1] != "completed_task":
        op_ground(agent)
