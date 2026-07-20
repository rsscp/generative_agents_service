import json

from persona.agent import Agent, AgentSetup
from generation.operations.module_operations import gen_plan, gen_grounding, gen_routine_selection
from persona.aid import PlanStep
from persona.cognitive_modules.utils import get_op_foundations, get_op_foundations_setup


def op_plan(agent: Agent):
    agent.recall.load_cache(
        agent.settings.planning.contract.memory_keys,
        agent.plan.goal
    )

    state, context, entities, *_ = get_op_foundations(agent)
    
    response = gen_plan(
        agent,
        state,
        context,
        entities
    )
    
    agent.plan.enqueue_steps([PlanStep(task=step) for step in response["plan_steps"]])


def op_ground(agent: Agent):
    state, context, entities, affordances = get_op_foundations(agent)

    plan_task = agent.plan.current_task() 

    response = gen_grounding(
        agent,
        state,
        context, 
        entities, 
        affordances,
        plan_task
    ) #TODO include in gen the logic for this to addon actions instead of generating an entirely new sequence. Or make the agent plan one action at a time, at the request of the client

    agent.plan.enqueue_actions(response)


def op_select_routine(agent: Agent):
    agent.recall.load_cache(
        agent.settings.planning.contract.memory_keys,
        agent.plan.goal
    )

    state, context, *_ = get_op_foundations(agent)
    
    routine, goal = gen_routine_selection(
        agent.routines,
        state,
        context
    )
    
    agent.set_plan(routine, goal)


def op_select_routine_setup(agent_setup: AgentSetup):
    state, context = get_op_foundations_setup(agent_setup)
    
    routine, goal = gen_routine_selection(
        agent_setup.routines,
        state,
        context
    )
    
    agent_setup.set_plan(routine, goal)


def op_plan_full(agent: Agent):
    op_plan(agent)
    
    while not agent.plan.steps:
        op_plan(agent)
    while not agent.plan.steps[-1].actions or agent.plan.steps[-1].actions[-1] != "completed_task":
        op_ground(agent)
