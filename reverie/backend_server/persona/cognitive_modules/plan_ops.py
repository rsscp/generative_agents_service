import json

from persona.agent import Agent, AgentSetup
from generation.operations.module_operations import gen_plan, gen_grounding, gen_routine_selection
from persona.aid import AgentRoutine, PlanStep, PlanStepLog, SegmentedGround, ToolCall
from persona.cognitive_modules.utils import get_op_foundations, get_op_foundations_setup
from standard import PlanStepLogSchema


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
    
    agent.plan.enqueue_steps([PlanStep(task=step.dict()) for step in response.plan_steps])
    agent.plan.do_log_steps([PlanStepLogSchema(task=step.dict()) for step in response.plan_steps])


def op_ground(agent: Agent, correction: bool):
    state, context, entities, affordances = get_op_foundations(agent)

    plan_task = agent.plan.current_task()
    failed_action = None

    if correction:
        failed_action = agent.plan.current_action

    reasoning, calls = gen_grounding(
        agent,
        state,
        context, 
        entities, 
        plan_task,
        failed_action
    ) #TODO include in gen the logic for this to addon actions instead of generating an entirely new sequence. Or make the agent plan one action at a time, at the request of the client

    
    if calls is not None:
        agent.plan.enqueue_actions(calls)
        agent.plan.do_log_actions(SegmentedGround(reasoning=reasoning, calls=calls))
    else:
        raise Exception("Grounding operation failed")


def op_select_routine(agent: Agent):
    agent.recall.load_cache(
        agent.settings.planning.contract.memory_keys,
        agent.plan.goal
    )

    state, context, *_ = get_op_foundations(agent)
    
    response = gen_routine_selection(
        agent.routines,
        state,
        context
    )

    routine = next(routine for routine in agent.routines if routine.name == response.routine_choice)
    goal = response.goal

    agent.set_plan(routine, goal)


def op_select_routine_setup(agent_setup: AgentSetup):
    state, context = get_op_foundations_setup(agent_setup)
    
    response = gen_routine_selection(
        agent_setup.routines,
        state,
        context
    )

    routine = next(routine for routine in agent_setup.routines if routine.name == response.routine_choice)
    goal = response.goal
    
    agent_setup.set_plan(routine, goal)
