import json

from persona.agent import Agent, AgentSetup
from generation.operations.module_operations import gen_plan, gen_grounding, gen_routine_selection
from persona.aid import PlanStep


def get_op_foundations(agent: Agent):
    core = agent.recall.core
    cache = agent.recall.cache

    context = {sec_name: [
        {
            "memory_description": node.core.description,
            "importance_percentage": node.core.poignancy 
        }
    for node in sec.values()] for sec_name, sec in cache.sections.items()}
    context["core"] = [
        {
            "memory_description": node.core.description,
            "importance_percentage": node.core.poignancy 
        }
    for node in core]

    state = {key: agent.blackboard.state[key] for key in agent.settings.planning.contract.state_keys}

    entities = [{
        "entity_id": entity.id,
        "description": entity.description
    } for entity in agent.blackboard.attended_entities.values()]

    affordances = [{
        "affordance_id": entity.id + "." + affordance,
        "description": entity.description,
        "affected_entity_id": entity.id
    } for entity in agent.blackboard.attended_entities.values()
      for affordance in entity.affordances]

    return state, context, entities, affordances


def get_op_foundations_setup(agent: AgentSetup):
    if agent.recall is None:
        raise Exception("Setup memory before requesting routine selection")
    if agent.plan_settings is None:
        raise Exception("Setup planning before requesting routine selection")
    
    core = agent.recall.core
    state = {key: agent.blackboard.state[key] for key in agent.plan_settings.contract.state_keys}
    context = { "core": [
        {
            "memory_description": node.core.description,
            "importance_percentage": node.core.poignancy 
        }    
    for node in core]}

    return state, context


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
    
    agent.plan.steps = [PlanStep(task=step) for step in response["plan_steps"]]
    agent.plan.advance_task()


def op_ground(agent: Agent):
    state, context, entities, affordances = get_op_foundations(agent)

    if not agent.plan.unplanned():
        plan_task = agent.plan.steps[agent.plan.task_index].task 
        actions_taken = agent.plan.current_action_sequence()

        response = gen_grounding(
            agent,
            state,
            context, 
            entities, 
            affordances,
            plan_task,
            actions_taken
        ) #TODO include in gen the logic for this to addon actions instead of generating an entirely new sequence. Or make the agent plan one action at a time, at the request of the client

        agent.plan.add_actions(response)
        agent.plan.advance_action(len(response))
        if response[-1].key == "completed_task":
            agent.plan.complete_current()


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
