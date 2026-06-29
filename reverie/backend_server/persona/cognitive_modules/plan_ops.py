import json

from persona.agent import Agent
from generation.operations.module_operations import gen_plan, gen_grounding
from persona.aid import PlanStep


def get_op_foundations(agent: Agent):
    core = agent.recall.core
    cache = agent.recall.cache.get_nodes()
    nodes = core + cache
    state = {key: agent.blackboard.state[key] for key in agent.settings.planning.contract.state_keys}
    context = [node.core.description for node in nodes]

    entities = [{
        "entity_id": entity.id,
        "description": entity.description
    } for entity in agent.blackboard.attended_entities]

    affordances = [{
        "affordance_id": entity.id + "." + affordance,
        "description": entity.description,
        "affected_entity_id": entity.id
    } for entity in agent.blackboard.attended_entities
      for affordance in entity.affordances]

    return state, context, entities, affordances


def op_plan(agent: Agent):
    agent.recall.load_cache(
        agent.settings.planning.contract.memory_keys,
        agent.goal
    )

    state, context, entities, affordances = get_op_foundations(agent)
    response = gen_plan(agent, state, context, entities)
    plan_steps = [PlanStep(task=step) for step in response["plan_steps"]]
    
    agent.plan.steps = plan_steps
    agent.plan.open_plan()


def op_ground(agent: Agent):
    state, context, entities, affordances = get_op_foundations(agent)

    plan_task = agent.plan.steps[agent.plan.task_index].task 
    actions_taken = agent.plan.current_action_sequence()

    if plan_task is not None and actions_taken is not None:
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

    for step in agent.plan.steps:
        if len(step.actions) == 0 or step.actions[-1].key != "completed_task":
            response = gen_grounding(agent, state, context, entities, affordances, step.task, step.actions) #TODO include in gen the logic for this to addon actions instead of generating an entirely new sequence. Or make the agent plan one action at a time, at the request of the client
            step.actions += response
            break

    if agent.plan.ungrounded():
        agent.plan.open_ground()

def op_plan_full(agent: Agent):
    op_plan(agent)
    
    while not agent.plan.steps:
        op_plan(agent)
    while not agent.plan.steps[-1].actions or agent.plan.steps[-1].actions[-1] != "completed_task":
        op_ground(agent)
