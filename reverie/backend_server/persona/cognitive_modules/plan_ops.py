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
    entities = list(set([entity for node in nodes for entity in node.core.entities_involved])) #TODO Is this list > to set > to list too goofy?

    return state, context, entities


def op_plan(agent: Agent):
    agent.recall.load_cache(
        agent.settings.planning.contract.memory_keys,
        agent.goal
    )

    state, context, entities = get_op_foundations(agent)
    response = gen_plan(agent, state, context, entities)
    plan_steps = [PlanStep(task=step) for step in response["plan_steps"]]
    
    agent.plan.steps = plan_steps
    agent.plan.open_plan()


def op_ground(agent: Agent):
    state, context, entities = get_op_foundations(agent)

    for step in agent.plan.steps:
        if len(step.actions) == 0 or step.actions[-1].key != "completed_task":
            response = gen_grounding(agent, state, context, entities, step.task, step.actions) #TODO include in gen the logic for this to addon actions instead of generating an entirely new sequence. Or make the agent plan one action at a time, at the request of the client
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
