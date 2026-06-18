import json

from persona.agent import Agent
from generation.operations.module_operations import gen_plan, gen_grounding
from persona.aid import PlanStep


def op_plan(agent: Agent):
    agent.recall.load_cache(
        agent.settings.planning.contract.memory_keys,
        agent.goal
    )
    state = agent.blackboard.state #TODO fetch contract specified fields
    core = agent.recall.core
    cache = agent.recall.cache.get_nodes()
    nodes = core + cache
    context = [{
        "description": node.core.description,
        "involved_game_enteties": node.core.entities_involved
    } for node in nodes]
    
    response = gen_plan(agent, state, context)
    plan_steps = [PlanStep(task=step) for step in response["plan_steps"]]
    
    agent.plan.steps = plan_steps
    print("Response -> I'm at plan_ops.py")
    print(json.dumps(response))
    agent.plan.open_plan()


def op_ground(agent: Agent):
    state = agent.blackboard.state
    core = agent.recall.core
    cache = agent.recall.cache.get_nodes()
    nodes = core + cache
    context = [{
        "description": node.core.description,
        "involved_game_enteties": node.core.entities_involved
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
    
    while not agent.plan.steps:
        op_plan(agent)
    while not agent.plan.steps[-1].actions or agent.plan.steps[-1].actions[-1] != "completed_task":
        op_ground(agent)
