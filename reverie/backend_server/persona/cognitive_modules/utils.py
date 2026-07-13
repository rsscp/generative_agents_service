from persona.agent import Agent, AgentSetup


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