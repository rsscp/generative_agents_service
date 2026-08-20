from persona.agent import Agent, AgentSetup


def get_op_foundations(agent: Agent):
    core = agent.recall.core
    cache = agent.recall.cache

    context = {
        sec_name: [
            {
                "memory": node.core.description,
                "entity_snapshots": node.core.entity_snapshots
            }
            for node in sec.values()
        ]
        for sec_name, sec in cache.sections.items()
    }
    context["core"] = [
        {
            "memory": node.core.description,
            "entity_snapshots": node.core.entity_snapshots
        }
        for node in core
    ]
    context = {
        sec_name: sec
        for sec_name, sec in context.items()
        if len(sec) > 0
    }

    state = {key: agent.blackboard.state[key] for key in agent.settings.planning.contract.state_keys}

    relevant_entity_ids = set([
        value
        for tool in agent.blackboard.generic_tools.values()
        for property in tool.arguments.values()
        if property.enum is not None
        for value in property.enum
    ])

    entities = [
        {
            "entity_id": entity.id,
            "description": entity.description,
            "tags": entity.tags
        }
        for entity in agent.blackboard.attended_entities.values()
        if entity.id in relevant_entity_ids
    ]

    affordances = [
        {
            "affordance_id": entity.id + "." + affordance,
            "description": entity.description,
            "affected_entity_id": entity.id
        }
        for entity in agent.blackboard.attended_entities.values()
        for affordance in entity.affordances
    ]

    return state, context, entities, affordances


def get_op_foundations_setup(agent: AgentSetup):
    if agent.recall is None:
        raise Exception("Setup memory before requesting routine selection")
    if agent.plan_settings is None:
        raise Exception("Setup planning before requesting routine selection")

    core = agent.recall.core
    state = {key: agent.blackboard.state[key] for key in agent.plan_settings.contract.state_keys}
    context = {
        "core": [
            {
                "memory": node.core.description,
                "entity_snapshots": node.core.entity_snapshots
            }
            for node in core
        ]
    }

    return state, context