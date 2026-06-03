from persona.agent import Agent
from llm_operations import plan_broad_request, plan_grounded_request
from persona.aid import PlanStep, Contract, ActionCall
from typing import Dict
from utils import merge


def plan_broad(agent: Agent):
    contract = agent.plan_contract
    relevant_state, relevant_memory = get_relevant_pieces(agent, contract)
    
    response = plan_broad_request(agent, relevant_state, relevant_memory)
    plan = [PlanStep(task=step) for step in response["plan_steps"]]
    
    agent.blackboard.curr_plan = plan


def plan_grounded(agent: Agent):
    contract = agent.plan_contract
    relevant_state, relevant_memory = get_relevant_pieces(agent, contract)

    for step in agent.blackboard.curr_plan:
        if len(step.actions) == 0 or step.actions[-1].key != "completed_task":
            response = plan_grounded_request(agent, relevant_state, relevant_memory, step.task, step.actions)
            step.actions += response
            break


def get_relevant_pieces(agent: Agent, contract: Contract):
    common_keys_state = set(contract.state_keys) & set(agent.blackboard.state.keys())
    common_keys_cache = set(contract.memory_keys) & set(agent.blackboard.cache_lists.keys())
    common_keys_memory = set(contract.memory_keys) & set(agent.recall.memory_lists.keys())
    
    relevant_state = {k: agent.blackboard.state[k] for k in common_keys_state}
    relevant_memory = relevance_filter(merge(
        {k: agent.blackboard.cache_lists[k] for k in common_keys_cache},
        {k: agent.recall.memory_lists[k] for k in common_keys_memory},
        lambda a, b : a + b
    ))

    return relevant_state, relevant_memory


def relevance_filter(memories: Dict[str, list]):
    memories_list = []
    [memories_list.extend(v) for k, v in memories.items()]

    return memories_list
