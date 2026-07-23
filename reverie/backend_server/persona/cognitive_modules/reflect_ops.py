"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reflect.py
Description: This defines the "Reflect" module for generative agents. 
"""
import sys
from typing import Any, Dict

from generation.operations.module_operations import gen_focal_points, gen_thought
from persona.agent import Agent
from persona.cognitive_modules.retrieve_ops import retrieve_request
from persona.memory_structures.memory_blocks.memory_box import node_from_raw
from persona.memory_structures.memory_blocks.node import CoreNode, RawNode
sys.path.append('../../')

import datetime
import random

from numpy import dot
from numpy.linalg import norm

from global_methods import *
from persona.prompt_template.run_gpt_prompt import *
from persona.prompt_template.gpt_structure import *
from persona.cognitive_modules.retrieve import *

from persona.cognitive_modules.utils import get_op_foundations, get_op_foundations_setup


def op_feed_event(agent: Agent, event: RawNode): #TODO make this call methods that lock shared resources
  node = node_from_raw(event, agent.recall.core)
  print(f"HELLLLLOOOOO >>> {event.description}")
  agent.recall.cache.add("events", node)
  process_event(agent, node.core.poignancy)


def process_event(agent: Agent, event_poignancy: int):
  agent.blackboard.importance_accumulator -= event_poignancy
  agent.blackboard.events_since_reflection += 1

  if should_reflect(agent):
    reflect(agent)
    reset_reflection(agent)


def should_reflect(agent: Agent) -> bool:
  return agent.blackboard.importance_accumulator >= agent.blackboard.reflection_config.importance_threshold


def reset_reflection(agent: Agent):
  agent.blackboard.importance_accumulator = 100
  agent.blackboard.events_since_reflection = 0


def reflect(agent: Agent):
  state, context, entities, *_ = get_op_foundations(agent)
  
  response = gen_thought(
    agent,
    state,
    context,
    entities
  )
  
  agent.recall.memory.add_core("thoughts", CoreNode(
    poignancy = response.thought.poignancy,
    description = response.thought.description,
    entity_keys = set(response.thought.entities_mentioned)
  ))
  agent.recall.cache.add_core("thoughts", CoreNode(
    poignancy = response.thought.poignancy,
    description = response.thought.description,
    entity_keys = set(response.thought.entities_mentioned)
  ))