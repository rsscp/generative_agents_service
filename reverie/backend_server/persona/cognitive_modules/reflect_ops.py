"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reflect.py
Description: This defines the "Reflect" module for generative agents. 
"""
import sys
from typing import Any, Dict

from generation.operations.module_operations import gen_node_poignancy, gen_thought
from persona.agent import Agent
from persona.aid import CoreNode
from persona.memory_structures.memory_blocks.memory_box import node_from_core
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


def op_feed_event(agent: Agent, event: CoreNode): #TODO make this call methods that lock shared resources
  node = node_from_core(event)
  agent.recall.cache.add("events", node)
  process_event(agent, event)


def process_event(agent: Agent, event: CoreNode):
  state, context, entities = get_op_foundations(agent)

  event_poignancy = gen_node_poignancy(agent, context, event.description)

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
  state, context, entities = get_op_foundations(agent)
  
  response = gen_thought(
    agent,
    state,
    context,
    entities
  )

  #TODO to fill entities_snapshot, I need the agent to maintain snapshots of entities, which is basically what already happens with entity knowledge, but there should be two entity datastructures, one in blackboard with only currently attended entities and their actual current state and another in recall that assumes entries are outdated snapshots of entites, meaning it can even have various entries for the same object.

  snapshots = {k: agent.recall.entity_snapshots[k] for k in response.entities_mentioned}
  
  agent.recall.memory.add_core("thoughts", CoreNode(
    description = response.description,
    entity_snapshots = snapshots
  ))
  agent.recall.cache.add_core("thoughts", CoreNode(
    description = response.description,
    entity_snapshots = snapshots
  ))