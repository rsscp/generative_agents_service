"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reflect.py
Description: This defines the "Reflect" module for generative agents. 
"""
import sys
from typing import Any, Dict

from generation.operations.module_operations import gen_focal_points
from persona.agent import Agent
from persona.cognitive_modules.retrieve_ops import retrieve_request
sys.path.append('../../')

import datetime
import random

from numpy import dot
from numpy.linalg import norm

from global_methods import *
from persona.prompt_template.run_gpt_prompt import *
from persona.prompt_template.gpt_structure import *
from persona.cognitive_modules.retrieve import *


def feed_event(agent: Agent, event: Dict[str, Any], event_weight: float = 1.0): #TODO make this call methods that lock shared resources
  if "events" not in agent.recall.cache.section_keys():
    agent.blackboard.cache.g["events"] = []
  agent.blackboard.cache["events"].append(event)
  process_event(agent, event_weight)

def process_event(agent: Agent, event_weight: float):
  agent.blackboard.importance_accumulator += event_weight
  agent.blackboard.events_since_reflection += 1

  if should_reflect(agent):
    reflect(agent)
    reset_reflection(agent)

def should_reflect(agent: Agent) -> bool:
  return agent.blackboard.importance_accumulator >= agent.blackboard.reflection_config.importance_threshold

def reset_reflection(agent: Agent):
  agent.blackboard.importance_accumulator = 0.0
  agent.blackboard.events_since_reflection = 0

def reflect(agent: Agent):
  contract = agent.settings.planning.contract
  relevant_state, relevant_memory = agent.get_relevant_pieces(contract)

  focal_points = gen_focal_points(agent, relevant_state, relevant_memory, length=3)
  retrieved = retrieve_request(agent, focal_points) #TODO make new retrieve on retrieve_alt.py

  #TODO complete reflect, make it get recent memories, actually call LLM for reflection, store thoughts, ...