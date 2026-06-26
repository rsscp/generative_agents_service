"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: persona.py
Description: Defines the Persona class that powers the agents in Reverie. 

Note (May 1, 2023) -- this is effectively GenerativeAgent class. Persona was
the term we used internally back in 2022, taking from our Social Simulacra 
paper.
"""
import math
import sys
import datetime
import random

from pydantic import BaseModel, Field
from persona.memory_structures.memory_blocks.node import CoreNode, Node, RawNode
from standard import BLOCKING_ACTIONS, DEFAULT_ACTIONS, PLAN_SCHEMA, PLAN_AUX_SCHEMAS, GROUND_SCHEMA, STANDARD_GROUNDING_INSTRUCTIONS, STANDARD_INSTRUCTIONS, STANDARD_PLANNING_INSTRUCTIONS

sys.path.append('../')

from global_methods import *

from persona.memory_structures.spatial_memory import *
from persona.memory_structures.associative_memory import *
from persona.memory_structures.scratch import *

from persona.cognitive_modules.perceive import *
from persona.cognitive_modules.retrieve import *
from persona.cognitive_modules.plan import *
from persona.cognitive_modules.reflect import *
from persona.cognitive_modules.execute import *
from persona.cognitive_modules.converse import *

from typing import Dict, Any, Optional
from threading import Lock

from persona.memory_structures.blackboard import Blackboard
from persona.memory_structures.recall import Recall
from api_classes import Contract, SchemaField
from persona.memory_structures.memory_blocks.memory_box import MemoryBox
from persona.aid import GroundingSettings, InteractionSettings, PlanStep, PlanningSettings, ReflectionSettings, Schema, Tool, Configuration, SchemaField, ToolCall


class AgentException(Exception):
  def __init__(self, message: str, reason: str):
    self.reason = reason
    super().__init__(message)


class MissingAgentRequirements(AgentException):
  def __init__(self, missing_requirements: list[str]):
    self.missing_requirements = missing_requirements
    self.message = (
      "Agent is missing required configuration: "
      + ", ".join(missing_requirements)
    )
    super().__init__(
      self.message,
      "Missing requirements to finalize agent"
    )


class RepeatedSchemaNames(AgentException):
  def __init__(self, repeated_names: set[str]):
    self.repeated_names = repeated_names
    self.message = (
      "The following schemas are already internally defined and their names must not be used:"
      + ", ".join(repeated_names)
    )
    super().__init__(
      self.message,
      "Provided schemas named after predefined system schemas"
    )

class Plan:
  
  def __init__(self):
    self.lock: Lock = Lock()

    self.INVALID_T_INDEX = -1
    self.INITIALIZED_T_INDEX = 0
    self.INVALID_A_INDEX = -2
    self.INITIALIZED_A_INDEX = -1
    

    self.steps: list[PlanStep] = []
    self.task_index: int = self.INVALID_T_INDEX
    self.action_index: int = self.INVALID_A_INDEX
    self.advance_lock: bool = False


  def reset_index(self):
    with self.lock:
      self.task_index = self.INVALID_T_INDEX
      self.action_index = self.INVALID_A_INDEX


  def unplanned(self):
    return self.task_index == self.INVALID_T_INDEX


  def ungrounded(self):
    return self.action_index == self.INVALID_A_INDEX
  

  def ready(self):
    result = \
      self.task_index >= self.INITIALIZED_T_INDEX and \
      self.action_index >= self.INITIALIZED_A_INDEX
    
    print(f"Ready: {result}")

    return result


  def open_plan(self):
    with self.lock:
      self.task_index = self.INITIALIZED_T_INDEX
      self.action_index = self.INVALID_A_INDEX


  def open_ground(self):
    with self.lock:
      if self.action_index == self.INVALID_A_INDEX:
        self.action_index = self.INITIALIZED_A_INDEX


  def clear_plan(self):
    with self.lock:
      self.reset_index()
      self.steps = []


  def allowed_jump(self) -> bool:
    result = \
      self.steps[self.task_index].actions[self.action_index].key == "completed_task" and \
      self.task_index + 1 < len(self.steps) and \
      len(self.steps[self.task_index + 1].actions) > 0

    return result

  
  def allowed_next(self) -> bool:
    result = self.action_index + 1 < len(self.steps[self.task_index].actions)

    print(f"Allowed Next: {result}")

    return result
  
  
  def allowed_reset(self) -> bool:
    result = \
      self.steps[self.task_index].actions[self.action_index].key == "completed_task" and \
      self.task_index == len(self.steps) - 1
    
    print(f"Allowed Reset: {result}")

    return result
  

  def advance_index(self) -> tuple[bool, Optional[ToolCall], str]:
    action = None
    hit = True
    key = ""

    task_index = self.task_index
    action_index = self.action_index

    if not self.ready():
      hit = False
    elif self.allowed_next():
      action_index += 1
    elif self.allowed_jump():
      task_index += 1
      action_index = self.INITIALIZED_A_INDEX
    elif self.allowed_reset():
      task_index = self.INVALID_T_INDEX
      action_index = self.INVALID_A_INDEX
      hit = False
    else:
      hit = False

    if hit:
      action = self \
        .steps[task_index] \
        .actions[action_index]
      key = action.key
    
    self.task_index = task_index
    self.action_index = action_index

    return hit, action, key
  

  def next_action(self) -> Optional[ToolCall]:
    hit, action, key = True, None, "completed_task"

    while key == "completed_task":
      hit, action, key = self.advance_index()

      print("\n\n\n")
      print(f"Index: ({self.task_index}, {self.action_index})")
      
    return action
      

class ModuleSettings(BaseModel):
  contract: Optional[Contract] = None
  instructions: Optional[list[str]] = None
  main_schema: Optional[Dict[str, SchemaField]] = None
  aux_schemas: Optional[Dict[str, Dict[str, SchemaField]]] = None


class AgentSettings(BaseModel):
  planning: PlanningSettings
  grounding: GroundingSettings
  reflection: ReflectionSettings
  interaction: InteractionSettings


class Agent:

  def __init__(
    self,
    goal: str,
    blackboard: Blackboard,
    recall: Recall,
    settings: AgentSettings
  ):
    self.lock = Lock()

    self.goal = goal
    self.settings = settings
    self.blackboard = blackboard
    self.recall = recall
    self.plan = Plan()

    pas_common_keys = \
      set(settings.planning.aux_schemas.keys()) & \
      set(PLAN_AUX_SCHEMAS.keys())

    if pas_common_keys:
      raise RepeatedSchemaNames(pas_common_keys)
    else:
      self.settings.planning.aux_schemas = PLAN_AUX_SCHEMAS | settings.planning.aux_schemas


  '''
  def merge_nodes(self, cache: Dict[str, Dict[str, Node]], memory: Dict[str, Dict[str, Node]]) -> list[Node]:
    cache_condensed: Dict[str, Node] = {embed: node for nodes in cache.values() for embed, node in nodes.items()}
    memory_condensed: Dict[str, Node] = {embed: node for nodes in memory.values() for embed, node in nodes.items()}

    embed_diff = set(memory_condensed.keys()).difference(cache_condensed.keys())
    result: list[Node] = [memory_condensed[embed] for embed in embed_diff]
    result.extend(cache_condensed.values())
    return result


  def get_relevant_pieces(self, contract: Contract) -> tuple[Dict[str, Any], list[Node]]:
    common_keys_state = set(contract.state_keys) & set(self.blackboard.state.keys())
    common_keys_cache = set(contract.memory_keys) & set(self.blackboard.cache.section_keys())
    common_keys_memory = set(contract.memory_keys) & set(self.recall.memory.section_keys())
    
    relevant_state = {k: self.blackboard.state[k] for k in common_keys_state}
    relevant_cache: Dict[str, Dict[str, Node]] = {k: self.blackboard.cache.sections[k] for k in common_keys_cache}
    relevant_memory: Dict[str, Dict[str, Node]] = {k: self.recall.memory.sections[k] for k in common_keys_memory}

    relevant_entities: set[str] = set()
    for section in relevant_cache.values():
      for node in section.values():
        relevant_entities.update(node.entities_involved)

    relevant_memory_sections = self.relevance_filter(relevant_entities, relevant_memory)
    relevant_nodes = self.merge_nodes(relevant_cache, relevant_memory_sections)

    return relevant_state, relevant_nodes


  def relevance_filter(self, keys: set[str], memory: Dict[str, Dict[str, Node]]) -> Dict[str, Dict[str, Node]]:
    result: Dict[str, Dict[str, Node]] = {k: {} for k in memory.keys()}
    
    for sec_name, section in memory.items():
      for embed, node in section.items():
        if bool(set(node.entities_involved) & keys):
          result[sec_name][embed] = node

    return result
  '''


# Temporary agent setup class with all optional fields
# When all requirements are filled, the final agent can be created
class AgentSetup:

  def __init__(
    self,
    goal: str,
    initial_state: Dict[str, Any],
  ):
    self.lock = Lock()

    self.goal = goal
    self.blackboard = Blackboard(initial_state)
    self.recall: Optional[Recall] = None
    self.config: Optional[Configuration] = None
    self.plan_settings: Optional[PlanningSettings] = None
    self.ground_settings: Optional[GroundingSettings] = None
    self.reflect_settings: Optional[ReflectionSettings] = None
    self.interact_settings: Optional[InteractionSettings] = None


  def set_config(self, config: Configuration):
    with self.lock:
      self.config = config


  def set_memory(self, core_nodes: list[CoreNode], node_sections: Dict[str, list[CoreNode]]):
    with self.lock:
      self.recall = Recall(core_nodes, node_sections)


  def set_actions(self, actions: list[Tool]):
    with self.lock:
      self.blackboard.set_tools(actions)
      #self.blackboard.tools += DEFAULT_ACTIONS


  # --- Planning requirements ---
  
  def setup_planning(self,
    instructions: list[str],
    contract: Contract,
    aux_schemas: Dict[str, Schema]
  ):
    with self.lock:
      self.plan_settings = PlanningSettings(
        instructions =
          instructions \
          + STANDARD_INSTRUCTIONS \
          + STANDARD_PLANNING_INSTRUCTIONS,
        contract = contract,
        main_schema = PLAN_SCHEMA,
        aux_schemas = aux_schemas
      )


  # --- Planning Grounded requirements ---

  def setup_grounding(self,
    instructions: list[str],
    contract: Contract
  ):
    with self.lock:
      self.ground_settings = GroundingSettings(
        instructions =
          instructions \
          + STANDARD_INSTRUCTIONS \
          + STANDARD_GROUNDING_INSTRUCTIONS,
        contract = contract,
        main_schema = GROUND_SCHEMA
      )


  # --- Reflection requirements ---

  def setup_reflection(self,
    instructions: list[str],
    main_schema: Schema,
    aux_schemas: Dict[str, Schema],
    contract: Contract
  ):
    with self.lock:
      self.reflect_settings = ReflectionSettings(
        instructions =
          instructions \
          + STANDARD_INSTRUCTIONS,
        main_schema = main_schema,
        aux_schemas = aux_schemas,
        contract = contract
      )


  # --- Reflection requirements ---

  def setup_interaction(self): #TODO Worry about interactions later
    self.interact_settings = InteractionSettings(
      instructions = [],
      main_schema = {},
      aux_schemas = {},
      contract = Contract(
        state_keys = [],
        memory_keys = []
      )
    )


  def create_agent(self) -> Agent:
    checks = {
      "memory": self.recall != None,
      "configuration": self.config != None,
      "planning settings": self.plan_settings != None,
      "grounding settings": self.ground_settings != None,
      "reflection settings": self.reflect_settings != None,
      "interaction settings": self.interact_settings != None,
    }

    missing = [k for k, v in checks.items() if v is False]
    if len(missing) > 0:
      raise MissingAgentRequirements(missing)
    
    assert self.recall is not None
    assert self.config is not None
    assert self.plan_settings is not None
    assert self.ground_settings is not None
    assert self.reflect_settings is not None
    assert self.interact_settings is not None

    settings = AgentSettings(
      planning = self.plan_settings,
      grounding = self.ground_settings,
      reflection = self.reflect_settings,
      interaction = self.interact_settings
    )

    return Agent(
      self.goal,
      self.blackboard,
      self.recall,
      settings
    )
