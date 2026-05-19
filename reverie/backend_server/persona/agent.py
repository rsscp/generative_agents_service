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

from typing import Dict, Any

from persona.memory_structures.blackboard import Blackboard
from persona.memory_structures.recall import Recall
from api_classes import Contract, MemoryList
from persona.aid import Action, Configuration, SchemaField

STANDARD_INSTRUCTIONS = [
  "Do not make up facts",
  "All information used on planning, apart from what is considered common sense, will be pulled from this message."
  "You response will follow the JSON structure specified in Schema.",
  "Always complying with JSON formating."
]


class MissingAgentRequirements(Exception):
    def __init__(self, missing_requirements: list[str]):
        self.missing_requirements = missing_requirements
        message = (
            "Agent is missing required configuration: "
            + ", ".join(missing_requirements)
        )
        super().__init__(message)


class Agent:

  def __init__(
    self,
    goal: str,
    blackboard: Blackboard,
    recall: Recall,
    config: Configuration,
    plan_contract: Contract,
    plan_instructions: list[str],
    plan_main_schema: Dict[str, SchemaField],
    plan_aux_schemas: Dict[str, Dict[str, SchemaField]],
    thought_contract: Contract,
    thought_instructions: list[str],
    thought_main_schema: Dict[str, SchemaField],
    thought_aux_schemas: Dict[str, Dict[str, SchemaField]]
  ):
    self.goal = goal
    self.blackboard = blackboard
    self.recall = recall
    self.config = config
    self.plan_contract = plan_contract
    self.plan_instructions = STANDARD_INSTRUCTIONS + plan_instructions
    self.plan_main_schema = plan_main_schema
    self.plan_aux_schemas = plan_aux_schemas
    self.thought_contract = thought_contract
    self.thought_instructions = STANDARD_INSTRUCTIONS + thought_instructions
    self.thought_main_schema = thought_main_schema
    self.thought_aux_schemas = thought_aux_schemas


# Temporary agent setup class with all optional fields
# When all requirements are filled, the final agent can be created
class AgentSetup:
  
  def __init__(
    self,
    goal: str,
    initial_state: Dict[str, Any],
  ):
    self.goal = goal
    self.blackboard = Blackboard(initial_state)

    self.recall = None
    self.config = None
    self.plan_contract = None
    self.plan_instructions = None
    self.plan_main_schema = None
    self.plan_aux_schemas = None
    self.thought_contract = None
    self.thought_instructions = None
    self.thought_main_schema = None
    self.thought_aux_schemas = None


  def set_config(self, config: Configuration):
    self.config = config


  def set_memory_lists(self, memory_lists: Dict[str, list]):
    self.recall = Recall(memory_lists)


  def set_actions(self, actions: Dict[str, Action]):
    self.blackboard.set_actions(actions)


  # --- Planning requirements ---

  def set_plan_contract(self, contract: Contract):
    self.plan_contract = contract
  
  def set_plan_req(
    self,
    instructions: list[str],
    main_schema: Dict[str, SchemaField],
    aux_schemas: Dict[str, Dict[str, SchemaField]]
  ):
    self.plan_instructions = instructions
    self.plan_main_schema = main_schema
    self.plan_aux_schemas = aux_schemas


  # --- Reflection requirements ---

  def set_thought_contract(self, contract: Contract):
    self.thought_contract = contract

  def set_thought_req(
    self,
    instructions: list[str],
    schema: Dict[str, SchemaField],
    aux_schemas: list[Dict[str, SchemaField]]
  ):
    self.thought_instructions = instructions
    self.thought_main_schema = schema
    self.thought_aux_schemas = aux_schemas


  def set_interaction_contracts(self):
    pass #TODO Worry about interactions later


  def create_agent(self) -> Agent:
    checks = {
      "memory": self.recall != None,
      "configuration": self.config != None,
      "contracts": self.plan_contract != None and self.thought_contract != None,
      "planning requirements": self.plan_instructions != None and self.plan_main_schema != None and self.plan_aux_schemas != None
    }
    
    print(f"{self.plan_instructions}, {self.plan_main_schema}")

    missing = [k for k, v in checks.items() if v is False]
    failed = len(missing) > 0
    
    if failed:
      raise MissingAgentRequirements(missing)
    
    assert self.recall is not None
    assert self.config is not None
    assert self.plan_contract is not None
    assert self.plan_instructions is not None
    assert self.plan_main_schema is not None
    assert self.plan_aux_schemas is not None
    #assert self.thought_contract is not None   TODO
    #assert self.thought_gen_ruleset is not None  TODO
    #assert self.thought_gen_schema is not None TODO
    
    return Agent(
      self.goal,
      self.blackboard,
      self.recall,
      self.config,
      self.plan_contract,
      self.plan_instructions,
      self.plan_main_schema,
      self.plan_aux_schemas,
      Contract(state_keys=[], memory_keys=[]), [], {}, {} #TODO
      #self.thought_contract,
      #self.thought_gen_ruleset,
      #self.thought_gen_schema
    )
