"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: associative_memory.py
Description: Defines the core long-term memory module for generative agents.

Note (May 1, 2023) -- this class is the Memory Stream module in the generative
agents paper. 
"""
import sys

from pydantic import BaseModel

sys.path.append('../../')

import json
import datetime

from global_methods import *
from typing import Dict, Optional
from persona.memory_structures.memory_blocks.memory_box import CacheBox, MemoryBox, node_from_core
from persona.memory_structures.memory_blocks.node import CoreNode, Node, RawNode
from persona.memory_structures.associative_memory import ConceptNode


STANDARD_RECENCY_THRESHOLD = 10
STANDARD_AGE_THRESHOLD = 1000000
STANDARD_EXPIRATION_TIME = 3600
STANDARD_POIGNANCY_THRESHOLD = 60
STANDARD_SEMANTIC_DISTANCE = 0.3


class RecallConfig(BaseModel):
  recency: int
  age: float
  expiration: float
  poignancy: int
  distance: float


class Recall: 
  def __init__(self,
    core_nodes: list[CoreNode],
    node_sections: Dict[str, list[CoreNode]],
  ):
    self.id_to_node = dict()

    self.seq_event = []
    self.seq_thought = []
    self.seq_interaction = []

    self.kw_to_event = dict()
    self.kw_to_thought = dict()
    self.kw_to_interaction = dict()

    self.kw_strength_event = dict()
    self.kw_strength_thought = dict()

    self.time = 0
    self.config = RecallConfig(
      recency = STANDARD_RECENCY_THRESHOLD,
      age = STANDARD_AGE_THRESHOLD,
      expiration = STANDARD_EXPIRATION_TIME,
      poignancy = STANDARD_POIGNANCY_THRESHOLD,
      distance = STANDARD_SEMANTIC_DISTANCE
    )

    self.core: list[Node] = [node_from_core(node) for node in core_nodes]
    self.memory: MemoryBox = MemoryBox(node_sections)    # TODO initialize memory box by calling embedding model for every node (entire node or just description?)
    self.cache: CacheBox = CacheBox({sec: [] for sec in node_sections.keys()})


  def load_cache(self,
    sections: list[str],
    subject: str,
  ):
    time_threshold: float = max(0, self.time - self.config.age)
    touched_threshold: float = max(0, self.time - self.config.expiration)

    self.load_cache_specific(
      sections,
      subject,
      self.config.recency,
      time_threshold,
      touched_threshold,
      self.config.poignancy,
      self.config.distance,
    )
  
  def load_cache_specific(self,
    sections: list[str],
    subject: str,
    recency: Optional[int] = None,
    time_threshold: Optional[float] = None,
    touched_threshold: Optional[float] = None,
    poignancy: Optional[int] = None,
    distance: Optional[float] = None,
  ):
    for sec in sections:
      relevant = self.memory.fetch(
        sec,
        subject,
        recency,
        time_threshold,
        touched_threshold,
        poignancy,
        distance
      )
      [self.cache.add(sec, node) for node in relevant]


  def load_cache_debug(self, subject: str):
    secs = self.memory.section_keys()
    result = {}

    time_threshold: float = max(0, self.time - self.config.age)
    touched_threshold: float = max(0, self.time - self.config.expiration)

    for sec_name in secs:
      result[sec_name] = self.memory.test_fetch(sec_name,
        subject,
        self.config.recency,
        time_threshold,
        touched_threshold,
        self.config.poignancy,
        self.config.distance,
      )

    return result


  def update(self, curr_time: float, new_batch: Optional[Dict[str, list[RawNode]]]):
    if curr_time < self.time:
      raise Exception("Provided game time is behind agent system accounted time") #TODO fazer exceção mais específica
  
    self.cache.cleanup(curr_time, STANDARD_AGE_THRESHOLD)

    if new_batch is not None:
      self.cache.load_batch(new_batch, self.core)
      self.memory.load_batch(new_batch, self.core)

    
  def save(self, out_json):
    r = dict()
    for count in range(len(self.id_to_node.keys()), 0, -1): 
      node_id = f"node_{str(count)}"
      node = self.id_to_node[node_id]

      r[node_id] = dict()
      r[node_id]["node_count"] = node.node_count
      r[node_id]["type_count"] = node.type_count
      r[node_id]["type"] = node.type
      r[node_id]["depth"] = node.depth

      r[node_id]["created"] = node.created.strftime('%Y-%m-%d %H:%M:%S')
      r[node_id]["expiration"] = None
      if node.expiration: 
        r[node_id]["expiration"] = (node.expiration
                                        .strftime('%Y-%m-%d %H:%M:%S'))

      r[node_id]["subject"] = node.subject
      r[node_id]["predicate"] = node.predicate
      r[node_id]["object"] = node.object

      r[node_id]["description"] = node.description
      r[node_id]["embedding_key"] = node.embedding_key
      r[node_id]["poignancy"] = node.poignancy
      r[node_id]["keywords"] = list(node.keywords)
      r[node_id]["filling"] = node.filling

    with open(out_json+"/nodes.json", "w") as outfile:
      json.dump(r, outfile)

    r = dict()
    r["kw_strength_event"] = self.kw_strength_event
    r["kw_strength_thought"] = self.kw_strength_thought
    with open(out_json+"/kw_strength.json", "w") as outfile:
      json.dump(r, outfile)

    with open(out_json+"/embeddings.json", "w") as outfile:
      json.dump(self.embeddings, outfile)


  def add_event(self, created, expiration, s, p, o, 
                      description, keywords, poignancy, 
                      embedding_pair, filling):
    # Setting up the node ID and counts.
    node_count = len(self.id_to_node.keys()) + 1
    type_count = len(self.seq_event) + 1
    node_type = "event"
    node_id = f"node_{str(node_count)}"
    depth = 0

    # Node type specific clean up. 
    if "(" in description: 
      description = (" ".join(description.split()[:3]) 
                     + " " 
                     +  description.split("(")[-1][:-1])

    # Creating the <ConceptNode> object.
    node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                       created, expiration, 
                       s, p, o, 
                       description, embedding_pair[0], 
                       poignancy, keywords, filling)

    # Creating various dictionary cache for fast access. 
    self.seq_event[0:0] = [node]
    keywords = [i.lower() for i in keywords]
    for kw in keywords: 
      if kw in self.kw_to_event: 
        self.kw_to_event[kw][0:0] = [node]
      else: 
        self.kw_to_event[kw] = [node]
    self.id_to_node[node_id] = node 

    # Adding in the kw_strength
    if f"{p} {o}" != "is idle":  
      for kw in keywords: 
        if kw in self.kw_strength_event: 
          self.kw_strength_event[kw] += 1
        else: 
          self.kw_strength_event[kw] = 1

    self.embeddings[embedding_pair[0]] = embedding_pair[1]

    return node


  def add_thought(self, created, expiration, s, p, o, 
                        description, keywords, poignancy, 
                        embedding_pair, filling):
    # Setting up the node ID and counts.
    node_count = len(self.id_to_node.keys()) + 1
    type_count = len(self.seq_thought) + 1
    node_type = "thought"
    node_id = f"node_{str(node_count)}"
    depth = 1 
    try: 
      if filling: 
        depth += max([self.id_to_node[i].depth for i in filling])
    except: 
      pass

    # Creating the <ConceptNode> object.
    node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                       created, expiration, 
                       s, p, o, 
                       description, embedding_pair[0], poignancy, keywords, filling)

    # Creating various dictionary cache for fast access. 
    self.seq_thought[0:0] = [node]
    keywords = [i.lower() for i in keywords]
    for kw in keywords: 
      if kw in self.kw_to_thought: 
        self.kw_to_thought[kw][0:0] = [node]
      else: 
        self.kw_to_thought[kw] = [node]
    self.id_to_node[node_id] = node 

    # Adding in the kw_strength
    if f"{p} {o}" != "is idle":  
      for kw in keywords: 
        if kw in self.kw_strength_thought: 
          self.kw_strength_thought[kw] += 1
        else: 
          self.kw_strength_thought[kw] = 1

    self.embeddings[embedding_pair[0]] = embedding_pair[1]

    return node


  def add_chat(self, created, expiration, s, p, o, 
                     description, keywords, poignancy, 
                     embedding_pair, filling): 
    # Setting up the node ID and counts.
    node_count = len(self.id_to_node.keys()) + 1
    type_count = len(self.seq_chat) + 1
    node_type = "chat"
    node_id = f"node_{str(node_count)}"
    depth = 0

    # Creating the <ConceptNode> object.
    node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                       created, expiration, 
                       s, p, o, 
                       description, embedding_pair[0], poignancy, keywords, filling)

    # Creating various dictionary cache for fast access. 
    self.seq_chat[0:0] = [node]
    keywords = [i.lower() for i in keywords]
    for kw in keywords: 
      if kw in self.kw_to_chat: 
        self.kw_to_chat[kw][0:0] = [node]
      else: 
        self.kw_to_chat[kw] = [node]
    self.id_to_node[node_id] = node 

    self.embeddings[embedding_pair[0]] = embedding_pair[1]
        
    return node


  def get_summarized_latest_events(self, retention): 
    ret_set = set()
    for e_node in self.seq_event[:retention]: 
      ret_set.add(e_node.spo_summary())
    return ret_set


  def get_str_seq_events(self): 
    ret_str = ""
    for count, event in enumerate(self.seq_event): 
      ret_str += f'{"Event", len(self.seq_event) - count, ": ", event.spo_summary(), " -- ", event.description}\n'
    return ret_str


  def get_str_seq_thoughts(self): 
    ret_str = ""
    for count, event in enumerate(self.seq_thought): 
      ret_str += f'{"Thought", len(self.seq_thought) - count, ": ", event.spo_summary(), " -- ", event.description}'
    return ret_str


  def get_str_seq_chats(self): 
    ret_str = ""
    for count, event in enumerate(self.seq_chat): 
      ret_str += f"with {event.object.content} ({event.description})\n"
      ret_str += f'{event.created.strftime("%B %d, %Y, %H:%M:%S")}\n'
      for row in event.filling: 
        ret_str += f"{row[0]}: {row[1]}\n"
    return ret_str


  def retrieve_relevant_thoughts(self, s_content, p_content, o_content): 
    contents = [s_content, p_content, o_content]

    ret = []
    for i in contents: 
      if i in self.kw_to_thought: 
        ret += self.kw_to_thought[i.lower()]

    ret = set(ret)
    return ret


  def retrieve_relevant_events(self, s_content, p_content, o_content): 
    contents = [s_content, p_content, o_content]

    ret = []
    for i in contents: 
      if i in self.kw_to_event: 
        ret += self.kw_to_event[i]

    ret = set(ret)
    return ret


  def get_last_chat(self, target_persona_name): 
    if target_persona_name.lower() in self.kw_to_chat: 
      return self.kw_to_chat[target_persona_name.lower()][0]
    else: 
      return False
