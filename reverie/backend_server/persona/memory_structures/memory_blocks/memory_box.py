from collections.abc import Callable
from functools import reduce
import math
from uuid import uuid4

from pydantic import Field
from typing import Any, Dict, Optional
from generation.operations.bound_operations import gen_node_poignancy
from generation.operations.embed_operations import gen_embedding
from persona.memory_structures.memory_blocks.node import CoreNode, MemoryBatch, Node, RawNode, MemorySection
import numpy as np
import random as rd

from generation.requests import EmbeddingArray
from utils import prune


def node_from_raw(raw_node: RawNode, core_nodes: list[Node], curr_time: float = 0) -> Node: #TODO generate poignancy
        return Node(CoreNode(
            poignancy = gen_node_poignancy(core_nodes, raw_node.description),
            description = raw_node.description,
            entities_involved = raw_node.entities_involved
        ), curr_time)


def node_from_core(core_node: CoreNode, curr_time: float = 0):
    return Node(core_node, curr_time)


class MemoryBox:

    def __init__(self,
        sections: Optional[Dict[str, list[CoreNode]]] = None
    ):
        self.sections: MemoryBatch = {}

        if sections is not None:
            for section_name, nodes in sections.items():
                self.sections[section_name] = {}
                for raw_node in nodes:
                    self.add_core(section_name, raw_node, 0)

    
    def load_batch(self, batch: Dict[str, list[RawNode]], core_nodes: list[Node], curr_time: float = 0):
        for sec_name, raw_nodes in batch.items():
            self.load_section(sec_name, raw_nodes, core_nodes, curr_time)


    def load_section(self, section: str, raw_nodes: list[RawNode], core_nodes: list[Node], curr_time: float = 0):
        for raw_node in raw_nodes:
            new_node = node_from_raw(raw_node, core_nodes, curr_time)
            self.add(section, new_node)


    def add(self, section: str, node: Node):
        self.sections[section][str(uuid4())] = node


    def add_core(self, section: str, complete_node: CoreNode, curr_time: float = 0):
        self.add(section, node_from_core(complete_node, curr_time))


    def add_raw(self, section: str, raw_node: RawNode, core_nodes: list[Node], curr_time: float = 0):
        node = node_from_raw(raw_node, core_nodes, curr_time)
        self.add(section, node)


    def get_by_id(self, section: str, id: str) -> Node:
        return self.sections[section][id]


    def get_by_index(self, section: str, index: int) -> Optional[Node]:
        values = self.sections[section].values()
        if index >= len(values):
            return list(values)[index]
        else:
            return None
        

    def get_nodes(self) -> list[Node]:
        return [node for sec in self.sections.values() for node in sec.values()]
        

    def get_rand_nodes(self, examples: int) -> list[Node]:
        keys: list[Node] = []
        nodes = [node for sec in self.sections.values() for node in sec.values()]
        left = min(examples, len(nodes))

        while left > 0:
            sec_index = int(rd.random() * len(self.sections.keys()))
            sec_key = self.section_keys()[sec_index]
            node_index = int(rd.random() * len(self.sections[sec_key]))
            node = list(self.sections[sec_key].values())[node_index]
            keys.append(node)
            left -= 1

        return keys 
        

    def fetch(self,
        section: str,
        subject: Optional[str] = None,
        cutoff: Optional[int] = None,
        time_threshold: Optional[float] = None,
        touched_threshold: Optional[float] = None,
        poignancy_threshold: Optional[int] = None,
        distance: Optional[float] = None
    ) -> list[Node]:
        values = list(self.sections[section].values())

        if cutoff is not None:
            values = values[-cutoff:]
        if time_threshold is not None:
            values = prune(lambda node: node.creation_time >= time_threshold, values)
        if touched_threshold is not None:
            values = prune(lambda node: node.touched_time >= touched_threshold, values)
        if poignancy_threshold is not None:
            values = list(filter(lambda node: node.core.poignancy >= poignancy_threshold, values))
        if subject is not None and distance is not None:
            embedding = gen_embedding(subject)
            values = list(filter(lambda node: self.cosine_similarity(node.embedding, embedding) > distance, values))

        return values
    

    def test_fetch(self,
        section: str,
        subject: str,
        cutoff: Optional[int] = None,
        time_threshold: Optional[float] = None,
        touched_threshold: Optional[float] = None,
        poignancy_threshold: Optional[int] = None,
        distance: Optional[float] = None
    ):
        result: Dict[str, Dict] = {key: {"node": node} for key, node in self.sections[section].items()}
        values = [node for node in self.sections[section].values()]

        def test_nodes(condition_name: str, condition: Callable[[Node], bool]):
            for key, item in result.items():
                node = item["node"]
                result[key][condition_name] = condition(node)

        if cutoff is not None:
            test_nodes("recency_check", lambda node: node in values[-cutoff:])
        if time_threshold is not None:
            test_nodes("age_check", lambda node: node.creation_time >= time_threshold)
        if touched_threshold is not None:
            test_nodes("expiration_check", lambda node: node.touched_time >= touched_threshold)
        if poignancy_threshold is not None:
            test_nodes("poignancy_check", lambda node: node.core.poignancy >= poignancy_threshold)
        if subject is not None and distance is not None:
            embedding = gen_embedding(subject)
            test_nodes("semantic_distance_check", lambda node: self.cosine_similarity(node.embedding, embedding) > distance)

        for key in result.keys():
            result[key]["node"] = result[key]["node"].core

        return result
    

    def fetch_scored(self): #TODO try to do this later, use above for now
        pass
    

    def cosine_similarity(self, a: EmbeddingArray, b: EmbeddingArray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0
        return float(np.dot(a, b) / denominator)

    
    def section_keys(self) -> list[str]:
        return list(self.sections.keys())
    

class CacheBox(MemoryBox):

    def try_touch(self, section: str, node_id: str, curr_time: float):
        try:
            self.sections[section][node_id].touch(curr_time)
            print(f"TOUCHED -> node {node_id} in {section}")
        except:
            pass

    def cleanup(self, time: float, max_age: float): #TODO Optimize, use prune-like

        threshold = time - max_age
        remove_keys: list[tuple[str, str]] = []

        for sec_name, section in self.sections.items():
            for key, node in section.items():
                if node.touched_time < threshold:
                    remove_keys.append((sec_name, key))

        [self.sections[key[0]].pop(key[1]) for key in remove_keys]


    def refresh(self, node_ids: list[str], curr_time: float): #TODO touches nodes fitting certain criteria, similar to get
        for sec_name in self.sections.keys():
            for id in node_ids:
                self.try_touch(sec_name, id, curr_time)