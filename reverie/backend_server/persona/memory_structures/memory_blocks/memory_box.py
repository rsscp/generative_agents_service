from functools import reduce
import math
from uuid import uuid4

from pydantic import Field
from typing import Dict, Optional
from llm_operations import gen_node_description, gen_node_poignancy, gen_node_reqs
from persona.memory_structures.memory_blocks.node import CoreNode, Node, RawNode, EmbeddingArray, MemorySection
import numpy as np
import random as rd

from reverie.backend_server.embed_operations import gen_embedding
from reverie.backend_server.persona.memory_structures.memory_blocks.node import CoreNode, Node, RawNode


def node_from_raw(raw_node: RawNode, core_nodes: list[Node]) -> Node: #TODO generate poignancy
        if raw_node.description is None:
            description, poignancy = gen_node_reqs(core_nodes, raw_node)
        else:
            description = raw_node.description
            poignancy = gen_node_poignancy(core_nodes, description)

        return Node(
            poignancy,
            description,
            raw_node.object,
            raw_node.entities_involved
        )


def node_from_core(core_node: CoreNode):
    return Node(
        core_node.poignancy,
        core_node.description,
        core_node.object,
        core_node.entities_involved
    )


class MemoryBox:

    def __init__(self,
        sections: Optional[Dict[str, list[CoreNode]]] = None
    ):
        self.sections: Dict[str, MemorySection] = {}

        if sections is not None:
            for section_name, nodes in sections.items():
                self.sections[section_name] = {}
                for raw_node in nodes:
                    self.add_core(section_name, raw_node)




    def add(self, section: str, node: Node):
        self.sections[section][str(uuid4())] = node


    def add_core(self, section: str, complete_node: CoreNode):
        self.add(section, node_from_core(complete_node))


    def add_raw(self, section: str, raw_node: RawNode, core_nodes: list[Node]):
        node = node_from_raw(raw_node, core_nodes)
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
        

    def get(self,
        section: str,
        subject: Optional[str] = None,
        recency: Optional[int] = None,
        threshold: Optional[int] = None,
        distance: Optional[float] = None
    ) -> list[Node]:
        values = list(self.sections[section].values())

        if recency is not None:
            values = values[-recency:]
        if threshold is not None:
            values = list(filter(lambda node: node.poignancy >= threshold, values))
        if subject is not None and distance is not None:
            embedding = gen_embedding(subject)
            values = list(filter(lambda node: self.cosine_similarity(node.embedding, embedding) > distance, values))

        return values
    

    def cosine_similarity(self, a: EmbeddingArray, b: EmbeddingArray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0
        return float(np.dot(a, b) / denominator)

    
    def section_keys(self) -> list[str]:
        return list(self.sections.keys())
    