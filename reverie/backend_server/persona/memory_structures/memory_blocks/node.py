from typing import Dict, Optional
from pydantic import Field, BaseModel
from generation.operations.embed_operations import gen_embedding
from generation.requests import EmbeddingArray
from persona.aid import Entity, Tool
import regex as re
from standard import ENTITY_FENCES


Embedding = tuple[float]

class CoreNode(BaseModel):
    description: str
    entity_snapshots: Dict[str, str]


class MemoryNode:
    core: CoreNode
    embedding: EmbeddingArray
    creation_time: float
    touched_time: float

    def __init__(self, core: CoreNode, time: float):
        self.core = core
        self.creation_time = time
        self.touched_time = time
        self.embedding = gen_embedding(core.description)
        self.checkConsistency()

    def checkConsistency(self):
        mentioned = extract_ids(self.core.description)
        
        if mentioned & set(self.core.entity_snapshots.keys()):
            raise Exception("Identified entities do not match references in this node's description")

    def touch(self, curr_time: float):
        self.touched_time = curr_time
    

MemorySection = Dict[str, MemoryNode]
MemoryBatch = Dict[str, MemorySection]


def extract_ids(description: str) -> set[str]:
    pattern = (
        rf"{re.escape(ENTITY_FENCES[0])}"
        r"([a-zA-Z0-9_-]+)"
        rf"{re.escape(ENTITY_FENCES[1])}"
    )
    mentioned: set[str] = set(re.findall(pattern, description))

    return mentioned
