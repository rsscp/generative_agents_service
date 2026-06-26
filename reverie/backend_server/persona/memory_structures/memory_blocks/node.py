from typing import Dict, Optional
from pydantic import Field, BaseModel
from generation.operations.embed_operations import gen_embedding
from generation.requests import EmbeddingArray


Embedding = tuple[float]

class CoreNode(BaseModel):
    poignancy: int
    description: str
    entities_involved: list[str]


class Node:
    core: CoreNode
    embedding: EmbeddingArray
    creation_time: float
    touched_time: float

    def __init__(self, core: CoreNode, time: float):
        self.core = core
        self.creation_time = time
        self.touched_time = time
        self.embedding = gen_embedding(core.description)

    def touch(self, curr_time: float):
        self.touched_time = curr_time


class RawNode(BaseModel):
    description: str
    entities_involved: list[str]


class EntityNode(BaseModel):
    description: str

MemorySection = Dict[str, Node]
MemoryBatch = Dict[str, MemorySection]
