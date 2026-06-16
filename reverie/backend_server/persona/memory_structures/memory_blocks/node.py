from typing import Dict, Optional
from numpy.typing import NDArray
from pydantic import Field, BaseModel
import numpy as np

from reverie.backend_server.embed_operations import gen_embedding


Embedding = tuple[float]
EmbeddingArray = NDArray[np.float32]

class CoreNode(BaseModel):
    object: Dict
    poignancy: int
    description: str
    entities_involved: list[str]

class Node(CoreNode):
    embedding: EmbeddingArray

    def __init__(self,
        poignancy: int,
        description: str,
        object: Dict,
        entities_involved: list[str]
    ):
        self.embedding = gen_embedding(self.description)

        super().__init__(
            object=object,
            poignancy=poignancy,
            description=description,
            entities_involved=entities_involved
        )

class RawNode(BaseModel):
    object: Dict
    description: Optional[str] = None
    entities_involved: list[str]

MemorySection = Dict[str, Node]
