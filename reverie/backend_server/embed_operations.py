from reverie.backend_server.llm_operations import embedding_request
from reverie.backend_server.persona.memory_structures.memory_blocks.node import EmbeddingArray


def gen_embedding(description: str) -> EmbeddingArray: #TODO
    return embedding_request(description)