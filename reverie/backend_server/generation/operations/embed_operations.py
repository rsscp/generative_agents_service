from generation.requests import EmbeddingArray, embedding_request

def gen_embedding(description: str) -> EmbeddingArray: #TODO
    return embedding_request(description)
