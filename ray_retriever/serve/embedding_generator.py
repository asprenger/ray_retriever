from typing import List, Dict
from ray import serve
from sentence_transformers import SentenceTransformer
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.schema import DocumentEmbedding

logger = get_logger()

@serve.deployment(name='EmbeddingGenerator')
class EmbeddingGenerator():
    """Calculate embeddings for a user query."""

    def __init__(self, model: str, batch_size:int, batch_wait_timeout_s:float):
        self.handle_batch.set_max_batch_size(batch_size)
        self.handle_batch.set_batch_wait_timeout_s(batch_wait_timeout_s)
        self.batch_size = batch_size
        logger.info(f"Loading {model}")
        self.model = SentenceTransformer(model)
        # Warmup embedding model
        self.model.encode(['Warmup...'], normalize_embeddings=True) 

    @serve.batch
    async def handle_batch(self, inputs: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(inputs, batch_size=self.batch_size, 
                                       normalize_embeddings=True) 
        return embeddings.tolist()
    
    async def calculate_embedding(self, query:str) -> DocumentEmbedding:
        query_embedding = await self.handle_batch(query)
        return DocumentEmbedding(embedding=query_embedding)
