from typing import List, Dict
from ray import serve
from langfuse import Langfuse
from sentence_transformers import SentenceTransformer
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.schema import DocumentEmbedding

logger = get_logger()

@serve.deployment(name='EmbeddingGenerator')
class EmbeddingGenerator():
    """Calculate the embedding vector for a user query. The embedding
    vector is calculated by a local embedding model. This task should
    be deployed on a GPU node.
    """

    def __init__(self, model: str, batch_size:int, batch_wait_timeout_s:float):
        self.handle_batch.set_max_batch_size(batch_size)
        self.handle_batch.set_batch_wait_timeout_s(batch_wait_timeout_s)
        self.batch_size = batch_size
        self.langfuse = Langfuse()
        logger.info(f"Loading {model}")
        self.model_name = model
        self.model = SentenceTransformer(model)

        # Warmup embedding model
        self.model.encode(['Warmup...'], normalize_embeddings=True) 

    @serve.batch
    async def handle_batch(self, inputs: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(inputs, batch_size=self.batch_size, 
                                       normalize_embeddings=True) 
        return embeddings.tolist()
    
    async def calculate_embedding(self, query:str, trace_id:str) -> DocumentEmbedding:
        span = self.langfuse.span(trace_id=trace_id, 
                                  name='query-embedding',
                                  metadata={'model':self.model_name})
        query_embedding = await self.handle_batch(query)
        span.end()
        return DocumentEmbedding(embedding=query_embedding)
