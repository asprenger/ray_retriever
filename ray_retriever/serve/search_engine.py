from typing import List, Dict
from ray import serve
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.async_weaviate_client import AsyncWeaviateClient
from ray_retriever.serve.schema import TextNode, NodeWithScore, DocumentEmbedding

logger = get_logger()

@serve.deployment(name='SearchEngine')
class SearchEngine():
    """Retrieve context nodes that help to answer a user query."""

    def __init__(self, weaviate_hostname:str, weaviate_port:int, index_name:str, similarity_top_n:int):
        self.weaviate_client = AsyncWeaviateClient(weaviate_hostname, weaviate_port)
        self.index_name = index_name
        self.similarity_top_n = similarity_top_n

    async def search(self, query_embedding:DocumentEmbedding) -> List[NodeWithScore]:
        result = await self.weaviate_client.find_similar_nodes(self.index_name, query_embedding.embedding, self.similarity_top_n, False)
        return [NodeWithScore(node=node, score=similarity) 
                for node, similarity in zip(result.nodes, result.similarities)] 