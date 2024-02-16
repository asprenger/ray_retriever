from typing import List, Dict, Union
from langfuse import Langfuse
from ray import serve
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.async_weaviate_client import AsyncWeaviateClient
from ray_retriever.serve.schema import TextNode, NodeWithScore, DocumentEmbedding

logger = get_logger()

@serve.deployment(name='SearchEngine')
class SearchEngine():
    """Retrieve context nodes that help to answer a user query."""

    def __init__(self, weaviate_hostname:str, weaviate_port:int, index_name:str, similarity_top_n:int):
        """_summary_

        

        Args:
            weaviate_hostname (str): _description_
            weaviate_port (int): _description_
            index_name (str): Index name. The index name is case sensitive.
            similarity_top_n (int): _description_
        """
        self.weaviate_client = AsyncWeaviateClient(weaviate_hostname, weaviate_port)
        self.index_name = index_name
        self.similarity_top_n = similarity_top_n
        self.langfuse = Langfuse()

    async def get_text_node(self, node_id:str, return_embedding:bool=False) -> Union[TextNode, None]:
        filter = {
            "path": "id",
            "operator": "Equal",
            "valueText": node_id
        }
        result = await self.weaviate_client.get_nodes(index_name=self.index_name, 
                                                      filter=filter, 
                                                      max_result_size=1, 
                                                      return_embeddings=return_embedding)
        if len(result) == 1:
            return result[0]
        else:
            return None
        

    async def search(self, query_embedding:DocumentEmbedding, trace_id:str) -> List[NodeWithScore]:
        span = self.langfuse.span(trace_id=trace_id, 
                                  name='context-retrieval',
                                  metadata={'index_name':self.index_name, 'top_n': self.similarity_top_n})
        result = await self.weaviate_client.find_similar_nodes(self.index_name, query_embedding.embedding, self.similarity_top_n, False)
        
        context_nodes =  [NodeWithScore(node=node, score=similarity) 
                          for node, similarity in zip(result.nodes, result.similarities)] 
        
        context_node_info = [node.node.metadata|{"node_id":node.node.id, "index_name":node.node.index_name} 
                             for node in context_nodes]

        span.end(output=context_node_info)

        return context_nodes