from typing import List, Dict, Optional
import time
from ray import serve
from llama_index.llms.base import ChatResponse
from llama_index.llms import ChatMessage, MessageRole
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.schema import NodeWithScore, RetrieverResponse
from ray_retriever.serve.llm import setup_openai_llm, setup_anyscale_llm

logger = get_logger()

@serve.deployment(name='ResponseGenerator')
class ResponseGenerator():

    def __init__(self, 
                 model_id:str,
                 anyscale_endpoint_key:Optional[str]=None, 
                 openai_api_key:Optional[str]=None,
                 llm_max_tokens:int=256,
                 temperature:float=0.0):
        
        if anyscale_endpoint_key and openai_api_key:
            raise ValueError('Only one of ["anyscale_endpoint_key", "openai_api_key"] must be set')

        if anyscale_endpoint_key:
            self.llm = setup_anyscale_llm(anyscale_endpoint_key=anyscale_endpoint_key,
                                          model=model_id,
                                          max_tokens=llm_max_tokens,
                                          temperature=temperature)
        elif openai_api_key:
            self.llm = setup_openai_llm(openai_api_key=openai_api_key,
                                        model=model_id,
                                        temperature=temperature,
                                        max_tokens=llm_max_tokens)
        else:
            raise ValueError('One of ["anyscale_endpoint_key", "openai_api_key"] must be set')

    async def generate_response(self, query:str, nodes: List[NodeWithScore]) -> RetrieverResponse:

        system_message = "You are a helpful assistant. Always give a consise answer, do not reply using a complete sentence."

        context = "\n".join([node.node.text for node in nodes])        

        user_message =(
            "Context information is below.\n"
            "---------------------\n"
            f"{context}\n"
            "---------------------\n"
            "If the question can not be answered from the context information say ONLY 'I do not know'."
            f"Answer the question: {query}\n"
        )

        message_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_message),
            ChatMessage(role=MessageRole.USER, content=user_message)
        ]

        start = time.perf_counter()
        #response:ChatResponse = await self.llm.achat(message_history)
        duration = time.perf_counter() - start
        
        #return RetrieverResponse(response=response.message.content)
        return RetrieverResponse(response='RESPONSE')