from typing import List, Dict, Optional
from ray import serve
from openai import AsyncOpenAI
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.schema import NodeWithScore, RetrieverResponse, TokenUsage

logger = get_logger()

@serve.deployment(name='ResponseGenerator')
class ResponseGenerator():

    def __init__(self, 
                 model_id:str,
                 anyscale_endpoint_key:Optional[str]=None, 
                 openai_api_key:Optional[str]=None,
                 llm_max_tokens:int=256,
                 temperature:float=0.0):
        
        if anyscale_endpoint_key:
            self.client = AsyncOpenAI(base_url="https://api.endpoints.anyscale.com/v1", 
                                      api_key=anyscale_endpoint_key)
        elif openai_api_key:
            self.client = AsyncOpenAI(api_key=openai_api_key)
        else:
            raise ValueError('One of ["anyscale_endpoint_key", "openai_api_key"] must be set')

        self.model_id = model_id
        self.max_tokens=llm_max_tokens
        self.temperature = temperature
        self.seed = 42

    async def generate_response(self, query:str, nodes: List[NodeWithScore]) -> RetrieverResponse:

        system_message = "You are a helpful assistant. Always give a consise answer, do not reply using a complete sentence."

        context = "\n".join([node.node.text for node in nodes])        

        user_message = (
            "Context information is below.\n"
            "---------------------\n"
            f"{context}\n"
            "---------------------\n"
            "If the question can not be answered from the context information say ONLY 'I do not know'."
            f"Answer the question: {query}\n"
        )
    
        messages = [
            {"role": "system", "content": system_message },
            {"role": "user", "content": user_message }
        ]

        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            seed=self.seed,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        usage = TokenUsage(completion_tokens=response.usage.completion_tokens, 
                           prompt_tokens=response.usage.prompt_tokens, 
                           total_tokens=response.usage.total_tokens)

        return RetrieverResponse(response=response.choices[0].message.content,
                                 finish_reason=response.choices[0].finish_reason,
                                 model=response.model,
                                 usage=usage)