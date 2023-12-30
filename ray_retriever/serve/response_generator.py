from typing import List, Dict, Optional
from ray import serve
from openai import AsyncOpenAI
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.schema import NodeWithScore, RetrieverResponse, TokenUsage
from ray_retriever.serve.prompt_manager import PromptManager

MODULE_NAME = 'response_generator'

logger = get_logger()

@serve.deployment(name='ResponseGenerator')
class ResponseGenerator():

    def __init__(self, 
                 model_id:str,
                 anyscale_endpoint_key:Optional[str]=None, 
                 openai_api_key:Optional[str]=None,
                 llm_max_tokens:int=256,
                 temperature:float=0.0,
                 seed:int=42):
        
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
        self.seed = seed
        self.prompt_manager = PromptManager()

    async def generate_response(self, query:str, context_nodes: List[NodeWithScore]) -> RetrieverResponse:
        "Answer a user query based on a set of context nodes."

        prompt = self.prompt_manager.get_prompt(MODULE_NAME, self.model_id)
        
        context = "\n".join([node.node.text for node in context_nodes])
        user_message = prompt.user_message.replace('{context}', context).replace('{query}', query)
    
        messages = [
            {"role": "system", "content": prompt.system_message },
            {"role": "user", "content": user_message }
        ]

        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            seed=self.seed,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        context_node_info = [node.node.metadata|{"id":node.node.id} for node in context_nodes]
        
        usage = TokenUsage(completion_tokens=response.usage.completion_tokens, 
                           prompt_tokens=response.usage.prompt_tokens, 
                           total_tokens=response.usage.total_tokens)

        return RetrieverResponse(response=response.choices[0].message.content,
                                 finish_reason=response.choices[0].finish_reason,
                                 model=response.model,
                                 usage=usage,
                                 context_node_info=context_node_info)