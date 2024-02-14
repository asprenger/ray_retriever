from typing import List, Dict, Optional
import os
from ray import serve
from langfuse import Langfuse
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.schema import NodeWithScore, RetrieverResponse, TokenUsage
from ray_retriever.serve.llm_config import LLMConfig

logger = get_logger()

@serve.deployment(name='ResponseGenerator')
class ResponseGenerator():

    def __init__(self, prompt_name:str):
        aiconfig_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'aiconfig.yaml')
        self.llm_config = LLMConfig(aiconfig_path=aiconfig_path)

        prompt = self.llm_config.get_prompt(prompt_name) 
        if prompt is None:
            raise ValueError(f'Unknown prompt "{prompt_name}"')
        
        self.prompt_name = prompt_name
        self.langfuse = Langfuse()
        
    async def generate_response(self, 
                                query:str, 
                                context_nodes: List[NodeWithScore], 
                                trace_id:str) -> RetrieverResponse:
        "Answer a user query based on a set of context nodes."

        context = "\n".join([node.node.text for node in context_nodes])

        prompt = self.llm_config.aiconfig.prompt_index[self.prompt_name]
        model_settings = dict(prompt.metadata.model.settings)
        generation = self.langfuse.generation(
            trace_id=trace_id,
            name="response-generation",
            input={'context':context, 'query':query},
            model_parameters=model_settings
        )

        result = await self.llm_config.run(prompt_name=self.prompt_name, 
                                           params={'context':context, 'query':query})

        context_node_info = [node.node.metadata | {"node_id":node.node.id, "index_name":node.node.index_name} 
                             for node in context_nodes]
        output = result.data.strip()
        usage = result.metadata['usage']

        generation.end(output=output,
                    metadata={
                        "prompt_name": self.prompt_name,
                        "context_nodes": context_node_info,
                        "finish_reason": result.metadata['finish_reason'],
                        "model": result.metadata['model'],
                        "prompt_tokens": usage['prompt_tokens'],
                        "completion_tokens": usage['completion_tokens'],  
                        "total_tokens": usage['total_tokens']
                        })

        token_usage = TokenUsage(prompt_tokens=usage['prompt_tokens'],
                        completion_tokens=usage['completion_tokens'],  
                        total_tokens=usage['total_tokens'])

        return RetrieverResponse(response=output,
                                finish_reason=result.metadata['finish_reason'],
                                model=result.metadata['model'],
                                usage=token_usage,
                                context_node_info=context_node_info)

