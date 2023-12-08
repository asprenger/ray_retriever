from llama_index.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anyscale import Anyscale

def setup_openai_llm(openai_api_key:str, 
                     model:str, 
                     temperature:float=0.0, 
                     max_tokens:int=128) -> LLM:
    return OpenAI(temperature=temperature, 
                  model=model, 
                  max_tokens=max_tokens, 
                  api_key=openai_api_key)

def setup_anyscale_llm(anyscale_endpoint_key:str, 
                     model:str, 
                     temperature:float=0.0, 
                     max_tokens:int=128) -> LLM:
    return Anyscale(model=model, 
                    temperature=temperature, 
                    max_tokens=max_tokens, 
                    api_key=anyscale_endpoint_key)