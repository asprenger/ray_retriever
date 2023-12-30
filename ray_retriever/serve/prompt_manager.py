from pydantic import BaseModel
import os
import json

class Prompt(BaseModel):
    system_message: str
    user_message: str

class PromptManager:

    def __init__(self, prompts_path:str=None):

        if not prompts_path:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            prompts_path = os.path.join(dir_path, 'prompts.json')

        with open(prompts_path, 'r') as file:
            prompts = json.load(file)

        self.prompts = {}
        for module,v in prompts.items():
            self.prompts[module] = dict((prompt['model_id'], Prompt(**prompt)) 
                                   for prompt in v)
            if 'default' not in self.prompts[module]:
                raise ValueError(f'"{prompts_path}" does not define a default prompt for module "{module}"')

            
    def get_prompt(self, module:str, model_id:str) -> Prompt:
        if module not in self.prompts:
            raise ValueError(f'No prompt defined for module "{module}"')
        if model_id in self.prompts[module]:
            return self.prompts[module][model_id]
        else:
            return self.prompts[module]['default']
