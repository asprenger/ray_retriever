from typing import List, Dict, Optional, Sequence, Callable, Any, Union
import os
import re
from aiconfig import AIConfigRuntime, InferenceOptions
from aiconfig.callback import CallbackEvent, CallbackManager
from aiconfig.schema import ExecuteResult
from ray_retriever.utils.logging_utils import get_logger

logger = get_logger()

def get_parameters_in_template(template) -> dict:

    # Regular expression pattern to match Handlebars tags
    re_pattern = r"{{[{]?(.*?)[}]?}}"

    # Find all Handlebars tags in the template
    tags = [match.group(1).strip() for match in re.finditer(re_pattern, template)]
    return tags

def missing_parameters(template:str, params:List[str]):
    template_param = get_parameters_in_template(template)
    return list(set(template_param) - set(params))

class LLMConfig():
    """Wrapper for AIConfig"""

    def __init__(self, aiconfig_path:str):
         
        if os.getenv("OPENAI_API_KEY") is not None:
            logger.info('Found OPENAI_API_KEY')
        if os.getenv("ANYSCALE_ENDPOINT_API_KEY") is not None:
            logger.info('Found ANYSCALE_ENDPOINT_API_KEY')

        def callback_handler(event: CallbackEvent):
            pass

        aiconfig = AIConfigRuntime.load(aiconfig_path)
        aiconfig.set_callback_manager(CallbackManager(callbacks=[callback_handler]))
        self.aiconfig = aiconfig

    async def run(self, 
                  prompt_name:str, 
                  params:Dict[str,Any]) -> ExecuteResult:
        """Run the prompt with the given parameters.

        Args:
            prompt_name (str): The prompt name
            params (Dict[str,Any]): The parameters

        Returns:
            ExecuteResult: Exection result
        """
        self._check_params(prompt_name, params.keys())
        inference_options = InferenceOptions(stream=False)
        result = await self.aiconfig.run(prompt_name=prompt_name, 
                                params=params, 
                                options=inference_options, 
                                run_with_dependencies=False,
                                seed=42)
        return result[0]
    
    def get_prompt(self, prompt_name:str):
        prompts = [prompt for prompt in self.aiconfig.prompts 
                   if prompt.name==prompt_name]
        if len(prompts) > 0:
            return prompts[0]
        else:
            return None

    def _check_params(self, prompt_name:str, param_names:List[str]) -> None:
        prompt = self.get_prompt(prompt_name)
        if prompt is None:
            raise ValueError(f'Unknown prompt: {prompt_name}')
        user_prompt_template = prompt.input
        system_prompt_template = prompt.metadata.model.settings['system_prompt']
        missing_params = missing_parameters(system_prompt_template, param_names)
        if len(missing_params) != 0:
            raise ValueError(f'Missing params for system_message in prompt "{prompt_name}": {missing_params}')
        missing_params = missing_parameters(user_prompt_template, param_names)
        if len(missing_params) != 0:
            raise ValueError(f'Missing params for user_message in prompt "{prompt_name}": {missing_params}')
