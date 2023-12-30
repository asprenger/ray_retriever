import os
import unittest
from ray_retriever.serve.prompt_manager import PromptManager

class PromptManagerCases(unittest.TestCase):

    def test_get_prompt(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        prompts_path = os.path.join(dir_path, 'test_prompts.json')
        prompt_manager = PromptManager(prompts_path)

        prompt = prompt_manager.get_prompt('response_generator', 'unknown_model')
        self.assertEqual(prompt.system_message, "Default system message")
        self.assertEqual(prompt.user_message, "Default user message")

        prompt = prompt_manager.get_prompt('response_generator', 'llama2-7b')
        self.assertEqual(prompt.system_message, "llama2-7b system message")
        self.assertEqual(prompt.user_message, "llama2-7b user message")

        try:
            prompt = prompt_manager.get_prompt('unknown_module', 'llama2-7b')
            self.fail('Expected ValueError')
        except ValueError:
            pass