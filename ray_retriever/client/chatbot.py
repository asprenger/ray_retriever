import argparse
import gradio as gr
from ray_retriever.client import sdk

# python -m ray_retriever.client.chatbot

def query(message, history):
    query_result = sdk.query(message)
    output = f"{query_result.response}<br>[Open Trace]({query_result.trace_url})"
    return output

parser = argparse.ArgumentParser(description='Ray Retriever Chatbot')
parser.add_argument('--verbose', type=bool, required=False,
                    help="Enable verbose logging.", default=False)
args = parser.parse_args()
verbose = args.verbose

examples = [
    "When has Alan Turing been born?",
    "What was the first capital of France?",
    "What is the shape of a qube?"
]

interface = gr.ChatInterface(fn=query, 
                        examples=examples, 
                        title="Ray Retriever UI", 
                        theme=gr.themes.Soft())

interface.launch(share=False)
