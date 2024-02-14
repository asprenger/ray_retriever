import gradio as gr
from ray_retriever.client import sdk

# python -m ray_retriever.client.chatbot

def query(message, history):
    print(type(message), message)
    query_result = sdk.query(message)
    output = f"{query_result.response}<br>[Open Trace]({query_result.trace_url})"
    return output

examples = [
    "When has Alan Turing been born?",
    "What was the first capital of France?",
    "What is the shape of a qube?"
]

demo = gr.ChatInterface(fn=query, 
                        examples=examples, 
                        title="Ray Retriever UI", 
                        theme=gr.themes.Soft())

demo.launch(share=False)
