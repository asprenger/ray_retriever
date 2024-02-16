from typing import Any, Dict, Iterator, List, Optional, Union, Annotated
import typer
import json
from rich import print as rp
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio
from ray_retriever.serve.async_weaviate_client import AsyncWeaviateClient

# python -m ray_retriever.index.index_admin

app = typer.Typer()

def progress_spinner():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

async def run_schema(index_name: str, 
                     hostname: str, 
                     port: int):
    aclient = AsyncWeaviateClient(hostname, port)
    try:
        schema = await aclient.get_schema(index_name)
        print(json.dumps(schema, indent=2))
    finally:
        await aclient.close()

@app.command()
def schema(index_name:str = 'Wikipedia',
           hostname:str = '127.0.0.1', 
           port:int = 9001):
    asyncio.run(run_schema(index_name, hostname, port))

async def run_content(index_name:str,
                      hostname:str, 
                      port:int):
    aclient = AsyncWeaviateClient(hostname, port)
    try:
        nodes = await aclient.get_all_nodes(index_name, max_result_size=5000)
        titles = sorted(list(set(node.metadata['title'] for node in nodes)))
        print(json.dumps(titles))
    finally:
        await aclient.close()

@app.command()
def content(index_name:str = 'Wikipedia',
            hostname:str = '127.0.0.1', 
            port:int = 9001):
    asyncio.run(run_content(index_name, hostname, port))

if __name__ == "__main__":
    app()
