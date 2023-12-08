from typing import Any, Dict, Iterator, List, Optional, Union, Annotated
import typer
import json
from rich import print as rp
from rich.progress import Progress, SpinnerColumn, TextColumn
from ray_retriever.client import sdk

# python -m semantic_search.client.cli query 'Where has Alan Turing been born?'

app = typer.Typer()

def progress_spinner():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

@app.command()
def models(metadata: Annotated[bool, "Whether to print metadata"] = False):
    """Get a list of the available models"""
    result = ['falcon-7b', 'falcon-13b','falcon-30b']
    if metadata:
        for model in result:
            rp(f"[bold]{model}:[/]")
            rp('__METADATA__')
    else:
        print("\n".join(result))

@app.command()
def query(query: Annotated[str, typer.Argument()]):
    with progress_spinner() as progress:
        progress.add_task(
                description=f"Run query...",
                total=None,
            )
        result = sdk.query(query)
        print(f"Answer: {result.response}")

@app.command()
def search(query: Annotated[str, typer.Argument()]):
    with progress_spinner() as progress:
        progress.add_task(
                description=f"Run search...",
                total=None,
            )
        result = sdk.search(query)
        print(json.dumps(result.nodes))

if __name__ == "__main__":
    app()
