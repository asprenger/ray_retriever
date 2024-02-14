from typing import Any, Dict, Iterator, List, Optional, Union, Annotated
import typer
from rich import print as rp
from rich.progress import Progress, SpinnerColumn, TextColumn
from ray_retriever.client import sdk

# python -m ray_retriever.client.cli query 'Where has Alan Turing been born?'

app = typer.Typer()

def progress_spinner():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

@app.command()
def query(query: Annotated[str, typer.Argument()], 
          hostname:Optional[str]=None, 
          port:Optional[int]=None):
    with progress_spinner() as progress:
        progress.add_task(
                description=f"Run query...",
                total=None,
            )
        result = sdk.query(query, hostname, port)        
        print(result.model_dump_json(indent=2))

@app.command()
def health(hostname:Optional[str]=None, 
           port:Optional[int]=None):
    with progress_spinner() as progress:
        progress.add_task(
                description=f"Check health...",
                total=None,
            )
        result = sdk.health(hostname, port)
        if result:
            print('OK')
        else:
            print('FAIL')

if __name__ == "__main__":
    app()