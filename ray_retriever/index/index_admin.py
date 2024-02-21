from typing import Any, Dict, Iterator, List, Optional, Union, Annotated
import typer
from typing_extensions import Annotated
import json
import requests
from rich import print as rp
from rich.progress import Progress, SpinnerColumn, TextColumn
import weaviate

# python -m ray_retriever.index.index_admin

WEAVIATE_HOSTNAME = '127.0.0.1'
WEAVIATE_PORT = 9001

app = typer.Typer()

def progress_spinner():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )

@app.command()
def ping(hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
         port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Ping Weaviate."""

    weaviate_url = f"http://{hostname}:{port}/"
    client = weaviate.Client(url=weaviate_url)
    if client.is_ready():
        print('OK')
    else:
        print('FAIL')


@app.command()
def schema(index: Annotated[str, typer.Argument(help="Index name")],
           hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
           port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Show index schema."""

    weaviate_url = f"http://{hostname}:{port}/"
    client = weaviate.Client(url=weaviate_url)
    schema = client.schema.get(index)
    print(json.dumps(schema, indent=2))

    
@app.command()
def meta(hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
         port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Show Weaviate metadata."""

    weaviate_url = f"http://{hostname}:{port}/"
    client = weaviate.Client(url=weaviate_url)
    print(json.dumps(client.get_meta(), indent=2))


@app.command()
def delete(index: Annotated[str, typer.Argument(help="Index name")],
           hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
           port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Delete index."""

    url = f"http://{hostname}:{port}/"
    client = weaviate.Client(url=url)
    client.schema.delete_class(index)


@app.command()
def backup(index: Annotated[str, typer.Argument(help="Index name")],
           backup_id: Annotated[str, typer.Argument(help="Backup ID")],
           hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
           port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Trigger index backup."""

    url = f"http://{hostname}:{port}/v1/backups/filesystem"
    payload = { "id": backup_id, "include": [index]}
    response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=5,
        )
    if response.status_code == 200:
        print('OK')
    else:
        print('FAIL')


@app.command()
def restore(backup_id: Annotated[str, typer.Argument(help="Backup ID")],
            hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
            port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Trigger backup restore."""

    url = f"http://{hostname}:{port}/v1/backups/filesystem/{backup_id}/restore"
    payload = {"id": backup_id}
    response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=5,
        )
    if response.status_code == 200:
        print('OK')
    else:
        print('FAIL')


def get_batch_with_cursor(client:weaviate.Client, 
                          collection_name:str, 
                          batch_size:int, 
                          return_embedding:bool, 
                          return_text:bool,
                          cursor=None):

    properties = ["title", "uri", "source"]
    if return_text:
        properties.append('text')

    additional_properties = ['id']
    if return_embedding:
        additional_properties.append('vector')    

    # Prepare the query to run through data
    query = (
        client.query.get(collection_name, properties)
            .with_additional(additional_properties)
            .with_limit(batch_size)
    )

    # Fetch the next set of results
    if cursor is not None:
        result = query.with_after(cursor).do()
    # Fetch the first set of results
    else:
        result = query.do()

    return result["data"]["Get"][collection_name]

@app.command()
def export(index: Annotated[str, typer.Argument(help="Index name")],
           batch_size: Annotated[int, typer.Option(help="Batch size")] = 100,
           return_embedding: Annotated[bool, typer.Option(help="Returns embedding vectors if set to True")] = False,
           return_text: Annotated[bool, typer.Option(help="Returns text properties if set to True")] = True,
           hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
           port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Export the content of an index"""

    weaviate_url = f"http://{hostname}:{port}/"
    client = weaviate.Client(url=weaviate_url)
    
    cursor = None
    while True:
        # Get the next batch of objects
        next_batch = get_batch_with_cursor(client, index, batch_size, return_embedding, 
                                           return_text, cursor)

        # Break the loop if empty – we are done
        if len(next_batch) == 0:
            break

        # Print the data in JSONL format
        for sample in next_batch:
            print(json.dumps(sample))

        # Move the cursor to the last returned uuid
        cursor=next_batch[-1]["_additional"]["id"]

if __name__ == "__main__":
    app()