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

def create_client(hostname:str, port:int):
    url = f"http://{hostname}:{port}/"
    return weaviate.Client(url=url)

@app.command()
def cluster(hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
         port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Show Weaviate cluster information."""

    client = create_client(hostname, port)    
    nodes_status = client.cluster.get_nodes_status()
    print(json.dumps(nodes_status, indent=2))


@app.command()
def ping(hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
         port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Ping Weaviate."""

    client = create_client(hostname, port)
    if client.is_ready():
        print('OK')
    else:
        print('FAIL')

@app.command()
def schema(index: Annotated[str, typer.Argument(help="Index name")],
           hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
           port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Show index schema."""

    client = create_client(hostname, port)
    schema = client.schema.get(index)
    print(json.dumps(schema, indent=2))


@app.command()
def node(hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
         port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Show Weaviate node information."""

    client = create_client(hostname, port)
    print(json.dumps(client.get_meta(), indent=2))


@app.command()
def delete(index: Annotated[str, typer.Argument(help="Index name")],
           hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
           port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Delete an index."""

    client = create_client(hostname, port)
    client.schema.delete_class(index)


@app.command()
def backup(index: Annotated[str, typer.Argument(help="Index name")],
           backup_id: Annotated[str, typer.Argument(help="Backup ID")],
           wait_for_completion: Annotated[bool, typer.Option(help="Whether to wait until the backup is done.")] = False,
           hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
           port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Trigger index backup."""

    client = create_client(hostname, port)
    client.backup.create(backup_id=backup_id, 
                         backend='filesystem', 
                         include_classes=[index],
                         wait_for_completion=wait_for_completion)


@app.command()
def restore(backup_id: Annotated[str, typer.Argument(help="Backup ID")],
            wait_for_completion: Annotated[bool, typer.Option(help="Whether to wait until the backup is done.")] = False,
            hostname: Annotated[str, typer.Option(help="Weaviate hostname")] = WEAVIATE_HOSTNAME, 
            port: Annotated[str, typer.Option(help="Weaviate port")] = WEAVIATE_PORT):
    """Trigger backup restore."""

    client = create_client(hostname, port)
    client.backup.restore(backup_id=backup_id,
                          backend='filesystem', 
                          wait_for_completion=wait_for_completion)


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
    """Export the content of an index."""

    client = create_client(hostname, port)
    
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