# README

A minimal scalable implementation of a retrieval augmented Q/A assistant based on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) and [Weaviate](https://weaviate.io/).

## Setup

Install from Github:

    pip install git+https://github.com/asprenger/ray_retriever

Install in develop mode:

    git clone https://github.com/asprenger/ray_retriever
    cd ray_retriever
    pip install -e .

## Weaviate

Download Weaviate release for local development:

  1. Check [Releases](https://github.com/weaviate/weaviate/releases) for the latest version
  2. Download binary:
    * MaxOSX: https://github.com/weaviate/weaviate/releases/download/v1.22.5/weaviate-v1.22.5-Darwin-all.zip
    * Linux (Intel/AMD): https://github.com/weaviate/weaviate/releases/download/v1.22.5/weaviate-v1.22.5-Linux-amd64.tar.gz

Start Weaviate:

    AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true PERSISTENCE_DATA_PATH=/tmp/weaviate_index /usr/local/weaviate-v1.22.5/weaviate --host 127.0.0.1 --port 9001 --scheme http

## Ray cluster

Start Ray cluster:

    ray start --head --disable-usage-stats

## Build Index

Build Weaviate index:

    python -m ray_retriever.index.build_wikipedia_index --dataset-size 1000 --num-partitions 4 --weaviate-url http://localhost:9001 --num-gpus 0.25
    
## Retriever Service

Start Retriever using a deployment configuration:

    serve run deploy-configs/retriever_serve.yaml

Use --non-blocking to start application in the background.

Query examples:

    curl --header "Content-Type: application/json" --data '{ "question":"What was Alan Turings middle name?"}' http://127.0.0.1:8000/query
    curl --header "Content-Type: application/json" --data '{ "question":"Where and when has Alan turing been born?" }' http://127.0.0.1:8000/query
    curl --header "Content-Type: application/json" --data '{ "question":"What is the shape of a cube?"}' http://127.0.0.1:8000/query
    curl --header "Content-Type: application/json" --data '{ "question":"How many sides does a cube have?"}' http://127.0.0.1:8000/query

## Run tests

    python -m unittest discover test "*_test.py"

## Swagger

FastAPI is able to generate a Swagger API specification:

    curl http://127.0.0.1:8000/openapi.json

## Embedding Models

Check [MTEB English leaderboard](https://huggingface.co/spaces/mteb/leaderboard)