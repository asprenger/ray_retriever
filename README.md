# README

A scalable implementation of a retrieval augmented QA assistant based on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) and [Weaviate](https://weaviate.io/).

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
    * MaxOSX: `wget https://github.com/weaviate/weaviate/releases/download/v1.22.5/weaviate-v1.22.5-Darwin-all.zip`
    * Linux (Intel/AMD): `wget https://github.com/weaviate/weaviate/releases/download/v1.22.5/weaviate-v1.22.5-Linux-amd64.tar.gz`

Start Weaviate:

    CLUSTER_HOSTNAME=weaviate_node_1 ENABLE_MODULES='backup-filesystem' BACKUP_FILESYSTEM_PATH='/tmp/backups' AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true PERSISTENCE_DATA_PATH=/tmp/weaviate_index /usr/local/weaviate-v1.22.5/weaviate --host 127.0.0.1 --port 9001 --scheme http

## Ray cluster

Start Ray cluster:

    ray start --head --disable-usage-stats

## Build Index

Build Weaviate index:

    RAY_DEDUP_LOGS=0 python -m ray_retriever.index.build_wikipedia_index --dataset-size 1000 --num-partitions 4 --weaviate-url http://localhost:9001 --num-gpus 0.25

## Restore index backup

Instead of building the index you can also restore an existing backup of the index.

Download backup:

    huggingface-cli download asprenger/wikipedia-20220301.simple-weaviate-backup --local-dir /tmp/backups/backup001 --local-dir-use-symlinks False

Start restore backup:

    curl \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "id": "backup001"
        }' \
    http://localhost:9001/v1/backups/filesystem/backup001/restore

The restoration happens in the background. The following line in the logfile indicates the backup has been started:

    {"action":"try_restore","backend":"filesystem","backup_id":"backup001","level":"info","msg":"","time":"2023-12-27T14:46:23Z","took":3461703}

The following log messages indicates a successful backup restoration:

    {"action":"restore","backup_id":"backup001","class":"Wikipedia","level":"info","msg":"successfully restored","time":"2023-12-27T14:46:52Z"}

## Retriever Service

The config file `deploy-configs/retriever_serve.yaml` must to be customized. The following values must be 
set:

 * OPENAI_API_KEY and/or ANYSCALE_ENDPOINT_KEY
 * response_prompt_name

Depending on the used hardware the number of replicas for the different tasks and GPU resources for the 
`EmbeddingGenerator` and `Reranker` should be adjusted.

Start Retriever:

    serve run deploy-configs/retriever_serve.yaml

Use --non-blocking to start application in the background.

Example queries and output for `meta-llama/Llama-2-70b-chat-hf`:

    curl --header "Content-Type: application/json" --data '{ "question":"What is the capital of France and when did it become the capital?"}' http://127.0.0.1:8000/query

    curl --header "Content-Type: application/json" --data '{ "question":"What was Alan Turings middle name?"}' http://127.0.0.1:8000/query
    {"response":"  Mathison"}

    curl --header "Content-Type: application/json" --data '{ "question":"Where and when has Alan turing been born?" }' http://127.0.0.1:8000/query
    {"response":"  Alan Turing was born in Maida Vale, London on June 23, 1912."}

    curl --header "Content-Type: application/json" --data '{ "question":"What is the shape of a cube?"}' http://127.0.0.1:8000/query
    {"response":"  A cube is a type of polyhedron with all right angles and whose height, width, and depth are all the same. It is a type of rectangular prism, which is itself a type of hexahedron."}

    curl --header "Content-Type: application/json" --data '{ "question":"How many sides does a cube have?"}' http://127.0.0.1:8000/query
    {"response":" 6"}

    curl --header "Content-Type: application/json" --data '{ "question":"How long is the term of a US supreme court judge?"}' http://127.0.0.1:8000/query
    {"response":"The term of a US Supreme Court judge is for life."}

    curl --header "Content-Type: application/json" --data '{ "question":"What is the capital of France?"}' http://127.0.0.1:8000/query
    {"response":"  Paris"}

    curl --header "Content-Type: application/json" --data '{ "question":"What is a Triceratops?" }' http://127.0.0.1:8000/query
    {"response":"  A Triceratops is a huge herbivorous ceratopsid dinosaur from the late Cretaceous period, characterized by its three horns on its head, bony beak, and a large frill on its neck."}

If the Retriever can not answer the question it should respond with "I do not know".

    curl --header "Content-Type: application/json" --data '{ "question":"What is the current temperature in Paris?"}' http://127.0.0.1:8000/query
    {"response":"I do not know."}

The current Retriever can only handle questions that can be directly answered from the retrieved context. It can not answer
complex questions that require multiple steps and planning.

## Run tests

    python -m unittest discover test "*_test.py"

## Swagger

FastAPI is able to generate a Swagger API specification:

    curl http://127.0.0.1:8000/openapi.json

## Embedding Models

Check [MTEB English leaderboard](https://huggingface.co/spaces/mteb/leaderboard)