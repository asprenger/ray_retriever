from typing import Any, List, Dict, Union, Optional, Sequence
import argparse
import logging
import weaviate
import openai
import ray
from llama_index.schema import Document, BaseNode
from llama_index.node_parser.interface import NodeParser
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from ray_retriever.constants import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger("ray")

# RAY_DEDUP_LOGS=0 python -m ray_retriever.index.index_wikipedia --dataset-path /tmp/wikipedia_embeddings --index-name WikiTest

class NopNodeParser(NodeParser):
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        return nodes

class IndexDataset:

    def __init__(self, weaviate_url:str, weaviate_index_name:str):

        self.weaviate_url = weaviate_url
        self.weaviate_index_name = weaviate_index_name

        # The LLM and embedding model components in the service context have OpenAI
        # default implementations that validate the existance of the API key. We do
        # not need these components but the service context will initialize them. The
        # fake key also raises an error in case OpenAI is called unexpectedly.
        # TODO get rid of this hack
        openai.api_key = "sk-000000000000000000000000000000000000000000000000"

        client = weaviate.Client(url=weaviate_url)
        vector_store = WeaviateVectorStore(weaviate_client=client, 
                                            index_name=weaviate_index_name)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build a service context. Configure a NOP NodeParser because the
        # documents are already split up into chunks.
        self.service_context = ServiceContext.from_defaults(node_parser=NopNodeParser())

    def __call__(self, batch):

        logger.info(f'Write {len(batch["text"])} records to index "{self.weaviate_index_name}"')

        documents = [Document(text=text, embedding=embedding.tolist(), metadata=metadata) 
                     for text, embedding, metadata in zip(batch['text'], batch['embedding'], batch['metadata'])]

        VectorStoreIndex.from_documents(documents, 
                                        storage_context=self.storage_context, 
                                        service_context=self.service_context)
        
        #return batch
        return {'results': []}


def main(weaviate_url:str, 
         index_name:str, 
         no_index_delete:bool, 
         dataset_path:str, 
         indexing_workers:int, 
         batch_size:int):

    # Delete the existing index (if it exists). This must be performed 
    # before submitting any remote tasks.
    if not no_index_delete:
        logger.info(f'Delete index: {index_name}')
        client = weaviate.Client(url=weaviate_url)
        client.schema.delete_class(index_name)
        del client

    # Connect to Ray cluster
    ray.init()

    # Read dataset
    ds = ray.data.read_parquet(dataset_path)
    logger.info(f"Dataset schema: {ds.schema()}")

    # Index data
    result = ds.map_batches(
        IndexDataset,
        fn_constructor_kwargs={
            "weaviate_url":weaviate_url, 
            "weaviate_index_name": index_name
        },
        batch_size=batch_size,
        concurrency=indexing_workers
    )

    # Trigger job execution
    result.take()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a Wikipedia index')
    parser.add_argument('--weaviate-url', type=str, default='http://127.0.0.1:9001/', help='Weaviate URL')
    parser.add_argument('--index-name', type=str, default='Wikipedia', help='Weaviate index name')
    parser.add_argument('--dataset-path', type=str, default='/tmp/wikipedia_embeddings', help='Input dataset path')
    parser.add_argument('--no-index-delete', action='store_true', help='Do not delete an existing index')
    parser.add_argument('--indexing-workers', type=int, default=1, help='Number of indexing workers')
    parser.add_argument('--index-map-batch-size', type=int, default=512, help='Batch size for the indexing worker map operation')
    args = parser.parse_args()
    main(weaviate_url=args.weaviate_url, index_name=args.index_name, 
         dataset_path=args.dataset_path, no_index_delete=args.no_index_delete,
         indexing_workers=args.indexing_workers, batch_size=args.index_map_batch_size)