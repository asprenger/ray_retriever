import os
from pathlib import Path
import shutil
import argparse
import logging
import tiktoken
import json
import ray
from datasets import load_dataset, VerificationMode
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ray_retriever.constants import DEFAULT_EMBEDDING_MODEL
from ray_retriever.utils.common_utils import current_time_millis

TIKTOKEN_MODEL = "gpt-3.5-turbo"
DATASET_ID = 'wikipedia'

logger = logging.getLogger("ray")

# RAY_DEDUP_LOGS=0 python -m ray_retriever.index.calculate_wikipedia_embeddings --dataset-size 50

class ChunkDocument:
    def __init__(self, chunk_size:int, chunk_overlap:int, dataset_source:str, min_doc_size:int):
        self.tokenizer = tiktoken.encoding_for_model(TIKTOKEN_MODEL)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dataset_source = dataset_source
        self.min_doc_size = min_doc_size

    def __call__(self, doc):

        def tokenizer_len_fn(text):
            return len(self.tokenizer.encode(text))

        if tokenizer_len_fn(doc['text']) < self.min_doc_size:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=tokenizer_len_fn
        )

        chunks = text_splitter.split_text(doc["text"])

        metadata = {
            "source": self.dataset_source,
            "uri": doc["url"],
            "title": doc["title"]
        }

        return [{"text": chunk, 'metadata': metadata.copy()|{"size": tokenizer_len_fn(chunk)}} 
                for chunk in chunks]

class CalculateEmbeddings:

    def __init__(self, embedding_model_id:str, device:str, batch_size:int):
        logger.info(f"Load embedding model {embedding_model_id}")
        self.embed_model = SentenceTransformer(embedding_model_id, device=device)
        self.batch_size = batch_size

    def __call__(self, batch):

        logger.info(f"Embedding batch size: {len(batch['text'])}")

        # Normalize embeddings so we can use a dot product to calculate cosine distance
        start_time = current_time_millis()
        embeddings = self.embed_model.encode(batch['text'], 
                                             batch_size=self.batch_size,
                                             normalize_embeddings=True)
        duration_s = (current_time_millis() - start_time) / 1000

        logger.info(f"duration: {duration_s} s, {len(batch['text']) / duration_s} samples/s")

        return {
            'text': batch['text'], 
            'embedding': embeddings.tolist(), 
            'metadata': batch['metadata']
        }

def main(dataset_size:int, split_doc_workers:int, embedding_workers:int, embedding_model_batch_size:int,
         embedding_map_batch_size:int, output_path:str, dataset_subset_name:str, chunk_size:int, 
         chunk_overlap:int, min_doc_size:int, num_gpus:float, num_output_partitions:int):

    if num_gpus > 0:
        device = 'cuda'
    else:
        num_gpus = 0
        device = 'cpu'

    shutil.rmtree(output_path, ignore_errors=True)

    # Load dataset into memory
    # dataset: List[{id: string, url: string, title: string, text: string}]
    split = 'train' if dataset_size is None else f'train[:{dataset_size}]'
    dataset = load_dataset(DATASET_ID, dataset_subset_name, 
                            split=split, 
                            verification_mode=VerificationMode.NO_CHECKS)

    # Connect to Ray cluster
    ray.init()

    # Create a Ray Dataset
    ds = ray.data.from_items([sample for sample in dataset]) # TODO parallelism=???
    logger.info(f"Dataset: {ds}")

    # Split documents into chunks
    dataset_source = f"{DATASET_ID}-{dataset_subset_name}"
    chunks_ds = ds.flat_map(ChunkDocument, 
                            concurrency=split_doc_workers,
                            fn_constructor_kwargs={
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "dataset_source": dataset_source,
                                "min_doc_size": min_doc_size
                            })
    num_chunks = chunks_ds.count()
    logger.info(f"Number of chunks: {num_chunks}")

    # Calculate embeddings
    embedding_ds = chunks_ds.map_batches(
        CalculateEmbeddings,
        fn_constructor_kwargs={
            "embedding_model_id":DEFAULT_EMBEDDING_MODEL, 
            "device": device, 
            "batch_size": embedding_model_batch_size
        },
        batch_size=embedding_map_batch_size, 
        num_gpus=num_gpus,
        concurrency=embedding_workers
    )

    # Write dataset to disk
    embedding_ds = embedding_ds.repartition(num_output_partitions)
    embedding_ds.write_parquet(output_path)

    # Write metadata files
    metadata = {
        "dataset": DATASET_ID,
        "dataset_subset": dataset_subset_name,
        "dataset_size": dataset_size,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_doc_size": min_doc_size,
        "num_chunks": num_chunks
    }
    with open(os.path.join(output_path, '_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    Path(os.path.join(output_path, '_SUCCESS')).touch()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate embeddings for a Wikipedia dataset')
    parser.add_argument('--subset-name', type=str, default="20220301.simple", help='Wikipedia subset name. (See: https://huggingface.co/datasets/wikipedia)')
    parser.add_argument('--dataset-size', type=int, help='Specify a size to reduce the size of the dataset')
    parser.add_argument('--output-path', type=str, default='/tmp/wikipedia_embeddings', help='Output directory')
    parser.add_argument('--num-output-partitions', type=int, default=10, help='Number of partitions in the output directory')
    parser.add_argument('--split-doc-workers', type=int, default=4, help='Number of workers for splitting documents')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size in number of tokens')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Chunk overlap in number of tokens')
    parser.add_argument('--min-doc-size', type=int, default=50, help='Minimum document size in number of tokens')
    parser.add_argument('--embedding-workers', type=int, default=4, help='Number of workers for embedding calculation')
    parser.add_argument('--embedding-map-batch-size', type=int, default=512, help='Batch size for the embedding worker map operation')
    parser.add_argument('--embedding-model-batch-size', type=int, default=32, help='Batch size for embedding calculation')
    parser.add_argument('--num-gpus', type=float, default=0, help='Fraction of GPUs allocated for embedding calculation for each worker')

    args = parser.parse_args()
    main(dataset_subset_name=args.subset_name, dataset_size=args.dataset_size, output_path=args.output_path,
         split_doc_workers=args.split_doc_workers, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, 
         min_doc_size=args.min_doc_size, embedding_workers=args.embedding_workers, num_gpus=args.num_gpus,
         embedding_model_batch_size=args.embedding_model_batch_size, embedding_map_batch_size=args.embedding_map_batch_size,
         num_output_partitions=args.num_output_partitions)