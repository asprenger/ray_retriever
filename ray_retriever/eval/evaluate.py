from typing import Any, Dict, List, Optional
import argparse
import yaml
from pydantic import BaseModel
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_utilization
)
from ragas import evaluate
from ragas.evaluation import Result
from ray_retriever.client import sdk

# python -m ray_retriever.eval.evaluate

class EvalSample(BaseModel):
    query:str
    answer:Optional[str] = None

def main(samples_path:str, dataset_save_path:Optional[str]):

    questions = []
    contexts = []
    responses = []
    # ground_truths: list[list[str]] - The ground truth answer to the questions. (only required if you are using context_recall)

    with open(samples_path, 'r') as stream:
        queries = yaml.safe_load(stream)

    for sample in queries['samples']:

        eval_sample = EvalSample.model_validate(sample)
        print(f"Query: {eval_sample.query}")

        # Retrieve a query response
        query_result = sdk.query(eval_sample.query, return_context_nodes=True)
        print(f"Response: {query_result.response}")
        print('Context nodes:')
        for context_node in query_result.context_nodes:
            print(f"{context_node['index_name']} - {context_node['node_id']}")

        # Lookup context that was used to answer the query
        context_parts = [sdk.get_text_node(context_node['node_id']).text 
                         for context_node in query_result.context_nodes]

        questions.append(eval_sample.query)
        contexts.append(context_parts)
        responses.append(query_result.response)

    ds = Dataset.from_dict({
        "question": questions, 
        "contexts": contexts,
        "answer": responses
        })

    if dataset_save_path is not None:
        ds.save_to_disk(dataset_save_path)

    # 'context_precision' requires columns ['ground_truth']
    # 'context_recall' requires columns ['ground_truth']

    metrics=[
        faithfulness, 
        answer_relevancy,
        context_utilization,
    ]

    eval_result = evaluate(ds, metrics=metrics)
    print(eval_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ray Retriever Chatbot')
    parser.add_argument('--samples-path', type=str, default='datasets/samples.yaml',
                        help="Path to store the collected dataset")
    parser.add_argument('--dataset-save-path', type=str, default='/tmp/eval_dataset',
                        help="Path to store the collected dataset")
    args = parser.parse_args()
    main(samples_path=args.samples_path, dataset_save_path=args.dataset_save_path)