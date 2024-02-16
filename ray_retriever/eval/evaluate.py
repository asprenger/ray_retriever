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
from ray_retriever.client import sdk

# python -m ray_retriever.eval.evaluate

class EvalSample(BaseModel):
    query:str

def main():

    with open('datasets/samples.yaml', 'r') as stream:
        queries = yaml.safe_load(stream)

    questions = []
    contexts = []
    responses = []

    # ground_truths: list[list[str]] - The ground truth answer to the questions. (only required if you are using context_recall)

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
    print(ds)

    ds.save_to_disk('/tmp/eval_dataset')

    result = evaluate(
        ds,
        metrics=[
            #context_precision,
            faithfulness,
            answer_relevancy,
            context_utilization,
            #context_recall,
        ],
    )

    print(result)

if __name__ == "__main__":
    main()    