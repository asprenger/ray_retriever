from typing import List, Dict
from ray import serve
from sentence_transformers import CrossEncoder
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.serve.schema import NodeWithScore

logger = get_logger()

DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH = 512 # TODO

@serve.deployment(name='Reranker')
class Reranker():
    """Rerank context nodes and return the top-n nodes"""

    def __init__(self, model: str, top_n:int, batch_size:int):
        self.top_n = top_n
        self.batch_size = batch_size
        logger.info(f"Loading {model}")
        self._model = CrossEncoder(model, max_length=DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH)
        # Warmup model
        self._model.predict([('query', 'document')])

    def rerank(self, query:str, documents: List[NodeWithScore]) -> List[NodeWithScore]:
        query_and_docs = [(query, doc.node.text) for doc in documents]
        scores = self._model.predict(query_and_docs, batch_size=self.batch_size)
        for node, score in zip(documents, scores):
            node.score = score
        top_n_nodes = sorted(documents, key=lambda x: -x.score if x.score else 0)[: self.top_n]
        return top_n_nodes