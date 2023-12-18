from typing import List, Dict, Optional, Sequence
from pydantic import BaseModel
from dataclasses import dataclass

class TextNode(BaseModel):
    id: str
    metadata: Dict
    text: str
    embedding: Optional[List[float]]

class NodeWithScore(BaseModel):
    node: TextNode
    score: Optional[float] = None

class RetrieverResponse(BaseModel):
    response: str

class DocumentEmbedding(BaseModel):
    embedding: List[float]

@dataclass
class VectorStoreQueryResult:
    """Vector store query result."""
    nodes: Optional[Sequence[TextNode]] = None
    similarities: Optional[List[float]] = None
