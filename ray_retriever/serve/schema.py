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

class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class RetrieverResponse(BaseModel):
    response: str
    finish_reason: str
    model: str
    usage: TokenUsage

class DocumentEmbedding(BaseModel):
    embedding: List[float]

@dataclass
class VectorStoreQueryResult:
    """Vector store query result."""
    nodes: Optional[Sequence[TextNode]] = None
    similarities: Optional[List[float]] = None
