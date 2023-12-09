from typing import List, Dict
from json.decoder import JSONDecodeError
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle, DeploymentResponse
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import FastAPI, Request, HTTPException
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL
from ray_retriever.serve.schema import RetrieverResponse
from ray_retriever.serve.search_engine import SearchEngine
from ray_retriever.serve.reranker import Reranker
from ray_retriever.serve.embedding_generator import EmbeddingGenerator
from ray_retriever.serve.response_generator import ResponseGenerator

"""
serve run deploy-configs/retriever_serve.yaml

curl --header "Content-Type: application/json" --data '{ "query":"What was Alan Turings middle name?"}' http://127.0.0.1:8000/query
"""

logger = get_logger()
app = FastAPI()

@serve.deployment(name='Retriever')
@serve.ingress(app)
class Retriever():
    """This class chains together the different stages of the RAG pipeline."""
    
    def __init__(self, embedding_generator, search_engine, reranker, response_generator):
        self._embedding_generator: DeploymentHandle = embedding_generator.options(
            use_new_handle_api=True,
        )
        self._search_engine: DeploymentHandle = search_engine.options(
            use_new_handle_api=True,
        )
        self._reranker: DeploymentHandle = reranker.options(
            use_new_handle_api=True,
        )
        self._response_generator: DeploymentHandle = response_generator.options(
            use_new_handle_api=True,
        )
    
    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    @app.post("/query")
    async def query(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            if 'question' not in payload:
                return JSONResponse(status_code=400, content='Missing parameter: "question"')
            query = payload['question'] 

            # Define the RAG pipeline. The response of each step in passed directly into the next step. 
            # We don’t need to await any of the intermediate responses, Ray Serve manages the await behavior 
            # under the hood.

            embedding_response: DeploymentResponse = self._embedding_generator.calculate_embedding.remote(query)
            search_response: DeploymentResponse = self._search_engine.search.remote(embedding_response)
            rerank_response: DeploymentResponse = self._reranker.rerank.remote(query, search_response)
            generated_response: DeploymentResponse = self._response_generator.generate_response.remote(query, rerank_response)

            # Wait for the respone of the chain
            response = await generated_response
            return JSONResponse(content={'response':response.response})
        
        except JSONDecodeError as e:
            return JSONResponse(status_code=400, content='Error parsing JSON request')
        except Exception as e:
            logger.error('Error in query()', exc_info=1)
            return JSONResponse(status_code=500, content='Internal error')

def deployment(args: Dict[str, str]) -> Application:

    embedding_model = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size = args.get('embedding_batch_size', 32)
    embedding_batch_wait_timeout_s = args.get('embedding_batch_wait_timeout_s', 0.1)

    weaviate_hostname = args.get('weaviate_hostname', "127.0.0.1")
    weaviate_port = args.get('weaviate_port', 9001)
    index_name = args.get('index_name', "Wikipedia") 
    similarity_top_n = args.get('similarity_top_n', 5)

    rerank_model = DEFAULT_RERANK_MODEL
    rerank_batch_size = args.get('rerank_batch_size', 32)
    rerank_top_n = args.get('rerank_top_n', 3)

    response_model = args.get('response_model')
    anyscale_endpoint_key = args.get('anyscale_endpoint_key', None)
    openai_api_key = args.get('openai_api_key', None)

    embedding_generator = EmbeddingGenerator.bind(model=embedding_model, batch_size=embedding_batch_size, 
                                                  batch_wait_timeout_s=embedding_batch_wait_timeout_s)
    search_engine = SearchEngine.bind(weaviate_hostname=weaviate_hostname, weaviate_port=weaviate_port, 
                                      index_name=index_name, similarity_top_n=similarity_top_n)
    reranker = Reranker.bind(model=rerank_model, top_n=rerank_top_n, batch_size=rerank_batch_size)
    response_generator = ResponseGenerator.bind(model_id=response_model, anyscale_endpoint_key=anyscale_endpoint_key, 
                                                openai_api_key=openai_api_key)

    return Retriever.bind(embedding_generator=embedding_generator, search_engine=search_engine,
                          reranker=reranker, response_generator=response_generator)
