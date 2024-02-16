from typing import List, Dict
from json.decoder import JSONDecodeError
from ray import serve
from ray.serve import Application
from ray.serve.handle import DeploymentHandle, DeploymentResponse
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import FastAPI, Request
from langfuse import Langfuse
from ray_retriever.utils.logging_utils import get_logger
from ray_retriever.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANK_MODEL
from ray_retriever.serve.schema import RetrieverResponse, TextNode
from ray_retriever.serve.search_engine import SearchEngine
from ray_retriever.serve.reranker import Reranker
from ray_retriever.serve.embedding_generator import EmbeddingGenerator
from ray_retriever.serve.response_generator import ResponseGenerator
from ray_retriever import __version__

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
        self.langfuse = Langfuse()
    
    @app.get("/health")
    async def health(self) -> Response:
        """Health check."""
        return Response(status_code=200)

    @app.get("/node/{node_id}")
    async def get_node(self, node_id:str) -> Dict:
        try:
            node = await self._search_engine.get_text_node.remote(node_id)
            if node is not None:
                return JSONResponse(content={
                    "id": node.id, 
                    "index_name": node.index_name, 
                    "metadata": node.metadata, 
                    "text": node.text})
            else:
                return Response(status_code=404, content=f'Text node "{node_id}" not found')
        except Exception as e:
            logger.error('Error in get_node()', exc_info=1)
            return JSONResponse(status_code=500, content='Internal error')

    @app.post("/query")
    async def query(self, request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            if 'question' not in payload:
                return JSONResponse(status_code=400, content='Missing parameter: "question"')
            query = payload['question'] 
            return_context_nodes = 'return_context_nodes' in payload and payload['return_context_nodes'] == True

            trace = self.langfuse.trace(
                name = "ray-retriever",
                metadata = {
                    "ray-retriever-version": __version__,
                },
                tags = ["production"], # TODO
                input = query
            )
            logger.info(f"Trace URL: {trace.get_trace_url()}")

            # Define the RAG pipeline. The response of each step in passed directly into the next step. 
            # We don’t need to await any of the intermediate responses, Ray Serve manages the await behavior 
            # under the hood.

            embedding_response: DeploymentResponse = self._embedding_generator.calculate_embedding.remote(query, trace.id)
            search_response: DeploymentResponse = self._search_engine.search.remote(embedding_response, trace.id)
            rerank_response: DeploymentResponse = self._reranker.rerank.remote(query, search_response, trace.id)
            generated_response: DeploymentResponse = self._response_generator.generate_response.remote(query, rerank_response, trace.id)

            # Wait for the final respone
            response:RetrieverResponse = await generated_response

            # Update Trace with output
            self.langfuse.trace(id=trace.id, output=response.response)

            response_content = {
                'response': response.response, 
                'trace_url': trace.get_trace_url()
            }
            if return_context_nodes:
                response_content['context_nodes'] = response.context_node_info

            return JSONResponse(content=response_content)
        
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

    response_prompt_name = args.get('response_prompt_name')

    embedding_generator = EmbeddingGenerator.bind(model=embedding_model, batch_size=embedding_batch_size, 
                                                  batch_wait_timeout_s=embedding_batch_wait_timeout_s)
    search_engine = SearchEngine.bind(weaviate_hostname=weaviate_hostname, weaviate_port=weaviate_port, 
                                      index_name=index_name, similarity_top_n=similarity_top_n)
    reranker = Reranker.bind(model=rerank_model, top_n=rerank_top_n, batch_size=rerank_batch_size)
    response_generator = ResponseGenerator.bind(prompt_name=response_prompt_name)

    return Retriever.bind(embedding_generator=embedding_generator, search_engine=search_engine,
                          reranker=reranker, response_generator=response_generator)
