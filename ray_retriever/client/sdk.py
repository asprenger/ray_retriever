from typing import List, Dict, Any
from pydantic import BaseModel
import requests

TIMEOUT = 10

class ResponseError(RuntimeError):
    def __init__(self, *args: object, **kwargs) -> None:
        self.response = kwargs.pop("response", None)
        super().__init__(*args)

class SearchResult(BaseModel):
    nodes:List[Dict]

class QueryResult(BaseModel):
    response: str
    response_gen_model: str
    query_time: int
    sources_metadata: List[Dict]=None

class SourceNode(BaseModel):
    id: str
    metadata: Dict
    text: str

class ServiceResource:
    """Stores information about the backend configuration."""

    def __init__(self, backend_url: str, bearer: str):
        self.backend_url = backend_url
        self.bearer = bearer
        self.headers = {"Authorization": self.bearer, "Content-Type": "application/json"}

def _get_service_backend():
    # Hardcode this for now
    backend_url = "http://127.0.0.1:8000"
    access_token = "123"
    bearer = f"Bearer {access_token}" if access_token else ""
    backend_url += "/" if not backend_url.endswith("/") else ""
    return ServiceResource(backend_url, bearer)

def _get_result(response: requests.Response) -> Dict[str, Any]:
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise ResponseError(
            f"Error decoding JSON from {response.url}. Text response: {response.text}",
            response=response,
        ) from e
    return result

def search(query:str) -> SearchResult:
    backend = _get_service_backend()
    url = backend.backend_url + "search"
    headers = backend.headers
    payload = {"question": query}
    response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
    if response.status_code == 200:
        result = _get_result(response)
        return SearchResult(nodes=result)
    else:
        raise ResponseError(f"Status code: {response.status_code} from: {response.url}. Text response: {response.text}", 
                            response=response)

def query(query:str, 
                  return_sources_metadata:bool=False) -> QueryResult:
    """Call QA service.

    Args:
        query (str): Query string

    Raises:
        ResponseError: Service error or non 200 status code
        requests.exceptions.RequestException: Network error (from requests package)

    Returns:
        QueryResult: Query service response
    """

    backend = _get_service_backend()
    url = backend.backend_url + "query"
    headers = backend.headers
    payload = { 
        "question": query, 
        "return_sources_metadata": 1 if return_sources_metadata else 0
    }
    response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
    if response.status_code == 200:
        result = _get_result(response)
        return QueryResult(**result)
    else:
        raise ResponseError(f"Status code: {response.status_code} from: {response.url}. Text response: {response.text}", 
                            response=response)
    

def get_source_node(node_id:str) -> SourceNode:
    backend = _get_service_backend()
    url = backend.backend_url + "source_node/" + node_id
    headers = backend.headers
    response = requests.get(
            url,
            headers=headers,
            timeout=TIMEOUT,
        )
    if response.status_code == 200:
        result = _get_result(response)
        return SourceNode(**result)
    else:
        raise ResponseError(f"Status code: {response.status_code} from: {response.url}. Text response: {response.text}", 
                            response=response)
