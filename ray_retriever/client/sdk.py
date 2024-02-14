from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import requests

HTTP_TIMEOUT = 10
DEFAULT_HOSTNAME='127.0.0.1'
DEFAULT_PORT=8000

class ResponseError(RuntimeError):
    def __init__(self, *args: object, **kwargs) -> None:
        self.response = kwargs.pop("response", None)
        super().__init__(*args)

class SearchResult(BaseModel):
    nodes:List[Dict]

class QueryResult(BaseModel):
    response: str
    trace_url: str

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

def _get_service_backend(hostname:Optional[str]=None, 
                         port:Optional[int]=None):
    url_hostname = hostname if hostname is not None else DEFAULT_HOSTNAME
    url_port = port if port is not None else DEFAULT_PORT
    backend_url = f"http://{url_hostname}:{url_port}"
    access_token = "secret_123" # dummy for now
    bearer = f"Bearer {access_token}" if access_token else ""
    backend_url += "/" if not backend_url.endswith("/") else ""
    return ServiceResource(backend_url, bearer)

def _get_result(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except requests.JSONDecodeError as e:
        raise ResponseError(
            f"Error decoding JSON from {response.url}. Text response: {response.text}",
            response=response,
        ) from e

def query(query:str, 
          hostname:Optional[str]=None, 
          port:Optional[int]=None) -> QueryResult:
    """Call QA service.

    Args:
        query (str): Query string

    Raises:
        ResponseError: Service error or non 200 status code
        requests.exceptions.RequestException: Network error (from requests package)

    Returns:
        QueryResult: Query service response
    """

    backend = _get_service_backend(hostname, port)
    url = backend.backend_url + "query"
    headers = backend.headers
    payload = { 
        "question": query, 
    }
    response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=HTTP_TIMEOUT,
        )
    if response.status_code == 200:
        result = _get_result(response)
        return QueryResult(**result)
    else:
        raise ResponseError(f"Status code: {response.status_code} from: {response.url}. Text response: {response.text}", 
                            response=response)
    
def health(hostname:Optional[str]=None, 
           port:Optional[int]=None) -> bool:
    backend = _get_service_backend(hostname, port)
    url = backend.backend_url + "health"
    headers = backend.headers
    response = requests.get(
            url,
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
    if response.status_code == 200:
        return True
    else:
        raise ResponseError(f"Status code: {response.status_code} from: {response.url}. Text response: {response.text}", 
                            response=response)