from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import json
from ray_retriever.serve.schema import TextNode

def get_node_similarity(entry: Dict, similarity_key: str = "distance") -> float:
    """Get converted node similarity from distance."""
    distance = entry["_additional"].get(similarity_key, 0.0)

    if distance is None:
        return 1.0

    # convert distance https://forum.weaviate.io/t/distance-vs-certainty-scores/258
    return 1.0 - float(distance)

def parse_get_response(response: Dict) -> Dict:
    """Parse get response from Weaviate."""
    if "errors" in response:
        raise ValueError("Invalid query, got errors: {}".format(response["errors"]))
    data_response = response["data"]
    if "Get" not in data_response:
        raise ValueError("Invalid query response, must be a Get query.")

    return data_response["Get"]

def to_node(entry:Dict, index_name:str) -> TextNode:
    print(entry)
    id = entry['_additional']['id']
    if 'vector' in entry['_additional']:
        embedding = entry['_additional']['vector']
    else:
        embedding = None
    node_content = json.loads(entry['_node_content'])
    metadata = node_content['metadata']
    text = entry['text']
    return TextNode(id=id, index_name=index_name, metadata=metadata, 
                    text=text, embedding=embedding)