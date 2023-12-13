from typing import Any, List, Optional, Sequence, Dict, Tuple
import json
import aiohttp
import weaviate
from ray_retriever.serve.schema import VectorStoreQueryResult, TextNode
from ray_retriever.serve.weaviate_utils import parse_get_response, get_node_similarity, to_node

class AsyncWeaviateClient():
    """
        Filter example:

            filter = {
                "path": "title",
                "operator": "Equal",
                "valueText": "Alan Turing"
            }

        Composit filter example:

            operands = [
                {"path": "title", "operator": "Equal", "valueText": "Alan Turing"},
                {"path": "title", "operator": "Equal", "valueText": "Addition"}
            ]
            filter = {"operands": operands, "operator": "Or"}

        Filter for Node IDs:

            operands = [
                {"path": ["id"], "operator": "Equal", "valueText": "09a3ac27-50ce-4724-9d06-2f6b37e55023"},
                {"path": ["id"], "operator": "Equal", "valueText": "6c1de1bf-d86d-4857-951b-966d9c5f1915"}
            ]
            filter = {"operands": operands, "operator": "Or"}

        Filter operands:

            * And
            * ContainsAll
            * ContainsAny
            * Equal
            * GreaterThan
            * GreaterThanEqual
            * IsNull
            * LessThan
            * LessThanEqual
            * Like
            * NotEqual
            * Or
            * WithinGeoRange        

    """

    def __init__(self, host:str, port:int):
        self.base_url = f"http://{host}:{port}/"
        # We need a weaviate.Client to access the query builder. There is
        # no easy way to initialize the builder on its own. The Client is
        # not used to access Weaviate. Note that the Client opens a 
        # connection pool to the server!
        self.client = weaviate.Client(url=self.base_url)
        self.http_session = aiohttp.ClientSession()

    async def get_schema(self, index_name:str) -> Dict:
        
        async with self.http_session.get(
            self.base_url + "v1/schema", 
            headers={'content-type': 'application/json'}) as response:
            schema = await response.json()

        classes = schema["classes"]
        classes_by_name = {c["class"]: c for c in classes}
        if index_name not in classes_by_name:
            raise ValueError(f"{index_name} schema does not exist.")
        schema = classes_by_name[index_name]
        return schema

    async def get_all_properties(self, index_name:str) -> List[str]:
        schema = await self.get_schema(index_name)
        return [p["name"] for p in schema["properties"]]

    async def find_similar_nodes(self, index_name:str, 
                                 embedding:List[float], 
                                 similarity_top_k:int,
                                 return_embeddings:bool=False,
                                 filter:Optional[Dict]=None) -> VectorStoreQueryResult:
        
        # Read all index properties from scheme
        all_properties = await self.get_all_properties(index_name)

        query_builder = self.client.query.get(index_name, all_properties)
        additional_properties = ["id", "distance", "score"]
        if return_embeddings:
            additional_properties.append("vector")
        query_builder = query_builder.with_additional(additional_properties)
        query_builder = query_builder.with_near_vector({"vector": embedding})
        query_builder = query_builder.with_limit(similarity_top_k)
        if filter:
            query_builder.with_where(filter)
        
        async with self.http_session.post(
            self.base_url + "v1/graphql", 
            headers={'content-type': 'application/json'}, 
            data=json.dumps({"query": query_builder.build()})) as response:
            query_result = await response.json()

        parsed_result = parse_get_response(query_result)
        entries = parsed_result[index_name]

        similarity_key = "distance"
        similarities = [get_node_similarity(entry, similarity_key) for entry in entries]

        nodes = [to_node(entry) for entry in entries]

        nodes = nodes[: similarity_top_k]
        node_idxs = [str(i) for i in range(len(nodes))]

        return VectorStoreQueryResult(nodes=nodes, ids=node_idxs, similarities=similarities)

    async def get_all_nodes(self, index_name:str, 
                            filter:Optional[Dict]=None,
                            return_embeddings:bool=False, 
                            max_result_size:int=100) -> List[TextNode]:

        # Read all index properties from scheme
        all_properties = await self.get_all_properties(index_name)

        query_builder = self.client.query.get(index_name, all_properties)
        additional_properties = ["id"]
        if return_embeddings:
            additional_properties.append("vector")
        query_builder = query_builder.with_additional(additional_properties)
        query_builder = query_builder.with_limit(max_result_size)
        if filter:
            query_builder.with_where(filter)

        async with self.http_session.post(
            self.base_url + "v1/graphql", 
            headers={'content-type': 'application/json'}, 
            data=json.dumps({"query": query_builder.build()})) as response:
            query_result = await response.json()

        parsed_result = parse_get_response(query_result)
        entries = parsed_result[index_name]

        nodes = [to_node(entry) for entry in entries]

        return nodes

    async def close(self):
        await self.http_session.close()