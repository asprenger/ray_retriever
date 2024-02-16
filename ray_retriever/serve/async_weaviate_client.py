from typing import Any, List, Optional, Sequence, Dict, Tuple
import json
import aiohttp
import weaviate
from ray_retriever.serve.schema import VectorStoreQueryResult, TextNode
from ray_retriever.serve.weaviate_utils import parse_get_response, get_node_similarity, to_node

class AsyncWeaviateClient():
    """Async Weaviate client"""

    def __init__(self, host:str, port:int):
        self.base_url = f"http://{host}:{port}/"
        # We need a weaviate.Client to access the query builder. There is
        # no easy way to initialize the builder on its own. The Client is
        # not used to access Weaviate. Note that the Client opens a 
        # connection pool to the server!
        self.client = weaviate.Client(url=self.base_url)
        self.http_session = aiohttp.ClientSession()

    async def get_schema(self, index_name:str) -> Dict:
        """Return the schema for an index.

        Args:
            index_name (str): Case-sensitive index name

        Raises:
            ValueError: Index does not exist

        Returns:
            Dict: Index schema
        """
        
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
        """Get Index property names.

        Args:
            index_name (str): Case-sensitive index name

        Raises:
            ValueError: Index does not exist

        Returns:
            List[str]: List of property names
        """
        schema = await self.get_schema(index_name)
        return [p["name"] for p in schema["properties"]]

    async def get_nodes(self, index_name:str, 
                        filter:Optional[Dict]=None,
                        return_embeddings:bool=False, 
                        max_result_size:int=100) -> List[TextNode]:
        """Fetch text nodes that match a filter.

        Weaviate has an internal query result limit of 10000 elements.

        Filter example:

            filter = {
                "path": "title",
                "operator": "Equal",
                "valueText": "Alan Turing"
            }

        Composite filter example:

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

        Args:
            index_name (str): Case-sensitive index name
            filter (Optional[Dict], optional): Query filter. Defaults to None.
            return_embeddings (bool, optional): Return embeddings in result. Defaults to False.
            max_result_size (int, optional): Max. result size. Defaults to 100.

        Raises:
            ValueError: Index does not exist
            
        Returns:
            List[TextNode]: Result set
        """

        # Read all index properties from schema
        all_properties = await self.get_all_properties(index_name)

        query_builder = self.client.query.get(index_name, all_properties)
        additional_properties = ["id"]
        if return_embeddings:
            additional_properties.append("vector")
        query_builder = query_builder.with_additional(additional_properties)
        if max_result_size:
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

        nodes = [to_node(entry, index_name) for entry in entries]

        return nodes


    async def find_similar_nodes(self, index_name:str, 
                                 embedding:List[float], 
                                 similarity_top_k:int,
                                 return_embeddings:bool=False,
                                 filter:Optional[Dict]=None) -> VectorStoreQueryResult:
        """Find the top-n nodes that are most similar to a given embedding vector.

        Weaviate has an internal query result limit of 10000 elements.

        See get_nodes() for filter examples.

        Args:
            index_name (str): Case-sensitive index name
            embedding (List[float]): The embedding vector
            similarity_top_k (int): Result set size
            return_embeddings (bool, optional): Return embeddings in result. Defaults to False.
            filter (Optional[Dict], optional): Query filter. Defaults to None.

        Returns:
            VectorStoreQueryResult: Query result
        """

        # Read all index properties from schema
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

        nodes = [to_node(entry, index_name) for entry in entries]

        nodes = nodes[: similarity_top_k]

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities)

    async def close(self):
        """Shutdown client"""
        self.client._connection.close()
        await self.http_session.close()