import pytest
import respx
import httpx
from gateway.main import distributed_retrieval, client_urls

@pytest.mark.asyncio
async def test_distributed_retrieval_fanout():

    test_urls = ["http://localhost:8001", "http://localhost:8002"]
    
    with respx.mock:
        # Mock Node A (Org A) - High Score Result
        respx.post("http://localhost:8001/search").mock(return_value=httpx.Response(200, json={
            "results": [
                {"name": "fast_sort", "content": "def sort()...", "score": 0.98, "source": "Org_A", "chunk_type": "function"}
            ]
        }))

        # Mock Node B (Org B) - Medium Score Result
        respx.post("http://localhost:8002/search").mock(return_value=httpx.Response(200, json={
            "results": [
                {"name": "bubble_sort", "content": "def bubble()...", "score": 0.75, "source": "Org_B", "chunk_type": "function"}
            ]
        }))


        import gateway.main
        gateway.main.client_urls = test_urls
        
        query_vector = [0.1] * 768
        aggregated_results = await distributed_retrieval(query_vector)

        # 3. Assertions
        assert len(aggregated_results) == 2
        
        # Verify Ranking (0.98 should be first)
        assert aggregated_results[0]["name"] == "fast_sort"
        assert aggregated_results[1]["name"] == "bubble_sort"
        
        # Verify metadata preservation
        assert aggregated_results[0]["source"] == "Org_A"
        assert aggregated_results[1]["source"] == "Org_B"

@pytest.mark.asyncio
async def test_gateway_partial_failure():
    """
    Verifies that if one client node is down, the Gateway still 
    returns results from the healthy nodes.
    """
    test_urls = ["http://localhost:8001", "http://localhost:8002"]
    
    with respx.mock:
        # Node A is healthy
        respx.post("http://localhost:8001/search").mock(return_value=httpx.Response(200, json={
            "results": [{"name": "healthy_node_code", "score": 0.9}]
        }))
        
        # Node B returns a 500 Internal Server Error
        respx.post("http://localhost:8002/search").mock(return_value=httpx.Response(500))

        import gateway.main
        gateway.main.client_urls = test_urls
        
        results = await distributed_retrieval([0.1]*768)

        assert len(results) == 1
        assert results[0]["name"] == "healthy_node_code"