import pytest
import respx
import httpx
from unittest.mock import patch, MagicMock
import gateway.main
from gateway.main import recommend, RecommendRequest

@pytest.mark.asyncio
async def test_llm_synthesis_flow():
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
    gateway.main.embedder = mock_embedder
    
    gateway.main.client_urls = ["http://localhost:8001", "http://localhost:8002"]

    query = "How do I implement a singleton pattern?"
    

    with respx.mock(assert_all_called=True) as respx_mock:
        # Mock Node A
        respx_mock.post("http://localhost:8001/search").mock(return_value=httpx.Response(
            200, 
            json={"results": [{"name": "singleton_dec", "content": "def singleton...", "score": 0.9, "source": "Org_A"}]}
        ))
        
        # Mock Node B
        respx_mock.post("http://localhost:8002/search").mock(return_value=httpx.Response(
            200, 
            json={"results": [{"name": "singleton_new", "content": "def __new__...", "score": 0.85, "source": "Org_B"}]}
        ))

        mock_ai_response = "```python\nclass Singleton:\n    _instance = None\n```"
        
        with patch("gateway.main.synthesize_code") as mock_synth:
            mock_synth.return_value = mock_ai_response
            
            request = RecommendRequest(query=query)
            # This triggers: Embedder -> Distributed Search -> LLM Synthesis
            response = await recommend(request)

            # Verify the response is exactly what our Mock LLM returned
            assert response.response == mock_ai_response
            
            assert len(response.sources) >= 2
            source_names = [s['name'] for s in response.sources]
            assert "singleton_dec" in source_names
            assert "singleton_new" in source_names
            
            mock_synth.assert_called_once()
            args, _ = mock_synth.call_args
            assert len(args[1]) >= 2