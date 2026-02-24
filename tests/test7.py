import pytest
import respx
import httpx
from fastapi.testclient import TestClient
from gateway.main import app

@respx.mock
@pytest.mark.asyncio
async def test_gateway_resilience_on_node_failure():
    """
    Scenario: Client B goes offline (Connection Refused).
    Expectation: Gateway returns results from Client A and doesn't crash.
    """
    # 1. Mock Client A (Healthy)
    respx.post("http://localhost:8001/search").mock(return_value=httpx.Response(
        200, 
        json={"results": [{"source": "ClientA_File.py", "content": "print('hello')", "score": 0.9}]}
    ))

    # 2. Mock Client B (Offline/Killed)
    # This simulates "Connection Refused"
    respx.post("http://localhost:8002/search").mock(side_effect=httpx.ConnectError("Connection Refused"))

    # 3. Setup Test Client for Gateway
    # Note: We use the TestClient provided by FastAPI
    with TestClient(app) as client:
        payload = {"query": "how to print hello", "context": ""}
        response = client.post("/recommend", json=payload)

        # 4. Assert Outcome
        assert response.status_code == 200  # Gateway did NOT crash
        data = response.json()
        
        # Verify results come from Client A
        assert any("ClientA_File.py" in str(src) for src in data["sources"])
        
        # Verify Client B's failure didn't block the response
        assert "response" in data
        print("Test Passed: Gateway returned partial results successfully.")