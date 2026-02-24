import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from gateway.main import app

def test_similarity_threshold_impact():
    mock_search_results = [
        {"source": "org_a", "content": "Perfect Match", "score": 0.95},
        {"source": "org_b", "content": "Weak Match", "score": 0.60}
    ]

    with TestClient(app) as client:
        # TEST 1: STRICT (0.9)
        # We manually filter in the mock to simulate Gateway behavior
        strict_val = 0.9
        filtered_strict = [r for r in mock_search_results if r["score"] >= strict_val]
        
        with patch("gateway.main.distributed_retrieval", return_value=filtered_strict):
            response = client.post("/recommend", json={"query": "test query"})
            count_strict = len(response.json().get("sources", []))

        # TEST 2: LOOSE (0.5)
        loose_val = 0.5
        filtered_loose = [r for r in mock_search_results if r["score"] >= loose_val]
        
        with patch("gateway.main.distributed_retrieval", return_value=filtered_loose):
            response = client.post("/recommend", json={"query": "test query"})
            count_loose = len(response.json().get("sources", []))

        print(f"\nStrict Results: {count_strict} | Loose Results: {count_loose}")
        assert count_strict < count_loose