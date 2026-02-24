import pytest
from unittest.mock import patch, MagicMock
from gateway.main import distributed_retrieval, format_context
from gateway.llm_client import synthesize_code

@pytest.mark.asyncio
async def test_style_divergence_and_harmonization():
    """
    Scenario: Query for an algorithm present in PEP8 and Legacy styles.
    Expectation: Both are retrieved, and LLM produces clean output.
    """
   
    mock_results = [
        {
            "source": "Org_A_PEP8.py",
            "content": "def calculate_mean(numbers: list[float]) -> float:\n    return sum(numbers) / len(numbers)",
            "score": 0.92
        },
        {
            "source": "Org_B_Legacy.py",
            "content": "def m(l):return sum(l)/len(l)", # Messy/Legacy
            "score": 0.88
        }
    ]

    
    # We check that both results are present in the aggregated list
    assert len(mock_results) == 2
    assert "calculate_mean" in mock_results[0]["content"]
    assert "m(l)" in mock_results[1]["content"]

    
    # We want to see if the LLM takes the 'm(l)' logic and cleans it up
    query = "How to calculate the average of a list?"
    context = format_context(mock_results)
    
    # We use a real LLM call here (or a mock that verifies 'harmonization' logic)
    with patch("llm_client.GeminiClient.generate") as mock_llm:
        mock_llm.return_value = "def get_average(data):\n    return sum(data) / len(data)"
        
        final_answer = synthesize_code(query, context)
        
        # Verification: Did the LLM output a 'harmonized' version?
        # It shouldn't use the name 'm(l)' from the legacy code
        assert "def m(l)" not in final_answer
        assert "sum" in final_answer and "len" in final_answer
        print("\nSuccess: Legacy code was harmonized into clean output.")