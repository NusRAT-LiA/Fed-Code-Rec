import pytest
from client_node.sanitizer import CodeSanitizer, sanitize_code

def test_privacy_sanitization_flow():
    """
    Validates that sensitive credentials are redacted from code snippets 
    before being returned in a search response.
    """
    # 1. Setup: Define a snippet containing sensitive data
    # Patterns: API Key, IP Address, and Database URL
    raw_snippet = """
    def connect_to_service():
        api_key = "sk_live_556677889900aabbccddeeff112233"
        server_ip = "192.168.1.105"
        db_url = "postgresql://admin:password123@localhost:5432/db"
        return True
    """

    # 2. Execution: Pass the snippet through the sanitizer
    sanitized_output = sanitize_code(raw_snippet)

    # 3. Assertions: Verify Expected Outcome
    # Check for redaction placeholders
    assert "[SECRET_REDACTED]" in sanitized_output
    assert "[IP_REDACTED]" in sanitized_output
    assert "[DB_CONNECTION]" in sanitized_output

    # Ensure raw sensitive data is NO LONGER present
    assert "sk_live_556677889900" not in sanitized_output
    assert "192.168.1.105" not in sanitized_output
    assert "password123" not in sanitized_output

    # 4. Statistical Verification
    from client_node.sanitizer import get_sanitizer_stats
    stats = get_sanitizer_stats()
    assert stats['api_key'] > 0
    assert stats['ipv4'] > 0
    assert stats['database_url'] > 0

def test_sanitization_integration_mock(monkeypatch):
    """
    Simulates a search response being intercepted and sanitized.
    """
    # Dummy search result as it would come from the Indexer/FAISS
    mock_search_results = [
        {
            "name": "config_auth",
            "content": 'aws_key = "AKIA1234567890123456"',
            "file": "config.py"
        }
    ]

    # Intercepting the response (simulating the Search API logic)
    sanitized_results = [
        {**res, "content": sanitize_code(res["content"])} 
        for res in mock_search_results
    ]

    # Verify the intercepted JSON payload
    assert "[AWS_KEY_REDACTED]" in sanitized_results[0]["content"]
    assert "AKIA" not in sanitized_results[0]["content"]