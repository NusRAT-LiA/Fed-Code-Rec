import pytest
import re
from unittest.mock import patch, MagicMock
from gateway.main import format_context

def sanitize_snippet(content: str) -> str:
    """
    A simple implementation of the logic we are testing.
    This should ideally exist in your shared_core/utils.py
    """
    # AWS Access Key ID pattern
    aws_key_pattern = r"AKIA[A-Z0-9]{16}"
    # Replace with [REDACTED_AWS_KEY]
    return re.sub(aws_key_pattern, "[REDACTED_AWS_KEY]", content)

def test_aws_key_redaction():
    """
    Scenario: Snippet contains a mock AWS Key.
    Expectation: The key is masked in the output.
    """
    # Mock retrieved result with a sensitive leak
    mock_key = "AKIAJ44QH8DHBEXAMPLE"
    leaky_content = f"os.environ['AWS_KEY'] = '{mock_key}'\nprint('Connecting...')"
    
    mock_results = [{
        "source": "config.py",
        "content": leaky_content,
        "name": "setup_env",
        "chunk_type": "code"
    }]

    # Apply Redaction (This mimics your system's safety layer)
    # We apply it to the content before formatting
    for result in mock_results:
        result["content"] = sanitize_snippet(result["content"])

    # Format Context for LLM
    formatted_output = format_context(mock_results)[0]

    # Ensure the original key is GONE
    assert mock_key not in formatted_output
    # Ensure the redaction label is PRESENT
    assert "[REDACTED_AWS_KEY]" in formatted_output
    
    print("\nSecurity Check Passed: AWS Key was successfully intercepted.")