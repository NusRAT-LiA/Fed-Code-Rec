import pytest
from client_node.sanitizer import CodeSanitizer

def test_internal_import_redaction():

    sanitizer = CodeSanitizer()
    
    raw_code = """
    from internal_core_logic import process_data
    import acme_encryption_utils as crypto
    from proprietary_auth.v1 import authenticator
    
    def run():
        return process_data(authenticator)
    """

    sanitized_code = sanitizer.sanitize_code(raw_code)

    assert "from [ORG]_package import process_data" in sanitized_code
    assert "import [ORG]_package as crypto" in sanitized_code
    assert "from [ORG]_package import authenticator" in sanitized_code

    assert "internal_core_logic" not in sanitized_code
    assert "acme_encryption_utils" not in sanitized_code
    assert "proprietary_auth" not in sanitized_code

    stats = sanitizer.get_redaction_stats()
    assert stats['internal_package'] == 3