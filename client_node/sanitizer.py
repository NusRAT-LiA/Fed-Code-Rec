"""
IP and Secret Sanitization Module
Removes sensitive information from code snippets before cross-org sharing
"""

import re
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeSanitizer:
    """Sanitizes code by redacting sensitive information."""
    
    def __init__(self):
        # Regex patterns for different types of sensitive data
        self.patterns = {
            'ipv4': (
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                '[IP_REDACTED]'
            ),
            'api_key': (
                r'["\']([a-zA-Z0-9_\-]{32,})["\']',
                '"[SECRET_REDACTED]"'
            ),
            'database_url': (
                r'(?:mongodb|postgresql|mysql|redis)://[^\s\'"]+',
                '[DB_CONNECTION]'
            ),
            'internal_package': (
                r'\b(from|import)\s+(acme|internal|proprietary)_\w+',
                r'\1 [ORG]_package'
            ),
            'aws_key': (
                r'AKIA[0-9A-Z]{16}',
                '[AWS_KEY_REDACTED]'
            ),
            'generic_secret': (
                r'(?:secret|password|token|key)\s*[=:]\s*["\']([^"\']{8,})["\']',
                r'\1="[SECRET_REDACTED]"'
            ),
        }
        
        self.redaction_count = {key: 0 for key in self.patterns.keys()}
    
    def sanitize_code(self, code: str) -> str:
        """
        Sanitize a code snippet by replacing sensitive patterns.
        
        Args:
            code: Raw code string
            
        Returns:
            Sanitized code string
        """
        sanitized = code
        
        for pattern_name, (pattern, replacement) in self.patterns.items():
            matches = re.findall(pattern, sanitized)
            if matches:
                self.redaction_count[pattern_name] += len(matches)
                sanitized = re.sub(pattern, replacement, sanitized)
                logger.debug(f"Redacted {len(matches)} {pattern_name} pattern(s)")
        
        return sanitized
    
    def get_redaction_stats(self) -> Dict[str, int]:
        """Return statistics on redactions performed."""
        return self.redaction_count.copy()
    
    def reset_stats(self):
        """Reset redaction counters."""
        self.redaction_count = {key: 0 for key in self.patterns.keys()}


# Global sanitizer instance
_sanitizer = CodeSanitizer()


def sanitize_code(code: str) -> str:
    """
    Convenience function to sanitize code.
    
    Args:
        code: Raw code string
        
    Returns:
        Sanitized code string
    """
    return _sanitizer.sanitize_code(code)


def sanitize_snippets(snippets: List[str]) -> List[str]:
    """
    Sanitize multiple code snippets.
    
    Args:
        snippets: List of code strings
        
    Returns:
        List of sanitized code strings
    """
    return [sanitize_code(snippet) for snippet in snippets]


def get_sanitizer_stats() -> Dict[str, int]:
    """Get global sanitization statistics."""
    return _sanitizer.get_redaction_stats()


if __name__ == "__main__":
    # Test the sanitizer
    test_cases = [
        
    ]
    
    print("=== Sanitization Tests ===\n")
    
    for i, code in enumerate(test_cases, 1):
        sanitized = sanitize_code(code)
        print(f"Test {i}:")
        print(f"  Original:  {code}")
        print(f"  Sanitized: {sanitized}")
        print()
    
    print("=== Redaction Statistics ===")
    stats = get_sanitizer_stats()
    for pattern_type, count in stats.items():
        if count > 0:
            print(f"  {pattern_type}: {count} redaction(s)")