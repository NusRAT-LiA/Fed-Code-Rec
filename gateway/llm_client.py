"""
LLM Client for Code Generation
Supports multiple providers: Gemini, OpenAI, Anthropic, and Ollama
"""

import sys
import os
import logging
from typing import List, Optional, Dict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class BaseLLMClient:
    """Base class for LLM clients."""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
    
    def generate(self, prompt: str, max_tokens: int = 2048, 
                 temperature: float = 0.7) -> str:
        """Generate response from LLM."""
        raise NotImplementedError


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str, system_prompt: str, 
                 model: str = "gemini-2.5-flash"):
        super().__init__(system_prompt)
        self.api_key = api_key
        self.model = model
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt
            )
            logger.info(f"Initialized Gemini client with model: {model}")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        """Generate using Gemini API."""
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error: Failed to generate response. {str(e)}"


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, system_prompt: str,
                 model: str = "gpt-4o-mini"):
        super().__init__(system_prompt)
        self.api_key = api_key
        self.model = model
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {model}")
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
    
    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        """Generate using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"Error: Failed to generate response. {str(e)}"


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, system_prompt: str,
                 model: str = "claude-sonnet-4-20250514"):
        super().__init__(system_prompt)
        self.api_key = api_key
        self.model = model
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client with model: {model}")
        except ImportError:
            raise ImportError(
                "anthropic not installed. "
                "Install with: pip install anthropic"
            )
    
    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        """Generate using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            return f"Error: Failed to generate response. {str(e)}"


class OllamaClient(BaseLLMClient):
    """Local Ollama client (fallback option)."""
    
    def __init__(self, system_prompt: str, model: str = "codellama:7b",
                 base_url: str = "http://localhost:11434"):
        super().__init__(system_prompt)
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        
        import requests
        self.requests = requests
        
        logger.info(f"Initialized Ollama client with model: {model}")
    
    def generate(self, prompt: str, max_tokens: int = 2048,
                 temperature: float = 0.7) -> str:
        """Generate using Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        try:
            response = self.requests.post(
                self.generate_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: {str(e)}"


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a code recommendation assistant.
You will receive code snippets from various organizations.

CRITICAL RULES:
1. DO NOT copy any snippet verbatim
2. SYNTHESIZE a generic solution based on patterns found
3. If context contains [REDACTED], treat it as a placeholder
4. Provide clean, production-ready code
5. Explain your reasoning briefly
"""


def create_llm_client(provider: str = "gemini", 
                     api_key: Optional[str] = None,
                     model: Optional[str] = None,
                     system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> BaseLLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        provider: LLM provider ("gemini", "openai", "anthropic", "ollama")
        api_key: API key for the provider (not needed for Ollama)
        model: Model name (optional, uses defaults)
        system_prompt: System prompt for the model
        
    Returns:
        Configured LLM client
        
    Environment Variables:
        GEMINI_API_KEY: For Gemini
        OPENAI_API_KEY: For OpenAI
        ANTHROPIC_API_KEY: For Anthropic
    """
    # Get API key from environment if not provided
    if api_key is None:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Create client based on provider
    if provider == "gemini":
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return GeminiClient(
            api_key=api_key,
            system_prompt=system_prompt,
            model=model or "gemini-2.5-flash"
        )
    
    elif provider == "openai":
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return OpenAIClient(
            api_key=api_key,
            system_prompt=system_prompt,
            model=model or "gpt-4o-mini"
        )
    
    elif provider == "anthropic":
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return AnthropicClient(
            api_key=api_key,
            system_prompt=system_prompt,
            model=model or "claude-sonnet-4-20250514"
        )
    
    elif provider == "ollama":
        return OllamaClient(
            system_prompt=system_prompt,
            model=model or "codellama:7b"
        )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Supported: gemini, openai, anthropic, ollama"
        )


def synthesize_code(user_query: str, context_snippets: List[str],
                    provider: str = "gemini",
                    api_key: Optional[str] = None) -> str:
    """
    Synthesize code recommendation from context snippets.
    
    Args:
        user_query: User's question
        context_snippets: List of relevant code snippets
        provider: LLM provider to use
        api_key: API key (optional if set in environment)
        
    Returns:
        Synthesized code recommendation
    """
    client = create_llm_client(provider=provider, api_key=api_key)
    
    # Build context section
    context_section = "# Relevant Code Examples\n\n"
    for i, snippet in enumerate(context_snippets, 1):
        context_section += f"## Example {i}\n```python\n{snippet}\n```\n\n"
    
    # Build full prompt
    prompt = f"""{context_section}

# User Question
{user_query}

# Your Task
Based on the examples above, provide a clean, synthesized code solution.
Remember:
- Do NOT copy code verbatim
- Synthesize patterns from the examples
- Provide production-ready code
- Explain your reasoning briefly
"""
    
    # Generate response
    response = client.generate(prompt)
    
    return response


if __name__ == "__main__":
    # Test the LLM client
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM client")
    parser.add_argument('--provider', default='gemini', 
                       choices=['gemini', 'openai', 'anthropic', 'ollama'],
                       help='LLM provider to use')
    parser.add_argument('--api-key', help='API key (or set via environment)')
    args = parser.parse_args()
    
    logger.info(f"Testing {args.provider} client...")
    
    test_snippets = [
        """
def authenticate_user(username, password):
    # Verify credentials
    hashed = hash_password(password)
    return check_database(username, hashed)
""",
        """
class AuthService:
    def login(self, credentials):
        if self.verify(credentials):
            return self.create_token()
        return None
"""
    ]
    
    query = "How do I implement user authentication?"
    
    try:
        result = synthesize_code(
            query, 
            test_snippets,
            provider=args.provider,
            api_key=args.api_key
        )
        
        logger.info("\n" + "="*60)
        logger.info("Generated Response:")
        logger.info("="*60)
        print(result)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.info("\nMake sure to set API key:")
        logger.info(f"  export GEMINI_API_KEY='your-key-here'")
        logger.info(f"  export OPENAI_API_KEY='your-key-here'")
        logger.info(f"  export ANTHROPIC_API_KEY='your-key-here'")