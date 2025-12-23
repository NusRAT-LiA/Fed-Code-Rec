# """
# Global Configuration for Federated RAG System
# Shared constants across all nodes
# """

# # Network Topology
# GATEWAY_PORT = 8000
# FL_SERVER_PORT = 8080
# CLIENT_PORTS = [8001, 8002, 8003]
# OLLAMA_PORT = 11434

# # Network Addresses
# FL_SERVER_ADDRESS = f"127.0.0.1:{FL_SERVER_PORT}"
# GATEWAY_URL = f"http://localhost:{GATEWAY_PORT}"
# OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}"

# # Model Configuration
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# VECTOR_DIMENSION = 384
# MAX_CHUNK_SIZE = 512  # tokens

# # FAISS Configuration
# TOP_K_RESULTS = 5
# SIMILARITY_THRESHOLD = 0.7

# # Privacy Configuration
# DP_EPSILON_TARGET = 1.0
# DP_DELTA = 1e-5
# DP_NOISE_MULTIPLIER = 1.1
# DP_MAX_GRAD_NORM = 1.0
# DP_EPSILON_WARNING = 5.0

# # Federated Learning Configuration
# FL_NUM_ROUNDS = 5
# FL_MIN_CLIENTS = 2
# FL_FRACTION_FIT = 1.0

# # Training Configuration
# LEARNING_RATE = 0.01
# BATCH_SIZE = 32
# LOCAL_EPOCHS = 1

# # File Paths
# GLOBAL_MODEL_PATH = "shared_core/global_model.pth"
# LOCAL_INDEX_PATH = "vector_store/index.faiss"
# LOCAL_METADATA_PATH = "vector_store/metadata.pkl"

# # LLM Configuration
# LLM_SYSTEM_PROMPT = """You are a code recommendation assistant.
# You will receive code snippets from various organizations.

# CRITICAL RULES:
# 1. DO NOT copy any snippet verbatim
# 2. SYNTHESIZE a generic solution based on patterns found
# 3. If context contains [REDACTED], treat it as a placeholder
# 4. Provide clean, production-ready code
# 5. Explain your reasoning briefly
# """

# LLM_MAX_TOKENS = 2048
# LLM_TEMPERATURE = 0.7

"""
Global Configuration for Federated RAG System
Shared constants across all nodes
"""

# Network Topology
GATEWAY_PORT = 8000
FL_SERVER_PORT = 8080
CLIENT_PORTS = [8001, 8002, 8003]
OLLAMA_PORT = 11434

# Network Addresses
FL_SERVER_ADDRESS = f"127.0.0.1:{FL_SERVER_PORT}"
GATEWAY_URL = f"http://localhost:{GATEWAY_PORT}"
OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}"

# Model Configuration
MODEL_NAME = "distilbert-base-uncased"
VECTOR_DIMENSION = 768
MAX_CHUNK_SIZE = 512  # tokens

# FAISS Configuration
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.1

# Privacy Configuration
DP_EPSILON_TARGET = 1.0
DP_DELTA = 1e-5
DP_NOISE_MULTIPLIER = 1.1
DP_MAX_GRAD_NORM = 1.0
DP_EPSILON_WARNING = 5.0

# Federated Learning Configuration
FL_NUM_ROUNDS = 5
FL_MIN_CLIENTS = 2
FL_FRACTION_FIT = 1.0

# Training Configuration
LEARNING_RATE = 0.01
BATCH_SIZE = 32
LOCAL_EPOCHS = 1

# File Paths
GLOBAL_MODEL_PATH = "shared_core/global_model.pth"
LOCAL_INDEX_PATH = "vector_store/index.faiss"
LOCAL_METADATA_PATH = "vector_store/metadata.pkl"

# LLM Configuration
LLM_PROVIDER = "gemini"  # Options: "gemini", "openai", "anthropic", "ollama"
LLM_MODEL = None  # Uses provider defaults if None
LLM_SYSTEM_PROMPT = """You are a code recommendation assistant.
You will receive code snippets from various organizations.

CRITICAL RULES:
1. DO NOT copy any snippet verbatim
2. SYNTHESIZE a generic solution based on patterns found
3. If context contains [REDACTED], treat it as a placeholder
4. Provide clean, production-ready code
5. Explain your reasoning briefly
"""

LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.7

# API Keys (loaded from environment variables)
# Set these in your environment:
#   export GEMINI_API_KEY="your-key-here"
#   export OPENAI_API_KEY="your-key-here"
#   export ANTHROPIC_API_KEY="your-key-here"