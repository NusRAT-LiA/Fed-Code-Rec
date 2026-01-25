# Federated Code Recommendation System - Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Class Details](#class-details)
4. [Interaction Diagrams](#interaction-diagrams)
5. [Data Flow](#data-flow)

---

## System Overview

The Federated Code Recommendation System is a distributed RAG (Retrieval-Augmented Generation) system that enables organizations to share code recommendations while preserving privacy. The system consists of:

- **Shared Core**: Common utilities for embedding, chunking, and configuration
- **Client Nodes**: Organization-specific nodes that index and search local code
- **FL Server**: Federated learning server for collaborative model training
- **Gateway**: Central aggregator that queries all client nodes and generates recommendations
- **LLM Client**: Multi-provider LLM integration for code synthesis

---

## Component Architecture

### High-Level Components

```
┌─────────────────┐
│   VS Code       │
│   Extension     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Gateway      │◄──────┐
│  (main.py)      │       │
└────────┬────────┘       │
         │                │
    ┌────┴────┬───────────┴───┐
    │         │               │
    ▼         ▼               ▼
┌────────┐ ┌────────┐    ┌────────┐
│Client A│ │Client B│    │Client C│
│(org_A) │ │(org_B) │    │(org_C) │
└────────┘ └────────┘    └────────┘
    │         │               │
    └─────────┴───────────────┘
              │
              ▼
    ┌─────────────────┐
    │  FL Server      │
    │  (server.py)    │
    └─────────────────┘
```

---

## Class Details

### 1. Shared Core Components

#### 1.1 CodeEmbedder (`shared_core/model.py`)

**Purpose**: Wraps DistilBERT model for consistent code embedding generation.

**Attributes**:
- `tokenizer` (AutoTokenizer): HuggingFace tokenizer for text preprocessing
- `model` (AutoModel): DistilBERT model for generating embeddings
- `embedding_dim` (int): Dimension of output embeddings (768 for DistilBERT)

**Methods**:
- `__init__(model_name: str = "distilbert-base-uncased")`: Initialize model and tokenizer
- `mean_pooling(model_output, attention_mask)`: Aggregates token embeddings using mean pooling
- `encode(text: Union[str, List[str]], convert_to_numpy: bool = True) -> np.ndarray`: Encodes text into embeddings (numpy array)
- `forward(text: Union[str, List[str]]) -> torch.Tensor`: PyTorch forward pass (returns tensor)
- `get_embedding_dimension() -> int`: Returns embedding dimension
- `save_weights(path: str)`: Saves model weights to disk
- `load_weights(path: str)`: Loads model weights from disk

**Interactions**:
- Used by: `CodeIndexer`, `Gateway`, `PrivacyPreservingClient`
- Uses: `AutoModel`, `AutoTokenizer` from transformers

**Factory Function**:
- `load_embedder(model_path: str = None) -> CodeEmbedder`: Creates and optionally loads a CodeEmbedder instance

---

#### 1.2 Chunker Functions (`shared_core/chunker.py`)

**Purpose**: AST-based code chunking using Tree-sitter to extract functions and classes.

**Functions** (not classes, but important components):
- `extract_functions_and_classes(code: str, filename: str = "unknown") -> List[Tuple[str, str, str]]`: Extracts (type, name, code) tuples from Python code
- `chunk_code_file(filepath: str) -> List[Tuple[str, str, str]]`: Reads file and extracts chunks
- `get_chunk_summary(chunk_type: str, name: str, code: str, max_lines: int = 5) -> str`: Creates human-readable summary

**Interactions**:
- Used by: `CodeIndexer.process_file()`
- Uses: `tree_sitter`, `tree_sitter_python`

---

#### 1.3 Configuration (`shared_core/config.py`)

**Purpose**: Centralized configuration constants.

**Constants**:
- Network: `GATEWAY_PORT`, `FL_SERVER_PORT`, `CLIENT_PORTS`, `OLLAMA_PORT`
- Model: `MODEL_NAME`, `VECTOR_DIMENSION`, `MAX_CHUNK_SIZE`
- FAISS: `TOP_K_RESULTS`, `SIMILARITY_THRESHOLD`
- Privacy: `DP_EPSILON_TARGET`, `DP_DELTA`, `DP_NOISE_MULTIPLIER`, `DP_MAX_GRAD_NORM`
- FL: `FL_NUM_ROUNDS`, `FL_MIN_CLIENTS`, `FL_FRACTION_FIT`
- Training: `LEARNING_RATE`, `BATCH_SIZE`, `LOCAL_EPOCHS`
- LLM: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_SYSTEM_PROMPT`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`

**Interactions**:
- Used by: All components for configuration

---

### 2. Client Node Components

#### 2.1 CodeIndexer (`client_node/indexer.py`)

**Purpose**: Builds and manages FAISS vector index for code chunks.

**Attributes**:
- `data_dir` (Path): Directory containing code files to index
- `index_path` (Path): Path to save/load FAISS index
- `metadata_path` (Path): Path to save/load metadata pickle file
- `embedder` (CodeEmbedder): Embedding model instance
- `index` (faiss.IndexFlatL2): FAISS L2 distance index
- `metadata` (List[Dict]): List of metadata dictionaries for each chunk

**Methods**:
- `__init__(data_dir: str, index_path: str, metadata_path: str)`: Initialize indexer with paths
- `collect_python_files() -> List[Path]`: Recursively finds all .py files in data directory
- `process_file(filepath: Path) -> List[Tuple[np.ndarray, Dict]]`: Processes single file into embeddings and metadata
- `build_index() -> int`: Builds FAISS index from all Python files, returns number of chunks
- `save_index()`: Saves FAISS index and metadata to disk
- `load_index() -> bool`: Loads existing index and metadata, returns success status
- `get_stats() -> Dict`: Returns statistics (total_chunks, functions, classes, files)

**Interactions**:
- Uses: `CodeEmbedder` (via `load_embedder()`), `chunk_code_file()`, `get_chunk_summary()`
- Used by: Indexing scripts, indirectly by `search_api.py` (loads saved index)

---

#### 2.2 Search API (`client_node/search_api.py`)

**Purpose**: FastAPI service for local FAISS retrieval with IP sanitization.

**Global State**:
- `app` (FastAPI): FastAPI application instance
- `index` (faiss.Index): FAISS index (loaded at startup)
- `metadata` (List[Dict]): Metadata list (loaded at startup)

**Pydantic Models**:
- `SearchRequest`: Request schema with `vector: List[float]`, `top_k: int`
- `SearchResult`: Result schema with `content: str`, `score: float`, `source: str`, `chunk_type: str`, `name: str`
- `SearchResponse`: Response schema with `results: List[SearchResult]`, `query_time_ms: float`

**Functions**:
- `load_local_index(index_path: str, metadata_path: str)`: Loads FAISS index and metadata into global state
- `startup_event()`: FastAPI startup hook that loads index
- `root()`: Health check endpoint
- `search(request: SearchRequest) -> SearchResponse`: Main search endpoint
- `get_stats()`: Returns index statistics

**Interactions**:
- Uses: `sanitize_code()` from `sanitizer.py`, FAISS index, metadata
- Used by: `Gateway.query_client_node()` (HTTP requests)

---

#### 2.3 CodeSanitizer (`client_node/sanitizer.py`)

**Purpose**: Removes sensitive information from code snippets before cross-org sharing.

**Attributes**:
- `patterns` (Dict[str, Tuple[str, str]]): Regex patterns and replacements for sensitive data
  - Keys: 'ipv4', 'api_key', 'database_url', 'internal_package', 'aws_key', 'generic_secret'
- `redaction_count` (Dict[str, int]): Counter for each pattern type

**Methods**:
- `__init__()`: Initializes regex patterns and counters
- `sanitize_code(code: str) -> str`: Applies all sanitization patterns to code
- `get_redaction_stats() -> Dict[str, int]`: Returns redaction statistics
- `reset_stats()`: Resets redaction counters

**Global Functions**:
- `sanitize_code(code: str) -> str`: Convenience function using global `_sanitizer` instance
- `sanitize_snippets(snippets: List[str]) -> List[str]`: Sanitizes multiple snippets
- `get_sanitizer_stats() -> Dict[str, int]`: Returns global statistics

**Interactions**:
- Used by: `search_api.py` (in `search()` endpoint before returning results)

---

#### 2.4 PrivacyPreservingClient (`client_node/train.py`)

**Purpose**: Federated learning client with manual differential privacy (DP-SGD).

**Attributes**:
- `model` (CodeEmbedder): Local embedding model
- `train_loader` (DataLoader): PyTorch data loader for training pairs
- `device` (str): Device for computation ("cpu" or "cuda")
- `criterion` (ContrastiveLoss): Loss function
- `optimizer` (torch.optim.SGD): SGD optimizer

**Methods**:
- `__init__(model: CodeEmbedder, train_loader: DataLoader, device: str = "cpu")`: Initialize client
- `fit(parameters, config) -> Tuple[List[np.ndarray], int, Dict]`: Federated training round
  - Updates model with global parameters
  - Trains on local data with DP-SGD (gradient clipping + noise)
  - Returns updated parameters, dataset size, metrics
- `get_parameters(config) -> List[np.ndarray]`: Extracts model parameters as numpy arrays
- `set_parameters(parameters)`: Updates model from numpy parameter arrays
- `evaluate(parameters, config) -> Tuple[float, int, Dict]`: Placeholder evaluation

**Interactions**:
- Uses: `CodeEmbedder`, `CodePairDataset`, `ContrastiveLoss`
- Used by: Flower framework (`fl.client.start_client()`)

---

#### 2.5 CodePairDataset (`client_node/train.py`)

**Purpose**: PyTorch Dataset for contrastive learning on code pairs from same file.

**Attributes**:
- `data` (List[dict]): Metadata list from index
- `file_groups` (Dict[str, List[int]]): Groups indices by source file
- `pairs` (List[Tuple[int, int]]): List of (idx1, idx2) pairs from same file

**Methods**:
- `__init__(metadata: List[dict])`: Builds file groups and pairs
- `__len__() -> int`: Returns number of pairs
- `__getitem__(idx) -> Tuple[str, str]`: Returns (code1, code2) tuple

**Interactions**:
- Used by: `PrivacyPreservingClient` (via DataLoader)

---

#### 2.6 ContrastiveLoss (`client_node/train.py`)

**Purpose**: Contrastive loss for training embeddings to be similar for related code.

**Attributes**:
- `margin` (float): Margin for contrastive loss (default 1.0)

**Methods**:
- `__init__(margin: float = 1.0)`: Initialize loss function
- `forward(embedding1, embedding2) -> torch.Tensor`: Computes pairwise distance loss

**Interactions**:
- Used by: `PrivacyPreservingClient.fit()`

---

#### 2.7 DataPipeline (`client_node/org_A/data/data_loader.py`)

**Purpose**: End-to-end data processing pipeline (example organization code).

**Attributes**:
- `config` (dict): Configuration dictionary
- `scaler` (StandardScaler): Scikit-learn scaler for normalization

**Methods**:
- `__init__(config)`: Initialize pipeline with config
- `fit_transform(data)`: Fit scaler and transform data
- `transform(data)`: Transform new data using fitted scaler

**Interactions**:
- Standalone utility class (example organization code)

---

#### 2.8 ModelManager (`client_node/org_A/data/model_trainer.py`)

**Purpose**: Manages model lifecycle (example organization code).

**Attributes**:
- `model_path` (str): Directory path for saving/loading models

**Methods**:
- `__init__(model_path: str = 'models/')`: Initialize with model directory
- `save_model(model, filename)`: Save model using joblib
- `load_model(filename)`: Load model from disk

**Interactions**:
- Standalone utility class (example organization code)

---

### 3. FL Server Components

#### 3.1 ModelCheckpointer (`fl_server/server.py`)

**Purpose**: Saves aggregated model after each federated round.

**Attributes**:
- `model_path` (Path): Path to save model checkpoints

**Methods**:
- `__init__(model_path: str = GLOBAL_MODEL_PATH)`: Initialize with checkpoint path
- `save_model(parameters: Parameters, round_num: int)`: Saves aggregated parameters to disk

**Interactions**:
- Used by: `CustomFedAvg.aggregate_fit()`

---

#### 3.2 CustomFedAvg (`fl_server/server.py`)

**Purpose**: Custom Federated Averaging strategy with model checkpointing.

**Inheritance**: Extends `flwr.server.strategy.FedAvg`

**Attributes**:
- `checkpointer` (ModelCheckpointer): Checkpointer instance

**Methods**:
- `__init__(checkpointer: ModelCheckpointer, *args, **kwargs)`: Initialize with checkpointer
- `aggregate_fit(server_round, results, failures) -> Tuple[Parameters, Dict]`: 
  - Calls parent aggregation
  - Saves checkpoint via checkpointer
  - Logs metrics

**Interactions**:
- Uses: `ModelCheckpointer`
- Used by: Flower server framework

---

### 4. Gateway Components

#### 4.1 Gateway Application (`gateway/main.py`)

**Purpose**: Central aggregator that queries distributed organization nodes and generates recommendations.

**Global State**:
- `app` (FastAPI): FastAPI application with CORS middleware
- `embedder` (CodeEmbedder): Global embedding model (loaded at startup)
- `client_urls` (List[str]): URLs of client nodes

**Pydantic Models**:
- `RecommendRequest`: Request with `query: str`, `context: str`
- `RecommendResponse`: Response with `response: str`, `sources: List[Dict]`, `query_time_ms: float`

**Functions**:
- `startup_event()`: Loads embedder and configures client URLs
- `root()`: Health check endpoint
- `query_client_node(client_url: str, query_vector: List[float], timeout: float = 5.0) -> Dict`: 
  - Queries single client node via HTTP POST
  - Returns search results or error
- `distributed_retrieval(query_vector: List[float]) -> List[Dict]`: 
  - Queries all client nodes in parallel
  - Aggregates and ranks results
- `format_context(results: List[Dict]) -> List[str]`: Formats results into context snippets
- `recommend(request: RecommendRequest) -> RecommendResponse`: 
  - Main recommendation endpoint
  - Embeds query, retrieves from all nodes, synthesizes with LLM
- `health_check()`: Detailed health check for all client nodes

**Interactions**:
- Uses: `CodeEmbedder` (via `load_embedder()`), `synthesize_code()` from `llm_client.py`
- Uses: HTTP client (`httpx`) to query client nodes
- Used by: VS Code extension (HTTP requests)

---

### 5. LLM Client Components

#### 5.1 LLMProvider Enum (`gateway/llm_client.py`)

**Purpose**: Enumeration of supported LLM providers.

**Values**:
- `GEMINI = "gemini"`
- `OPENAI = "openai"`
- `ANTHROPIC = "anthropic"`
- `OLLAMA = "ollama"`

---

#### 5.2 BaseLLMClient (`gateway/llm_client.py`)

**Purpose**: Abstract base class for LLM clients.

**Attributes**:
- `system_prompt` (str): System prompt for the LLM

**Methods**:
- `__init__(system_prompt: str)`: Initialize with system prompt
- `generate(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str`: Abstract method

**Interactions**:
- Base class for: `GeminiClient`, `OpenAIClient`, `AnthropicClient`, `OllamaClient`

---

#### 5.3 GeminiClient (`gateway/llm_client.py`)

**Purpose**: Google Gemini API client.

**Inheritance**: Extends `BaseLLMClient`

**Attributes**:
- `api_key` (str): Gemini API key
- `model` (str): Model name (default: "gemini-2.5-flash")
- `client` (GenerativeModel): Google Generative AI model instance

**Methods**:
- `__init__(api_key: str, system_prompt: str, model: str = "gemini-2.5-flash")`: Initialize Gemini client
- `generate(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str`: Generate using Gemini API

**Interactions**:
- Uses: `google.generativeai`

---

#### 5.4 OpenAIClient (`gateway/llm_client.py`)

**Purpose**: OpenAI API client.

**Inheritance**: Extends `BaseLLMClient`

**Attributes**:
- `api_key` (str): OpenAI API key
- `model` (str): Model name (default: "gpt-4o-mini")
- `client` (OpenAI): OpenAI client instance

**Methods**:
- `__init__(api_key: str, system_prompt: str, model: str = "gpt-4o-mini")`: Initialize OpenAI client
- `generate(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str`: Generate using OpenAI API

**Interactions**:
- Uses: `openai` library

---

#### 5.5 AnthropicClient (`gateway/llm_client.py`)

**Purpose**: Anthropic Claude API client.

**Inheritance**: Extends `BaseLLMClient`

**Attributes**:
- `api_key` (str): Anthropic API key
- `model` (str): Model name (default: "claude-sonnet-4-20250514")
- `client` (Anthropic): Anthropic client instance

**Methods**:
- `__init__(api_key: str, system_prompt: str, model: str = "claude-sonnet-4-20250514")`: Initialize Anthropic client
- `generate(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str`: Generate using Anthropic API

**Interactions**:
- Uses: `anthropic` library

---

#### 5.6 OllamaClient (`gateway/llm_client.py`)

**Purpose**: Local Ollama client (fallback option).

**Inheritance**: Extends `BaseLLMClient`

**Attributes**:
- `model` (str): Model name (default: "codellama:7b")
- `base_url` (str): Ollama server URL (default: "http://localhost:11434")
- `generate_url` (str): Full URL for generate endpoint
- `requests`: Requests library module

**Methods**:
- `__init__(system_prompt: str, model: str = "codellama:7b", base_url: str = "http://localhost:11434")`: Initialize Ollama client
- `generate(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str`: Generate using Ollama API

**Interactions**:
- Uses: `requests` library for HTTP calls

---

#### 5.7 Factory Functions (`gateway/llm_client.py`)

**Functions**:
- `create_llm_client(provider: str = "gemini", api_key: Optional[str] = None, model: Optional[str] = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> BaseLLMClient`: 
  - Factory function to create appropriate LLM client
  - Reads API keys from environment variables if not provided
- `synthesize_code(user_query: str, context_snippets: List[str], provider: str = "gemini", api_key: Optional[str] = None) -> str`: 
  - High-level function to synthesize code from context
  - Creates client, formats prompt, generates response

**Interactions**:
- Used by: `Gateway.recommend()` endpoint

---

## Interaction Diagrams

### 1. Indexing Flow

```
User/Script
    │
    ▼
CodeIndexer.__init__()
    │
    ├─► load_embedder() ──► CodeEmbedder
    │
    ▼
CodeIndexer.build_index()
    │
    ├─► collect_python_files()
    │
    ├─► process_file()
    │   │
    │   ├─► chunk_code_file() ──► extract_functions_and_classes()
    │   │
    │   └─► embedder.encode() ──► CodeEmbedder.encode()
    │
    └─► save_index()
        │
        ├─► faiss.write_index()
        └─► pickle.dump(metadata)
```

### 2. Search Flow

```
Gateway.recommend()
    │
    ├─► embedder.encode(query) ──► CodeEmbedder.encode()
    │
    ├─► distributed_retrieval()
    │   │
    │   └─► query_client_node() [for each client]
    │       │
    │       └─► HTTP POST ──► Client Search API
    │           │
    │           └─► search() endpoint
    │               │
    │               ├─► faiss.index.search()
    │               │
    │               └─► sanitize_code() ──► CodeSanitizer.sanitize_code()
    │
    ├─► format_context()
    │
    └─► synthesize_code() ──► LLM Client
        │
        └─► create_llm_client() ──► [GeminiClient|OpenAIClient|AnthropicClient|OllamaClient]
            │
            └─► client.generate()
```

### 3. Federated Learning Flow

```
FL Server (CustomFedAvg)
    │
    ├─► fl.server.start_server()
    │
    └─► aggregate_fit()
        │
        └─► ModelCheckpointer.save_model()

Client (PrivacyPreservingClient)
    │
    ├─► fl.client.start_client()
    │
    ├─► fit(parameters, config)
    │   │
    │   ├─► set_parameters() ──► model.load_state_dict()
    │   │
    │   ├─► Training Loop:
    │   │   │
    │   │   ├─► model(text1_batch) ──► CodeEmbedder.forward()
    │   │   ├─► model(text2_batch) ──► CodeEmbedder.forward()
    │   │   ├─► criterion(emb1, emb2) ──► ContrastiveLoss.forward()
    │   │   ├─► loss.backward()
    │   │   ├─► clip_grad_norm_() [DP-SGD]
    │   │   ├─► Add Gaussian noise [DP-SGD]
    │   │   └─► optimizer.step()
    │   │
    │   └─► get_parameters() ──► model.state_dict()
    │
    └─► Returns: (parameters, num_samples, metrics)
```

### 4. Class Dependency Graph

```
┌─────────────────┐
│  CodeEmbedder   │◄────┐
└────────┬────────┘     │
         │              │
    ┌────┴────┐         │
    │         │         │
    ▼         ▼         │
┌─────────┐ ┌──────────────┐
│CodeIndex│ │PrivacyPreserv│
│  er     │ │  ingClient   │
└────┬────┘ └──────────────┘
     │              │
     │              │
     ▼              ▼
┌─────────┐    ┌──────────┐
│SearchAPI│    │CustomFedA│
│         │    │    vg    │
└────┬────┘    └─────┬────┘
     │               │
     │               │
     ▼               ▼
┌─────────┐    ┌──────────┐
│CodeSanit│    │ModelCheck│
│  izer   │    │ pointer  │
└─────────┘    └──────────┘
     │
     │
     ▼
┌─────────┐
│ Gateway │
└────┬────┘
     │
     ▼
┌──────────┐
│LLMClient │
│(various) │
└──────────┘
```

---

## Data Flow

### 1. Indexing Pipeline

1. **CodeIndexer** scans `data/` directory for `.py` files
2. **Chunker** extracts functions and classes using Tree-sitter
3. **CodeEmbedder** generates embeddings for each chunk
4. **FAISS** index stores embeddings
5. **Metadata** pickle stores chunk information (file, type, name, content)

### 2. Query Pipeline

1. **Gateway** receives query from VS Code extension
2. **CodeEmbedder** embeds query text
3. **Gateway** sends query vector to all client nodes (parallel HTTP requests)
4. **Client Search API** searches local FAISS index
5. **CodeSanitizer** sanitizes results before returning
6. **Gateway** aggregates and ranks results from all nodes
7. **LLM Client** synthesizes recommendation from context snippets
8. **Gateway** returns recommendation to VS Code extension

### 3. Training Pipeline

1. **FL Server** initializes with global model
2. **Clients** load local training data (code pairs from metadata)
3. **Server** sends global parameters to clients
4. **Clients** train locally with DP-SGD:
   - Forward pass through CodeEmbedder
   - Contrastive loss on code pairs
   - Gradient clipping
   - Gaussian noise addition
5. **Clients** return updated parameters
6. **Server** aggregates parameters (FedAvg)
7. **ModelCheckpointer** saves aggregated model
8. Repeat for multiple rounds

---

## Key Design Patterns

1. **Factory Pattern**: `load_embedder()`, `create_llm_client()`
2. **Strategy Pattern**: Different LLM providers implement `BaseLLMClient`
3. **Singleton Pattern**: Global `_sanitizer` instance
4. **Repository Pattern**: `CodeIndexer` manages FAISS index and metadata
5. **Federated Learning Pattern**: `PrivacyPreservingClient` implements Flower client interface

---

## Summary

This system implements a federated code recommendation architecture with:

- **Privacy**: Code sanitization and differential privacy in training
- **Distributed Search**: Multiple organization nodes with local indexes
- **Centralized Aggregation**: Gateway coordinates queries and synthesis
- **Flexible LLM Integration**: Support for multiple LLM providers
- **Federated Learning**: Collaborative model training without data sharing

Each component is designed with clear responsibilities and well-defined interfaces, enabling modularity and maintainability.

