"""
Federated RAG Gateway
Central aggregator that queries distributed organization nodes
"""

import sys
import os
from pathlib import Path
import logging
import asyncio
from typing import List, Dict
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_core.model import load_embedder
from shared_core.config import (
    GATEWAY_PORT,
    CLIENT_PORTS,
    GLOBAL_MODEL_PATH,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    LLM_PROVIDER,
    LLM_MODEL
)
from llm_client import synthesize_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API Models
class RecommendRequest(BaseModel):
    """Recommendation request from VS Code extension."""
    query: str
    context: str = ""  # Optional local code context


class RecommendResponse(BaseModel):
    """Recommendation response."""
    response: str
    sources: List[Dict]
    query_time_ms: float


# Global state
app = FastAPI(title="Federated RAG Gateway")
embedder = None
client_urls = []


# Enable CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup."""
    global embedder, client_urls
    
    logger.info("="*60)
    logger.info("Initializing Federated RAG Gateway")
    logger.info("="*60)
    
    # Load global embedding model
    logger.info("Loading global embedding model...")
    try:
        embedder = load_embedder(model_path=GLOBAL_MODEL_PATH)
        logger.info("✓ Global model loaded")
    except Exception as e:
        logger.warning(f"Could not load federated model: {e}")
        logger.info("Using fresh model (federated training may not be complete)")
        embedder = load_embedder()
    
    # Configure client URLs
    client_urls = [f"http://localhost:{port}" for port in CLIENT_PORTS]
    logger.info(f"Configured {len(client_urls)} client nodes:")
    for url in client_urls:
        logger.info(f"  - {url}")
    
    logger.info("="*60)
    logger.info("Gateway ready")
    logger.info("="*60)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Federated RAG Gateway",
        "client_nodes": len(client_urls),
        "model_loaded": embedder is not None
    }


async def query_client_node(client_url: str, query_vector: List[float],
                            timeout: float = 5.0) -> Dict:
    """
    Query a single organization client node.
    
    Args:
        client_url: Base URL of client node
        query_vector: Query embedding vector
        timeout: Request timeout in seconds
        
    Returns:
        Search results from client node
    """
    search_url = f"{client_url}/search"
    
    # payload = {
    #     "vector": query_vector,
    #     "top_k": TOP_K_RESULTS
    # }
    payload = {
    # This single line handles 2D arrays, 1D arrays, and nested lists automatically
    "vector": np.array(query_vector).flatten().tolist(),
    "top_k": TOP_K_RESULTS
}
    # DEBUG: print out payload and URL before request
    logger.info(f"DEBUG outgoing POST to {search_url} with payload: {payload}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                search_url,
                json=payload,
                timeout=timeout
            )
            logger.info(f"DEBUG got status {response.status_code}, text: {response.text}")
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException:
        logger.warning(f"Timeout querying {client_url}")
        return {"results": [], "error": "timeout"}
    except httpx.RequestError as e:
        logger.warning(f"Error querying {client_url}: {e}")
        return {"results": [], "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error with {client_url}: {e}")
        return {"results": [], "error": str(e)}


async def distributed_retrieval(query_vector: List[float]) -> List[Dict]:
    """
    Query all client nodes in parallel and aggregate results.
    
    Args:
        query_vector: Query embedding vector
        
    Returns:
        Aggregated and ranked results from all nodes
    """
    logger.info(f"Querying {len(client_urls)} organization nodes...")
    
    # Query all clients in parallel
    tasks = [
        query_client_node(url, query_vector)
        for url in client_urls
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate results
    all_results = []
    successful_queries = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Client {i} failed: {result}")
            continue
        
        if "error" in result:
            logger.warning(f"Client {i} error: {result['error']}")
            continue
        
        # Extract results
        client_results = result.get("results", [])
        all_results.extend(client_results)
        successful_queries += 1
        
        logger.info(f"Client {i}: {len(client_results)} results")
    
    logger.info(f"Retrieved {len(all_results)} total results from {successful_queries} nodes")
    
    # Sort by score (descending)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take top results
    top_results = all_results[:TOP_K_RESULTS * 2]  # Get more for diversity
    
    return top_results


def format_context(results: List[Dict]) -> List[str]:
    """
    Format search results into context snippets.
    
    Args:
        results: List of search results
        
    Returns:
        List of formatted code snippets
    """
    snippets = []
    
    for result in results:
        content = result.get("content", "")
        source = result.get("source", "unknown")
        chunk_type = result.get("chunk_type", "code")
        name = result.get("name", "unknown")
        
        # Add metadata comment
        header = f"# Source: {source} | Type: {chunk_type} | Name: {name}\n"
        snippet = header + content
        
        snippets.append(snippet)
    
    return snippets


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Generate code recommendation using federated RAG.
    
    Args:
        request: User query and optional context
        
    Returns:
        AI-generated code recommendation
    """
    import time
    start_time = time.time()
    
    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded"
        )
    
    logger.info("="*60)
    logger.info(f"New recommendation request")
    logger.info(f"Query: {request.query[:100]}...")
    logger.info("="*60)
    
    # Combine query with local context
    full_query = request.query
    if request.context:
        full_query = f"{request.query}\n\nLocal context:\n{request.context}"
    
    # Embed query
    logger.info("Embedding query...")
    query_embedding = embedder.encode(full_query)
    query_vector = query_embedding.tolist()
    
    # Distributed retrieval
    logger.info("Starting distributed retrieval...")
    results = await distributed_retrieval(query_vector)
    
    if not results:
        logger.warning("No results found from any client node")
        return RecommendResponse(
            response="Sorry, I couldn't find relevant code examples to help with your query.",
            sources=[],
            query_time_ms=(time.time() - start_time) * 1000
        )
    
    # Format context for LLM
    logger.info(f"Formatting {len(results)} results for LLM...")
    context_snippets = format_context(results)
    
    # Generate recommendation using LLM
    logger.info(f"Generating recommendation with {LLM_PROVIDER.upper()}...")
    try:
        recommendation = synthesize_code(
            request.query, 
            context_snippets,
            provider=LLM_PROVIDER
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendation: {str(e)}"
        )
    
    # Prepare response
    query_time = (time.time() - start_time) * 1000
    
    # Format sources for response
    sources = [
        {
            "source": r.get("source"),
            "type": r.get("chunk_type"),
            "name": r.get("name"),
            "score": r.get("score")
        }
        for r in results[:5]  # Top 5 sources
    ]
    
    logger.info(f"Request completed in {query_time:.2f}ms")
    logger.info("="*60)
    
    return RecommendResponse(
        response=recommendation,
        sources=sources,
        query_time_ms=query_time
    )


@app.get("/health")
async def health_check():
    """Detailed health check."""
    # Check client nodes
    client_status = []
    
    for url in client_urls:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/", timeout=2.0)
                status = "healthy" if response.status_code == 200 else "unhealthy"
        except Exception:
            status = "unreachable"
        
        client_status.append({
            "url": url,
            "status": status
        })
    
    return {
        "gateway": "healthy",
        "model_loaded": embedder is not None,
        "client_nodes": client_status
    }


def main():
    """Start the gateway server."""
    import uvicorn
    
    logger.info("Starting Federated RAG Gateway...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=GATEWAY_PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()