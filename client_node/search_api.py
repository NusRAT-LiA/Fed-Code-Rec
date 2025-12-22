"""
Organization Search API
FastAPI service for local FAISS retrieval with IP sanitization
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import pickle
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_core.config import (
    LOCAL_INDEX_PATH,
    LOCAL_METADATA_PATH,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD
)
from sanitizer import sanitize_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API Models
class SearchRequest(BaseModel):
    """Search request schema."""
    vector: List[float]
    top_k: int = TOP_K_RESULTS


class SearchResult(BaseModel):
    """Individual search result."""
    content: str
    score: float
    source: str
    chunk_type: str
    name: str


class SearchResponse(BaseModel):
    """Search response schema."""
    results: List[SearchResult]
    query_time_ms: float


# Global state
app = FastAPI(title="Organization Search Node")
index = None
metadata = None


def load_local_index(index_path: str = LOCAL_INDEX_PATH, 
                     metadata_path: str = LOCAL_METADATA_PATH):
    """
    Load FAISS index and metadata from disk.
    
    Args:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata pickle file
    """
    global index, metadata
    
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run indexer.py first to build the index."
        )
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            "Run indexer.py first to build the index."
        )
    
    # Load FAISS index
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    
    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded index with {len(metadata)} chunks")
    logger.info(f"Index size: {index.ntotal} vectors")


@app.on_event("startup")
async def startup_event():
    """Initialize index on startup."""
    try:
        load_local_index()
        logger.info("Search API ready")
    except FileNotFoundError as e:
        logger.error(f"Failed to load index: {e}")
        logger.error("API will not function properly until index is built")


@app.get("/")
async def root():
    """Health check endpoint."""
    if index is None or metadata is None:
        return {
            "status": "error",
            "message": "Index not loaded. Run indexer.py first."
        }
    
    return {
        "status": "healthy",
        "index_size": len(metadata),
        "vectors": index.ntotal if index else 0
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for similar code chunks.
    
    Args:
        request: Search request with query vector
        
    Returns:
        List of sanitized code results
    """
    if index is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run indexer.py to build index."
        )
    
    import time
    start_time = time.time()
    
    # Convert query vector to numpy array
    query_vector = np.array(request.vector, dtype='float32').reshape(1, -1)
    
    # Search FAISS index
    k = min(request.top_k, len(metadata))
    distances, indices = index.search(query_vector, k)
    
    # Process results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:  # Invalid index
            continue
        
        # Convert distance to similarity score (L2 distance -> similarity)
        # Lower distance = higher similarity
        score = 1.0 / (1.0 + dist)
        
        # Filter by threshold
        if score < SIMILARITY_THRESHOLD:
            continue
        
        # Get metadata
        meta = metadata[idx]
        
        # Sanitize code before returning
        sanitized_content = sanitize_code(meta['content'])
        
        result = SearchResult(
            content=sanitized_content,
            score=float(score),
            source=meta['file'],
            chunk_type=meta['type'],
            name=meta['name']
        )
        
        results.append(result)
    
    query_time = (time.time() - start_time) * 1000
    
    logger.info(f"Search completed in {query_time:.2f}ms, returned {len(results)} results")
    
    return SearchResponse(
        results=results,
        query_time_ms=query_time
    )


@app.get("/stats")
async def get_stats():
    """Get statistics about the local index."""
    if index is None or metadata is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Calculate statistics
    file_counts = {}
    type_counts = {"function": 0, "class": 0}
    
    for meta in metadata:
        # Count by file
        filename = meta['file']
        file_counts[filename] = file_counts.get(filename, 0) + 1
        
        # Count by type
        chunk_type = meta['type']
        if chunk_type in type_counts:
            type_counts[chunk_type] += 1
    
    return {
        "total_chunks": len(metadata),
        "total_vectors": index.ntotal,
        "unique_files": len(file_counts),
        "functions": type_counts["function"],
        "classes": type_counts["class"],
        "files": file_counts
    }


def main():
    """Start the search API server."""
    parser = argparse.ArgumentParser(description="Start organization search API")
    parser.add_argument('--port', type=int, default=8001, help='Port to run on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Starting Organization Search API")
    logger.info("="*60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info("="*60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()