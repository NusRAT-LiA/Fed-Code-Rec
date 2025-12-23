"""
FAISS Index Builder
Walks data directory, chunks code, embeds, and builds searchable vector index
"""

# --- CRITICAL: ENV VARS MUST BE FIRST ---
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import sys

# --- CRITICAL: IMPORT ORDER FIX ---
# We must import torch BEFORE faiss to prevent macOS Segmentation Faults
import torch 
import faiss 

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_core.model import load_embedder
from shared_core.chunker import chunk_code_file, get_chunk_summary
from shared_core.config import LOCAL_INDEX_PATH, LOCAL_METADATA_PATH

# DistilBERT dimension
VECTOR_DIMENSION = 768 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeIndexer:
    """Builds and manages FAISS index for code chunks."""
    
    def __init__(self, data_dir: str = "./data", 
                 index_path: str = LOCAL_INDEX_PATH,
                 metadata_path: str = LOCAL_METADATA_PATH):
        self.data_dir = Path(data_dir)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Create vector_store directory
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedder
        logger.info("Loading embedding model...")
        self.embedder = load_embedder()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        self.metadata = []
        
        logger.info(f"Indexer initialized. Data dir: {self.data_dir}")
    
    def collect_python_files(self) -> List[Path]:
        """Find all Python files in data directory."""
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return []
        
        python_files = list(self.data_dir.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")
        return python_files
    
    def process_file(self, filepath: Path) -> List[Tuple[np.ndarray, Dict]]:
        """Process a single Python file into embeddings and metadata."""
        results = []
        chunks = chunk_code_file(str(filepath))
        
        if not chunks:
            logger.warning(f"No chunks extracted from {filepath}")
            return results
        
        for chunk_type, name, code in chunks:
            try:
                clean_code = str(code).strip()
                if not clean_code:
                    continue
                    
                # Create embedding
                embedding = self.embedder.encode(clean_code)
                
                # Check dimensions
                if embedding.shape[1] != VECTOR_DIMENSION:
                    logger.warning(f"Dimension mismatch: Got {embedding.shape[1]}, expected {VECTOR_DIMENSION}")
                    continue

                # Create metadata
                metadata = {
                    'file': str(filepath.relative_to(self.data_dir)),
                    'type': chunk_type,
                    'name': name,
                    'content': clean_code,
                    'summary': get_chunk_summary(chunk_type, name, clean_code),
                    'lines': len(clean_code.split('\n'))
                }
                
                results.append((embedding, metadata))
            except Exception as e:
                logger.error(f"Failed to process chunk {name} in {filepath}: {e}")
                continue
        
        return results
    
    def build_index(self) -> int:
        """Build FAISS index from all Python files."""
        python_files = self.collect_python_files()
        
        if not python_files:
            logger.error("No Python files found to index")
            return 0
        
        all_embeddings = []
        all_metadata = []
        
        for filepath in python_files:
            logger.info(f"Processing: {filepath.name}")
            file_results = self.process_file(filepath)
            
            for embedding, metadata in file_results:
                all_embeddings.append(embedding)
                all_metadata.append(metadata)
        
        if not all_embeddings:
            logger.error("No code chunks were successfully embedded")
            return 0
        
        # Stack embeddings
        embeddings_array = np.vstack(all_embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        self.metadata = all_metadata
        
        logger.info(f"Built index with {len(all_metadata)} chunks")
        return len(all_metadata)
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"Saved FAISS index to {self.index_path}")
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata to {self.metadata_path}")
    
    def load_index(self) -> bool:
        """Load existing FAISS index and metadata."""
        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded index with {len(self.metadata)} chunks")
            return True
        except FileNotFoundError:
            logger.warning("Index files not found")
            return False
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the index."""
        return {
            'total_chunks': len(self.metadata),
            'functions': sum(1 for m in self.metadata if m['type'] == 'function'),
            'classes': sum(1 for m in self.metadata if m['type'] == 'class'),
            'files': len(set(m['file'] for m in self.metadata))
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS index from code files")
    parser.add_argument('--data-dir', default='./data', help='Directory containing code files')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild even if index exists')
    args = parser.parse_args()
    
    indexer = CodeIndexer(data_dir=args.data_dir)
    
    if not args.rebuild and indexer.load_index():
        logger.info("Index already exists. Use --rebuild to recreate.")
        stats = indexer.get_stats()
        logger.info(f"Index stats: {stats}")
        return
    
    logger.info("Building new index...")
    num_chunks = indexer.build_index()
    
    if num_chunks > 0:
        indexer.save_index()
        stats = indexer.get_stats()
        logger.info(f"Successfully indexed {num_chunks} code chunks")
        logger.info(f"Stats: {stats}")
    else:
        logger.error("Failed to build index")
        sys.exit(1)


if __name__ == "__main__":
    main()