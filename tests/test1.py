import pytest
import numpy as np
import os
from pathlib import Path
from client_node.indexer import CodeIndexer

class MockEmbedder:
    def encode(self, text):

        return np.random.rand(1, 768).astype('float32')

def test_indexer_workflow(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_path = tmp_path / "vector_store" / "code_index.faiss"
    metadata_path = tmp_path / "vector_store" / "metadata.pkl"

    sample_code = """
def calculate_sum(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
"""
    file_path = data_dir / "sample_math.py"
    file_path.write_text(sample_code.strip())


    indexer = CodeIndexer(
        data_dir=str(data_dir),
        index_path=str(index_path),
        metadata_path=str(metadata_path)
    )
    
    indexer.embedder = MockEmbedder()

    num_chunks = indexer.build_index()


    assert num_chunks > 0, "No chunks were indexed!"
    
    stats = indexer.get_stats()
    assert stats['files'] == 1
    assert stats['total_chunks'] == num_chunks
    assert stats['functions'] >= 1

    indexer.save_index()
    assert index_path.exists(), "FAISS index file was not saved."
    assert metadata_path.exists(), "Metadata pickle was not saved."

    new_indexer = CodeIndexer(
        data_dir=str(data_dir),
        index_path=str(index_path),
        metadata_path=str(metadata_path)
    )
    success = new_indexer.load_index()
    assert success is True
    assert len(new_indexer.metadata) == num_chunks