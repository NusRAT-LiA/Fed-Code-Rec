import pytest
import numpy as np
from shared_core.model import CodeEmbedder

def test_semantic_link_database_connect():
    embedder = CodeEmbedder()
    code_snippet = "def database_connect(user, pswd): return auth.get_session(user, pswd)"
    user_query = "How to connect to DB"
    
    # These are currently (1, 768)
    code_vec = embedder.encode(code_snippet)
    query_vec = embedder.encode(user_query)

    # Calculate Cosine Similarity (Flattening to 1D)
    # .flatten() converts (1, 768) -> (768,)
    v1 = code_vec.flatten()
    v2 = query_vec.flatten()
    
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    print(f"Semantic Similarity Score: {similarity:.4f}")
    
    assert similarity > 0.6