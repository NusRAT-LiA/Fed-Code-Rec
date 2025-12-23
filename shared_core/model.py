"""
Code Embedding Model Wrapper
Wraps DistilBERT for consistent encoding.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Union, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeEmbedder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super(CodeEmbedder, self).__init__()
        logger.info(f"Initializing CodeEmbedder with model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, text: Union[str, List[str]], convert_to_numpy: bool = True) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
        return embeddings
    
    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))

def load_embedder(model_path: str = None) -> CodeEmbedder:
    embedder = CodeEmbedder()
    if model_path:
        try:
            embedder.load_weights(model_path)
        except FileNotFoundError:
            pass
    return embedder