"""
Privacy-Preserving Federated Learning Client
Uses MANUAL DP-SGD (No Opacus dependency) for maximum stability.
"""

import sys
import os
import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import flwr as fl
from typing import List
import pickle

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_core.model import CodeEmbedder
from shared_core.config import (
    FL_SERVER_ADDRESS, DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM,
    LEARNING_RATE, BATCH_SIZE, LOCAL_METADATA_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodePairDataset(Dataset):
    def __init__(self, metadata: List[dict]):
        self.data = metadata
        self.file_groups = {}
        for idx, meta in enumerate(metadata):
            filename = meta['file']
            if filename not in self.file_groups:
                self.file_groups[filename] = []
            self.file_groups[filename].append(idx)
        
        self.pairs = []
        for filename, indices in self.file_groups.items():
            if len(indices) >= 2:
                for i in range(len(indices) - 1):
                    self.pairs.append((indices[i], indices[i + 1]))
        
        if len(self.pairs) == 0:
            logger.warning("WARNING: Dataset is empty.")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        return self.data[idx1]['content'], self.data[idx2]['content']

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, embedding1, embedding2):
        distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        return torch.mean(distance)

class PrivacyPreservingClient(fl.client.NumPyClient):
    def __init__(self, model: CodeEmbedder, train_loader: DataLoader, device: str = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = ContrastiveLoss()
        
        # Standard Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        
        logger.info("Initialized Client with Manual DP-SGD")
        logger.info(f"DP Params: Max Norm={DP_MAX_GRAD_NORM}, Noise Multiplier={DP_NOISE_MULTIPLIER}")

    def fit(self, parameters, config):
        # 1. Update Local Model
        self.set_parameters(parameters)
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for text1_batch, text2_batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Forward
            emb1 = self.model(text1_batch)
            emb2 = self.model(text2_batch)
            loss = self.criterion(emb1, emb2)
            loss.backward()
            
            # --- MANUAL PRIVACY MECHANISM START ---
            
            # 1. Clip Gradients (Global Norm)
            # This bounds the sensitivity of the update
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=DP_MAX_GRAD_NORM)
            
            # 2. Add Noise
            # We add Gaussian noise to the gradients *after* clipping
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM,
                        size=p.grad.shape,
                        device=self.device
                    )
                    p.grad += noise

            # --- MANUAL PRIVACY MECHANISM END ---

            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Training complete. Loss: {avg_loss:.4f}")
        
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"loss": avg_loss}
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

def load_training_data(data_dir: str = "./") -> DataLoader:
    metadata_path = Path(data_dir) / LOCAL_METADATA_PATH
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    dataset = CodePairDataset(metadata)
    
    n_samples = len(dataset)
    actual_batch_size = min(BATCH_SIZE, n_samples)
    
    return DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=False  # Safe to use False now since we don't rely on Opacus stacking
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./')
    parser.add_argument('--server', default=FL_SERVER_ADDRESS)
    args = parser.parse_args()
    
    model = CodeEmbedder()
    try:
        train_loader = load_training_data(args.data_dir)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    
    client = PrivacyPreservingClient(model, train_loader)
    fl.client.start_client(server_address=args.server, client=client.to_client())

if __name__ == "__main__":
    main()