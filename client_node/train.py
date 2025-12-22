"""
Privacy-Preserving Federated Learning Client
Trains embedding model locally with Differential Privacy using Opacus
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
from opacus import PrivacyEngine
from typing import List, Tuple
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_core.model import CodeEmbedder
from shared_core.config import (
    FL_SERVER_ADDRESS,
    DP_NOISE_MULTIPLIER,
    DP_MAX_GRAD_NORM,
    DP_DELTA,
    DP_EPSILON_WARNING,
    LEARNING_RATE,
    BATCH_SIZE,
    LOCAL_EPOCHS,
    LOCAL_METADATA_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodePairDataset(Dataset):
    """Dataset for contrastive learning on code pairs."""
    
    def __init__(self, metadata: List[dict]):
        """
        Args:
            metadata: List of chunk metadata from indexer
        """
        self.data = metadata
        
        # Group chunks by file for positive pairs
        self.file_groups = {}
        for idx, meta in enumerate(metadata):
            filename = meta['file']
            if filename not in self.file_groups:
                self.file_groups[filename] = []
            self.file_groups[filename].append(idx)
        
        # Create positive pairs (chunks from same file)
        self.pairs = []
        for filename, indices in self.file_groups.items():
            if len(indices) >= 2:
                for i in range(len(indices) - 1):
                    self.pairs.append((indices[i], indices[i + 1]))
        
        logger.info(f"Created dataset with {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        text1 = self.data[idx1]['content']
        text2 = self.data[idx2]['content']
        return text1, text2


class ContrastiveLoss(nn.Module):
    """Simple contrastive loss for similar code pairs."""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2):
        """
        Compute contrastive loss for positive pairs.
        
        Args:
            embedding1: First embeddings (batch_size, dim)
            embedding2: Second embeddings (batch_size, dim)
        """
        # For positive pairs, minimize distance
        distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        loss = torch.mean(distance)
        return loss


class PrivacyPreservingClient(fl.client.NumPyClient):
    """Flower client with Differential Privacy."""
    
    def __init__(self, model: CodeEmbedder, train_loader: DataLoader, 
                 device: str = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = ContrastiveLoss()
        
        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=LEARNING_RATE
        )
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=DP_NOISE_MULTIPLIER,
            max_grad_norm=DP_MAX_GRAD_NORM,
        )
        
        logger.info("Privacy engine initialized")
        logger.info(f"  Noise multiplier: {DP_NOISE_MULTIPLIER}")
        logger.info(f"  Max grad norm: {DP_MAX_GRAD_NORM}")
        logger.info(f"  Target delta: {DP_DELTA}")
    
    def get_parameters(self, config):
        """Return current model parameters."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """Update model with parameters from server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)
    
    def fit(self, parameters, config):
        """
        Train model locally with differential privacy.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration
            
        Returns:
            Updated parameters, number of examples, metrics
        """
        # Update local model with global parameters
        self.set_parameters(parameters)
        
        # Train for one epoch (local training)
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for text1_batch, text2_batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Get embeddings
            emb1 = self.model(text1_batch)
            emb2 = self.model(text2_batch)
            
            # Compute loss
            loss = self.criterion(emb1, emb2)
            
            # Backward pass with privacy
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate privacy budget spent
        epsilon = self.privacy_engine.get_epsilon(delta=DP_DELTA)
        
        logger.info(f"Training complete:")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Privacy spent: ε={epsilon:.2f} (δ={DP_DELTA})")
        
        # Warning if privacy budget exceeded
        if epsilon > DP_EPSILON_WARNING:
            logger.warning(f"⚠️  Privacy budget exceeded threshold ({DP_EPSILON_WARNING})!")
            logger.warning(f"   Current ε={epsilon:.2f}")
        
        # Return updated parameters
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"loss": avg_loss, "epsilon": epsilon}
        )
    
    def evaluate(self, parameters, config):
        """Evaluate model (not implemented for prototype)."""
        return 0.0, 0, {}


def load_training_data(data_dir: str = "./") -> DataLoader:
    """
    Load training data from metadata.
    
    Args:
        data_dir: Directory containing metadata
        
    Returns:
        DataLoader for training
    """
    metadata_path = Path(data_dir) / LOCAL_METADATA_PATH
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            "Run indexer.py first to build the index."
        )
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    logger.info(f"Loaded metadata with {len(metadata)} chunks")
    
    # Create dataset and dataloader
    dataset = CodePairDataset(metadata)
    
    # Use small batch size for DP
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    return dataloader


def main():
    """Main entry point for federated client."""
    parser = argparse.ArgumentParser(description="Start FL client with DP")
    parser.add_argument('--data-dir', default='./', help='Data directory')
    parser.add_argument('--server', default=FL_SERVER_ADDRESS, help='FL server address')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Starting Privacy-Preserving FL Client")
    logger.info("="*60)
    logger.info(f"Server: {args.server}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info("="*60)
    
    # Initialize model
    logger.info("Initializing model...")
    model = CodeEmbedder()
    
    # Load training data
    logger.info("Loading training data...")
    try:
        train_loader = load_training_data(args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Create client
    client = PrivacyPreservingClient(model, train_loader)
    
    # Connect to server
    logger.info(f"Connecting to FL server at {args.server}...")
    
    try:
        fl.client.start_numpy_client(
            server_address=args.server,
            client=client
        )
        
        logger.info("\n" + "="*60)
        logger.info("Federated training complete!")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nClient stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise


if __name__ == "__main__":
    main()