"""
Federated Learning Server
Coordinates distributed training using Flower framework with FedAvg strategy
"""

import sys
import os
from pathlib import Path
import logging
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_core.config import (
    FL_SERVER_PORT, 
    FL_NUM_ROUNDS, 
    FL_MIN_CLIENTS,
    FL_FRACTION_FIT,
    GLOBAL_MODEL_PATH
)
from shared_core.model import CodeEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCheckpointer:
    """Save model after each federated round."""
    
    def __init__(self, model_path: str = GLOBAL_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model checkpoints will be saved to: {self.model_path}")
    
    def save_model(self, parameters: Parameters, round_num: int):
        """
        Save aggregated model parameters.
        
        Args:
            parameters: Federated model parameters
            round_num: Current training round
        """
        try:
            # Convert parameters to PyTorch state dict format
            # Note: This is simplified - in production you'd need proper conversion
            param_dict = {}
            for i, param in enumerate(parameters.tensors):
                param_dict[f'layer_{i}'] = torch.tensor(param)
            
            # Save checkpoint
            torch.save(param_dict, self.model_path)
            logger.info(f"Round {round_num}: Saved global model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")


def create_strategy(checkpointer: ModelCheckpointer) -> FedAvg:
    """
    Create Federated Averaging strategy with callbacks.
    
    Args:
        checkpointer: Model checkpointer instance
        
    Returns:
        Configured FedAvg strategy
    """
    
    def on_fit_config(server_round: int):
        """Configure client training for each round."""
        config = {
            "server_round": server_round,
            "local_epochs": 1,
        }
        return config
    
    def aggregate_fit(server_round, results, failures):
        """Aggregate model updates and save checkpoint."""
        # Call default aggregation
        aggregated_parameters, aggregated_metrics = FedAvg.aggregate_fit(
            None, server_round, results, failures
        )
        
        if aggregated_parameters:
            # Save checkpoint after aggregation
            checkpointer.save_model(aggregated_parameters, server_round)
            
            # Log metrics
            if aggregated_metrics:
                logger.info(f"Round {server_round} metrics: {aggregated_metrics}")
        
        return aggregated_parameters, aggregated_metrics
    
    # Create strategy with custom callbacks
    strategy = FedAvg(
        fraction_fit=FL_FRACTION_FIT,
        fraction_evaluate=0.0,  # No federated evaluation for this prototype
        min_fit_clients=FL_MIN_CLIENTS,
        min_available_clients=FL_MIN_CLIENTS,
        on_fit_config_fn=on_fit_config,
        # Note: aggregate_fit customization requires subclassing in newer Flower versions
    )
    
    return strategy


class CustomFedAvg(FedAvg):
    """Custom FedAvg with model checkpointing."""
    
    def __init__(self, checkpointer: ModelCheckpointer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpointer = checkpointer
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate and save model."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters:
            self.checkpointer.save_model(aggregated_parameters, server_round)
            
            # Log aggregation info
            logger.info(f"Round {server_round} complete:")
            logger.info(f"  - Clients participated: {len(results)}")
            logger.info(f"  - Failures: {len(failures)}")
            
            if aggregated_metrics:
                logger.info(f"  - Metrics: {aggregated_metrics}")
        
        return aggregated_parameters, aggregated_metrics


def main():
    """Start Federated Learning server."""
    logger.info("="*60)
    logger.info("Starting Federated Learning Server")
    logger.info("="*60)
    logger.info(f"Server address: 0.0.0.0:{FL_SERVER_PORT}")
    logger.info(f"Number of rounds: {FL_NUM_ROUNDS}")
    logger.info(f"Min clients required: {FL_MIN_CLIENTS}")
    logger.info("="*60)
    
    # Initialize checkpointer
    checkpointer = ModelCheckpointer()
    
    # Create strategy with checkpointing
    strategy = CustomFedAvg(
        checkpointer=checkpointer,
        fraction_fit=FL_FRACTION_FIT,
        fraction_evaluate=0.0,
        min_fit_clients=FL_MIN_CLIENTS,
        min_available_clients=FL_MIN_CLIENTS,
    )
    
    # Start Flower server
    try:
        fl.server.start_server(
            server_address=f"0.0.0.0:{FL_SERVER_PORT}",
            config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS),
            strategy=strategy,
        )
        
        logger.info("\n" + "="*60)
        logger.info("Federated Learning Complete!")
        logger.info(f"Final model saved to: {GLOBAL_MODEL_PATH}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()