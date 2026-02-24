import pytest
import os
import torch
import numpy as np
from pathlib import Path
from shared_core.model import CodeEmbedder
from shared_core.config import GLOBAL_MODEL_PATH

def test_checkpoint_persistence_and_recovery():
    """
    Scenario: Ensure the FL Server saves progress and can resume.
    Expectation: Reloaded weights match the last saved state.
    """
    # Setup - Create a model and modify weights to simulate "Training Progress"
    model = CodeEmbedder()
    original_weights = [p.clone() for p in model.parameters()]
    
    # Manually change one weight value to simulate a Round 3 update
    with torch.no_grad():
        list(model.parameters())[0].fill_(0.5)
    
    updated_weights = [p.clone() for p in model.parameters()]
    
    # Save logic (Simulates Server at the end of Round 3)
    model.save_weights(GLOBAL_MODEL_PATH)
    assert Path(GLOBAL_MODEL_PATH).exists()

    # Simulate "Kill Process" & "Restart Server"
    # We create a fresh model instance (starts from scratch)
    new_model = CodeEmbedder()
    
    # Verify fresh model doesn't have the updated weights yet
    assert not torch.equal(list(new_model.parameters())[0], list(model.parameters())[0])

    # Load logic (Expected Outcome)
    if Path(GLOBAL_MODEL_PATH).exists():
        new_model.load_weights(GLOBAL_MODEL_PATH)
    
    # Final Assertion: Round 3 weights recovered?
    recovered_weight = list(new_model.parameters())[0]
    expected_weight = list(model.parameters())[0]
    
    assert torch.equal(recovered_weight, expected_weight), "Model did not resume from checkpoint!"
    print(f"\nCheckpoint Recovery Successful: Weights match Round 3 state.")