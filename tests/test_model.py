import torch
import pytest
from omegaconf import OmegaConf
from sensoformer.models.network import Sensoformer

@pytest.fixture
def model_cfg():
    """Mock configuration for the model."""
    # Returns a DictConfig via OmegaConf
    return OmegaConf.create({
        'model': {
            'in_channels': 6,
            'num_scalar_features': 20,
            'embed_dim': 32, 
            'heads': 2,
            'dropout': 0.1,
            'layers': 1
        }
    })

def test_sensoformer_instantiation(model_cfg):
    # This should now work because Sensoformer.__init__ handles the config object
    model = Sensoformer(model_cfg)
    assert isinstance(model, torch.nn.Module)
    assert model.embed_dim == 32

def test_sensoformer_forward_pass(model_cfg):
    model = Sensoformer(model_cfg)
    
    # Dummy Batch
    B, S, C, L = 2, 5, 12, 100
    wf = torch.randn(B, S, C, L)
    ft = torch.randn(B, S, 20)
    mask = torch.ones(B, S)
    
    preds, weights = model(wf, ft, mask)
    
    assert preds.shape == (B, 6)
    assert weights.shape == (B, S)