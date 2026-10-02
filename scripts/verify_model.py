import sys
import os
import torch
import logging

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensoformer.models.network import Sensoformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_sensoformer():
    logger.info("Initializing Model Verification...")
    
    # 1. Define hyperparameters (Simulating the Hydra Config Structure)
    # The new Sensoformer expects a dict with a 'model' key containing specific field names
    mock_cfg = {
        'model': {
            'in_channels': 6,           
            'num_scalar_features': 20,
            'embed_dim': 128,           # Renamed: station_feature_dim -> embed_dim
            'heads': 4,                 # Renamed: num_heads -> heads
            'layers': 2,                # Renamed: num_layers -> layers
            'dropout': 0.1
        }
    }
    
    # 2. Instantiate Model
    try:
        # Pass the config dictionary directly, NOT unpacked (**params)
        model = Sensoformer(mock_cfg)
        model.eval() # Set to eval mode
        logger.info("✅ Model instantiated successfully.")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model Parameter Count:")
        logger.info(f"   Total:     {total_params:,}")
        logger.info(f" ✅ Trainable: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"❌ Model instantiation failed: {e}")
        # Print detailed traceback for debugging
        import traceback
        traceback.print_exc()
        return

    # 3. Create Dummy Inputs
    BATCH_SIZE = 32
    NUM_STATIONS = 50
    CHANNELS = 12
    LENGTH = 101 # e.g., 20 seconds @ 10Hz
    
    # Waveforms: (B, S, 12, L)
    dummy_waveforms = torch.randn(BATCH_SIZE, NUM_STATIONS, CHANNELS, LENGTH)
    
    # Scalar Features: (B, S, 20)
    dummy_features = torch.randn(BATCH_SIZE, NUM_STATIONS, mock_cfg['model']['num_scalar_features'])
    
    # Mask: (B, S). Let's simulate some padding.
    # Set the last 2 stations of the first event to 0 (padded)
    dummy_mask = torch.ones(BATCH_SIZE, NUM_STATIONS)
    dummy_mask[0, -2:] = 0 
    
    logger.info(f"Input Shapes: Waveforms {dummy_waveforms.shape}, Features {dummy_features.shape}")

    # 4. Forward Pass
    try:
        predictions, attn_weights = model(dummy_waveforms, dummy_features, dummy_mask)
        logger.info("✅ Forward pass successful.")
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Check Output Shapes
    expected_pred_shape = (BATCH_SIZE, 6) # Mag + 5 MT components
    expected_attn_shape = (BATCH_SIZE, NUM_STATIONS)
    
    if predictions.shape == expected_pred_shape:
        logger.info(f"✅ Prediction shape correct: {predictions.shape}")
    else:
        logger.error(f"❌ Prediction shape mismatch: Expected {expected_pred_shape}, Got {predictions.shape}")

    if attn_weights.shape == expected_attn_shape:
        logger.info(f"✅ Attention weights shape correct: {attn_weights.shape}")
    else:
        logger.error(f"❌ Attention weights shape mismatch: Expected {expected_attn_shape}, Got {attn_weights.shape}")

    # 6. Check Output Range (Tanh should be -1 to 1)
    if predictions.max() <= 1.0 and predictions.min() >= -1.0:
        logger.info("✅ Output range correct (within [-1, 1] due to Tanh).")
    else:
        logger.warning(f"⚠️ Output range suspicious: Max {predictions.max()}, Min {predictions.min()}")

    # 7. Check Masking Logic (Padded stations should have 0 attention)
    # Event 0, last 2 stations were masked.
    masked_attn = attn_weights[0, -2:]
    
    # Use torch.allclose for float comparison
    if torch.allclose(masked_attn, torch.zeros_like(masked_attn), atol=1e-6):
        logger.info("✅ Masking logic correct: Padded stations have 0 attention.")
    else:
        logger.error(f"❌ Masking logic failed: Padded stations have attention {masked_attn}")

if __name__ == "__main__":
    verify_sensoformer()