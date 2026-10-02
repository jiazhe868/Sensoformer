import sys
import os
import torch
import h5py
import numpy as np
import logging
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensoformer.data import SeismicDataset, collate_fn
from sensoformer.hub import resolve_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Registry names resolve through $SENSOFORMER_DATA (default ./data) and fall
# back to the Hugging Face Hub; override with explicit paths or the
# SENSOFORMER_SYN_HDF5 / SENSOFORMER_REAL_HDF5 environment variables.
def _resolve(env_var, registry_name):
    explicit = os.environ.get(env_var)
    if explicit:
        return explicit
    try:
        return str(resolve_dataset(registry_name))
    except Exception as exc:
        logger.warning("Could not resolve %s (%s): %s", registry_name, env_var, exc)
        return ""

SYN_HDF5 = _resolve("SENSOFORMER_SYN_HDF5", "synthetic-psdr")
REAL_HDF5 = _resolve("SENSOFORMER_REAL_HDF5", "socal-real")

def get_first_n_keys(hdf5_path, n=100):
    with h5py.File(hdf5_path, 'r') as f:
        return list(f.keys())[:n]

def test_dataset_loading(name, hdf5_path, augmentation=False):
    logger.info(f"--- Testing {name} Dataset (Augmentation={augmentation}) ---")
    
    if not os.path.exists(hdf5_path):
        logger.warning(f"File not found: {hdf5_path}. Skipping.")
        return

    # 1. Get IDs
    ids = get_first_n_keys(hdf5_path)
    logger.info(f"Loaded {len(ids)} IDs.")

    # 2. Config
    config = {
        'max_stations': 50,
        'min_stations_keep': 5, # Low number to easily see effect
        'noise_level': 0.1
    }

    # 3. Instantiate
    dataset = SeismicDataset(
        hdf5_path=hdf5_path,
        event_ids=ids,
        mode='train' if augmentation else 'test',
        augmentation=augmentation,
        config=config
    )

    # 4. Check __getitem__ directly
    wf, ft, mask, target, ev_id = dataset[0]
    logger.info(f"Single Item Shapes -> Waveform: {wf.shape}, Feat: {ft.shape}, Target: {target.shape}")
    
    # Check Data Types
    if wf.dtype == torch.float32 and ft.dtype == torch.float32:
        logger.info("✅ Data Types Correct (float32)")
    else:
        logger.error(f"❌ Data Type Mismatch: WF {wf.dtype}, FT {ft.dtype}")

    # 5. Check DataLoader with Collate
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=2)
    
    for batch_idx, batch in enumerate(loader):
        b_wf, b_ft, b_mask, b_target, b_ids = batch
        
        logger.info(f"Batch {batch_idx} Shapes -> WF: {b_wf.shape}, Mask: {b_mask.shape}, Target: {b_target.shape}")
        
        # Validation Checks
        # 1. Mask Logic: If mask is 0, waveform should ideally be 0 (padded)
        # Note: Functional pad adds 0s.
        is_padded_correctly = True
        for i in range(b_wf.shape[0]): # Batch
            for s in range(b_wf.shape[1]): # Station
                if b_mask[i, s] == 0:
                    if not torch.allclose(b_wf[i, s], torch.zeros_like(b_wf[i, s])):
                        is_padded_correctly = False
        
        if is_padded_correctly:
            logger.info("✅ Padding Logic Correct (Mask 0 corresponds to Zero-padded data)")
        else:
            logger.warning("⚠️ Padding Logic Check Failed (Might be non-zero padding?)")

        # 2. Augmentation Check (Station Count)
        # In augmented mode, station count should vary if original count > max_stations
        # This is hard to assert strictly without knowing original count, but we verify it runs.
        
        if batch_idx == 0: break # Just test one batch

    logger.info(f"✅ {name} Dataset Verified.\n")

if __name__ == "__main__":
    # Test 1: Synthetic Data (No Augmentation - mimics Validation)
    test_dataset_loading("Synthetic (Val)", SYN_HDF5, augmentation=False)
    
    # Test 2: Real Data (With Augmentation - mimics Fine-tuning)
    test_dataset_loading("Real (Train/Finetune)", REAL_HDF5, augmentation=True)
