import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import logging
from pathlib import Path
import os
import sys
from tqdm import tqdm

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sensoformer.models.network import Sensoformer
from sensoformer.data.dataset import SeismicDataset, collate_fn
from sensoformer.ext import MTDecomposer
from sensoformer.utils.visualization import (
    plot_scatter_matrix, 
    plot_beachball_comparison, 
    plot_kagan_histogram
)
from sensoformer.utils.physics import kagan_angle 
from sensoformer.utils.metrics import FocalLossForRegression

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    
    # 0. Environment Setup
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    # Output directory is handled by Hydra (outputs/YYYY-MM-DD/HH-MM-SS)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Inference Run ID: {output_dir.name}")
    
    # 1. Recreate Data Split 
    data_path = cfg.data.path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HDF5 file not found: {data_path}")

    log.info("Recreating data splits to identify Test Set...")
    with h5py.File(data_path, 'r') as h5f:
        all_ids = list(h5f.keys())
    
    val_ratio = cfg.training.get('val_size', 0.1)
    test_ratio = cfg.training.get('test_size', 0.1)
    
    # Deterministic split
    _, temp_ids = train_test_split(all_ids, test_size=(val_ratio + test_ratio), random_state=cfg.seed)
    _, test_ids = train_test_split(temp_ids, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=cfg.seed)
    
    log.info(f"Found {len(test_ids)} events in Test Set.")

    # 2. Dataset & Loader
    # Note: augmentations are False for inference
    test_set = SeismicDataset(
        data_path, test_ids, mode='test', 
        augmentation=False, config=cfg.data.aug_params
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.data.batch_size, 
        collate_fn=collate_fn, 
        num_workers=4
    )

    # 3. Load Model & Weights
    log.info(f"Initializing Model: {cfg.model.name}")
    model = Sensoformer(cfg)
    
    # Determine Checkpoint Path
    if 'model_path' in cfg and cfg.model_path:
        ckpt_path = cfg.model_path
    else:
        # Fallback: Assume we are running in the same dir structure or user points config to it
        ckpt_path = cfg.training.get('pretrained_ckpt', None)
        if not ckpt_path:
            # Try finding local best model if running inside a training output dir
            ckpt_path = cfg.training.ckpt_name

    log.info(f"Loading weights from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Load State Dict (Handle DataParallel & Mismatches)
    state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # Strip 'module.'
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_preds = []
    all_targets = []
    all_ids = []
    total_loss = 0.0
    
    # Loss setup for evaluation metric
    criterion_mag = nn.MSELoss()
    # Use Focal Loss if finetuning, else MSE, or just MSE for standard reporting
    criterion_mt = FocalLossForRegression(gamma=1.5) if cfg.training.type == 'finetune' else nn.MSELoss()

    log.info("Starting Inference Loop...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            wf, ft, mask, target, ids = batch
            wf = wf.to(device)
            ft = ft.to(device)
            mask = mask.to(device)
            target = target.to(device)
            
            # Forward
            preds, _ = model(wf, ft, mask)
            
            # Loss Calculation (for reporting)
            l_mag = criterion_mag(preds[:, 0], target[:, 0])
            l_mt = criterion_mt(preds[:, 1:], target[:, 1:])
            total_loss += (l_mag + l_mt).item()
            
            # Store results (CPU)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_ids.extend(ids)
            
    avg_loss = total_loss / len(test_loader)
    log.info(f"Final Test Loss: {avg_loss:.4f}")

    # 5. Data Post-Processing
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Denormalize Magnitude
    # Assumption: Normalized to [-1, 1] from range [2.0, 8.0]
    # x_norm = 2 * (x - min) / (max - min) - 1
    # x = (x_norm + 1) / 2 * (max - min) + min
    MIN_MAG, MAX_MAG = 2.0, 8.0
    
    pred_mag = (all_preds[:, 0] + 1) / 2 * (MAX_MAG - MIN_MAG) + MIN_MAG
    true_mag = (all_targets[:, 0] + 1) / 2 * (MAX_MAG - MIN_MAG) + MIN_MAG
    
    mag_mae = np.mean(np.abs(pred_mag - true_mag))
    log.info(f"Magnitude MAE: {mag_mae:.4f}")
    
    # Combine denormalized mag with raw MT components (MT is already normalized -1 to 1)
    plot_preds = np.column_stack([pred_mag, all_preds[:, 1:]])
    plot_targets = np.column_stack([true_mag, all_targets[:, 1:]])
    
    log.info("Calculating Kagan Angles for full test set...")
    try:
        decomposer = MTDecomposer()
        all_kagans = []
        
        # Helper to pick primary plane
        def get_sdr(mt):
            sdr1, sdr2 = decomposer.mt_to_sdr(mt)
            if -90 <= sdr1[2] <= 90: return sdr1
            return sdr2

        # Loop through all samples (CPU calculation is fast enough for ~10k items)
        for i in tqdm(range(len(plot_targets)), desc="Calc Metrics"):
            mt_t = plot_targets[i, 1:] # Skip mag
            mt_p = plot_preds[i, 1:]
            
            sdr_t = get_sdr(mt_t)
            sdr_p = get_sdr(mt_p)
            
            # Unpack for physics function
            k = kagan_angle(*sdr_t, *sdr_p)
            all_kagans.append(k)
            
        all_kagans = np.array(all_kagans)
        log.info(f"Mean Kagan: {np.mean(all_kagans):.2f}, Median: {np.median(all_kagans):.2f}")

        plot_kagan_histogram(
            all_kagans, 
            save_path=output_dir / "kagan_histogram.pdf"
        )
        
    except Exception as e:
        log.error(f"Metric calculation failed: {e}")

    # 6. Visualization Generation
    
    # A. Scatter Matrix
    log.info("Generating Scatter Plots...")
    param_names = ['Mw', 'Mxx', 'Myy', 'Mxy', 'Mxz', 'Myz']
    plot_scatter_matrix(
        plot_preds, 
        plot_targets, 
        param_names, 
        event_ids=all_ids,
        save_path=output_dir / "scatter_results.pdf"
    )
    
    # B. Beachball Comparison
    log.info("Generating Beachball Comparison (This may take a moment)...")
    try:
        decomposer = MTDecomposer() # Requires src/sensoformer/ext/mtdcmp.so
        
        # Pass indices 1:6 for MT components (skipping magnitude)
        plot_beachball_comparison(
            mt_true=plot_targets[:, 1:],
            mt_pred=plot_preds[:, 1:],
            event_ids=all_ids,
            magnitudes=true_mag,
            decomposer=decomposer,
            num_samples=100,
            save_path=output_dir / "beachball_grid.pdf"
        )
    except Exception as e:
        log.error(f"Failed to generate beachballs. Is mtdcmp.so compiled? Error: {e}")

    log.info(f"✅ Inference Complete. All results saved to: {output_dir}")

if __name__ == "__main__":
    main()