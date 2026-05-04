import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import logging
from pathlib import Path
import os
import sys
from tqdm import tqdm
import json
from sensoformer.utils.visualization import plot_learning_curve

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sensoformer.models.network import Sensoformer
from sensoformer.data.dataset import SeismicDataset, collate_fn
from sensoformer.utils.metrics import FocalLossForRegression

log = logging.getLogger(__name__)

def create_weighted_sampler(hdf5_path, event_ids):
    """
    Creates a WeightedRandomSampler to balance the Moment Tensor distribution.
    This is critical for handling the 'Rare Event' problem in seismology.
    """
    log.info("Computing sampler weights based on MT distribution...")
    params = []
    
    # Efficiently read attributes without loading waveforms
    with h5py.File(hdf5_path, 'r') as h5f:
        for eid in event_ids:
            attrs = h5f[eid].attrs
            # Collecting normalized MT components
            params.append([
                attrs.get('Mxx', 0), attrs.get('Myy', 0), 
                attrs.get('Mxy', 0), attrs.get('Mxz', 0), attrs.get('Myz', 0)
            ])
            
    params = np.array(params)
    
    # Create 5D Histogram bins
    bins = [np.linspace(-1, 1, 10) for _ in range(5)]
    hist_nd, _ = np.histogramdd(params, bins=bins)
    
    # Calculate weights (Inverse Frequency)
    # Add epsilon to avoid division by zero
    bin_weights = 1.0 / (hist_nd + 1e-6)
    
    # Assign weight to each sample
    sample_weights = []
    for i in range(len(params)):
        indices = []
        for dim in range(5):
            # Find bin index for each component
            idx = np.digitize(params[i, dim], bins[dim]) - 1
            idx = np.clip(idx, 0, 8) # Clip to valid bin range
            indices.append(idx)
        
        weight = bin_weights[tuple(indices)]
        sample_weights.append(weight)
        
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    log.info("Sampler created successfully.")
    return sampler

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 0. Global Setup
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    log.info(f"Run ID: {output_dir.name}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Data Splitting (Reproducible)
    data_path = cfg.data.path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HDF5 file not found at: {data_path}")

    log.info("Reading HDF5 keys...")
    with h5py.File(data_path, 'r') as h5f:
        all_ids = list(h5f.keys())
    
    val_ratio = cfg.training.get('val_size', 0.1)
    test_ratio = cfg.training.get('test_size', 0.1)
    
    # Train/Temp Split
    train_ids, temp_ids = train_test_split(
        all_ids, 
        test_size=(val_ratio + test_ratio), 
        random_state=cfg.seed
    )
    # Val/Test Split
    val_ids, test_ids = train_test_split(
        temp_ids, 
        test_size=(test_ratio / (val_ratio + test_ratio)), 
        random_state=cfg.seed
    )
    
    log.info(f"Data Split: {len(train_ids)} Train, {len(val_ids)} Val, {len(test_ids)} Test")

    # 2. Dataset Initialization
    # Pass augmentation params from config
    train_set = SeismicDataset(
        data_path, train_ids, mode='train', 
        augmentation=True, config=cfg.data.aug_params
    )
    val_set = SeismicDataset(
        data_path, val_ids, mode='val', 
        augmentation=False, config=cfg.data.aug_params
    )

    # 3. Sampler Strategy
    sampler = None
    shuffle = True
    if cfg.training.type == 'finetune':
        # Apply weighted sampling for real data to balance rare mechanisms
        try:
            sampler = create_weighted_sampler(data_path, train_ids)
            shuffle = False # Sampler implies shuffle=False
        except Exception as e:
            log.warning(f"Failed to create sampler: {e}. Falling back to standard shuffling.")

    # 4. DataLoaders
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.data.batch_size, 
        shuffle=shuffle, 
        sampler=sampler,
        collate_fn=collate_fn, 
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg.data.batch_size, 
        collate_fn=collate_fn, 
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    # 5. Model Initialization
    log.info(f"Initializing Model: {cfg.model.name}")
    model = Sensoformer(cfg)
    
    # 6. Mode Handling: Pretrain vs Finetune
    if cfg.training.type == 'finetune':
        ckpt = cfg.training.pretrained_ckpt
        log.info(f"Finetuning Mode. Loading weights from: {ckpt}")
        
        if os.path.exists(ckpt):
            state_dict = torch.load(ckpt, map_location='cpu')
            
            # Handle DataParallel keys (strip 'module.' prefix if present)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict, strict=False)
            log.info("Weights loaded successfully.")
        else:
            log.error(f"Checkpoint file {ckpt} not found! Aborting.")
            raise FileNotFoundError(f"Checkpoint {ckpt} missing.")

        # Differential Learning Rates
        optimizer = torch.optim.AdamW([
            # Backbone Group 
            {'params': model.station_encoder.parameters(), 'lr': cfg.training.lr_backbone},
            {'params': model.event_aggregator.parameters(), 'lr': cfg.training.lr_backbone},
            {'params': model.attention_pooling.parameters(), 'lr': cfg.training.lr_backbone}, # Added pooling layer
            # Head Group 
            {'params': model.moment_tensor_head.parameters(), 'lr': cfg.training.lr_head},
            {'params': model.magnitude_head.parameters(), 'lr': cfg.training.lr_head}
        ], weight_decay=cfg.training.weight_decay)
        
        # Loss: Focal Loss for better handling of outliers
        criterion_mt = FocalLossForRegression(gamma=cfg.training.focal_gamma)
        
    else:
        # Pretraining Mode
        log.info("Pretraining Mode. Starting from scratch.")
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.training.lr, 
            weight_decay=cfg.training.weight_decay
        )
        # Loss: Standard MSE
        criterion_mt = nn.MSELoss()

    model = model.to(device)
    
    # Wrap in DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion_mag = nn.MSELoss()

    # 7. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_path = output_dir / cfg.training.ckpt_name
    history = {
        'train_loss': [],
        'val_loss': []
    }
    log.info(f"Starting training for {cfg.training.epochs} epochs...")
    
    for epoch in range(cfg.training.epochs):
        # --- TRAIN STEP ---
        model.train()
        train_loss = 0.0
        
        # Tqdm bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]")
        for batch in pbar:
            wf, ft, mask, target, _ = batch
            wf = wf.to(device, non_blocking=True)
            ft = ft.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward
            preds, _ = model(wf, ft, mask)
            
            # Loss Calculation
            # Preds: [Mag, Mxx, Myy, Mxy, Mxz, Myz]
            l_mag = criterion_mag(preds[:, 0], target[:, 0])
            l_mt = criterion_mt(preds[:, 1:], target[:, 1:])
            
            loss = l_mag + l_mt
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION STEP ---
        model.eval()
        val_loss = 0.0
        val_loss_mag = 0.0
        val_loss_mt = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                wf, ft, mask, target, _ = batch
                wf = wf.to(device)
                ft = ft.to(device)
                mask = mask.to(device)
                target = target.to(device)
                
                preds, _ = model(wf, ft, mask)
                
                # Validation Metric (Keep consistent with Training Loss for tracking)
                l_m = criterion_mag(preds[:, 0], target[:, 0])
                l_t = criterion_mt(preds[:, 1:], target[:, 1:])
                
                val_loss += (l_m + l_t).item()
                val_loss_mag += l_m.item()
                val_loss_mt += l_t.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mag = val_loss_mag / len(val_loader)
        avg_val_mt = val_loss_mt / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        log.info(
            f"Epoch {epoch+1} Summary: "
            f"Train Loss={avg_train_loss:.4f} | "
            f"Val Loss={avg_val_loss:.4f} (Mag={avg_val_mag:.4f}, MT={avg_val_mt:.4f})"
        )

        # --- EARLY STOPPING & SAVING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            # Save logic for DataParallel
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, save_path)
            log.info(f"⭐️ New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            log.info(f"No improvement. Patience: {epochs_no_improve}/{cfg.training.patience}")
            
            if epochs_no_improve >= cfg.training.patience:
                log.info("⏹ Early stopping triggered.")
                break

    log.info("Saving training history...")
    
    # Save raw history data as JSON
    history_path = output_dir / "loss_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Generate Plot (start from epoch 3)
    plot_learning_curve(
        history, 
        save_path=output_dir / "learning_curve.png",
        start_epoch=3 
    )

    log.info("Training complete.")

if __name__ == "__main__":
    main()