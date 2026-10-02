import os
import tempfile
import multiprocessing.util

# Permanently redirect multiprocessing & PyTorch IPC temporary directories to /dev/shm (RAM disk)
os.environ["TMPDIR"] = "/dev/shm"
os.environ["TEMP"] = "/dev/shm"
os.environ["TMP"] = "/dev/shm"
tempfile.tempdir = "/dev/shm"

def _redirect_pymp_tempdir():
    path = "/dev/shm/pymp_shared"
    os.makedirs(path, exist_ok=True)
    return path

multiprocessing.util.get_temp_dir = _redirect_pymp_tempdir

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import os
import sys
from tqdm import tqdm
import json

# Ensure src is in python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sensoformer.models.network import Sensoformer
from sensoformer.data.dataset import SeismicDataset, collate_fn
from sensoformer.utils.metrics import FocalLossForRegression, OrientationCosineLoss
from sensoformer.ext import MTDecomposer
from sensoformer.utils.physics import kagan_angle
from sensoformer.utils.visualization import (
    plot_learning_curve,
    plot_scatter_matrix,
    plot_beachball_comparison,
    plot_kagan_histogram
)

log = logging.getLogger(__name__)

def create_weighted_sampler(hdf5_path, event_ids):
    """
    Creates a WeightedRandomSampler to balance the Moment Tensor distribution.
    This is critical for handling the 'Rare Event' problem in seismology.
    """
    log.info("Computing sampler weights based on MT distribution...")
    params = []
    
    with h5py.File(hdf5_path, 'r') as h5f:
        for eid in event_ids:
            attrs = h5f[eid].attrs
            params.append([
                attrs.get('Mxx', 0), attrs.get('Myy', 0), 
                attrs.get('Mxy', 0), attrs.get('Mxz', 0), attrs.get('Myz', 0)
            ])
            
    params = np.array(params)
    
    bins = [np.linspace(-1, 1, 10) for _ in range(5)]
    hist_nd, _ = np.histogramdd(params, bins=bins)
    
    bin_weights = 1.0 / (hist_nd + 1e-6)
    
    sample_weights = []
    for i in range(len(params)):
        indices = []
        for dim in range(5):
            idx = np.digitize(params[i, dim], bins[dim]) - 1
            idx = np.clip(idx, 0, 8)
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
    torch.set_num_threads(4)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    log.info(f"Run ID: {output_dir.name}")
    log.info(f"Output Directory: {output_dir}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Save exact resolved config YAML in output dir
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 1. Data Splitting & Smoke Test Check
    data_path = cfg.data.path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HDF5 file not found at: {data_path}")

    log.info("Reading HDF5 keys...")
    with h5py.File(data_path, 'r') as h5f:
        all_ids = list(h5f.keys())
    
    is_smoke_test = cfg.get('smoke_test', False) or cfg.training.get('smoke_test', False)
    epochs = 2 if is_smoke_test else cfg.training.epochs
    patience = 2 if is_smoke_test else cfg.training.patience
    
    if is_smoke_test:
        log.info("🔥 [SMOKE TEST MODE] Truncating dataset to 20 events and setting epochs=2.")
        all_ids = all_ids[:20]

    val_ratio = cfg.training.get('val_size', 0.15)
    test_ratio = cfg.training.get('test_size', 0.15)
    
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
    
    # Optional label-efficiency mode: train on a random subset of the training
    # split (val/test untouched, so results stay comparable across fractions).
    train_fraction = float(cfg.training.get('train_fraction', 1.0))
    if train_fraction < 1.0:
        n_full = len(train_ids)
        train_ids, _ = train_test_split(
            train_ids,
            train_size=train_fraction,
            random_state=cfg.seed
        )
        log.info(f"Label-efficiency mode: {len(train_ids)}/{n_full} training "
                 f"events (train_fraction={train_fraction}).")

    log.info(f"Data Split: {len(train_ids)} Train, {len(val_ids)} Val, {len(test_ids)} Test")

    # 2. Dataset Initialization
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
    if cfg.training.type in ['finetune', 'real_from_scratch']:
        try:
            sampler = create_weighted_sampler(data_path, train_ids)
            shuffle = False
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

    # 5. Model Initialization (Dynamic choice: sensoformer, gnn, deepsets, deeponet)
    model_name = str(cfg.model.name).lower()
    log.info(f"Initializing Model Architecture: {model_name}")
    
    if model_name == "gnn":
        from sensoformer.models.gnn import GNNSensoformer
        model = GNNSensoformer(cfg)
    elif model_name == "deeponet":
        from sensoformer.models.deeponet import DeepONet
        model = DeepONet(cfg)
    elif model_name == "linear_baseline":
        from sensoformer.models.simple_baselines import LinearBaseline
        model = LinearBaseline(cfg)
    elif model_name == "mlp_baseline":
        from sensoformer.models.simple_baselines import MLPBaseline
        model = MLPBaseline(cfg)
    else:
        # Default for sensoformer and ablation_deepsets
        from sensoformer.models.network import Sensoformer
        model = Sensoformer(cfg)
    
    # 6. Mode Handling & Optimizer Setup
    if cfg.training.type in ['finetune', 'real_from_scratch']:
        ckpt = cfg.training.get('pretrained_ckpt', None)
        
        if ckpt and os.path.exists(ckpt):
            log.info(f"Finetuning Mode. Loading weights from: {ckpt}")
            state_dict = torch.load(ckpt, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            log.info("Weights loaded successfully.")
        else:
            log.info("Real-From-Scratch Mode. Full Sensoformer initialized with random weights; skipping synthetic pretraining.")

        lr_backbone = float(cfg.training.get('lr_backbone', cfg.training.get('lr', 1e-4)))
        lr_head = float(cfg.training.get('lr_head', cfg.training.get('lr', 1e-4)))
        
        log.info(f"Using Learning Rates -> Backbone: {lr_backbone:.2e}, Head: {lr_head:.2e}")
        
        if hasattr(model, 'station_encoder') and hasattr(model, 'event_aggregator'):
            optimizer = torch.optim.AdamW([
                {'params': model.station_encoder.parameters(), 'lr': lr_backbone},
                {'params': model.event_aggregator.parameters(), 'lr': lr_backbone},
                {'params': model.attention_pooling.parameters(), 'lr': lr_backbone},
                {'params': model.magnitude_head.parameters(), 'lr': lr_head},
                {'params': model.moment_tensor_head.parameters(), 'lr': lr_head}
            ], weight_decay=cfg.training.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr_backbone, 
                weight_decay=cfg.training.weight_decay
            )
        
        focal_gamma = cfg.training.get('focal_gamma', 1.5)
        criterion_mt = FocalLossForRegression(gamma=focal_gamma)
    else:
        log.info("Pretraining Mode.")
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.training.lr, 
            weight_decay=cfg.training.weight_decay
        )
        criterion_mt = nn.MSELoss()

    model = model.to(device)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion_mag = nn.MSELoss()

    # Optional scale-invariant orientation loss on the deviatoric MT
    # (smooth Kagan-angle surrogate; see OrientationCosineLoss). Off by
    # default: orientation_loss_weight=0 reproduces the standard objective.
    orientation_weight = float(cfg.training.get('orientation_loss_weight', 0.0))
    criterion_orient = OrientationCosineLoss() if orientation_weight > 0 else None
    if criterion_orient is not None:
        log.info(f"Orientation cosine loss enabled (weight={orientation_weight}).")

    # 7. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_path = output_dir / cfg.training.ckpt_name
    history = {
        'train_loss': [],
        'val_loss': []
    }
    log.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            wf, ft, mask, target, _ = batch
            wf = wf.to(device, non_blocking=True)
            ft = ft.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            preds, _ = model(wf, ft, mask)
            
            l_mag = criterion_mag(preds[:, 0], target[:, 0])
            l_mt = criterion_mt(preds[:, 1:], target[:, 1:])
            loss = l_mag + l_mt
            if criterion_orient is not None:
                loss = loss + orientation_weight * criterion_orient(
                    preds[:, 1:], target[:, 1:])

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation Step
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
                l_m = criterion_mag(preds[:, 0], target[:, 0])
                l_t = criterion_mt(preds[:, 1:], target[:, 1:])

                batch_val = l_m + l_t
                if criterion_orient is not None:
                    batch_val = batch_val + orientation_weight * criterion_orient(
                        preds[:, 1:], target[:, 1:])
                val_loss += batch_val.item()
                val_loss_mag += l_m.item()
                val_loss_mt += l_t.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mag = val_loss_mag / len(val_loader)
        avg_val_mt = val_loss_mt / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        log.info(
            f"Epoch {epoch+1:03d}/{epochs:03d} | "
            f"Train Loss={avg_train_loss:.4f} | "
            f"Val Loss={avg_val_loss:.4f} (Mag={avg_val_mag:.4f}, MT={avg_val_mt:.4f})"
        )

        # Early Stopping & Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, save_path)
            log.info(f"  --> New best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            log.info(f"  --> No improvement. Patience: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                log.info("⏹ Early stopping triggered.")
                break

    log.info("Saving training history...")
    history_path = output_dir / "loss_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    plot_learning_curve(
        history, 
        save_path=output_dir / "learning_curve.png",
        start_epoch=1 
    )

    # 8. Compute Validation-Set Kagan Angle on Best Saved Model (Validation-only evaluation)
    val_kagan_mean = None
    val_kagan_median = None
    if os.path.exists(save_path):
        if model_name == "gnn":
            from sensoformer.models.gnn import GNNSensoformer
            val_eval_model = GNNSensoformer(cfg)
        elif model_name == "deeponet":
            from sensoformer.models.deeponet import DeepONet
            val_eval_model = DeepONet(cfg)
        elif model_name == "linear_baseline":
            from sensoformer.models.simple_baselines import LinearBaseline
            val_eval_model = LinearBaseline(cfg)
        elif model_name == "mlp_baseline":
            from sensoformer.models.simple_baselines import MLPBaseline
            val_eval_model = MLPBaseline(cfg)
        else:
            from sensoformer.models.network import Sensoformer
            val_eval_model = Sensoformer(cfg)
        state_dict = torch.load(save_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
        val_eval_model.load_state_dict(new_state_dict)
        val_eval_model.to(device)
        val_eval_model.eval()

        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                wf, ft, mask, target, _ = batch
                wf = wf.to(device); ft = ft.to(device); mask = mask.to(device)
                preds, _ = val_eval_model(wf, ft, mask)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(target.cpu().numpy())
                
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        MIN_MAG, MAX_MAG = 2.0, 8.0
        val_pred_mag = (val_preds[:, 0] + 1) / 2 * (MAX_MAG - MIN_MAG) + MIN_MAG
        val_true_mag = (val_targets[:, 0] + 1) / 2 * (MAX_MAG - MIN_MAG) + MIN_MAG
        
        try:
            decomposer = MTDecomposer()
            val_kagans = []
            def get_sdr(mt):
                sdr1, sdr2 = decomposer.mt_to_sdr(mt)
                return sdr1 if -90 <= sdr1[2] <= 90 else sdr2

            for i in range(len(val_targets)):
                sdr_t = get_sdr(val_targets[i, 1:])
                sdr_p = get_sdr(val_preds[i, 1:])
                val_kagans.append(kagan_angle(*sdr_t, *sdr_p))
                
            val_kagans = np.array(val_kagans)
            val_kagan_mean = float(np.mean(val_kagans))
            val_kagan_median = float(np.median(val_kagans))
            log.info(f"Validation Kagan Angle -> Mean: {val_kagan_mean:.2f}°, Median: {val_kagan_median:.2f}°")
        except Exception as e:
            log.error(f"Validation Kagan calculation failed: {e}")

    # Create Summary JSON
    summary_metrics = {
        'seed': cfg.seed,
        'learning_rate': float(cfg.training.get('lr', 1e-4)),
        'best_val_loss': best_val_loss,
        'val_kagan_mean': val_kagan_mean,
        'val_kagan_median': val_kagan_median,
        'is_smoke_test': is_smoke_test
    }

    eval_test = cfg.training.get('eval_test', True)
    
    if eval_test and os.path.exists(save_path):
        log.info("=" * 60)
        log.info("Evaluating Best Checkpoint on Real-World Test Set...")
        log.info("=" * 60)
        
        test_set = SeismicDataset(
            data_path, test_ids, mode='test', 
            augmentation=False, config=cfg.data.aug_params
        )
        test_loader = DataLoader(
            test_set, 
            batch_size=cfg.data.batch_size, 
            collate_fn=collate_fn, 
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        
        all_preds = []
        all_targets = []
        all_test_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                wf, ft, mask, target, ids = batch
                wf = wf.to(device); ft = ft.to(device); mask = mask.to(device)
                preds, _ = val_eval_model(wf, ft, mask)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_test_ids.extend(ids)
                
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        MIN_MAG, MAX_MAG = 2.0, 8.0
        pred_mag = (all_preds[:, 0] + 1) / 2 * (MAX_MAG - MIN_MAG) + MIN_MAG
        true_mag = (all_targets[:, 0] + 1) / 2 * (MAX_MAG - MIN_MAG) + MIN_MAG
        
        mag_errors = np.abs(pred_mag - true_mag)
        mag_mae = float(np.mean(mag_errors))
        
        plot_preds = np.column_stack([pred_mag, all_preds[:, 1:]])
        plot_targets = np.column_stack([true_mag, all_targets[:, 1:]])
        
        test_kagans = []
        try:
            decomposer = MTDecomposer()
            def get_sdr(mt):
                sdr1, sdr2 = decomposer.mt_to_sdr(mt)
                return sdr1 if -90 <= sdr1[2] <= 90 else sdr2

            for i in range(len(plot_targets)):
                sdr_t = get_sdr(plot_targets[i, 1:])
                sdr_p = get_sdr(plot_preds[i, 1:])
                test_kagans.append(kagan_angle(*sdr_t, *sdr_p))
                
            test_kagans = np.array(test_kagans)
            kagan_mean = float(np.mean(test_kagans))
            kagan_median = float(np.median(test_kagans))
            
            plot_kagan_histogram(
                test_kagans, 
                save_path=output_dir / "kagan_histogram.pdf"
            )
        except Exception as e:
            log.error(f"Test Kagan calculation failed: {e}")
            test_kagans = [np.nan] * len(all_test_ids)
            kagan_mean, kagan_median = None, None

        # Plot Scatter & Beachballs
        param_names = ['Mw', 'Mxx', 'Myy', 'Mxy', 'Mxz', 'Myz']
        plot_scatter_matrix(
            plot_preds, plot_targets, param_names, 
            event_ids=all_test_ids, save_path=output_dir / "scatter_results.pdf"
        )
        try:
            plot_beachball_comparison(
                mt_true=plot_targets[:, 1:], mt_pred=plot_preds[:, 1:],
                event_ids=all_test_ids, magnitudes=true_mag,
                decomposer=decomposer, num_samples=100,
                save_path=output_dir / "beachball_grid.pdf"
            )
        except Exception as e:
            log.error(f"Beachball comparison failed: {e}")
            
        # Save Per-Event CSV Predictions
        df_predictions = pd.DataFrame({
            'event_id': all_test_ids,
            'true_Mw': true_mag,
            'true_Mxx': all_targets[:, 1],
            'true_Myy': all_targets[:, 2],
            'true_Mxy': all_targets[:, 3],
            'true_Mxz': all_targets[:, 4],
            'true_Myz': all_targets[:, 5],
            'pred_Mw': pred_mag,
            'pred_Mxx': all_preds[:, 1],
            'pred_Myy': all_preds[:, 2],
            'pred_Mxy': all_preds[:, 3],
            'pred_Mxz': all_preds[:, 4],
            'pred_Myz': all_preds[:, 5],
            'kagan_angle': test_kagans,
            'magnitude_error': mag_errors
        })
        
        csv_path = output_dir / "per_event_predictions.csv"
        df_predictions.to_csv(csv_path, index=False)
        log.info(f"Saved per-event predictions to: {csv_path}")

        summary_metrics.update({
            'test_magnitude_mae': mag_mae,
            'test_kagan_mean': kagan_mean,
            'test_kagan_median': kagan_median,
            'num_test_events': len(all_test_ids)
        })

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
        
    log.info(f"Saved metrics summary to: {summary_path}")
    log.info("Training and evaluation run complete.")

if __name__ == "__main__":
    main()