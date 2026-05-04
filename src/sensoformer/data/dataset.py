import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SeismicDataset(Dataset):
    """
    Unified dataset for Seismic Waveforms (Synthetic & Real).
    Handles dynamic station sampling, augmentation, and data normalization.
    """

    def __init__(
        self, 
        hdf5_path: str, 
        event_ids: List[str], 
        mode: str = 'train',
        augmentation: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            hdf5_path: Path to the HDF5 file.
            event_ids: List of event IDs to include.
            mode: 'train', 'val', or 'test'.
            augmentation: Whether to apply data augmentation (noise, dropouts).
            config: Dictionary containing hyperparameters. 
                    e.g. {'max_stations': 50, 'noise_level': 0.05}
        """
        self.hdf5_path = hdf5_path
        self.event_ids = event_ids
        self.mode = mode
        self.augmentation = augmentation
        
        # Default Configuration
        self.cfg = config or {}
        self.max_stations = self.cfg.get('max_stations', 50)
        self.min_stations_keep = self.cfg.get('min_stations_keep', 30)
        self.noise_level = self.cfg.get('noise_level', 0.05)
        
        self.h5f: Optional[h5py.File] = None
        
        logger.info(f"Dataset init: {len(self.event_ids)} events. Mode={mode}, Aug={augmentation}")

    def __len__(self) -> int:
        return len(self.event_ids)

    def _open_h5(self):
        """
        Lazy loading of HDF5 file handler. 
        Essential for PyTorch DataLoader with num_workers > 0.
        """
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_path, 'r')

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        self._open_h5()
        event_id = self.event_ids[idx]
        
        try:
            group = self.h5f[event_id]
            
            # 1. Load Raw Data
            # Shapes: Waveforms (N_sta, 12, Time), Features (N_sta, Feat_dim)
            waveforms = group['waveforms'][:]  
            features = group['features'][:]    
            
            # 2. Station Selection Logic
            total_stations = waveforms.shape[0]
            indices = np.arange(total_stations)

            if self.mode == 'train' and self.augmentation:
                # [Training Strategy]: 
                # If we have many stations, randomly sample a subset to force permutation invariance.
                # If we have few stations, keep them all (or handle min requirements).
                if total_stations > self.max_stations:
                    # Randomly drop stations down to a range [min_keep, max_stations]
                    # This simulates network variability.
                    num_keep = np.random.randint(self.min_stations_keep, self.max_stations + 1)
                    indices = np.random.choice(total_stations, size=num_keep, replace=False)
                # Note: If total_stations < max_stations, we currently keep all. 
                # You could add logic to drop even fewer stations if desired.
                
            else:
                # [Val/Test Strategy]: Deterministic Behavior
                # If there are too many stations, take the first N (or highest SNR if sorted).
                # For reproducibility, we usually slice deterministically.
                if total_stations > self.max_stations:
                    indices = np.arange(self.max_stations)

            # Apply selection
            waveforms = waveforms[indices]
            features = features[indices]

            # 3. Waveform Augmentation (Noise Injection)
            if self.mode == 'train' and self.augmentation:
                signal_std = waveforms.std()
                if signal_std > 0:
                    # Scale noise by signal amplitude
                    noise_scale = self.noise_level * np.random.rand() * signal_std
                    noise = np.random.normal(0, noise_scale, waveforms.shape)
                    waveforms += noise

            # 4. Target Extraction & Normalization
            attrs = group.attrs
            
            # Magnitude Normalization: Map [2.0, 8.0] -> [-1.0, 1.0]
            # (val - min) / (max - min) * 2 - 1
            mag = attrs['magnitude']
            scaled_mag = 2 * (mag - 2.0) / (8.0 - 2.0) - 1.0
            
            # Moment Tensor Components (Assuming normalized Mxx..Myz)
            # Order: Mag, Mxx, Myy, Mxy, Mxz, Myz
            mt_keys = ['Mxx', 'Myy', 'Mxy', 'Mxz', 'Myz']
            mt_values = [attrs[k] for k in mt_keys]
            
            targets = np.array([scaled_mag] + mt_values, dtype=np.float32)

            # 5. Return Tensors
            # Note: Mask is generated in collate_fn, returning placeholder here
            return (
                torch.from_numpy(waveforms.astype(np.float32)),
                torch.from_numpy(features.astype(np.float32)),
                torch.zeros(waveforms.shape[0]), # Placeholder for mask
                torch.from_numpy(targets),
                str(event_id)
            )
            
        except Exception as e:
            logger.error(f"Error loading event {event_id}: {e}")
            # Return a dummy sample or raise, depending on preference. 
            # Raising allows DataLoader to catch it if configured, or crash early.
            raise e

def collate_fn(batch):
    """
    Custom collate function to handle variable number of stations (Padding).
    """
    # Sort by number of stations (descending) for efficiency (optional but good for pack_padded_sequence)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    waveforms, features, _, targets, event_ids = zip(*batch)
    
    # Find max stations in this batch
    max_stations = waveforms[0].shape[0]
    
    padded_waveforms = []
    padded_features = []
    attention_masks = []
    
    for wf, feat in zip(waveforms, features):
        num_stations = wf.shape[0]
        padding_len = max_stations - num_stations
        
        # Pad Waveforms: (Station, Channel, Time) -> Pad last dim of station
        # Torch pad syntax: (last_dim_left, last_dim_right, 2nd_last_left, ...)
        # We want to pad dim 0 (Stations). 
        # Since W is 3D, we need 6 values. 
        # Time(0,0), Channel(0,0), Station(0, padding_len)
        padded_wf = torch.nn.functional.pad(wf, (0, 0, 0, 0, 0, padding_len))
        padded_waveforms.append(padded_wf)
        
        # Pad Features: (Station, Feat)
        padded_ft = torch.nn.functional.pad(feat, (0, 0, 0, padding_len))
        padded_features.append(padded_ft)
        
        # Create Mask: 1 for Real, 0 for Padding
        mask = torch.zeros(max_stations, dtype=torch.float32)
        mask[:num_stations] = 1.0
        attention_masks.append(mask)

    return (
        torch.stack(padded_waveforms),
        torch.stack(padded_features),
        torch.stack(attention_masks),
        torch.stack(targets),
        list(event_ids)
    )
