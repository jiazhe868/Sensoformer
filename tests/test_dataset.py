import pytest
import h5py
import torch
import numpy as np
import os
from sensoformer.data.dataset import SeismicDataset, collate_fn

@pytest.fixture
def temp_hdf5(tmp_path):
    """Creates a dummy HDF5 file for testing."""
    d = tmp_path / "test_data.hdf5"
    with h5py.File(d, 'w') as f:
        # Create 10 dummy events
        for i in range(10):
            grp = f.create_group(f"ev_{i}")
            # Random stations between 5 and 10
            n_sta = np.random.randint(5, 11)
            
            # Waveforms: (N, 12, 100)
            grp.create_dataset('waveforms', data=np.random.randn(n_sta, 12, 100))
            # Features: (N, 20)
            grp.create_dataset('features', data=np.random.randn(n_sta, 20))
            
            # Attributes
            grp.attrs['magnitude'] = 5.0
            for k in ['Mxx', 'Myy', 'Mxy', 'Mxz', 'Myz']:
                grp.attrs[k] = 0.5
                
    return str(d)

def test_dataset_loading(temp_hdf5):
    """Test basic __getitem__ functionality."""
    ids = [f"ev_{i}" for i in range(10)]
    dataset = SeismicDataset(temp_hdf5, ids, mode='test', augmentation=False)
    
    wf, ft, mask, target, eid = dataset[0]
    
    assert isinstance(wf, torch.Tensor)
    assert wf.shape[1] == 12 # Channels
    assert target.shape == (6,) # Mag + 5 MT

def test_collate_fn(temp_hdf5):
    """Test batching and padding logic."""
    ids = [f"ev_{i}" for i in range(4)]
    dataset = SeismicDataset(temp_hdf5, ids)
    
    # Manually collect a list to simulate DataLoader
    batch_list = [dataset[i] for i in range(4)]
    
    wf_batch, ft_batch, mask_batch, target_batch, ids_batch = collate_fn(batch_list)
    
    # 1. Check Batch Dimension
    assert wf_batch.shape[0] == 4
    
    # 2. Check Masking Logic
    # The mask should sum to the number of stations for that event
    real_station_counts = sorted([item[0].shape[0] for item in batch_list], reverse=True)
    mask_sums = mask_batch.sum(dim=1).tolist()
    
    assert mask_sums == real_station_counts