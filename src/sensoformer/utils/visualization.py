import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
import logging

# Local imports
from .physics import plot_beachball, kagan_angle
from ..ext import MTDecomposer 

logger = logging.getLogger(__name__)

# --- 1. Scatter Plots ---

def plot_scatter_matrix(
    preds: np.ndarray, 
    targets: np.ndarray, 
    param_names: List[str],
    event_ids: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Generates a scatter plot matrix comparing Predictions vs Ground Truth.
    """
    num_params = preds.shape[1]
    rows = int(np.ceil(np.sqrt(num_params)))
    cols = int(np.ceil(num_params / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4*rows))
    axes = axes.flatten()
    
    for i in range(num_params):
        ax = axes[i]
        x = targets[:, i]
        y = preds[:, i]
        
        name = param_names[i] if i < len(param_names) else f"Param {i}"
        
        if "Strike" in name:
            diff = np.abs(x - y)
            mae = np.mean(np.minimum(diff, 360 - diff))
        else:
            mae = np.mean(np.abs(x - y))
            
        ax.scatter(x, y, alpha=0.5, s=10)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]), 
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        
        ax.set_title(f'{name}\nMAE: {mae:.3f}')
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Pred {name}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')

        if event_ids is not None:
            errors = np.abs(x - y)
            top_k_indices = np.argsort(errors)[-5:] 
            # for idx in top_k_indices:
            #     ax.text(x[idx], y[idx], event_ids[idx], fontsize=7, color='red', rotation=15)

    for j in range(num_params, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved scatter matrix to {save_path}")
    return fig


# --- 2. Beachball Comparison ---

def plot_beachball_comparison(
    mt_true: np.ndarray,
    mt_pred: np.ndarray,
    event_ids: List[str],
    magnitudes: np.ndarray,
    decomposer: MTDecomposer,
    num_samples: int = 100,
    save_path: Optional[str] = None
):
    """
    Generates a grid of True (Black) vs Pred (Blue) Beachballs.
    """
    n_total = len(mt_true)
    if n_total > num_samples:
        indices = np.random.choice(n_total, num_samples, replace=False)
    else:
        indices = np.arange(n_total)
        num_samples = n_total
    
    cols = 10
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2.0, rows * 2.2))
    if rows == 1: axes = axes.reshape(1, -1)
    
    # Pre-define helper to pick plane
    def select_plane(pair):
        sdr1, sdr2 = pair
        # Heuristic: keep plane with rake closer to 0 (optional) or just consistent check
        if -90 <= sdr1[2] <= 90: return sdr1
        return sdr2

    for idx, i in enumerate(tqdm(indices, desc="Plotting Beachballs")):
        row = idx // cols
        col_true = (idx % cols) * 2
        col_pred = col_true + 1
        
        sdr_t_pair = decomposer.mt_to_sdr(mt_true[i])
        sdr_p_pair = decomposer.mt_to_sdr(mt_pred[i])
        
        if np.all(sdr_t_pair[0] == 0):
            continue

        sdr_t = select_plane(sdr_t_pair)
        sdr_p = select_plane(sdr_p_pair)
        
        # sdr_t is [strike, dip, rake], sdr_p is [strike, dip, rake]
        kagan = kagan_angle(*sdr_t, *sdr_p)
        
        # Plot True (Black)
        ax_t = axes[row, col_true]
        title_t = f"{event_ids[i]}\nM{magnitudes[i]:.1f}"
        # Also unpack for plot_beachball
        plot_beachball(ax_t, *sdr_t, color='black', title=title_t)
        
        # Plot Pred (Blue)
        ax_p = axes[row, col_pred]
        title_p = f"Kagan\n{kagan:.0f}°"
        plot_beachball(ax_p, *sdr_p, color='blue', title=title_p)

    # Hide unused axes
    for j in range(num_samples * 2, rows * cols * 2):
        r = j // (cols * 2)
        c = j % (cols * 2)
        axes[r, c].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved beachball comparison to {save_path}")


# --- 3. Attention Visualization ---

def plot_attention_vs_azimuth(
    attn_weights: np.ndarray,
    azimuths: np.ndarray,
    station_names: List[str],
    event_id: str,
    save_path: Optional[str] = None
):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markerline, stemlines, baseline = ax.stem(
        azimuths, attn_weights, 
        linefmt='grey', markerfmt='o', basefmt='k-'
    )
    plt.setp(markerline, markersize=8, alpha=0.8)
    
    top_indices = np.argsort(attn_weights)[-5:]
    for i in top_indices:
        ax.text(
            azimuths[i], attn_weights[i] + 0.01, 
            station_names[i], 
            ha='center', va='bottom', fontsize=9, rotation=0, fontweight='bold'
        )
        
    ax.set_xlim(0, 360)
    ax.set_ylim(bottom=0, top=max(attn_weights.max()*1.2, 0.1))
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel('Station Azimuth (deg)')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Event {event_id}: Attention Distribution')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)


# --- 4. Grad-CAM Utilities ---

class GradCamHooks:
    def __init__(self):
        self.activations = None
        self.gradients = None

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

def generate_gradcam_heatmap(
    model: torch.nn.Module,
    waveforms: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
    target_layer: torch.nn.Module,
    target_output_idx: int = 1,
    station_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    hooks = GradCamHooks()
    f_handle = target_layer.register_forward_hook(hooks.forward_hook)
    b_handle = target_layer.register_full_backward_hook(hooks.backward_hook)
    
    model.zero_grad()
    preds, _ = model(waveforms, features, mask)
    score = preds[0, target_output_idx]
    score.backward()
    
    f_handle.remove()
    b_handle.remove()
    
    grads = hooks.gradients
    acts = hooks.activations
    
    sta_grads = grads[station_idx]
    sta_acts = acts[station_idx]
    
    weights = torch.mean(sta_grads, dim=1)
    
    cam = torch.zeros(sta_acts.shape[1], device=waveforms.device)
    for i, w in enumerate(weights):
        cam += w * sta_acts[i, :]
        
    cam = F.relu(cam)
    
    original_len = waveforms.shape[-1]
    cam_np = cam.cpu().detach().numpy()
    
    x_cam = np.linspace(0, original_len-1, len(cam_np))
    x_orig = np.arange(original_len)
    f_interp = interp1d(x_cam, cam_np, kind='linear', fill_value="extrapolate")
    heatmap = f_interp(x_orig)
    
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    return heatmap, waveforms[0, station_idx].cpu().numpy()

def plot_gradcam(
    heatmap: np.ndarray,
    waveform: np.ndarray,
    channels: List[int],
    channel_names: List[str],
    title: str,
    save_path: Optional[str] = None
):
    fig, axs = plt.subplots(len(channels), 1, figsize=(6, 2*len(channels)), sharex=True)
    if len(channels) == 1: axs = [axs]
    
    colors = [(1, 0, 0, 0), (1, 0, 0, 0.5)] 
    cmap_red = matplotlib.colors.LinearSegmentedColormap.from_list('trans_red', colors)
    
    heatmap_img = heatmap.reshape(1, -1)
    x = np.arange(len(heatmap))
    
    for i, ch_idx in enumerate(channels):
        ax = axs[i]
        wave_data = waveform[ch_idx]
        
        ax.plot(x, wave_data, 'k-', linewidth=1, zorder=2)
        
        y_min, y_max = np.min(wave_data), np.max(wave_data)
        y_rng = (y_max - y_min) or 1.0
        extent = [x[0], x[-1], y_min - 0.1*y_rng, y_max + 0.1*y_rng]
        
        ax.imshow(
            heatmap_img, aspect='auto', cmap=cmap_red, 
            extent=extent, zorder=1
        )
        
        ax.set_ylabel(channel_names[i])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, linestyle=':', alpha=0.6)

    axs[-1].set_xlabel('Time Samples')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

def plot_kagan_histogram(
    kagan_angles: np.ndarray,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plots a histogram of Kagan angles with statistics.
    
    Args:
        kagan_angles: 1D numpy array of Kagan angles in degrees.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Statistics
    mean_val = np.mean(kagan_angles)
    median_val = np.median(kagan_angles)
    
    # Plot Histogram
    bins = np.arange(0, 180, 4)
    counts, _, _ = ax.hist(
        kagan_angles, bins=bins, 
        color='skyblue', edgecolor='black', alpha=0.7, rwidth=0.85
    )
    
    # Vertical lines for stats
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.1f}°')
    
    # Styling
    ax.set_xlabel('Kagan Angle (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution of Focal Mechanisms')
    ax.set_xlim(0, 180)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved Kagan histogram to {save_path}")
    return fig

def plot_learning_curve(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    start_epoch: int = 1
) -> Figure:
    """
    Plots training and validation loss curves.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists.
        save_path: Path to save the figure.
        start_epoch: Epoch number to start plotting from (useful to skip early spikes).
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    if not train_loss or not val_loss:
        logger.warning("No training history found (lists are empty). Skipping Learning Curve plot.")
        return None
    total_epochs = len(train_loss)
    epochs = np.arange(1, total_epochs + 1)
    
    if start_epoch > total_epochs:
        logger.info(f"Total epochs ({total_epochs}) < start_epoch ({start_epoch}). Resetting start_epoch to 1.")
        start_epoch = 1
        
    # Filter by start_epoch
    mask = epochs >= start_epoch
    epochs = epochs[mask]
    train_loss = np.array(train_loss)[mask]
    val_loss = np.array(val_loss)[mask]
    
    if len(epochs) == 0:
        logger.warning("No epochs left to plot after filtering. Skipping.")
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(epochs, train_loss, 'o-', label='Training Loss', markersize=6)
    ax.plot(epochs, val_loss, 'o-', label='Validation Loss', markersize=6)
    
    # Get final test metric for title if available
    final_mse = val_loss[-1] if len(val_loss) > 0 else 0.0
    
    ax.set_title(f'Learning Curve (Epochs {start_epoch}-{epochs[-1]}) | Final Val Loss: {final_mse:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='-', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved learning curve to {save_path}")
    return fig
