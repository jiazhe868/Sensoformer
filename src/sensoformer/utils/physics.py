import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from obspy.imaging.beachball import beach
from typing import Union, Tuple, Optional

def kagan_angle(
    strike1: Union[float, np.ndarray], 
    dip1: Union[float, np.ndarray], 
    rake1: Union[float, np.ndarray], 
    strike2: Union[float, np.ndarray], 
    dip2: Union[float, np.ndarray], 
    rake2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculates the Kagan angle (minimum 3D rotation) between two sets of focal mechanisms.
    
    This function handles the double-couple symmetry ambiguity (4 symmetries).
    Inputs can be scalars or numpy arrays.

    Args:
        strike1, dip1, rake1: Mechanism 1 parameters (degrees).
        strike2, dip2, rake2: Mechanism 2 parameters (degrees).

    Returns:
        The minimum rotation angle in degrees [0, 120].
    """
    # Ensure inputs are numpy arrays
    strike1, dip1, rake1 = np.asarray(strike1), np.asarray(dip1), np.asarray(rake1)
    strike2, dip2, rake2 = np.asarray(strike2), np.asarray(dip2), np.asarray(rake2)

    # Calculate quaternions
    q1 = _quat_fps_vec(strike1, dip1, rake1)
    q2 = _quat_fps_vec(strike2, dip2, rake2)

    is_scalar = strike1.ndim == 0
    
    # Initialize result containers
    min_angles = np.full(strike1.shape, 180.0)

    # Iterate through the 4 symmetry operations
    for i in range(1, 5):
        q_dum = _f4r1_vec(q1, q2, i)
        angles, _, _ = _sphcoor_vec(q_dum)
        
        # Update minimum angles
        if is_scalar:
            if angles < min_angles:
                min_angles = angles
        else:
            update_mask = angles < min_angles
            min_angles[update_mask] = angles[update_mask]
            
    return min_angles

def plot_beachball(
    ax: Axes, 
    strike: float, 
    dip: float, 
    rake: float, 
    color: str = 'black', 
    title: str = "",
    width: int = 50
):
    """
    Plots a focal mechanism (beachball) on a given matplotlib axes using ObsPy.
    
    Args:
        ax: The matplotlib axes to plot on.
        strike, dip, rake: Focal mechanism parameters in degrees.
        color: Color for the COMPRESSIONAL quadrants (background).
        title: Title for the subplot.
        width: Visual width of the beachball.
    """
    fm = [strike, dip, rake]
    
    # ObsPy's beach() logic:
    # facecolor -> Tensional quadrants
    # bgcolor -> Compressional quadrants
    bball = beach(
        fm, 
        facecolor='white',  
        bgcolor=color,      
        edgecolor='black',  
        xy=(0, 0),          
        width=width,            
        axes=ax,            
        zorder=10
    )
    
    ax.add_collection(bball)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=8)

# --------------------------------------------------------------------
# Private Vectorized Helper functions (Logic preserved from kagan.py)
# --------------------------------------------------------------------

def _quat_fps_vec(strike, dip, rake):
    """Calculates rotation quaternions from focal mechanisms."""
    err = 1.e-15
    strike, dip, rake = np.asarray(strike), np.asarray(dip), np.asarray(rake)
    is_scalar = strike.ndim == 0
    
    strike_rad, dip_rad, rake_rad = np.deg2rad([strike, dip, rake])
    cdd, sdd = np.cos(strike_rad), np.sin(strike_rad)
    cda, sda = np.cos(dip_rad), np.sin(dip_rad)
    csa, ssa = np.cos(rake_rad), np.sin(rake_rad)

    s1, s2, s3 = csa * sdd - ssa * cda * cdd, -csa * cdd - ssa * cda * sdd, -ssa * sda
    v1, v2, v3 = sda * cdd, sda * sdd, -cda
    an1, an2, an3 = s2 * v3 - v2 * s3, v1 * s3 - s1 * v3, s1 * v2 - v1 * s2

    d2 = 1. / np.sqrt(2.)
    t1, t2, t3 = (v1 + s1) * d2, (v2 + s2) * d2, (v3 + s3) * d2
    p1, p2, p3 = (v1 - s1) * d2, (v2 - s2) * d2, (v3 - s3) * d2

    u0_sq, u1_sq = (t1 + p2 + an3 + 1.) / 4., (t1 - p2 - an3 + 1.) / 4.
    u2_sq, u3_sq = (-t1 + p2 - an3 + 1.) / 4., (-t1 - p2 + an3 + 1.) / 4.
    
    u_sq_stack = np.stack([u0_sq, u1_sq, u2_sq, u3_sq], axis=0)

    if is_scalar:
        imax = np.argmax(u_sq_stack, axis=0).item()
        u0, u1, u2, u3 = 0.0, 0.0, 0.0, 0.0
    else:
        imax = np.argmax(u_sq_stack, axis=0)
        u0, u1, u2, u3 = (np.zeros_like(strike) for _ in range(4))
    
    for i in range(4):
        mask = (imax == i)
        if is_scalar:
            if not mask: continue
            if i == 0:
                u0 = np.sqrt(u0_sq) if u0_sq > 0 else err
                if u0 == 0: u0 = err
                u3, u2, u1 = (t2 - p1)/(4.*u0), (an1 - t3)/(4.*u0), (p3 - an2)/(4.*u0)
            elif i == 1:
                u1 = np.sqrt(u1_sq) if u1_sq > 0 else err
                if u1 == 0: u1 = err
                u2, u3, u0 = (t2 + p1)/(4.*u1), (an1 + t3)/(4.*u1), (p3 - an2)/(4.*u1)
            elif i == 2:
                u2 = np.sqrt(u2_sq) if u2_sq > 0 else err
                if u2 == 0: u2 = err
                u1, u0, u3 = (t2 + p1)/(4.*u2), (an1 - t3)/(4.*u2), (p3 + an2)/(4.*u2)
            else: 
                u3 = np.sqrt(u3_sq) if u3_sq > 0 else err
                if u3 == 0: u3 = err
                u0, u1, u2 = (t2 - p1)/(4.*u3), (an1 + t3)/(4.*u3), (p3 + an2)/(4.*u3)
        else:
            if not np.any(mask): continue
            u_denom = np.zeros_like(strike)
            if i == 0:
                u_denom[mask] = np.sqrt(u0_sq[mask])
                u_denom[u_denom == 0.0] = err
                u0[mask] = u_denom[mask]
                u3[mask], u2[mask], u1[mask] = (t2[mask]-p1[mask])/(4.*u_denom[mask]), (an1[mask]-t3[mask])/(4.*u_denom[mask]), (p3[mask]-an2[mask])/(4.*u_denom[mask])
            elif i == 1:
                u_denom[mask] = np.sqrt(u1_sq[mask])
                u_denom[u_denom == 0.0] = err
                u1[mask] = u_denom[mask]
                u2[mask], u3[mask], u0[mask] = (t2[mask]+p1[mask])/(4.*u_denom[mask]), (an1[mask]+t3[mask])/(4.*u_denom[mask]), (p3[mask]-an2[mask])/(4.*u_denom[mask])
            elif i == 2:
                u_denom[mask] = np.sqrt(u2_sq[mask])
                u_denom[u_denom == 0.0] = err
                u2[mask] = u_denom[mask]
                u1[mask], u0[mask], u3[mask] = (t2[mask]+p1[mask])/(4.*u_denom[mask]), (an1[mask]-t3[mask])/(4.*u_denom[mask]), (p3[mask]+an2[mask])/(4.*u_denom[mask])
            else:
                u_denom[mask] = np.sqrt(u3_sq[mask])
                u_denom[u_denom == 0.0] = err
                u3[mask] = u_denom[mask]
                u0[mask], u1[mask], u2[mask] = (t2[mask]-p1[mask])/(4.*u_denom[mask]), (an1[mask]+t3[mask])/(4.*u_denom[mask]), (p3[mask]+an2[mask])/(4.*u_denom[mask])

    if is_scalar:
        quat = np.array([u1, u2, u3, u0])
        norm = np.linalg.norm(quat)
        return quat / (norm if norm != 0 else 1.0)
    else:
        quat = np.stack([u1, u2, u3, u0], axis=1)
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return quat / norm

def _sphcoor_vec(quat):
    """Converts rotation quaternions to axis-angle."""
    q = np.copy(quat)
    if q.ndim == 1:
        if q[3] < 0: q *= -1
        q[3] = np.clip(q[3], -1.0, 1.0)
        angle = 2. * np.rad2deg(np.arccos(q[3]))
        # (Theta/Phi calculation omitted for brevity as mainly angle is used for Kagan)
        return angle, 0.0, 0.0 
    else:
        neg_mask = q[:, 3] < 0
        q[neg_mask] *= -1
        q[:, 3] = np.clip(q[:, 3], -1.0, 1.0)
        angle = 2. * np.rad2deg(np.arccos(q[:, 3]))
        return angle, np.zeros_like(angle), np.zeros_like(angle)

def _quatp_vec(q1, q2):
    """Quaternion product."""
    if q1.ndim == 1 and q2.ndim == 2:
        q3 = np.zeros_like(q2)
        q3[:,0] = q1[3]*q2[:,0] + q1[2]*q2[:,1] - q1[1]*q2[:,2] + q1[0]*q2[:,3]
        q3[:,1] = -q1[2]*q2[:,0] + q1[3]*q2[:,1] + q1[0]*q2[:,2] + q1[1]*q2[:,3]
        q3[:,2] = q1[1]*q2[:,0] - q1[0]*q2[:,1] + q1[3]*q2[:,2] + q1[2]*q2[:,3]
        q3[:,3] = -q1[0]*q2[:,0] - q1[1]*q2[:,1] - q1[2]*q2[:,2] + q1[3]*q2[:,3]
        return q3
    elif q1.ndim == 2 and q2.ndim == 2:
        q3 = np.zeros_like(q1)
        q3[:,0] = q1[:,3]*q2[:,0] + q1[:,2]*q2[:,1] - q1[:,1]*q2[:,2] + q1[:,0]*q2[:,3]
        q3[:,1] = -q1[:,2]*q2[:,0] + q1[:,3]*q2[:,1] + q1[:,0]*q2[:,2] + q1[:,1]*q2[:,3]
        q3[:,2] = q1[:,1]*q2[:,0] - q1[:,0]*q2[:,1] + q1[:,3]*q2[:,2] + q1[:,2]*q2[:,3]
        q3[:,3] = -q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2] + q1[:,3]*q2[:,3]
        return q3
    elif q1.ndim == 1 and q2.ndim == 1:
        q3 = np.zeros_like(q1)
        q3[0] = q1[3]*q2[0] + q1[2]*q2[1] - q1[1]*q2[2] + q1[0]*q2[3]
        q3[1] = -q1[2]*q2[0] + q1[3]*q2[1] + q1[0]*q2[2] + q1[1]*q2[3]
        q3[2] = q1[1]*q2[0] - q1[0]*q2[1] + q1[3]*q2[2] + q1[2]*q2[3]
        q3[3] = -q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] + q1[3]*q2[3]
        return q3
    raise ValueError("Shape mismatch in quaternion product")

def _quatd_vec(q1, q2):
    qc1 = q1 * np.array([-1, -1, -1, 1])
    return _quatp_vec(qc1, q2)

def _boxtest_vec(q1, icode):
    quat_sym = np.array([0., 0., 0., 1.])
    if icode == 1: quat_sym = np.array([1., 0., 0., 0.])
    elif icode == 2: quat_sym = np.array([0., 1., 0., 0.])
    elif icode == 3: quat_sym = np.array([0., 0., 1., 0.])
    
    q2 = _quatp_vec(quat_sym, q1) if icode != 4 else np.copy(q1)
    
    if q2.ndim == 1:
        if q2[3] < 0: q2 *= -1
    else:
        q2[q2[:,3] < 0] *= -1
    return q2

def _f4r1_vec(q1, q2, icode):
    qr1 = _boxtest_vec(q1, icode)
    return _quatd_vec(qr1, q2)