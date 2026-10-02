"""Shared source-parameter utilities for the data pipeline."""
import numpy as np


def sdr2mxyz_norm(strike, dip, rake):
    """Strike/dip/rake (degrees) -> normalized deviatoric moment-tensor
    components (Mxx, Myy, Mxy, Mxz, Myz) with unit scalar moment
    (Mzz = -(Mxx + Myy) implied by the zero-trace constraint).

    Aki & Richards convention; numerical port of the original
    sdr2mxyz.py used to build all training HDF5 files.
    """
    d2r = np.pi / 180.0
    sstr, cstr = np.sin(strike * d2r), np.cos(strike * d2r)
    sstr2, cstr2 = 2 * sstr * cstr, 1 - 2 * sstr * sstr
    sdip, cdip = np.sin(dip * d2r), np.cos(dip * d2r)
    sdip2, cdip2 = 2 * sdip * cdip, 1 - 2 * sdip * sdip
    crak, srak = np.cos(rake * d2r), np.sin(rake * d2r)

    mxx = -(sdip * crak * sstr2 + sdip2 * srak * sstr * sstr)
    mxy = (sdip * crak * cstr2 + 0.5 * sdip2 * srak * sstr2)
    mxz = -(cdip * crak * cstr + cdip2 * srak * sstr)
    myy = (sdip * crak * sstr2 - sdip2 * srak * cstr * cstr)
    myz = (cdip2 * srak * cstr - cdip * crak * sstr)
    return mxx, myy, mxy, mxz, myz


def parse_synthetic_event_name(name):
    """Parse 'ev_{strike}_{dip}_{rake}_{depth}_{mag}' directory names created
    by generate_synthetic_events.py. Returns a dict or None."""
    parts = name.split("_")
    if len(parts) != 6 or parts[0] != "ev":
        return None
    try:
        return {
            "event_id": name,
            "strike": float(parts[1]),
            "dip": float(parts[2]),
            "rake": float(parts[3]),
            "depth": float(parts[4]),
            "magnitude": float(parts[5]),
        }
    except ValueError:
        return None
