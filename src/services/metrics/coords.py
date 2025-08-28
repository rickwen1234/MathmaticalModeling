# src/metrics/coords.py
from __future__ import annotations
import math
import numpy as np


__all__ = [
"polar_series", "orthogonal_series", "coords_df", "export_node_coords_csv"
]


def _unit(w):
    w = np.asarray(w, dtype=float)
    n = np.linalg.norm(w)
    if n < 1e-15:
        raise ValueError("axis vector has near-zero norm")
    return w / n


def _gram_schmidt(e1, e2=None):
    e1 = _unit(e1)
    if e2 is None:
        # Choose any perpendicular vector
        e2 = np.array([-e1[1], e1[0]], dtype=float)
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = _unit(e2)
    return e1, e2


def polar_series(t, pos, node: int = 0, origin=(0.0, 0.0), ref_axis=(1.0, 0.0), unwrap=True):
    """Return the polar coordinate time series (r(t), θ(t)) for a given node.
    - origin: The pole (default is origin)
    - ref_axis: Reference direction for θ=0 (default is +x axis)
    - unwrap: Whether to unwrap θ (keep continuity across ±π)
    Returns an array of shape (T, 2) and field names ["r", "theta"].
    """
    p = pos[:, node, :] - np.asarray(origin, dtype=float)
    r = np.linalg.norm(p, axis=1)
    th0 = math.atan2(ref_axis[1], ref_axis[0])
    th = np.arctan2(p[:, 1], p[:, 0]) - th0
    if unwrap:
        th = np.unwrap(th)
    return np.stack([r, th], axis=1), ("r", "theta")


def orthogonal_series(t, pos, node: int = 0, origin=(0.0, 0.0), axis_x=(1.0, 0.0), axis_y=None):
    """Project node coordinates onto any orthogonal basis (e_x, e_y): returns (u(t), v(t)).
    - axis_x: Main axis direction; axis_y is optional, if provided it will be orthogonalized and normalized with axis_x (Gram-Schmidt).
    - origin: Origin of the new coordinate system (default is world origin).
    Returns an array of shape (T, 2) and field names ["u", "v"].
    """
    e1, e2 = _gram_schmidt(np.asarray(axis_x, float), None if axis_y is None else np.asarray(axis_y, float))
    p = pos[:, node, :] - np.asarray(origin, dtype=float)
    u = p @ e1
    v = p @ e2
    return np.stack([u, v], axis=1), ("u", "v")


def coords_df(t, pos, node: int, mode: str = "polar", **kw):
    import pandas as pd
    if mode == "polar":
        arr, names = polar_series(t, pos, node=node, **kw)
    elif mode == "orth":
        arr, names = orthogonal_series(t, pos, node=node, **kw)
    else:
        raise ValueError("mode must be 'polar' or 'orth'")
    df = pd.DataFrame({
        "time_s": np.asarray(t, dtype=float),
        "node": int(node),
        names[0]: arr[:, 0],
        names[1]: arr[:, 1],
    })
    return df


def export_node_coords_csv(t, pos, node: int, mode: str, out_csv: str, **kw):
    df = coords_df(t, pos, node=node, mode=mode, **kw)
    df.to_csv(out_csv, index=False)
    return out_csv