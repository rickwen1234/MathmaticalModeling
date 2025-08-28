"""
Geometry utilities for spiral and arc calculations used in chain/path modeling.
Includes vector normalization, rotation, spiral arc length, tangent calculation, and inverse solutions.
"""
# src/services/geometry.py
from __future__ import annotations
import math
import numpy as np

__all__ = [
    "unit", "rot90", "F_arc", "spiral_xy_tangent",
    "invert_F_bisection", "invert_F_newton"
]

EPS = 1e-15

def unit(v: np.ndarray) -> np.ndarray:
    # Normalize a vector to unit length; returns zero vector if norm is too small
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)

def rot90(v, sgn: int = +1) -> np.ndarray:
    # Rotate a 2D vector by 90 degrees (sgn=+1 for CCW, -1 for CW)
    x, y = v
    return np.array([-sgn*y, sgn*x], dtype=float)

# s(θ) for Archimedean spiral r=bθ, θ>=0
# s(θ) = 0.5*b*(θ*sqrt(θ^2+1) + asinh(θ))

def F_arc(b: float, theta: float) -> float:
    # Calculate arc length s(theta) for Archimedean spiral r=b*theta, theta>=0
    return 0.5*b*(theta*math.sqrt(theta*theta+1.0) + math.asinh(theta))

def spiral_xy_tangent(b: float, theta: float, motion_sign: int):
    # Compute (x, y) position and tangent vector for spiral at given theta
    r = b*theta
    ct, st = math.cos(theta), math.sin(theta)
    x, y = r*ct, r*st
    dx_dth = b*ct - r*st
    dy_dth = b*st + r*ct
    t = unit(np.array([dx_dth, dy_dth], dtype=float)) * motion_sign
    return np.array([x, y], dtype=float), t

# Inverse solution: s(θ)=S_target → θ (two strategies)

def invert_F_bisection(b: float, S_target: float, theta_hint: float = 1.0, tol: float = 1e-12) -> float:
    # Inverse solution: find theta such that s(theta) = S_target using bisection method
    lo, hi = 0.0, max(theta_hint, 1.0)
    def Fth(th): return F_arc(b, th)
    while Fth(hi) < S_target:
        hi *= 2.0
        if hi > 1e7: break
    for _ in range(130):
        mid = 0.5*(lo+hi)
        if Fth(mid) < S_target: lo = mid
        else: hi = mid
        if hi - lo < tol: break
    return 0.5*(lo+hi)

def invert_F_newton(b: float, S_target: float, theta_init: float) -> float:
    # Inverse solution: find theta such that s(theta) = S_target using Newton's method
    th = max(theta_init, 1e-9)
    for _ in range(20):
        Fth = 0.5*b*(th*math.sqrt(th*th+1.0) + math.asinh(th))
        dF = b*math.sqrt(max(th*th + 1.0, 1e-30))
        delta = (Fth - S_target) / dF
        th_new = th - delta
        if th_new <= 0:
            th_new = 0.5*th
        if abs(delta) < 1e-12:
            return th_new
        th = th_new
    return th