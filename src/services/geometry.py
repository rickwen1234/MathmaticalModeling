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
    "invert_F_bisection", "invert_F_newton", "segments_intersect"
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

def rect_vertices_from_pose(pos: np.ndarray,
                            tan: np.ndarray,
                            length: float,
                            width: float,
                            anchor_offset_front: float) -> np.ndarray:
    """
    Generate the 4 vertices of a rectangle (bench unit) given:
      - pos: 2D position (the anchor point lying on the rectangle's centerline).
              In this project, 'pos' is typically the handle position on path.
      - tan: 2D unit tangent direction of the rectangle's longitudinal axis (along bench length).
      - length: full rectangle length (e.g., 2.20 m for body/tail).
      - width:  full rectangle width  (e.g., 0.30 m).
      - anchor_offset_front: distance from the anchor point 'pos' to the *front edge* along +tan.
              (e.g., 0.275 m when the handle hole is 27.5 cm from the front.)
    Returns:
      np.ndarray of shape (4, 2), vertices ordered as:
        [front-left, front-right, back-right, back-left]
      with "left/right" defined by +normal = rot90(tan, +1).
    """
    pos = np.asarray(pos, dtype=float)
    t = unit(np.asarray(tan, dtype=float))
    n = unit(rot90(t, +1))

    # Centers of the front and back edges along the centerline
    front_center = pos + t*anchor_offset_front
    back_center  = pos - t*(length - anchor_offset_front)

    hw = 0.5*float(width)

    # Build vertices: front edge then back edge; left/right via +n/-n
    v_front_left  = front_center + hw*n
    v_front_right = front_center - hw*n
    v_back_right  = back_center  - hw*n
    v_back_left   = back_center  + hw*n

    return np.vstack([v_front_right, v_back_right, v_back_left, v_front_left]) # (4,2) array, clockwise order starting from front-right

def segments_intersect(p1: np.ndarray, p2: np.ndarray,
                       q1: np.ndarray, q2: np.ndarray) -> bool:
    """
    Check if two line segments (p1-p2 and q1-q2) intersect in 2D.

    Uses the cross product method to determine if the segments straddle
    each other, which indicates intersection.

    Args:
        p1, p2: Endpoints of the first segment
        q1, q2: Endpoints of the second segment

    Returns:
        bool: True if the segments intersect, False otherwise
    """

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Check if the segments straddle each other
    d1 = cross_product(p1, p2, q1)
    d2 = cross_product(p1, p2, q2)
    d3 = cross_product(q1, q2, p1)
    d4 = cross_product(q1, q2, p2)

    # If both pairs of cross products have opposite signs, the segments intersect
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Check for collinear cases (all cross products are zero)
    if abs(d1) < EPS and abs(d2) < EPS and abs(d3) < EPS and abs(d4) < EPS:
        # Check if segments are overlapping along the x-axis
        if max(p1[0], p2[0]) < min(q1[0], q2[0]) or max(q1[0], q2[0]) < min(p1[0], p2[0]):
            return False
        # Check if segments are overlapping along the y-axis
        if max(p1[1], p2[1]) < min(q1[1], q2[1]) or max(q1[1], q2[1]) < min(p1[1], p2[1]):
            return False
        return True

    return False