# src/services/path.py
from __future__ import annotations
import math
import numpy as np
from .geometry import unit, rot90, F_arc, spiral_xy_tangent, invert_F_bisection, invert_F_newton

__all__ = [
    "PiecewisePath", "FastPiecewisePath", "FastInSpiralOnly"
]

class PiecewisePath:
    """
    Composition: Inner spiral (θ decreases) → S-curve (two externally tangent circles) → Outer spiral; symmetric about the origin.
    The zero point of s is at the tangent point P between the inner spiral and the turning circle; s > 0 is along the forward direction.
    """
    def __init__(self, b: float, R_turn: float, R1_R2_ratio: float = 2.0):
        self.b = float(b)
        self.R_turn = float(R_turn)
        self.k = float(R1_R2_ratio)
        # 入口/出口点与切向
        self.theta_P = self.R_turn / self.b
        self.P,  self.t_in  = spiral_xy_tangent(self.b, self.theta_P, motion_sign=-1)
        self.Pp, self.t_out = -self.P, -self.t_in
        self.n_in  = unit(rot90(self.t_in, +1))
        self.n_out = unit(rot90(self.t_out, +1))
    # S-curve external tangency calculation
        self.R2, self.R1, self.C1, self.C2, self.Q = self._solve_S_arcs()
        self._setup_arcs()
    # Precompute lengths
        self.L1 = self.R1*self.phi1
        self.L2 = self.R2*self.phi2
        self.LS = self.L1 + self.L2
        self.F_thetaP = F_arc(self.b, self.theta_P)

    def _solve_S_arcs(self):
        s1 = s2 = -1.0  # 左法向，匹配内向运动的切向方向
        def residual(R2: float):
            R1 = self.k*R2
            C1 = self.P  + s1*R1*self.n_in
            C2 = self.Pp + s2*R2*self.n_out
            D  = np.linalg.norm(C2 - C1)
            return D - (R1 + R2), (R1, C1, C2)
        a, bR = 1e-4, max(1.0, self.R_turn*2.0)
        fa, _ = residual(a); fb, _ = residual(bR)
        it = 0
        while fa*fb > 0 and it < 80:
            bR *= 1.5
            fb, _ = residual(bR)
            it += 1
        for _ in range(120):
            mid = 0.5*(a+bR)
            fm, payload = residual(mid)
            if abs(fm) < 1e-12:
                a = bR = mid
                break
            if fa*fm < 0: bR, fb = mid, fm
            else: a, fa = mid, fm
        R2 = 0.5*(a+bR)
        R1, C1, C2 = residual(R2)[1]
        u = unit(C2 - C1)
        Q = C1 + u*R1
        return float(R2), float(R1), C1, C2, Q

    def _setup_arcs(self):
        def signed_ang(a, b, orientation):
            ang = math.atan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1])
            if orientation >= 0: return ang if ang >= 0 else (2*math.pi + ang)
            else: return ang if ang <= 0 else (ang - 2*math.pi)
    # Arc 1: P→Q
        r1P = unit(self.P - self.C1)
        r1Q = unit(self.Q - self.C1)
        tan_ccw = rot90(r1P, +1); tan_cw = rot90(r1P, -1)
        ori1 = +1 if float(np.dot(tan_ccw, self.t_in)) >= float(np.dot(tan_cw, self.t_in)) else -1
        phi1 = abs(signed_ang(r1P, r1Q, ori1))
        self.r1P, self.ori1, self.phi1 = r1P, ori1, phi1
    # Arc 2: Q→P'
        r2Q = unit(self.Q - self.C2)
        r2P = unit(self.Pp - self.C2)
        tan2_ccw = rot90(r2P, +1); tan2_cw = rot90(r2P, -1)
        ori2 = +1 if float(np.dot(tan2_ccw, self.t_out)) >= float(np.dot(tan2_cw, self.t_out)) else -1
        phi2 = abs(signed_ang(r2Q, r2P, ori2))
        self.r2Q, self.ori2, self.phi2 = r2Q, ori2, phi2

    def eval(self, s: float):
    # s < 0: inner spiral; 0..L1: arc 1; L1..LS: arc 2; s > LS: outer spiral (central symmetry mapping)
        if s < 0:
            S_target = self.F_thetaP + (-s)
            th = invert_F_bisection(self.b, S_target, theta_hint=self.theta_P+1.0)
            return spiral_xy_tangent(self.b, th, motion_sign=-1)
        elif s <= self.L1:
            dphi = s/self.R1
            c, sn = math.cos(self.ori1*dphi), math.sin(self.ori1*dphi)
            rx = c*self.r1P[0] - sn*self.ori1*self.r1P[1]
            ry = c*self.r1P[1] + sn*self.ori1*self.r1P[0]
            pos = self.C1 + self.R1*np.array([rx, ry])
            tan = unit(rot90([rx, ry], +1 if self.ori1>=0 else -1))
            return pos, tan
        elif s <= self.LS:
            dphi = (s - self.L1)/self.R2
            c, sn = math.cos(self.ori2*dphi), math.sin(self.ori2*dphi)
            rx = c*self.r2Q[0] - sn*self.ori2*self.r2Q[1]
            ry = c*self.r2Q[1] + sn*self.ori2*self.r2Q[0]
            pos = self.C2 + self.R2*np.array([rx, ry])
            tan = unit(rot90([rx, ry], +1 if self.ori2>=0 else -1))
            return pos, tan
        else:
            S_target = self.F_thetaP + (s - self.LS)
            th = invert_F_bisection(self.b, S_target, theta_hint=self.theta_P+1.0)
            pos, tan = spiral_xy_tangent(self.b, th, motion_sign=+1)
            return -pos, -tan

class FastPiecewisePath(PiecewisePath):
    """Fast eval with Newton hint caching (recommended for long simulations)"""
    def __init__(self, b: float, R_turn: float, R1_R2_ratio: float = 2.0):
        super().__init__(b, R_turn, R1_R2_ratio)
        self.theta_hint_in  = self.theta_P
        self.theta_hint_out = self.theta_P

    def eval(self, s: float):
        if s < 0:
            S_target = self.F_thetaP + (-s)
            self.theta_hint_in = invert_F_newton(self.b, S_target, self.theta_hint_in + 1e-9)
            return spiral_xy_tangent(self.b, self.theta_hint_in, motion_sign=-1)
        elif s <= self.L1:
            return super().eval(s)
        elif s <= self.LS:
            return super().eval(s)
        else:
            S_target = self.F_thetaP + (s - self.LS)
            self.theta_hint_out = invert_F_newton(self.b, S_target, self.theta_hint_out + 1e-9)
            pos, tan = spiral_xy_tangent(self.b, self.theta_hint_out, motion_sign=+1)
            return -pos, -tan

class FastInSpiralOnly:
    """Only inner spiral segment (for Problem 1), with θ hint caching"""
    def __init__(self, b: float, theta_start: float):
        self.b = float(b)
        self.theta0 = float(theta_start)
        self.F_theta0 = F_arc(self.b, self.theta0)
        self._theta_hint = self.theta0

    def eval(self, s_head_negative: float):
        S_target = self.F_theta0 + (-s_head_negative)
        self._theta_hint = invert_F_newton(self.b, S_target, self._theta_hint + 1e-9)
        return spiral_xy_tangent(self.b, self._theta_hint, motion_sign=-1)