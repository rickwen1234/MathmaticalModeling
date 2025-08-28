# src/services/rigid_chain.py
from __future__ import annotations
import numpy as np
from typing import List

__all__ = ["solve_chain_s_incremental"]

def solve_chain_s_incremental(path, s_head: float, link_lengths: List[float], s_prev=None, s_head_prev=None):
        """
        Incremental rigid chain solution: Given the head arc length s_head, solve for each s_i sequentially backward.
        - If the previous frame solution s_prev and s_head_prev are provided, use ds_head as the initial offset,
            apply a small window bracketing + bisection for convergence, which significantly improves speed.
        - path only needs to implement eval(s) -> (pos, tan).
        """
        s_vals = [float(s_head)]
        p_prev, _ = path.eval(s_head)
        if s_prev is None:
            step_seed = 0.75
            for L in link_lengths:
                s_lo = s_vals[-1]
                s_hi = s_lo + max(L*0.5, step_seed)
                # Expand upper bound until distance ≥ L
                for _ in range(20):
                    p_hi, _ = path.eval(s_hi)
                    if np.linalg.norm(p_hi - p_prev) >= L:
                        break
                    s_hi += max(L*0.3, 0.25)
                # Bisection
                for _ in range(18):
                    s_mid = 0.5*(s_lo + s_hi)
                    p_mid, _ = path.eval(s_mid)
                    d_mid = np.linalg.norm(p_mid - p_prev)
                    if abs(d_mid - L) < 5e-9:
                        s_lo = s_hi = s_mid
                        break
                    if d_mid < L:
                        s_lo = s_mid
                    else:
                        s_hi = s_mid
                s_i = 0.5*(s_lo + s_hi)
                s_vals.append(s_i)
                p_prev, _ = path.eval(s_i)
                step_seed = s_i - s_vals[-2]
            return s_vals
        else:
            # Use previous frame as approximate seed
            ds_head = float(s_head - s_head_prev)
            for idx, L in enumerate(link_lengths):
                s_guess = s_prev[idx+1] + ds_head
                br = max(0.2, 0.2*L)
                s_lo = max(s_vals[-1], s_guess - br)
                s_hi = s_guess + br
                d_lo = np.linalg.norm(path.eval(s_lo)[0] - p_prev)
                d_hi = np.linalg.norm(path.eval(s_hi)[0] - p_prev)
                it = 0
                while not (d_lo <= L <= d_hi) and it < 12:
                    s_lo = s_hi
                    d_lo = d_hi
                    s_hi = s_hi + max(0.15*L, 0.2)
                    d_hi = np.linalg.norm(path.eval(s_hi)[0] - p_prev)
                    it += 1
                for _ in range(16):
                    s_mid = 0.5*(s_lo + s_hi)
                    d_mid = np.linalg.norm(path.eval(s_mid)[0] - p_prev)
                    if abs(d_mid - L) < 5e-9:
                        s_lo = s_hi = s_mid
                        break
                    if d_mid < L:
                        s_lo = s_mid
                    else:
                        s_hi = s_mid
                s_i = 0.5*(s_lo + s_hi)
                s_vals.append(s_i)
                p_prev, _ = path.eval(s_i)
            return s_vals