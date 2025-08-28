# 目标
把现有的“螺线几何 + 连续 S 弯 + 刚性链解算 + 可视化/导出”改造成**四层结构**，每层职责单一、可测试、可替换，并通过“项目内调用”的轻量 API 串起来，避免编辑器卡顿与单文件臃肿。

---

## 目录结构（建议）
```
project/
  pyproject.toml        # 或 requirements.txt
  src/
    services/
      geometry.py       # 第1层：几何原语/积分与反解（纯函数）
      path.py           # 第2层：路径构造（螺线+S弯），仅依赖 geometry
      rigid_chain.py    # 第3层：刚性链解算器（只依赖第2层的 path.eval）
    apps/
      common_io.py      # 第4层：可视化/导出工具（matplotlib+pandas）
      problem1.py       # 应用脚本：调用 1-3 层 + 导出
      problem3.py       # 应用脚本：最小螺距判据与对比图
      problem4.py       # 应用脚本：S弯+链仿真+导出
  tests/
    test_geometry.py
    test_path.py
    test_rigid_chain.py
  README.md
```

> 依赖方向：**apps → rigid_chain → path → geometry**（单向）。

---

## 第1层：`services/geometry.py`
```python
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
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)

def rot90(v, sgn: int = +1) -> np.ndarray:
    x, y = v
    return np.array([-sgn*y, sgn*x], dtype=float)

# s(θ) for Archimedean spiral r=bθ, θ>=0
# s(θ) = 0.5*b*(θ*sqrt(θ^2+1) + asinh(θ))

def F_arc(b: float, theta: float) -> float:
    return 0.5*b*(theta*math.sqrt(theta*theta+1.0) + math.asinh(theta))

def spiral_xy_tangent(b: float, theta: float, motion_sign: int):
    r = b*theta
    ct, st = math.cos(theta), math.sin(theta)
    x, y = r*ct, r*st
    dx_dth = b*ct - r*st
    dy_dth = b*st + r*ct
    t = unit(np.array([dx_dth, dy_dth], dtype=float)) * motion_sign
    return np.array([x, y], dtype=float), t

# 反解 s(θ)=S_target → θ（两种策略）

def invert_F_bisection(b: float, S_target: float, theta_hint: float = 1.0, tol: float = 1e-12) -> float:
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
```

---

## 第2层：`services/path.py`
```python
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
    组成：内螺线(θ递减) → S 弯(两圆外切) → 外螺线；关于原点中心对称。
    s 的零点取在内螺线与转弯圆相切点 P；s>0 沿前进方向。
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
        # S 弯外切解算
        self.R2, self.R1, self.C1, self.C2, self.Q = self._solve_S_arcs()
        self._setup_arcs()
        # 预计算长度
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
        # 弧1：P→Q
        r1P = unit(self.P - self.C1)
        r1Q = unit(self.Q - self.C1)
        tan_ccw = rot90(r1P, +1); tan_cw = rot90(r1P, -1)
        ori1 = +1 if float(np.dot(tan_ccw, self.t_in)) >= float(np.dot(tan_cw, self.t_in)) else -1
        phi1 = abs(signed_ang(r1P, r1Q, ori1))
        self.r1P, self.ori1, self.phi1 = r1P, ori1, phi1
        # 弧2：Q→P'
        r2Q = unit(self.Q - self.C2)
        r2P = unit(self.Pp - self.C2)
        tan2_ccw = rot90(r2P, +1); tan2_cw = rot90(r2P, -1)
        ori2 = +1 if float(np.dot(tan2_ccw, self.t_out)) >= float(np.dot(tan2_cw, self.t_out)) else -1
        phi2 = abs(signed_ang(r2Q, r2P, ori2))
        self.r2Q, self.ori2, self.phi2 = r2Q, ori2, phi2

    def eval(self, s: float):
        # s<0 内螺线； 0..L1 弧1； L1..LS 弧2； s>LS 外螺线（中心对称映射）
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
    """带 Newton 提示缓存的快速 eval（长时仿真推荐）"""
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
    """仅内螺线段（用于 Problem 1），带 θ 提示缓存"""
    def __init__(self, b: float, theta_start: float):
        self.b = float(b)
        self.theta0 = float(theta_start)
        self.F_theta0 = F_arc(self.b, self.theta0)
        self._theta_hint = self.theta0

    def eval(self, s_head_negative: float):
        S_target = self.F_theta0 + (-s_head_negative)
        self._theta_hint = invert_F_newton(self.b, S_target, self._theta_hint + 1e-9)
        return spiral_xy_tangent(self.b, self._theta_hint, motion_sign=-1)
```

---

## 第3层：`services/rigid_chain.py`
```python
# src/services/rigid_chain.py
from __future__ import annotations
import numpy as np
from typing import List

__all__ = ["solve_chain_s_incremental"]

def solve_chain_s_incremental(path, s_head: float, link_lengths: List[float], s_prev=None, s_head_prev=None):
    """
    增量式刚性链解：给定头部弧长 s_head，逐节向后解出 s_i。
    - 若提供上一帧解 s_prev 与 s_head_prev，则以 ds_head 为初值偏移，
      用小窗口括域+二分收敛，速度显著提升。
    - path 只需实现 eval(s) -> (pos, tan)。
    """
    s_vals = [float(s_head)]
    p_prev, _ = path.eval(s_head)
    if s_prev is None:
        step_seed = 0.75
        for L in link_lengths:
            s_lo = s_vals[-1]
            s_hi = s_lo + max(L*0.5, step_seed)
            # 扩展上界直到距离 ≥ L
            for _ in range(20):
                p_hi, _ = path.eval(s_hi)
                if np.linalg.norm(p_hi - p_prev) >= L:
                    break
                s_hi += max(L*0.3, 0.25)
            # 二分
            for _ in range(18):
                s_mid = 0.5*(s_lo + s_hi)
                p_mid, _ = path.eval(s_mid)
                d_mid = np.linalg.norm(p_mid - p_prev)
                if abs(d_mid - L) < 5e-9:
                    s_lo = s_hi = s_mid
                    break
                if d_mid < L: s_lo = s_mid
                else:         s_hi = s_mid
            s_i = 0.5*(s_lo + s_hi)
            s_vals.append(s_i)
            p_prev, _ = path.eval(s_i)
            step_seed = s_i - s_vals[-2]
        return s_vals

    # 用上一帧做近似种子
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
            s_lo = s_hi; d_lo = d_hi
            s_hi = s_hi + max(0.15*L, 0.2)
            d_hi = np.linalg.norm(path.eval(s_hi)[0] - p_prev)
            it += 1
        for _ in range(16):
            s_mid = 0.5*(s_lo + s_hi)
            d_mid = np.linalg.norm(path.eval(s_mid)[0] - p_prev)
            if abs(d_mid - L) < 5e-9:
                s_lo = s_hi = s_mid
                break
            if d_mid < L: s_lo = s_mid
            else:         s_hi = s_mid
        s_i = 0.5*(s_lo + s_hi)
        s_vals.append(s_i)
        p_prev, _ = path.eval(s_i)
    return s_vals
```

---

## 第4层：`apps/common_io.py`（导出/可视化 API）
```python
# src/apps/common_io.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["export_chain_csv", "export_chain_xlsx", "plot_speed_heatmap", "plot_path"]

def export_chain_csv(path, t_arr, positions, velocities, labels, out_csv):
    speed = np.linalg.norm(velocities, axis=2)
    rows = []
    for ti, t in enumerate(t_arr):
        for j in range(positions.shape[1]):
            rows.append({
                "time_s": float(t),
                "handle_index": j,
                "handle_label": labels[j] if j < len(labels) else f"node_{j}",
                "x_m": float(positions[ti,j,0]),
                "y_m": float(positions[ti,j,1]),
                "vx_mps": float(velocities[ti,j,0]),
                "vy_mps": float(velocities[ti,j,1]),
                "speed_mps": float(speed[ti,j]),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def export_chain_xlsx(t_arr, positions, velocities, labels, out_xlsx, snapshots_times=None, snap_indices=None):
    speed = np.linalg.norm(velocities, axis=2)
    rows = []
    for ti, t in enumerate(t_arr):
        for j in range(positions.shape[1]):
            rows.append({
                "time_s": float(t),
                "handle_index": j,
                "handle_label": labels[j] if j < len(labels) else f"node_{j}",
                "x_m": float(positions[ti,j,0]),
                "y_m": float(positions[ti,j,1]),
                "vx_mps": float(velocities[ti,j,0]),
                "vy_mps": float(velocities[ti,j,1]),
                "speed_mps": float(speed[ti,j]),
            })
    df_traj = pd.DataFrame(rows)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_traj.to_excel(writer, index=False, sheet_name="trajectory")
        if snapshots_times is not None and snap_indices is not None:
            sn = []
            for t in snapshots_times:
                i = int(np.where(t_arr==t)[0][0])
                for j in snap_indices:
                    sn.append({
                        "time_s": float(t),
                        "handle_index": j,
                        "handle_label": labels[j] if j < len(labels) else f"node_{j}",
                        "x_m": float(positions[i,j,0]),
                        "y_m": float(positions[i,j,1]),
                        "vx_mps": float(velocities[i,j,0]),
                        "vy_mps": float(velocities[i,j,1]),
                        "speed_mps": float(speed[i,j]),
                    })
            pd.DataFrame(sn).to_excel(writer, index=False, sheet_name="snapshots")

def plot_speed_heatmap(speed, t_arr, title, out_png=None):
    plt.figure(figsize=(9,4.8))
    plt.imshow(speed, aspect="auto", origin="lower", extent=[0, speed.shape[1]-1, float(t_arr[0]), float(t_arr[-1])])
    plt.xlabel("Handle index (0=head_front, N-1=tail_rear)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="Speed (m/s)")
    if out_png: plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()

def plot_path(in_xy, out_xy, arc1_xy, arc2_xy, R_turn, title, out_png=None):
    plt.figure(figsize=(7,7))
    plt.plot(in_xy[:,0],  in_xy[:,1],  lw=1.0, label="Inward spiral")
    plt.plot(out_xy[:,0], out_xy[:,1], lw=1.0, label="Outward spiral")
    plt.plot(arc1_xy[:,0], arc1_xy[:,1], lw=2, label="S-turn arc1")
    plt.plot(arc2_xy[:,0], arc2_xy[:,1], lw=2, label="S-turn arc2")
    circle = plt.Circle((0,0), R_turn, fill=False)
    plt.gca().add_patch(circle)
    plt.gca().set_aspect("equal")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title(title)
    plt.legend(); plt.grid(True)
    if out_png: plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
```

---

## 应用脚本示例

### `apps/problem1.py`
```python
from __future__ import annotations
import numpy as np
from services.path import FastInSpiralOnly
from services.rigid_chain import solve_chain_s_incremental
from apps.common_io import export_chain_csv, plot_speed_heatmap

# 几何参数
p1_pitch = 0.55
b1 = p1_pitch/(2*np.pi)
theta_start = 32*np.pi
path = FastInSpiralOnly(b1, theta_start)

# 链参数
L_head = 3.41 - 2*0.275
L_body = 2.20 - 2*0.275
link_lengths = [L_head] + [L_body]*222
Nnodes = len(link_lengths)+1

# 时间轴（可配置）
t = np.arange(0, 301, 10.0)
pos = np.zeros((len(t), Nnodes, 2))
vel = np.zeros_like(pos)

s_prev = None
s_head_prev = None
for i, ti in enumerate(t):
    s_head = -ti
    s_list = solve_chain_s_incremental(path, s_head, link_lengths, s_prev, s_head_prev)
    for j, s in enumerate(s_list):
        pos[i, j, :] = path.eval(s)[0]
    s_prev = s_list; s_head_prev = s_head

# 速度
dt = t[1]-t[0]
vel[1:-1,:,:] = (pos[2:,:,:]-pos[:-2,:,:])/(2*dt)
vel[0,:,:]    = (pos[1,:,:]-pos[0,:,:])/dt
vel[-1,:,:]   = (pos[-1,:,:]-pos[-2,:,:])/dt
speed = np.linalg.norm(vel, axis=2)

# 导出 & 图
labels = ["head_front"] + [f"bench_{i}_front" for i in range(1,223)] + ["tail_rear"]
export_chain_csv(path, t, pos, vel, labels, out_csv="result1_chain.csv")
plot_speed_heatmap(speed, t, "Problem 1 (p=0.55 m): speed heatmap")
```

### `apps/problem3.py`
```python
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

R_turn = 4.5
w = 0.30
p_from_width = w
p_from_16th = R_turn/16.0
p_min = max(p_from_width, p_from_16th)
print({"p_min": p_min, "from_width": p_from_width, "from_16th": p_from_16th})

# 可选：画对比图
for p in [p_from_16th, p_from_width]:
    b = p/(2*np.pi)
    th = np.linspace(0, 12*2*np.pi, 1200)
    r = b*th
    x = r*np.cos(th); y = r*np.sin(th)
    plt.figure(figsize=(6,6))
    plt.plot(x,y, lw=1.2, label=f"spiral (p={p:.5f} m)")
    circle = plt.Circle((0,0), R_turn, fill=False)
    plt.gca().add_patch(circle)
    plt.gca().set_aspect("equal"); plt.grid(True)
    plt.legend(); plt.title(f"Pitch feasibility check (p={p:.5f} m)")
    plt.show()
```

### `apps/problem4.py`
```python
from __future__ import annotations
import numpy as np
from services.path import FastPiecewisePath
from services.rigid_chain import solve_chain_s_incremental
from apps.common_io import export_chain_xlsx, plot_speed_heatmap

R_turn = 4.5
p4_pitch = 1.7
b4 = p4_pitch/(2*np.pi)
path = FastPiecewisePath(b=b4, R_turn=R_turn, R1_R2_ratio=2.0)

L_head = 3.41 - 2*0.275
L_body = 2.20 - 2*0.275
link_lengths = [L_head] + [L_body]*222
Nnodes = len(link_lengths)+1

t = np.arange(-100, 101, 5.0)
pos = np.zeros((len(t), Nnodes, 2))
vel = np.zeros_like(pos)

s_prev = None
s_head_prev = None
for i, ti in enumerate(t):
    s_head = ti
    s_list = solve_chain_s_incremental(path, s_head, link_lengths, s_prev, s_head_prev)
    for j, s in enumerate(s_list):
        pos[i, j, :] = path.eval(s)[0]
    s_prev = s_list; s_head_prev = s_head

# 速度/导出/图
dt = t[1]-t[0]
vel[1:-1,:,:] = (pos[2:,:,:]-pos[:-2,:,:])/(2*dt)
vel[0,:,:]    = (pos[1,:,:]-pos[0,:,:])/dt
vel[-1,:,:]   = (pos[-1,:,:]-pos[-2,:,:])/dt
speed = np.linalg.norm(vel, axis=2)

labels = ["head_front"] + [f"bench_{i}_front" for i in range(1,223)] + ["tail_rear"]
export_chain_xlsx(t, pos, vel, labels, out_xlsx="result4_chain.xlsx",
                  snapshots_times=[-100,-50,0,50,100], snap_indices=[0,1,51,101,151,201,223])
plot_speed_heatmap(speed, t, "Problem 4: speed heatmap (rigid-chain, S-turn)")
```

---

## 测试建议（最小集）
- `test_geometry.py`：
  - `F_arc(b,0)==0`；`invert_F_bisection/invert_F_newton` 互验；
  - `spiral_xy_tangent` 返回单位切向（范数≈1）。
- `test_path.py`：
  - `PiecewisePath.eval(0-)` 与 `eval(0+)` 的切向连续性；
  - S 弯的外切几何：`||C2-C1||≈R1+R2`。
- `test_rigid_chain.py`：
  - 两节链在直线/圆弧上的解与解析解对齐；
  - 增量解与从零解的一致性（误差 < 1e-6）。

---

## 性能与并行
- 反解：长程仿真优先 `invert_F_newton`，并给每条螺线维护 **θ 提示缓存**（随 s 单调）。
- 解算：用上一帧解作种子；括域窗口随链节长动态设置；所有求值都走 `path.eval`。
- 并行：把 apps 层拆到 **进程池/Job 队列** 或前端 WebWorker（JS 版）执行，UI 只拉静态导出。

---

## 运行
```
# Problem 1
python -m src.apps.problem1
# Problem 3
python -m src.apps.problem3
# Problem 4
python -m src.apps.problem4
```

> 若要提高分辨率，只改 `t = np.arange(...)` 步长即可；其它模块无需改动。



---

## 轻量化 `common_io` 与解耦：把“计算 vs. 作图”拆到专用模块
为避免 `common_io` 变成“万金油大杂烩”，把**数值派生**与**绘图呈现**分离：

```
src/
  services/
    geometry.py
    path.py
    rigid_chain.py
  metrics/
    kinematics.py        # 速度/加速度等派生量计算
    coords.py            # ← 新增：坐标系变换（极坐标/正交坐标/可选弗雷内）
  viz/
    heatmap.py           # 所有热图渲染
    trace.py             # 单节点轨迹/速度曲线
  apps/
    common_io.py         # 作为外观/门面，仅薄薄转发
```

### `metrics/kinematics.py`
```python
# src/metrics/kinematics.py
import numpy as np

__all__ = [
    "velocity_from_positions", "acceleration_from_positions",
    "speed_from_velocities", "accel_mag_from_positions",
]

def _assert_uniform_dt(t):
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        raise ValueError("t_arr must be uniform for finite differences")
    return float(dt[0])

def velocity_from_positions(t, pos):
    dt = _assert_uniform_dt(t)
    v = np.zeros_like(pos)
    v[1:-1] = (pos[2:] - pos[:-2])/(2*dt)
    v[0]    = (pos[1] - pos[0]) / dt
    v[-1]   = (pos[-1] - pos[-2]) / dt
    return v

def acceleration_from_positions(t, pos, v=None):
    if v is None:
        v = velocity_from_positions(t, pos)
    dt = _assert_uniform_dt(t)
    a = np.zeros_like(pos)
    a[1:-1] = (v[2:] - v[:-2])/(2*dt)
    a[0]    = (v[1]  - v[0]) / dt
    a[-1]   = (v[-1] - v[-2]) / dt
    return a

def speed_from_velocities(v):
    return np.linalg.norm(v, axis=2)

def accel_mag_from_positions(t, pos, v=None):
    a = acceleration_from_positions(t, pos, v)
    return np.linalg.norm(a, axis=2)
```

### `metrics/coords.py`
```python
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
        # choose any perpendicular
        e2 = np.array([-e1[1], e1[0]], dtype=float)
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = _unit(e2)
    return e1, e2

def polar_series(t, pos, node: int = 0, origin=(0.0, 0.0), ref_axis=(1.0, 0.0), unwrap=True):
    """给定节点的极坐标时间序列 (r(t), θ(t))。
    - origin: 极点（默认原点）
    - ref_axis: θ=0 的参考方向（默认 +x 轴）
    - unwrap: 是否解缠绕 θ（跨越 ±π 时保持连续）
    返回形状 (T, 2) 的数组，以及字段名 ["r", "theta"].
    """
    p = pos[:, node, :] - np.asarray(origin, dtype=float)
    r = np.linalg.norm(p, axis=1)
    th0 = math.atan2(ref_axis[1], ref_axis[0])
    th = np.arctan2(p[:, 1], p[:, 0]) - th0
    if unwrap:
        th = np.unwrap(th)
    return np.stack([r, th], axis=1), ("r", "theta")

def orthogonal_series(t, pos, node: int = 0, origin=(0.0, 0.0), axis_x=(1.0, 0.0), axis_y=None):
    """将节点坐标投影到任意正交基 (e_x, e_y)：返回 (u(t), v(t))。
    - axis_x: 主轴方向；axis_y 可选，若给出则与 axis_x 正交归一化（Gram-Schmidt）。
    - origin: 新坐标系的原点（默认世界原点）。
    返回形状 (T, 2) 的数组，以及字段名 ["u", "v"].
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
```

### `viz/heatmap.py`
```python
# src/viz/heatmap.py
import numpy as np
import matplotlib.pyplot as plt
from ..metrics.kinematics import (
    velocity_from_positions, speed_from_velocities, accel_mag_from_positions,
)

__all__ = ["render_heatmap", "heatmap_speed", "heatmap_accel"]

def render_heatmap(data, t, label, title, out_png=None):
    plt.figure(figsize=(9,4.8))
    plt.imshow(
        data, aspect="auto", origin="lower",
        extent=[0, data.shape[1]-1, float(t[0]), float(t[-1])]
    )
    plt.xlabel("Handle index (0=head_front, N-1=tail_rear)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.colorbar(label=label)
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()

def heatmap_speed(t, pos, v=None, title=None, out_png=None):
    if v is None:
        v = velocity_from_positions(t, pos)
    data = speed_from_velocities(v)
    render_heatmap(data, t, "Speed (m/s)", title or "Speed heatmap", out_png)

def heatmap_accel(t, pos, v=None, title=None, out_png=None):
    data = accel_mag_from_positions(t, pos, v)
    render_heatmap(data, t, "Acceleration (m/s²)", title or "Acceleration heatmap", out_png)
```

### `viz/trace.py`
```python
# src/viz/trace.py
import numpy as np
import matplotlib.pyplot as plt
from ..metrics.kinematics import velocity_from_positions

__all__ = ["node_path", "node_velocity"]

def node_path(t, pos, node=0, title=None, out_png=None):
    p = pos[:, node, :]
    plt.figure(figsize=(6,6))
    plt.plot(p[:,0], p[:,1])
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title(title or f"Node {node} Position")
    plt.grid(True); plt.gca().set_aspect("equal")
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()

def node_velocity(t, pos=None, vel=None, node=0, title=None, out_png=None):
    if vel is None:
        if pos is None:
            raise ValueError("need pos or vel")
        vel = velocity_from_positions(t, pos)
    v = vel[:, node, :]
    plt.figure(figsize=(9,4.8))
    plt.plot(t, v[:,0], label="vx (m/s)")
    plt.plot(t, v[:,1], label="vy (m/s)")
    plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
    plt.title(title or f"Node {node} Velocity")
    plt.legend(); plt.grid(True)
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
```

### `apps/common_io.py`（门面/薄包装）
```python
# src/apps/common_io.py
from ..viz.heatmap import heatmap_speed, heatmap_accel
from ..viz.trace import node_path, node_velocity
from ..viz.merge import merge as merge_plots, merge_paths
from ..metrics.coords import polar_series, orthogonal_series, coords_df, export_node_coords_csv

__all__ = [
    # 可视化
    "heatmap_speed", "heatmap_accel", "node_path", "node_velocity",
    # 合并
    "merge_plots", "merge_paths",
    # 坐标导出
    "polar_series", "orthogonal_series", "coords_df", "export_node_coords_csv",
]

# 兼容旧 API（可选）：

def plot_heatmap(t_arr, positions, velocities=None, mode="speed", title=None, out_png=None):
    if mode == "speed":
        return heatmap_speed(t_arr, positions, velocities, title, out_png)
    if mode == "acc":
        return heatmap_accel(t_arr, positions, velocities, title, out_png)
    raise ValueError("mode must be 'speed' or 'acc'")
```python
# src/apps/common_io.py
from ..viz.heatmap import heatmap_speed, heatmap_accel
from ..viz.trace import node_path, node_velocity
from ..metrics.coords import polar_series, orthogonal_series, coords_df, export_node_coords_csv

__all__ = [
    # 可视化
    "heatmap_speed", "heatmap_accel", "node_path", "node_velocity",
    # 坐标导出
    "polar_series", "orthogonal_series", "coords_df", "export_node_coords_csv",
]

# 兼容旧 API（可选）：

def plot_heatmap(t_arr, positions, velocities=None, mode="speed", title=None, out_png=None):
    if mode == "speed":
        return heatmap_speed(t_arr, positions, velocities, title, out_png)
    if mode == "acc":
        return heatmap_accel(t_arr, positions, velocities, title, out_png)
    raise ValueError("mode must be 'speed' or 'acc'")
```

### 使用示例
```python
from apps.common_io import (
    polar_series, orthogonal_series, export_node_coords_csv,
)

# 极坐标（以原点为极点，+x 为参考轴；θ 解缠绕）
arr, names = polar_series(t_arr, positions, node=101, origin=(0,0), ref_axis=(1,0), unwrap=True)
# 导出 CSV
export_node_coords_csv(t_arr, positions, node=101, mode="polar", out_csv="node101_polar.csv")

# 正交坐标（以某自定义轴 e_x、与其正交的 e_y）
arr2, names2 = orthogonal_series(t_arr, positions, node=101, origin=(0,0), axis_x=(0.6,0.8))
export_node_coords_csv(t_arr, positions, node=101, mode="orth", out_csv="node101_orth.csv", axis_x=(0.6,0.8))
```

### 扩展（可选）
- 若需要**随路径切向的弗雷内坐标**（切向/法向），可在 `metrics/coords.py` 补：
  - 用 `kinematics.velocity_from_positions` 得到切向单元 `t̂(t)`，法向 `n̂(t)=R90·t̂(t)`，
  - 将 `p(t)-p(t0)` 在 `{t̂(t), n̂(t)}` 上投影（或只输出速度在切/法方向的分量）。
- 若需角度范围统一，可在 `polar_series` 增加 `normalize_to=(-pi, pi)` 或 `(0, 2pi)` 的后处理。

---

## 新增：`viz/merge.py`（把多张图合成一张）
```python
# src/viz/merge.py
from __future__ import annotations
import io, math
from typing import Iterable, Union

try:
    from PIL import Image, ImageOps
except Exception as e:
    raise RuntimeError("viz.merge requires Pillow. pip install pillow") from e

# 支持的对象：matplotlib Figure、Axes、PNG 文件路径
MatObj = Union["matplotlib.figure.Figure", "matplotlib.axes.Axes", str]

def _to_image(obj: MatObj, dpi: int = 150, tight: bool = True) -> Image.Image:
    if isinstance(obj, str):
        im = Image.open(obj)
        return im.convert("RGBA") if im.mode != "RGBA" else im
    # 延迟导入 matplotlib 以减少常驻依赖
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    if isinstance(obj, Axes):
        fig = obj.figure
    elif isinstance(obj, Figure):
        fig = obj
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else None)
    buf.seek(0)
    im = Image.open(buf)
    return im.convert("RGBA") if im.mode != "RGBA" else im

def merge(objects: Iterable[MatObj], out_png: str | None = None, *, ncols: int = 2,
          pad: int = 10, bg: str = "white", dpi: int = 150, equalize: str = "height") -> Image.Image:
    """把多张图（Figure/Axes/PNG 路径）拼成一张面板图。

    - ncols: 每行列数；行数自动 = ceil(N/ncols)
    - pad: 砖块与画布的像素间距
    - bg: 背景颜色（"white" 或 "transparent"）
    - equalize: "height" | "width"，对齐各子图的高度或宽度
    - 返回 Pillow Image；如给出 out_png 则保存
    """
    imgs = [_to_image(o, dpi=dpi) for o in objects]
    if not imgs:
        raise ValueError("no images to merge")

    # 尺寸对齐
    if equalize == "height":
        h = max(im.height for im in imgs)
        imgs = [ImageOps.contain(im, (int(im.width * h / im.height), h)) if im.height != h else im for im in imgs]
    elif equalize == "width":
        w = max(im.width for im in imgs)
        imgs = [ImageOps.contain(im, (w, int(im.height * w / im.width))) if im.width != w else im for im in imgs]

    n = len(imgs)
    rows = math.ceil(n / ncols)
    colw = [0] * ncols
    rowh = [0] * rows
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, ncols)
        colw[c] = max(colw[c], im.width)
        rowh[r] = max(rowh[r], im.height)

    total_w = sum(colw) + pad * (ncols + 1)
    total_h = sum(rowh) + pad * (rows + 1)
    base = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 0) if bg == "transparent" else bg)

    y = pad
    for r in range(rows):
        x = pad
        for c in range(ncols):
            idx = r * ncols + c
            if idx >= n:
                break
            im = imgs[idx]
            dx = (colw[c] - im.width) // 2
            dy = (rowh[r] - im.height) // 2
            base.paste(im, (x + dx, y + dy), mask=im if im.mode == "RGBA" else None)
            x += colw[c] + pad
        y += rowh[r] + pad

    out = base if bg == "transparent" else base.convert("RGB")
    if out_png:
        out.save(out_png, dpi=(dpi, dpi))
    return out

# 便捷：仅合并 PNG 路径

def merge_paths(paths: Iterable[str], out_png: str, **kw) -> str:
    merge(paths, out_png=out_png, **kw)
    return out_png
```

### 合并面板：快速使用
```python
from apps.common_io import merge_plots
merge_plots(["heatmap_speed.png", "node101_path.png", "node101_vel.png"], out_png="panel.png", ncols=2)
# 或
# merge_plots([fig1, ax2, "some_saved_plot.png"], out_png="panel.png", ncols=3, bg="transparent")
```

