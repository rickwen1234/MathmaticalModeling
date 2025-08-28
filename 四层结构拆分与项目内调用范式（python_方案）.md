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

