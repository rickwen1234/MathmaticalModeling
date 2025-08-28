from __future__ import annotations
import numpy as np
from services.path import FastInSpiralOnly
from services.rigid_chain import solve_chain_s_incremental
from apps.common_io import export_chain_csv, plot_speed_heatmap, heatmap_accel

 # Geometric parameters
p1_pitch = 0.55
b1 = p1_pitch/(2*np.pi)
theta_start = 32*np.pi
path = FastInSpiralOnly(b1, theta_start)

 # Chain parameters
L_head = 3.41 - 2*0.275
L_body = 2.20 - 2*0.275
link_lengths = [L_head] + [L_body]*222
Nnodes = len(link_lengths)+1

 # Time axis (configurable)
t = np.arange(0, 301, 10.0)
pos = np.zeros((len(t), Nnodes, 2))
vel = np.zeros_like(pos)

HEAD = 0
TAIL = Nnodes - 1

s_prev = None
s_head_prev = None
for i, ti in enumerate(t):
    s_head = -ti
    s_list = solve_chain_s_incremental(path, s_head, link_lengths, s_prev, s_head_prev)
    for j, s in enumerate(s_list):
        pos[i, j, :] = path.eval(s)[0]
    s_prev = s_list; s_head_prev = s_head

 # Velocity calculation
dt = t[1]-t[0]
vel[1:-1,:,:] = (pos[2:,:,:]-pos[:-2,:,:])/(2*dt)
vel[0,:,:]    = (pos[1,:,:]-pos[0,:,:])/dt
vel[-1,:,:]   = (pos[-1,:,:]-pos[-2,:,:])/dt
speed = np.linalg.norm(vel, axis=2)

 # Export & plot
labels = ["head_front"] + [f"bench_{i}_front" for i in range(1,223)] + ["tail_rear"]
export_chain_csv(path, t, pos, vel, labels, out_csv="result1_chain.csv")
plot_speed_heatmap(speed, t, "Problem 1 (p=0.55 m): speed heatmap")
heatmap_accel(t, pos, title="Problem 1 (p=0.55 m): acceleration heatmap")

# Export snapshots for specified times and node indices
import pandas as pd
snapshot_times = [0, 60, 120, 180, 240, 300]
snapshot_indices = [0, 1, 51, 101, 151, 201, 223]
rows = []
for t_snap in snapshot_times:
    # Find the index in t closest to t_snap
    i = int(np.where(t == t_snap)[0][0])
    for j in snapshot_indices:
        rows.append({
            "time_s": float(t[i]),
            "handle_index": j,
            "handle_label": labels[j] if j < len(labels) else f"node_{j}",
            "x_m": float(pos[i, j, 0]),
            "y_m": float(pos[i, j, 1]),
            "vx_mps": float(vel[i, j, 0]),
            "vy_mps": float(vel[i, j, 1]),
            "speed_mps": float(speed[i, j]),
        })
pd.DataFrame(rows).to_csv("problem1_snapshots.csv", index=False)