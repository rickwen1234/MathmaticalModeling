from __future__ import annotations
import numpy as np
import pandas as pd
from services.geometry import rect_vertices_from_pose, segments_intersect
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
t = np.arange(0, 301, 1)
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


# Detect interactions (self-contact) between bench rectangles at each time step
# Assumptions:
# - Two bench units interact if any of their rectangle edges intersect.
# - Adjacent nodes (connected links) are skipped since they are expected to touch.
# - Rectangle parameters (width, anchor offset) use typical values from geometry comments.

width = 0.30
anchor_offset_front = 0.275

def _unit_vec(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n

interaction_rows = []
for i_t, t_val in enumerate(t):
    # build rectangles for each node and test pairwise intersections (skip adjacent)
    verts_cache = [None] * Nnodes
    for a in range(Nnodes):
        # approximate tangent along the chain using neighbor difference
        if a < Nnodes - 1:
            tan = _unit_vec(pos[i_t, a, :] - pos[i_t, a+1, :])
        else:
            tan = _unit_vec(pos[i_t, a, :] - pos[i_t, a-1, :])
        length_a = link_lengths[a] if a < len(link_lengths) else link_lengths[-1]
        verts_cache[a] = rect_vertices_from_pose(pos[i_t, a, :], tan, length_a, width, anchor_offset_front)

    for a in range(Nnodes):
        for b in range(a + 2, Nnodes):
            # skip immediate neighbors (a+1) because they are connected
            va = verts_cache[a]
            vb = verts_cache[b]
            intersected = False
            # check all edge pairs
            for ia in range(4):
                p1, p2 = va[ia], va[(ia + 1) % 4]
                for ib in range(4):
                    q1, q2 = vb[ib], vb[(ib + 1) % 4]
                    if segments_intersect(np.asarray(p1), np.asarray(p2), np.asarray(q1), np.asarray(q2)):
                        intersected = True
                        break
                if intersected:
                    break
            if intersected:
                interaction_rows.append({
                    "time_s": float(t_val),
                    "node_a": int(a),
                    "node_b": int(b),
                })

pd.DataFrame(interaction_rows).to_csv("problem2_interactions.csv", index=False)
print(f"Recorded {len(interaction_rows)} interaction events; unique pairs: {len(set((r['node_a'], r['node_b']) for r in interaction_rows))}")