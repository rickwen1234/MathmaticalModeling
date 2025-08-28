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

# Velocity / Export / Plot
dt = t[1]-t[0]
vel[1:-1,:,:] = (pos[2:,:,:]-pos[:-2,:,:])/(2*dt)
vel[0,:,:]    = (pos[1,:,:]-pos[0,:,:])/dt
vel[-1,:,:]   = (pos[-1,:,:]-pos[-2,:,:])/dt
speed = np.linalg.norm(vel, axis=2)

labels = ["head_front"] + [f"bench_{i}_front" for i in range(1,223)] + ["tail_rear"]
export_chain_xlsx(t, pos, vel, labels, out_xlsx="result4_chain.xlsx",
                  snapshots_times=[-100,-50,0,50,100], snap_indices=[0,1,51,101,151,201,223])
plot_speed_heatmap(speed, t, "Problem 4: speed heatmap (rigid-chain, S-turn)")