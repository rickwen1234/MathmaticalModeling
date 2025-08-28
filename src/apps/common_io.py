"""
Common input/output utilities for chain simulation and visualization.
Includes functions for exporting chain data to CSV/XLSX and plotting results.
"""
# src/apps/common_io.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from services.viz.trace import node_path, node_velocity
from services.viz.heatmap import heatmap_accel, heatmap_speed

__all__ = [
    "export_chain_csv", "export_chain_xlsx", "plot_speed_heatmap", "plot_path",
    "plot_heatmap", "heatmap_speed", "heatmap_accel", "node_path", "node_velocity"
]

def export_chain_csv(path, t_arr, positions, velocities, labels, out_csv):
    # Export chain simulation data to a CSV file
    # Calculate speed for each handle at each time step
    # Build rows for each handle and time
    # Save to CSV
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
    # Export chain simulation data to an Excel file
    # Calculate speed for each handle at each time step
    # Build rows for each handle and time
    # Write trajectory data to Excel
    # If snapshot times and indices are provided, export those as well
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
    # Plot a heatmap of handle speeds over time
    # Show and optionally save the heatmap
    plt.figure(figsize=(9,4.8))
    plt.imshow(speed, aspect="auto", origin="lower", extent=[0, speed.shape[1]-1, float(t_arr[0]), float(t_arr[-1])])
    plt.xlabel("Handle index (0=head_front, N-1=tail_rear)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="Speed (m/s)")
    if out_png: plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()

def plot_path(in_xy, out_xy, arc1_xy, arc2_xy, R_turn, title, out_png=None):
    # Plot the simulated path including spirals and S-turn arcs
    # Show and optionally save the path plot
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

def plot_heatmap(t_arr, positions, velocities=None, mode="speed", title=None, out_png=None):
    if mode == "speed":
        return heatmap_speed(t_arr, positions, velocities, title, out_png)
    if mode == "acc":
        return heatmap_accel(t_arr, positions, velocities, title, out_png)
    raise ValueError("mode must be 'speed' or 'acc'")