# src/service/viz/heatmap.py
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