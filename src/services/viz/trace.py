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