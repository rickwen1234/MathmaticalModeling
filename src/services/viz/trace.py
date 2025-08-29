# src/viz/trace.py
import numpy as np
import matplotlib.pyplot as plt
from ..metrics.kinematics import velocity_from_positions, accel_mag_from_positions


__all__ = ["node_path", "node_velocity", "node_accelerate", "node_accelelerate"]


def node_path(t, pos, node=0, title=None, out_png=None):
    """
    Plot the trajectory (path) of a specific node in 2D space over time.
    - t: time array
    - pos: position array (T, N, 2)
    - node: index of the node to plot
    - title: plot title
    - out_png: optional output file path to save the figure
    """
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
    """
    Plot the velocity components (vx, vy) of a specific node over time.
    - t: time array
    - pos: position array (T, N, 2), optional if vel is provided
    - vel: velocity array (T, N, 2), optional
    - node: index of the node to plot
    - title: plot title
    - out_png: optional output file path to save the figure
    """
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


def node_accelerate(t, pos=None, vel=None, acc=None, node=0, title=None, out_png=None):
    """
    Plot the acceleration magnitude of a specific node over time.
    Accepts either:
      - raw positions (`pos`) with optional `vel` (will compute accel internally), or
      - a precomputed acceleration array `acc` with shape (T, N, 2).
    - t: time array
    - pos: position array (T, N, 2), optional if `acc` is provided
    - vel: velocity array (T, N, 2), optional helper when pos provided
    - acc: acceleration array (T, N, 2), optional
    - node: index of the node to plot
    - title: plot title
    - out_png: optional output file path to save the figure
    """
    if acc is None:
        if pos is None:
            raise ValueError("need pos or acc")
        # accel_mag_from_positions handles computing acceleration from pos (and optional vel)
        mag = accel_mag_from_positions(t, pos, v=vel)
    else:
        # acc is expected with last-dimension components; compute magnitude
        mag = np.linalg.norm(acc, axis=2)

    plt.figure(figsize=(9, 4.8))
    plt.plot(t, mag[:, node], label="|a| (m/s²)")
    plt.xlabel("Time (s)"); plt.ylabel("Acceleration (m/s²)")
    plt.title(title or f"Node {node} Acceleration Magnitude")
    plt.legend(); plt.grid(True)
    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()

# Backwards-compatible alias to accommodate a previous misspelling in imports
def node_accelelerate(*args, **kwargs):
    """Alias for node_accelerate (keeps compatibility with older code that used the misspelling)."""
    return node_accelerate(*args, **kwargs)