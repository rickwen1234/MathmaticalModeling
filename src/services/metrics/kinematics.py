# src/metrics/kinematics.py
import numpy as np


__all__ = [
"velocity_from_positions", "acceleration_from_positions",
"speed_from_velocities", "accel_mag_from_positions",
]


def _assert_uniform_dt(t):
    """
    Ensure that the time array t has uniform intervals.
    Raises ValueError if intervals are not uniform.
    Returns the time step dt.
    """
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        raise ValueError("t_arr must be uniform for finite differences")
    return float(dt[0])


def velocity_from_positions(t, pos):
    """
    Calculate velocity from position data using central differences for interior points
    and forward/backward differences for endpoints.
    t: time array
    pos: position array (T, ...)
    Returns velocity array of same shape as pos.
    """
    dt = _assert_uniform_dt(t)
    v = np.zeros_like(pos)
    v[1:-1] = (pos[2:] - pos[:-2])/(2*dt)
    v[0] = (pos[1] - pos[0]) / dt
    v[-1] = (pos[-1] - pos[-2]) / dt
    return v


def acceleration_from_positions(t, pos, v=None):
    """
    Calculate acceleration from position data (or velocity if provided).
    Uses central differences for interior points and forward/backward for endpoints.
    t: time array
    pos: position array (T, ...)
    v: optional velocity array
    Returns acceleration array of same shape as pos.
    """
    if v is None:
        v = velocity_from_positions(t, pos)
    dt = _assert_uniform_dt(t)
    a = np.zeros_like(pos)
    a[1:-1] = (v[2:] - v[:-2])/(2*dt)
    a[0] = (v[1] - v[0]) / dt
    a[-1] = (v[-1] - v[-2]) / dt
    return a


def speed_from_velocities(v):
    """
    Compute speed (magnitude of velocity) from velocity array.
    v: velocity array (T, N, D)
    Returns speed array (T, N)
    """
    return np.linalg.norm(v, axis=2)


def accel_mag_from_positions(t, pos, v=None):
    """
    Compute acceleration magnitude from position data (or velocity if provided).
    t: time array
    pos: position array (T, ...)
    v: optional velocity array
    Returns acceleration magnitude array (T, N)
    """
    a = acceleration_from_positions(t, pos, v)
    return np.linalg.norm(a, axis=2)