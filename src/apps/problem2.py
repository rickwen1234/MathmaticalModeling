from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

R_turn = 4.5
w = 0.30
p_from_width = w
p_from_16th = R_turn/16.0
p_min = max(p_from_width, p_from_16th)
print({"p_min": p_min, "from_width": p_from_width, "from_16th": p_from_16th})

# Optional: plot comparison chart
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