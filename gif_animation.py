import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Minimal Chladni simulation class (simplified) ---
class ChladniSimulator:
    def __init__(self, resolution=200, max_mode=10, gamma=0.06, k=1.0):
        self.resolution = resolution
        self.max_mode = max_mode
        self.gamma = gamma
        self.k = k

        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

        self.mode_shapes = []
        self.mode_frequencies = []
        for m in range(1, max_mode+1):
            for n in range(1, max_mode+1):
                self.mode_frequencies.append(self.k*np.sqrt(m**2+n**2))
                self.mode_shapes.append(np.sin(m*np.pi*self.X)*np.sin(n*np.pi*self.Y))
        self.mode_shapes = np.array(self.mode_shapes)
        self.mode_frequencies = np.array(self.mode_frequencies)

    def compute_displacement(self, f):
        weights = 1.0 / ((f - self.mode_frequencies)**2 + self.gamma**2)
        Z = np.sum(weights[:, np.newaxis, np.newaxis] * self.mode_shapes, axis=0)
        return Z

# --- Create the GIF ---
sim = ChladniSimulator(resolution=400, max_mode=25, gamma=0.06, k=1.0)

fig, ax = plt.subplots(figsize=(6,6))
Z = sim.compute_displacement(5.0)                                           
im = ax.imshow(np.abs(Z)**0.2, cmap='plasma', origin='lower', extent=[0,1,0,1])
ax.set_axis_off()

# Frequency range for GIF
frequencies = np.linspace(5, 15, 60)  # 60 frames, slower sweep for clarity

def update(frame):                                                          
    f = frequencies[frame]
    Z = sim.compute_displacement(f)
    im.set_data(np.abs(Z)**0.2)
    return [im]

ani = FuncAnimation(fig, update, frames=len(frequencies), blit=True)

# Save as GIF (slower to make nodal lines visible)
ani.save("chladni_simulation.gif", writer=PillowWriter(fps=5))  # fps=5 slows it down

plt.close(fig)

