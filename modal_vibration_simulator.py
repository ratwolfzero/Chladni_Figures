import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class MembraneAnimation:
    def __init__(self):
        # --- Parameters ---
        self.Lx, self.Ly = 1.0, 1.0
        self.c = 1.0
        self.modes = [(3,5), (5,3)]
        self.amplitudes = [1.0, 0.6, 0.4]
        self.Nx, self.Ny = 80, 80
        self.T = 6.0
        self.fps = 60
        self.frames_total = int(self.T * self.fps)
        
        self.setup_components()
        self.setup_figure()
    
    def setup_components(self):
        # Create grid
        x = np.linspace(0, self.Lx, self.Nx)
        y = np.linspace(0, self.Ly, self.Ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Precompute spatial components
        self.spatial_components = []
        self.omegas = []
        for (m, n), A in zip(self.modes, self.amplitudes):
            shape = A * np.sin(m * np.pi * self.X / self.Lx) * np.sin(n * np.pi * self.Y / self.Ly)
            omega = np.pi * self.c * np.sqrt((m/self.Lx)**2 + (n/self.Ly)**2)
            self.spatial_components.append(shape)
            self.omegas.append(omega)
        
        self.spatial_components = np.array(self.spatial_components)
        self.omegas = np.array(self.omegas)
    
    def displacement(self, t):
        # Vectorized calculation
        time_factors = np.cos(self.omegas * t)
        return np.sum(self.spatial_components * time_factors[:, np.newaxis, np.newaxis], axis=0)
    
    def setup_figure(self):
        self.fig = plt.figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("displacement")
        self.ax.set_zlim(-2, 2)
        
        # Initial surface
        Z0 = self.displacement(0)
        self.surf = self.ax.plot_surface(self.X, self.Y, Z0, cmap="plasma", 
                                       antialiased=True, rstride=1, cstride=1, alpha=0.8)
    
    def update(self, frame):
        t = frame / self.fps
        Z = self.displacement(t)
        
        # Update the surface data
        self.surf.remove()
        self.surf = self.ax.plot_surface(self.X, self.Y, Z, cmap="plasma",
                                       antialiased=True, rstride=1, cstride=1, alpha=0.8)
        return self.surf,
    
    def animate(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.frames_total,
                                         interval=1000/self.fps, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()

# Create and run the animation
anim = MembraneAnimation()
anim.animate()
