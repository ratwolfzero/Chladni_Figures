import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os


class MembraneAnimation:
    def __init__(self, save_gif=False, gif_filename="membrane_animation.gif", gif_fps=15):
        # --- Parameters ---
        self.Lx, self.Ly = 1.0, 1.0
        self.c = 1.0
        self.modes = [(3, 5), (5, 3)]
        self.amplitudes = [1.0, 0.6, 0.4]
        self.Nx, self.Ny = 100, 100
        self.T = 6.0  # Total animation duration in seconds

        # Display settings
        self.display_fps = 50
        self.frames_total = int(self.T * self.display_fps)

        # GIF settings
        self.save_gif = save_gif
        self.gif_filename = gif_filename
        self.gif_fps = gif_fps  # Separate FPS for GIF
        self.gif_writer = None

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
            shape = A * np.sin(m * np.pi * self.X / self.Lx) * \
                np.sin(n * np.pi * self.Y / self.Ly)
            omega = np.pi * self.c * np.sqrt((m/self.Lx)**2 + (n/self.Ly)**2)
            self.spatial_components.append(shape)
            self.omegas.append(omega)

        self.spatial_components = np.array(self.spatial_components)
        self.omegas = np.array(self.omegas)

    def setup_figure(self):
        self.fig = plt.figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(
            f"Free Vibration Superposition (Modes {self.modes})", pad=10
        )
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("displacement Z")
        self.ax.set_zlim(-2, 2)

        # Initial surface
        Z0 = self.displacement(0)
        self.surf = self.ax.plot_surface(self.X, self.Y, Z0, cmap="plasma",
                                         antialiased=True, rstride=1, cstride=1, alpha=0.8)

    def displacement(self, t):
        # Vectorized calculation
        time_factors = np.cos(self.omegas * t)
        return np.sum(self.spatial_components * time_factors[:, np.newaxis, np.newaxis], axis=0)

    def update(self, frame):
        # Calculate time based on display FPS for consistent timing
        t = frame / self.display_fps
        Z = self.displacement(t)

        # Update the surface data
        self.surf.remove()
        self.surf = self.ax.plot_surface(self.X, self.Y, Z, cmap="plasma",
                                         antialiased=True, rstride=1, cstride=1, alpha=0.8)
        return self.surf,

    def animate(self):
        # Set up GIF writer if saving is enabled
        if self.save_gif:
            from matplotlib.animation import PillowWriter
            self.gif_writer = PillowWriter(fps=self.gif_fps, bitrate=1200)
            print(
                f"Saving animation as {self.gif_filename} at {self.gif_fps} FPS...")

            # Calculate how many frames to save for the GIF to match the duration
            gif_frames_total = int(self.T * self.gif_fps)
        else:
            gif_frames_total = self.frames_total

        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=gif_frames_total if self.save_gif else self.frames_total,
            interval=1000/self.display_fps,  # Display interval
            blit=False,
            repeat=True
        )

        plt.tight_layout()

        # Save GIF if enabled
        if self.save_gif:
            self.ani.save(self.gif_filename, writer=self.gif_writer, dpi=80)
            print(
                f"GIF saved! Duration: {self.T}s, FPS: {self.gif_fps}, Frames: {gif_frames_total}")

        plt.show()

    def save_animation(self, filename=None, fps=15, bitrate=1200):
        """Save the animation with specified FPS"""
        if filename:
            self.gif_filename = filename

        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=fps, bitrate=bitrate)

        # Create new animation with correct frame count for the desired FPS
        gif_frames_total = int(self.T * fps)

        temp_ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=gif_frames_total,
            interval=1000/self.display_fps,
            blit=False,
            repeat=True
        )

        print(f"Saving animation as {self.gif_filename} at {fps} FPS...")
        temp_ani.save(self.gif_filename, writer=writer, dpi=80)
        print(f"Animation saved! Duration: {self.T}s, FPS: {fps}")


# Example usage:
if __name__ == "__main__":
    # Option 1: Save GIF with same timing as display (15 FPS)
    anim = MembraneAnimation(
        save_gif=False, gif_filename="membrane_slow.gif", gif_fps=15)
    anim.animate()

    # Option 2: Show first, then save with custom FPS
    # anim = MembraneAnimation()
    # anim.animate()
    # anim.save_animation("membrane_custom.gif", fps=12)  # Even slower
