import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import List, Tuple
import sounddevice as sd  # pip install sounddevice


class ChladniSimulator:
    """
    A class to simulate and visualize Chladni figures for a square membrane.
    Handles the core computation and data management.
    """

    def __init__(self, resolution: int = 200, max_mode: int = 15, gamma: float = 0.03, k: float = 1.0):
        self.resolution = resolution
        self.max_mode = max_mode
        self.gamma = gamma
        self.k = k

        # Create the spatial grid
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

        # Precompute mode shapes and frequencies for performance
        self.mode_shapes = []
        self.mode_frequencies = []
        self._precompute_modes()

        # Sorted list of eigenfrequencies for UI
        self.eigenfrequencies = [
            (m, n, f_mn) for (m, n), f_mn in zip(
                [(m, n) for m in range(1, max_mode + 1)
                 for n in range(1, max_mode + 1)],
                self.mode_frequencies
            )
        ]
        self.eigenfrequencies.sort(key=lambda x: x[2])

    def _precompute_modes(self):
        """Precompute all mode shapes and store them."""
        for m in range(1, self.max_mode + 1):
            for n in range(1, self.max_mode + 1):
                f_mn = self.k * np.sqrt(m**2 + n**2)
                self.mode_frequencies.append(f_mn)
                mode_shape = np.sin(m * np.pi * self.X) * np.sin(n * np.pi * self.Y)
                self.mode_shapes.append(mode_shape)
        self.mode_frequencies = np.array(self.mode_frequencies)
        self.mode_shapes = np.array(self.mode_shapes)

    def compute_displacement(self, f: float) -> np.ndarray:
        weights = 1.0 / ((f - self.mode_frequencies) ** 2 + self.gamma ** 2)
        Z = np.sum(weights[:, np.newaxis, np.newaxis] * self.mode_shapes, axis=0)
        return Z

    def find_close_modes(self, f: float, tolerance: float = 0.1) -> List[Tuple[int, int, float]]:
        return [(m, n, fmn) for m, n, fmn in self.eigenfrequencies if abs(f - fmn) < tolerance]


def play_tone(frequencies, duration: float = 1.0, sample_rate: int = 44100,
              amplitude: float = 0.2, gamma: float = 0.01):
    """
    Play a combination of sine waves for given frequencies.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.zeros_like(t)
    for f in frequencies:
        wave += np.sin(2 * np.pi * f * t)
    wave /= max(len(frequencies), 1)  # normalize
    wave *= amplitude * np.exp(-gamma * t)  # damping
    sd.play(wave, samplerate=sample_rate)
    sd.wait()


class ChladniUI:
    """Handles the matplotlib UI for Chladni simulation."""

    def __init__(self, simulator: ChladniSimulator, show_axes: bool = True):
        self.simulator = simulator
        self.show_axes = show_axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)

        self.init_freq = 5.0
        Z_init = self.simulator.compute_displacement(self.init_freq)
        self.im = self.ax.imshow(
            np.abs(Z_init) ** 0.2, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
        self.cbar = plt.colorbar(self.im, ax=self.ax, label='Displacement (|Z|^0.2)')

        self._setup_axes()
        self._setup_widgets()

        self.scan_ani = None
        self.update(self.init_freq)

    def _setup_axes(self):
        if self.show_axes:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_xlabel('')
            self.ax.set_ylabel('')

    def _setup_widgets(self):
        ax_freq = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.freq_slider = Slider(ax_freq, 'Frequency', 0.0, 20.0,
                                  valinit=self.init_freq, valstep=0.01)
        self.freq_slider.on_changed(self.update)

        ax_jump = plt.axes([0.1, 0.05, 0.15, 0.03])
        self.jump_button = Button(ax_jump, 'Next Resonance')
        self.jump_button.on_clicked(self.jump_to_next_resonance)

        ax_scan = plt.axes([0.27, 0.05, 0.1, 0.03])
        self.scan_button = Button(ax_scan, 'Auto Scan')
        self.scan_button.on_clicked(self.start_scan)
																												   
        ax_stop = plt.axes([0.39, 0.05, 0.1, 0.03])
        self.stop_button = Button(ax_stop, 'Stop Scan')
        self.stop_button.on_clicked(self.stop_scan)

        ax_play = plt.axes([0.51, 0.05, 0.1, 0.03])
        self.play_button = Button(ax_play, 'Play Sound')
        self.play_button.on_clicked(self.play_current_frequency)

    def update(self, val: float):
        f = self.freq_slider.val
        Z = self.simulator.compute_displacement(f)
        Z_plot = np.abs(Z) ** 0.2
        self.im.set_array(Z_plot)
        self.im.set_clim(Z_plot.min(), Z_plot.max())
        self.cbar.update_normal(self.im)

        matching_modes = self.simulator.find_close_modes(f)
        if matching_modes:
            modes_str = ", ".join([f"({m},{n})" for m, n, _ in matching_modes])
            title = f"Chladni Figures: f = {f:.3f}, Modes {modes_str}: f_mn ≈ {matching_modes[0][2]:.3f}"
        else:
            title = f"Chladni Figures: f = {f:.3f}"
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    def jump_to_next_resonance(self, event):
        current_f = self.freq_slider.val
        next_f = min([fmn for _, _, fmn in self.simulator.eigenfrequencies if fmn > current_f],
                     default=self.simulator.eigenfrequencies[0][2])
        self.freq_slider.set_val(next_f)

    def start_scan(self, event):
        from matplotlib.animation import FuncAnimation
        if self.scan_ani is not None:
            self.scan_ani.event_source.stop()

        def update_scan(frame):
            f = self.freq_slider.val + 0.05
            if f > 20.0:
                f = 0.0
            self.freq_slider.set_val(f)
            return self.im,

        self.scan_ani = FuncAnimation(
            self.fig, update_scan, interval=50, blit=True, cache_frame_data=False)
        self.fig.canvas.draw_idle()

    def stop_scan(self, event):
        if self.scan_ani is not None:
            self.scan_ani.event_source.stop()
            self.scan_ani = None

    def play_current_frequency(self, event):
        f = self.freq_slider.val
        scale_factor = 50  # map simulation freq to audible Hz
        matching_modes = self.simulator.find_close_modes(f, tolerance=0.5)
        if matching_modes:
            audible_freqs = [fmn * scale_factor for _, _, fmn in matching_modes]
        else:
            audible_freqs = [f * scale_factor]
        play_tone(audible_freqs, duration=1.5, gamma=self.simulator.gamma)

    def show(self):
        plt.show()


def main():
    simulator = ChladniSimulator(resolution=200, max_mode=15, gamma=0.01, k=1.0)
    ui = ChladniUI(simulator, show_axes=False)
    ui.show()


if __name__ == "__main__":
    main()
