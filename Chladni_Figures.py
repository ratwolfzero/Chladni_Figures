import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import TypeAlias

# =========================================================
# 🎛️ Simulation Configuration
# =========================================================


class Config:
    # Frequency controls
    FREQ_RANGE = (1.0, 20.0)        # Range for frequency slider
    FREQ_STEP = 0.01                # Step size for frequency slider
    INIT_FREQ = 5.0                 # Initial frequency

    # Damping (gamma) controls
    GAMMA_RANGE = (0.001, 0.15)     # Range for damping slider
    GAMMA_STEP = 0.001              # Step size for damping slider
    INIT_GAMMA = 0.01               # Default damping

    # Mode / resonance parameters
    MAX_MODE = 15                   # m,n up to this number
    RESONANCE_TOL = 0.02            # Tolerance for resonance tagging
    MODE_WEIGHT_THRESHOLD = 1.1     # % threshold for list inclusion
    MAX_DISPLAY_MODES = None        # Max number of modes shown (None = all)

    # Simulation resolution & scaling
    RESOLUTION = 200                # Grid resolution
    K = 1.0                         # Frequency scale factor

    # UI / Animation
    SCAN_SPEED = 0.03               # Frequency increment during Auto Scan
    SHOW_AXES = False               # Toggle axes visibility
# =========================================================


Mode: TypeAlias = tuple[int, int, float]


class ChladniSimulator:
    """Simulate Chladni figures for a square membrane."""

    def __init__(self):
        self.resolution = Config.RESOLUTION
        self.max_mode = Config.MAX_MODE
        self.gamma = Config.INIT_GAMMA
        self.k = Config.K

        # Spatial grid
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

        # Precompute modes
        self.mode_shapes = []
        self.mode_frequencies = []
        self._precompute_modes()

        # Sorted eigenfrequencies
        self.eigenfrequencies = [
            (m, n, f_mn) for (m, n), f_mn in zip(
                [(m, n) for m in range(1, self.max_mode + 1)
                 for n in range(1, self.max_mode + 1)],
                self.mode_frequencies
            )
        ]
        self.eigenfrequencies.sort(key=lambda x: x[2])

    def _precompute_modes(self) -> None:
        for m in range(1, self.max_mode + 1):
            for n in range(1, self.max_mode + 1):
                f_mn = self.k * np.sqrt(m ** 2 + n ** 2)
                self.mode_frequencies.append(f_mn)
                mode_shape = np.sin(m * np.pi * self.X) * \
                    np.sin(n * np.pi * self.Y)
                self.mode_shapes.append(mode_shape)

        self.mode_frequencies = np.array(self.mode_frequencies)
        self.mode_shapes = np.array(self.mode_shapes, dtype=np.float32)

    def compute_displacement(self, f: float) -> np.ndarray:
        weights = 1.0 / ((f - self.mode_frequencies) ** 2 + self.gamma ** 2)
        Z = np.sum(weights[:, np.newaxis, np.newaxis]
                   * self.mode_shapes, axis=0)
        return Z


class ChladniUI:
    """Matplotlib UI for Chladni simulator."""

    def __init__(self, simulator: ChladniSimulator):
        self.simulator = simulator
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.info_ax = self.fig.add_subplot(gs[1])
        self.info_ax.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.32, top=0.95)

        # Initialize plot
        self.init_freq = Config.INIT_FREQ
        Z_init = self.simulator.compute_displacement(self.init_freq)
        self.im = self.ax.imshow(
            np.abs(Z_init) ** 0.2, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
        self.cbar = plt.colorbar(
            self.im, ax=self.ax, label='Displacement (|Z|^0.2)')
        self._setup_axes()
        self._setup_widgets()

        self.mode_text = self.info_ax.text(
            0, 1, '', va='top', ha='left', fontsize=12)
        self.scan_ani = None
        self.update(self.init_freq)

    def _setup_axes(self) -> None:
        if Config.SHOW_AXES:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_xlabel('')
            self.ax.set_ylabel('')

    def _setup_widgets(self) -> None:
        ax_freq = plt.axes([0.05, 0.25, 0.8, 0.03])
        self.freq_slider = Slider(
            ax_freq, 'f.', *Config.FREQ_RANGE, valinit=Config.INIT_FREQ, valstep=Config.FREQ_STEP)
        self.freq_slider.on_changed(self.update)

        ax_gamma = plt.axes([0.05, 0.2, 0.8, 0.03])
        self.gamma_slider = Slider(
            ax_gamma, 'γ', *Config.GAMMA_RANGE, valinit=Config.INIT_GAMMA, valstep=Config.GAMMA_STEP)
        self.gamma_slider.on_changed(self.update_gamma)

        ax_prev = plt.axes([0.05, 0.1, 0.08, 0.04])
        self.prev_button = Button(ax_prev, '◀')
        self.prev_button.on_clicked(self.jump_to_prev_resonance)

        ax_next = plt.axes([0.14, 0.1, 0.08, 0.04])
        self.next_button = Button(ax_next, '▶')
        self.next_button.on_clicked(self.jump_to_next_resonance)

        plt.text(x=0.135, y=0.06, s="Resonance Navigation",
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 transform=self.fig.transFigure)

        ax_scan = plt.axes([0.25, 0.1, 0.1, 0.04])
        self.scan_button = Button(ax_scan, 'Auto Scan')
        self.scan_button.on_clicked(self.start_scan)

        ax_stop = plt.axes([0.37, 0.1, 0.1, 0.04])
        self.stop_button = Button(ax_stop, 'Stop Scan')
        self.stop_button.on_clicked(self.stop_scan)

    def update(self, val: float) -> None:
        f = round(self.freq_slider.val, 2)
        Z = self.simulator.compute_displacement(f)
        Z_plot = np.abs(Z) ** 0.2

        self.im.set_array(Z_plot)
        self.im.set_clim(Z_plot.min(), Z_plot.max())
        self.cbar.update_normal(self.im)

        # --- Resonance / degenerate modes ---
        idx_closest = np.argmin(np.abs(self.simulator.mode_frequencies - f))
        f_closest = self.simulator.mode_frequencies[idx_closest]

        mode_list = [(m, n) for m in range(1, self.simulator.max_mode + 1)
                     for n in range(1, self.simulator.max_mode + 1)]
        degenerate_modes = [(m, n) for idx, (m, n) in enumerate(mode_list)
                            if abs(self.simulator.mode_frequencies[idx] - f_closest) < 1e-6]

        title = f"Chladni Figures: f = {f:.2f}"
        if abs(f - f_closest) < Config.RESONANCE_TOL:
            deg_modes_str = ', '.join(
                [f"({m},{n})" for m, n in degenerate_modes])
            title += f"  ← Resonance: {deg_modes_str} f_mn={f_closest:.2f}"

        self.ax.set_title(title)

        # --- Detailed mode table ---
        weights = 1.0 / ((f - self.simulator.mode_frequencies)
                         ** 2 + self.simulator.gamma ** 2)
        total_weight = np.sum(weights)
        percentages = (weights / total_weight) * \
            100 if total_weight > 0 else np.zeros_like(weights)

        modes_info = []
        for idx, (m, n) in enumerate(mode_list):
            fmn = self.simulator.mode_frequencies[idx]
            perc = percentages[idx]
            if perc > Config.MODE_WEIGHT_THRESHOLD:
                modes_info.append((m, n, fmn, perc))

        modes_info.sort(key=lambda x: x[3], reverse=True)
        text_str = "Contributing Modes (%):\n\n"
        max_modes = Config.MAX_DISPLAY_MODES or len(modes_info)
        for m, n, fmn, perc in modes_info[:max_modes]:
            text_str += f"({m},{n}) f={fmn:.2f}: {perc:.1f}%\n"
        self.mode_text.set_text(text_str)

        self.fig.canvas.draw_idle()

    def update_gamma(self, val: float) -> None:
        self.simulator.gamma = val
        self.update(self.freq_slider.val)

    def jump_to_next_resonance(self, event) -> None:
        current_f = self.freq_slider.val
        next_f = min(
            [fmn for _, _, fmn in self.simulator.eigenfrequencies if fmn > current_f],
            default=self.simulator.eigenfrequencies[0][2] if self.simulator.eigenfrequencies else current_f
        )
        self.freq_slider.set_val(next_f)

    def jump_to_prev_resonance(self, event) -> None:
        current_f = self.freq_slider.val
        prev_f = max(
            [fmn for _, _, fmn in self.simulator.eigenfrequencies if fmn < current_f],
            default=self.simulator.eigenfrequencies[-1][2] if self.simulator.eigenfrequencies else current_f
        )
        self.freq_slider.set_val(prev_f)

    def start_scan(self, event) -> None:
        if self.scan_ani is not None:
            self.scan_ani.event_source.stop()

        def update_scan(frame):
            f = self.freq_slider.val + Config.SCAN_SPEED
            if f > Config.FREQ_RANGE[1]:
                f = Config.FREQ_RANGE[0]
            self.freq_slider.set_val(f)
            return self.im,

        self.scan_ani = FuncAnimation(
            self.fig, update_scan, interval=50, blit=True, cache_frame_data=False)
        self.fig.canvas.draw_idle()

    def stop_scan(self, event) -> None:
        if self.scan_ani is not None:
            self.scan_ani.event_source.stop()
            self.scan_ani = None

    def show(self) -> None:
        plt.show()


def main() -> None:
    simulator = ChladniSimulator()
    ui = ChladniUI(simulator)
    ui.show()


if __name__ == "__main__":
    main()
