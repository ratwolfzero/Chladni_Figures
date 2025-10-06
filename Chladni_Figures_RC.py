import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import TypeAlias

class Config:
    # =========================================================
    # 🎛️ Frequency controls
    # =========================================================
    FREQ_RANGE = (1.0, 20.0)        # Range for driving frequency slider (Hz)
    FREQ_STEP = 0.01                # Frequency slider step size
    INIT_FREQ = 5.0                 # Initial driving frequency (Hz)

    # =========================================================
    # ⚖️ Damping (gamma) controls
    # =========================================================
    GAMMA_RANGE = (0.001, 0.15)     # Range for damping coefficient γ
    GAMMA_STEP = 0.001              # Damping slider step size
    INIT_GAMMA = 0.01               # Default damping value (γ)

    # =========================================================
    # 🔢 Mode / resonance parameters
    # =========================================================
    MAX_MODE = 15                   # Maximum mode indices m,n to compute
    RESONANCE_TOL = 0.02            # Frequency tolerance for resonance detection
    MODE_WEIGHT_THRESHOLD = 1.1     # Minimum % weight for mode to be listed
    MAX_DISPLAY_MODES = None        # Limit number of modes shown (None = all)

    # =========================================================
    # 🧮 Simulation grid & scaling
    # =========================================================
    RESOLUTION = 200                # Grid resolution for spatial mode shapes
    K = 1.0                         # Frequency scaling factor for eigenmodes

    # =========================================================
    # 🖥️ UI & animation behavior
    # =========================================================
    SCAN_SPEED = 0.03               # Frequency step per frame during Auto Scan
    SHOW_AXES = False               # Toggle for showing coordinate axes

    # =========================================================
    # 📈 Resonance curve plot settings
    # =========================================================
    RESONANCE_CURVE_RANGE = 1       # Frequency range around resonance (Hz)
    RESONANCE_CURVE_SAMPLES = 10000 # Number of sampling points per Lorentzian



Mode: TypeAlias = tuple[int, int, float]


# =========================================================
# 🧮 Chladni Simulator
# =========================================================
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

        # Precompute mode shapes and frequencies
        self.mode_shapes = []
        self.mode_frequencies = []
        self._precompute_modes()

        # Sorted eigenfrequencies
        self.eigenfrequencies = [
            (m, n, f_mn)
            for (m, n), f_mn in zip(
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
                mode_shape = np.sin(m * np.pi * self.X) * np.sin(n * np.pi * self.Y)
                self.mode_shapes.append(mode_shape)

        self.mode_frequencies = np.array(self.mode_frequencies)
        self.mode_shapes = np.array(self.mode_shapes, dtype=np.float32)

    def compute_displacement(self, f: float) -> np.ndarray:
        weights = 1.0 / ((f - self.mode_frequencies) ** 2 + self.gamma ** 2)
        Z = np.sum(weights[:, np.newaxis, np.newaxis] * self.mode_shapes, axis=0)
        return Z

    def compute_lorentzian_weights(self, f_range: np.ndarray, target_freq: float) -> np.ndarray:
        return 1.0 / ((f_range - target_freq) ** 2 + self.gamma ** 2)

    def get_closest_resonance_info(self, current_f: float) -> tuple[float, list[tuple[int, int]]]:
        idx_closest = np.argmin(np.abs(self.mode_frequencies - current_f))
        f_closest = self.mode_frequencies[idx_closest]

        mode_list = [(m, n) for m in range(1, self.max_mode + 1)
                     for n in range(1, self.max_mode + 1)]
        degenerate_modes = [(m, n) for idx, (m, n) in enumerate(mode_list)
                            if abs(self.mode_frequencies[idx] - f_closest) < 1e-6]

        return f_closest, degenerate_modes

    def get_mode_weight_at_frequency(self, f: float) -> np.ndarray:
        return 1.0 / ((f - self.mode_frequencies) ** 2 + self.gamma ** 2)


# =========================================================
# 📈 Resonance Curve Window
# =========================================================
class ResonanceCurveWindow:
    """Separate window for displaying Lorentzian resonance curves."""

    def __init__(self, simulator: ChladniSimulator, main_ui):
        self.simulator = simulator
        self.main_ui = main_ui
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title('Lorentzian Resonance Curves')

        self.current_f = self.main_ui.freq_slider.val
        self.current_resonance_freq, self.current_modes = self.simulator.get_closest_resonance_info(self.current_f)
        self.current_gamma = self.simulator.gamma
        self.current_marker = None

        self.setup_curve()

        # Connect updates
        self.main_ui.freq_slider.on_changed(self.update_current_freq)
        self.main_ui.gamma_slider.on_changed(self.update_gamma)

    def setup_curve(self) -> None:
        self.ax.clear()

        self.current_resonance_freq, self.current_modes = self.simulator.get_closest_resonance_info(self.current_f)

        # Frequency range and sampling
        f_min = max(Config.FREQ_RANGE[0], self.current_resonance_freq - Config.RESONANCE_CURVE_RANGE)
        f_max = min(Config.FREQ_RANGE[1], self.current_resonance_freq + Config.RESONANCE_CURVE_RANGE)
        self.f_range = np.linspace(f_min, f_max, Config.RESONANCE_CURVE_SAMPLES)

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.current_modes)))

        for (m, n), color in zip(self.current_modes, colors):
            lorentzian = self.simulator.compute_lorentzian_weights(self.f_range, self.current_resonance_freq)
            self.ax.plot(self.f_range, lorentzian, '-', color=color, linewidth=2, label=f'Mode ({m},{n})')

        # Add resonance frequency and marker
        max_weight = 1.0 / (self.simulator.gamma ** 2)
        self.ax.axvline(self.current_resonance_freq, color='red', linestyle='--', alpha=0.7,
                        label=f'Resonance: f={self.current_resonance_freq:.2f}')
        self.add_current_marker()

        # Info text and FWHM
        fwhm = 2 * self.simulator.gamma
        half_max = max_weight / 2
        self.ax.axhline(half_max, color='gray', linestyle=':', alpha=0.5, label='Half Maximum')
        self.ax.text(0.02, 0.98, f'γ = {self.simulator.gamma:.3f}', transform=self.ax.transAxes,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.ax.text(0.02, 0.88, f'FWHM = {fwhm:.3f}', transform=self.ax.transAxes,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        modes_str = ', '.join([f"({m},{n})" for m, n in self.current_modes])
        self.ax.set_title(f'Lorentzian Resonance Curves for Mode(s): {modes_str}')

        self.ax.set_xlabel('Driving Frequency (f)')
        self.ax.set_ylabel('Resonance Weight')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(f_min, f_max)
        self.ax.set_ylim(0, max_weight * 1.1)

        self.fig.canvas.draw_idle()

    def add_current_marker(self) -> None:
        current_weights = self.simulator.get_mode_weight_at_frequency(self.current_f)
        mode_list = [(m, n) for m in range(1, self.simulator.max_mode + 1)
                     for n in range(1, self.simulator.max_mode + 1)]

        resonance_weight = 0
        for idx, (m, n) in enumerate(mode_list):
            if abs(self.simulator.mode_frequencies[idx] - self.current_resonance_freq) < 1e-6:
                resonance_weight = current_weights[idx]
                break

        self.current_marker, = self.ax.plot(self.current_f, resonance_weight, 'go', markersize=8,
                                            label=f'Current: f={self.current_f:.2f}, weight={resonance_weight:.3f}')

    def update_current_freq(self, val: float) -> None:
        if not plt.fignum_exists(self.fig.number):
            return

        new_f = self.main_ui.freq_slider.val
        new_resonance_freq, new_modes = self.simulator.get_closest_resonance_info(new_f)

        resonance_changed = False
        if abs(new_resonance_freq - self.current_resonance_freq) > 1e-6:
            resonance_changed = True
        if set(new_modes) != set(self.current_modes):
            resonance_changed = True
        if hasattr(self, 'f_range') and (new_f < self.f_range[0] or new_f > self.f_range[-1]):
            resonance_changed = True

        self.current_f = new_f
        if resonance_changed:
            self.current_resonance_freq = new_resonance_freq
            self.current_modes = new_modes
            self.setup_curve()
        else:
            self.update_current_marker()

    def update_gamma(self, val: float) -> None:
        if plt.fignum_exists(self.fig.number):
            self.current_gamma = self.simulator.gamma
            self.setup_curve()

    def update_current_marker(self) -> None:
        if self.current_marker is not None:
            self.current_marker.remove()

        current_weights = self.simulator.get_mode_weight_at_frequency(self.current_f)
        mode_list = [(m, n) for m in range(1, self.simulator.max_mode + 1)
                     for n in range(1, self.simulator.max_mode + 1)]

        resonance_weight = 0
        for idx, (m, n) in enumerate(mode_list):
            if abs(self.simulator.mode_frequencies[idx] - self.current_resonance_freq) < 1e-6:
                resonance_weight = current_weights[idx]
                break

        self.current_marker, = self.ax.plot(self.current_f, resonance_weight, 'go', markersize=8,
                                            label=f'Current: f={self.current_f:.2f}, weight={resonance_weight:.3f}')
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


# =========================================================
# 🖥️ Main Chladni UI
# =========================================================
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
        self.resonance_window = None
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

        # New Resonance Curve button
        ax_resonance = plt.axes([0.50, 0.1, 0.15, 0.04])
        self.resonance_button = Button(ax_resonance, 'Resonance Curves')
        self.resonance_button.on_clicked(self.open_resonance_curve)

    def open_resonance_curve(self, event) -> None:
        """Open the resonance curve window."""
        if self.resonance_window is None or not plt.fignum_exists(self.resonance_window.fig.number):
            self.resonance_window = ResonanceCurveWindow(self.simulator, self)
        plt.figure(self.resonance_window.fig.number)
        plt.show()

    def update(self, val: float) -> None:
        f = round(self.freq_slider.val, 2)
        Z = self.simulator.compute_displacement(f)
        Z_plot = np.abs(Z) ** 0.2

        self.im.set_array(Z_plot)
        self.im.set_clim(Z_plot.min(), Z_plot.max())
        self.cbar.update_normal(self.im)

        # --- Resonance / degenerate modes ---
        f_closest, degenerate_modes = self.simulator.get_closest_resonance_info(
            f)

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

        mode_list = [(m, n) for m in range(1, self.simulator.max_mode + 1)
                     for n in range(1, self.simulator.max_mode + 1)]
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


# =========================================================
# 🚀 Main Entry
# =========================================================
def main() -> None:
    simulator = ChladniSimulator()
    ui = ChladniUI(simulator)
    ui.show()


if __name__ == "__main__":
    main()
