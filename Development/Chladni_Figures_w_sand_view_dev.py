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
    FREQ_RANGE = (1.0, 20.0)  # Range for driving frequency slider (Hz)
    FREQ_STEP = 0.01  # Frequency slider step size
    INIT_FREQ = 5.0  # Initial driving frequency (Hz)
    # =========================================================
    # ⚖️ Damping (gamma) controls
    # =========================================================
    GAMMA_RANGE = (0.001, 0.05)  # Range for damping coefficient γ
    GAMMA_STEP = 0.001  # Damping slider step size
    INIT_GAMMA = 0.01  # Default damping value (γ)
    # =========================================================
    # 🔢 Mode / resonance parameters
    # =========================================================
    MAX_MODE = 15  # Maximum mode indices m,n to compute
    RESONANCE_TOL = 0.02  # Frequency tolerance for resonance detection
    MODE_WEIGHT_THRESHOLD = 1.0  # Minimum % weight for mode to be listed
    MAX_DISPLAY_MODES = None  # Limit number of modes shown (None = all)
    EPS_FREQ_COMPARE = 1e-6  # Small epsilon for frequency equality check
    # =========================================================
    # 🧮 Simulation grid & scaling
    # =========================================================
    RESOLUTION = 200  # Grid resolution for spatial mode shapes
    K = 1.0  # Frequency scaling factor for eigenmodes
    # Exponent for magnitude visualization (|Z|^exp)
    VISUAL_EXPONENT = 0.2
    # =========================================================
    # 🖥️ UI & animation behavior
    # =========================================================
    SCAN_SPEED = 0.02  # Frequency step per frame during Auto Scan
    SHOW_AXES = False  # Toggle for showing coordinate axes
    # =========================================================
    # 📈 Resonance curve plot settings
    # =========================================================
    RESONANCE_CURVE_RANGE = 1  # Frequency range around resonance (Hz)
    RESONANCE_CURVE_SAMPLES = 20000  # Number of sampling points per Lorentzian
    # =========================================================
    # 🏖️ Sand simulation settings
    # =========================================================
    SAND_EXP_SCALE = 0.05  # Scale for exponential probability decay
    NUM_GRAINS = 30000  # Number of sand grains to simulate
    SAND_NOISE_STD = 0.5  # Standard deviation for noise in sand positions
    SAND_SIZE = 0.1  # Size of sand grains in scatter plot
    SAND_COLOR = 'black'  # Color of sand grains

Mode: TypeAlias = tuple[int, int, float]

# =========================================================
# 🧮 Chladni Simulator
# =========================================================
class ChladniSimulator:
    """Simulate Chladni figures for a square membrane."""

    def __init__(self):
        self.resolution = Config.RESOLUTION
        self.max_mode = Config.MAX_MODE
        self._gamma = Config.INIT_GAMMA
        self.k = Config.K
        # Spatial grid
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)
        # Precompute mode indices and frequencies vectorized
        ms, ns = np.meshgrid(np.arange(1, self.max_mode + 1),
                             np.arange(1, self.max_mode + 1))
        self._modes = list(zip(ms.ravel(), ns.ravel()))
        self._mode_frequencies = self.k * \
            np.sqrt(np.array([m**2 + n**2 for m, n in self._modes]))
        # Precompute all mode shapes as 3D array (modes × res × res)
        self._mode_shapes = np.array([
            np.sin(m * np.pi * self.X) * np.sin(n * np.pi * self.Y)
            for m, n in self._modes
        ], dtype=np.float64)
        # Sorted list of (m, n, frequency)
        self._eigenfrequencies = [
            (m, n, f_mn)
            for (m, n), f_mn in zip(self._modes, self._mode_frequencies)
        ]
        self._eigenfrequencies.sort(key=lambda x: x[2])

    @property
    def gamma(self) -> float:
        return self._gamma

    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma

    def compute_displacement(self, f: float) -> np.ndarray:
        """Compute total displacement field as weighted sum of mode shapes."""
        weights = self.get_mode_weights_at_frequency(f)
        Z = np.tensordot(weights, self._mode_shapes, axes=(0, 0))
        return Z

    def compute_lorentzian_weights(self, f_range: np.ndarray, target_freq: float) -> np.ndarray:
        return 1.0 / ((f_range - target_freq) ** 2 + self.gamma ** 2)

    def get_lorentzian_weight_at_freq(self, f: float, target_freq: float) -> float:
        return 1.0 / ((f - target_freq) ** 2 + self.gamma ** 2)

    def get_closest_resonance_info(self, current_f: float) -> tuple[float, list[tuple[int, int]]]:
        """Find nearest eigenfrequency and all degenerate modes at that frequency."""
        idx_closest = np.argmin(np.abs(self._mode_frequencies - current_f))
        f_closest = self._mode_frequencies[idx_closest]
        degenerate_modes = [
            mode for idx, mode in enumerate(self._modes)
            if abs(self._mode_frequencies[idx] - f_closest) < Config.EPS_FREQ_COMPARE
        ]
        return f_closest, degenerate_modes

    def get_mode_weights_at_frequency(self, f: float) -> np.ndarray:
        return 1.0 / ((f - self._mode_frequencies) ** 2 + self.gamma ** 2)

    def get_contributing_modes(self, f: float, threshold: float = Config.MODE_WEIGHT_THRESHOLD) -> list[tuple[int, int, float, float]]:
        weights = self.get_mode_weights_at_frequency(f)
        total_weight = np.sum(weights)
        if total_weight == 0:
            return []
        percentages = (weights / total_weight) * 100
        modes_info = [
            (m, n, self._mode_frequencies[i], percentages[i])
            for i, (m, n) in enumerate(self._modes)
            if percentages[i] > threshold
        ]
        modes_info.sort(key=lambda x: x[3], reverse=True)
        return modes_info

    def get_sand_coordinates_from_Z(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate sand grain positions using importance sampling based on |Z|."""
        max_z = np.max(np.abs(Z))
        if max_z == 0:
            max_z = 1e-12
        # Probability ~ exp(-|displacement| / scale)
        p = np.exp(-np.abs(Z) / (max_z * Config.SAND_EXP_SCALE))
        p_flat = p.ravel()
        total = p_flat.sum()
        if total > 0:
            p_flat /= total
        else:
            # Rare fallback — uniform
            p_flat.fill(1.0 / p_flat.size)
        indices = np.random.choice(
            p_flat.size,
            size=Config.NUM_GRAINS,
            p=p_flat
        )
        y_idx, x_idx = np.unravel_index(
            indices, (self.resolution, self.resolution)
        )
        # Add small Gaussian noise for more natural distribution
        x = x_idx + np.random.normal(0, Config.SAND_NOISE_STD, size=len(x_idx))
        y = y_idx + np.random.normal(0, Config.SAND_NOISE_STD, size=len(y_idx))
        return x, y

    def get_next_resonance_frequency(self, current_f: float) -> float:
        freqs = [fmn for _, _, fmn in self._eigenfrequencies if fmn > current_f]
        if not freqs:
            return self._eigenfrequencies[0][2] if self._eigenfrequencies else current_f
        return min(freqs)

    def get_previous_resonance_frequency(self, current_f: float) -> float:
        freqs = [fmn for _, _, fmn in self._eigenfrequencies if fmn < current_f]
        if not freqs:
            return self._eigenfrequencies[-1][2] if self._eigenfrequencies else current_f
        return max(freqs)

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
        self.current_f_raw = self.main_ui.freq_slider.val
        self.current_f_display = round(self.current_f_raw, 2)
        self.current_resonance_freq, self.current_modes = self.simulator.get_closest_resonance_info(
            self.current_f_display)
        self.current_gamma = self.simulator.gamma
        self.current_marker = None
        self.setup_curve()
        # Connect updates
        self.main_ui.freq_slider.on_changed(self.update_current_freq)
        self.main_ui.gamma_slider.on_changed(self.update_gamma)

    def setup_curve(self) -> None:
        self.ax.clear()
        self.current_resonance_freq, self.current_modes = self.simulator.get_closest_resonance_info(
            self.current_f_display)
        f_min = max(
            Config.FREQ_RANGE[0], self.current_resonance_freq - Config.RESONANCE_CURVE_RANGE)
        f_max = min(
            Config.FREQ_RANGE[1], self.current_resonance_freq + Config.RESONANCE_CURVE_RANGE)
        self.f_range = np.linspace(
            f_min, f_max, Config.RESONANCE_CURVE_SAMPLES)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.current_modes)))
        for (m, n), color in zip(self.current_modes, colors):
            lorentzian = self.simulator.compute_lorentzian_weights(
                self.f_range, self.current_resonance_freq)
            self.ax.plot(self.f_range, lorentzian, '-', color=color,
                         linewidth=2, label=f'Mode ({m},{n})')
        max_weight = self.simulator.get_lorentzian_weight_at_freq(self.current_resonance_freq, self.current_resonance_freq)
        self.ax.axvline(self.current_resonance_freq, color='red', linestyle='--', alpha=0.7,
                        label=f'Resonance: f={self.current_resonance_freq:.2f}')
        self.add_current_marker()
        fwhm = 2 * self.simulator.gamma
        half_max = max_weight / 2
        self.ax.axhline(half_max, color='gray', linestyle=':',
                        alpha=0.5, label='Half Maximum')
        self.ax.text(0.02, 0.98, f'γ = {self.simulator.gamma:.3f}', transform=self.ax.transAxes,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.ax.text(0.02, 0.88, f'FWHM = {fwhm:.3f}', transform=self.ax.transAxes,
                     va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        modes_str = ', '.join([f"({m},{n})" for m, n in self.current_modes])
        self.ax.set_title(
            f'Lorentzian Resonance Curves for dominant Mode(s): {modes_str}\n The height of the amplitude remains constant, only the width changes, with the y-axis scaled by a variable weight.')
        self.ax.set_xlabel('Driving Frequency (f)')
        self.ax.set_ylabel('Resonance Weight')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(f_min, f_max)
        self.ax.set_ylim(0, max_weight * 1.1)
        self.fig.canvas.draw_idle()

    def add_current_marker(self) -> None:
        resonance_weight = self.simulator.get_lorentzian_weight_at_freq(
            self.current_f_raw, self.current_resonance_freq)
        self.current_marker, = self.ax.plot(self.current_f_raw, resonance_weight, 'go', markersize=8,
                                            label=f'Current: f={self.current_f_display:.2f}, weight={resonance_weight:.3f}')

    def update_current_freq(self, val: float) -> None:
        if not plt.fignum_exists(self.fig.number):
            return
        new_f_raw = self.main_ui.freq_slider.val
        new_f_display = round(new_f_raw, 2)
        new_resonance_freq, new_modes = self.simulator.get_closest_resonance_info(
            new_f_display)
        resonance_changed = False
        if abs(new_resonance_freq - self.current_resonance_freq) > Config.EPS_FREQ_COMPARE:
            resonance_changed = True
        if set(new_modes) != set(self.current_modes):
            resonance_changed = True
        if hasattr(self, 'f_range') and (new_f_raw < self.f_range[0] or new_f_raw > self.f_range[-1]):
            resonance_changed = True
        self.current_f_raw = new_f_raw
        self.current_f_display = new_f_display
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
        resonance_weight = self.simulator.get_lorentzian_weight_at_freq(
            self.current_f_raw, self.current_resonance_freq)
        self.current_marker, = self.ax.plot(self.current_f_raw, resonance_weight, 'go', markersize=8,
                                            label=f'Current: f={self.current_f_display:.2f}, weight={resonance_weight:.3f}')
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
    """Matplotlib UI for Chladni simulator with phase view and sand view toggle."""

    def __init__(self, simulator: ChladniSimulator):
        self.simulator = simulator
        self.view_mode = 'magnitude'  # 'magnitude', 'phase', 'sand'
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.info_ax = self.fig.add_subplot(gs[1])
        self.info_ax.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.95)
        # Store original axes limits
        self.orig_xlim = (0, 1)
        self.orig_ylim = (0, 1)
        # Initialize plots
        Z_init = self.simulator.compute_displacement(Config.INIT_FREQ)
        plot_data = np.abs(Z_init) ** Config.VISUAL_EXPONENT
        self.imshow_artist = self.ax.imshow(
            plot_data, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
        self.scatter_artist = self.ax.scatter(
            [], [], s=Config.SAND_SIZE, c=Config.SAND_COLOR, marker='.')
        self.scatter_artist.set_visible(False)
        # Colorbar created once
        self.cbar = self.fig.colorbar(
            self.imshow_artist,
            ax=self.ax,
            label=f'Displacement (|Z|^{Config.VISUAL_EXPONENT})'
        )
        self._setup_axes()
        self._setup_widgets()
        self.mode_text = self.info_ax.text(
            0, 1, '', va='top', ha='left', fontsize=12)
        self.mode_text.set_fontfamily('Monospace')
        self.scan_ani = None
        self.resonance_window = None
        self.update(Config.INIT_FREQ)

    def _setup_axes(self) -> None:
        self.ax.set_facecolor('white')
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
            ax_freq, 'f.', *Config.FREQ_RANGE,
            valinit=Config.INIT_FREQ, valstep=Config.FREQ_STEP)
        self.freq_slider.on_changed(self.update)
        ax_gamma = plt.axes([0.05, 0.2, 0.8, 0.03])
        self.gamma_slider = Slider(
            ax_gamma, 'γ', *Config.GAMMA_RANGE,
            valinit=Config.INIT_GAMMA, valstep=Config.GAMMA_STEP)
        self.gamma_slider.on_changed(self.update_gamma)
        ax_prev = plt.axes([0.05, 0.1, 0.08, 0.04])
        self.prev_button = Button(ax_prev, '◀')
        self.prev_button.on_clicked(self.jump_to_prev_resonance)
        ax_next = plt.axes([0.14, 0.1, 0.08, 0.04])
        self.next_button = Button(ax_next, '▶')
        self.next_button.on_clicked(self.jump_to_next_resonance)
        plt.text(0.135, 0.06, "Resonance Navigation",
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 transform=self.fig.transFigure)
        ax_scan = plt.axes([0.25, 0.1, 0.1, 0.04])
        self.scan_button = Button(ax_scan, 'Auto Scan')
        self.scan_button.on_clicked(self.start_scan)
        ax_stop = plt.axes([0.37, 0.1, 0.1, 0.04])
        self.stop_button = Button(ax_stop, 'Stop Scan')
        self.stop_button.on_clicked(self.stop_scan)
        ax_resonance = plt.axes([0.69, 0.1, 0.15, 0.04])
        self.resonance_button = Button(ax_resonance, 'Show Resonance Curves')
        self.resonance_button.on_clicked(self.open_resonance_curve)
        ax_toggle = plt.axes([0.50, 0.1, 0.16, 0.04])
        self.toggle_button = Button(ax_toggle, 'Toggle to Phase View')
        self.toggle_button.on_clicked(self.toggle_view)

    def toggle_view(self, event) -> None:
        modes = ['magnitude', 'phase', 'sand']
        labels = ['Magnitude View', 'Phase View', 'Sand View']
        current_idx = modes.index(self.view_mode)
        next_idx = (current_idx + 1) % 3
        self.view_mode = modes[next_idx]
        next_next_idx = (next_idx + 1) % 3
        self.toggle_button.label.set_text(f'Toggle to {labels[next_next_idx]}')
        self.update(self.freq_slider.val)

    def open_resonance_curve(self, event) -> None:
        if self.resonance_window is None or not plt.fignum_exists(self.resonance_window.fig.number):
            self.resonance_window = ResonanceCurveWindow(self.simulator, self)
        plt.figure(self.resonance_window.fig.number)
        plt.show()

    def update(self, val: float) -> None:
        f_compute = val
        f_display = round(val, 2)
        # ─── Single expensive computation ───────────────────────────────
        Z = self.simulator.compute_displacement(f_compute)
        # ────────────────────────────────────────────────────────────────
        label = ''
        if self.view_mode == 'sand':
            self.imshow_artist.set_visible(False)
            self.scatter_artist.set_visible(True)
            if hasattr(self, 'cbar'):
                self.cbar.ax.set_visible(False)
            x, y = self.simulator.get_sand_coordinates_from_Z(Z)
            x_plot = x / (self.simulator.resolution - 1)
            y_plot = y / (self.simulator.resolution - 1)
            offsets = np.column_stack((x_plot, y_plot))
            self.scatter_artist.set_offsets(offsets)
        else:
            self.scatter_artist.set_visible(False)
            self.imshow_artist.set_visible(True)
            if self.view_mode == 'phase':
                plot_data = Z
                cmap = 'coolwarm'
                max_abs = np.max(np.abs(Z))
                vmin = -max_abs
                vmax = max_abs
                label = 'Signed Displacement (Phase View)'
            else:  # magnitude
                plot_data = np.abs(Z) ** Config.VISUAL_EXPONENT
                cmap = 'plasma'
                vmin = 0
                vmax = np.max(plot_data)
                label = f'Displacement (|Z|^{Config.VISUAL_EXPONENT})'
            self.imshow_artist.set_data(plot_data)
            self.imshow_artist.set_cmap(cmap)
            self.imshow_artist.set_clim(vmin=vmin, vmax=vmax)
            if hasattr(self, 'cbar'):
                self.cbar.ax.set_visible(True)
                self.cbar.set_label(label)
            else:
                self.cbar = self.fig.colorbar(
                    self.imshow_artist, ax=self.ax, label=label)
        # Window title
        title_prefix = {
            'magnitude': 'Magnitude View',
            'phase': 'Phase View',
            'sand': 'Sand Simulation'
        }[self.view_mode]
        self.fig.canvas.manager.set_window_title(
            f'Chladni Simulator — {title_prefix}')
        self.ax.set_xlim(self.orig_xlim)
        self.ax.set_ylim(self.orig_ylim)
        f_closest, degenerate_modes = self.simulator.get_closest_resonance_info(
            f_display)
        title = f"f = {f_display:.2f}"
        if abs(f_display - f_closest) < Config.RESONANCE_TOL:
            deg_modes_str = ', '.join(
                [f"({m},{n})" for m, n in degenerate_modes])
            title += f" ← Resonance: {deg_modes_str} f_mn={f_closest:.2f}"
        self.ax.set_title(title)
        # Mode contribution text
        modes_info = self.simulator.get_contributing_modes(f_compute)
        max_modes = Config.MAX_DISPLAY_MODES or len(modes_info)
        text_str = (
            "Contributing Modes (%)\n\n"
            f"{'Mode (m,n)':<10} {'f_mn':>5} {'Weight %':>12}\n"
            + "-" * 32 + "\n"
        )
        for m, n, fmn, perc in modes_info[:max_modes]:
            text_str += f"({m:>2},{n:<2}) {fmn:>8.2f} {perc:>10.1f}\n"
        self.mode_text.set_text(text_str)
        self.fig.canvas.draw_idle()

    def update_gamma(self, val: float) -> None:
        self.simulator.set_gamma(val)
        self.update(self.freq_slider.val)

    def jump_to_next_resonance(self, event) -> None:
        current_f = self.freq_slider.val
        next_f = self.simulator.get_next_resonance_frequency(current_f)
        self.freq_slider.set_val(next_f)

    def jump_to_prev_resonance(self, event) -> None:
        current_f = self.freq_slider.val
        prev_f = self.simulator.get_previous_resonance_frequency(current_f)
        self.freq_slider.set_val(prev_f)

    def start_scan(self, event) -> None:
        if self.scan_ani is not None:
            self.scan_ani.event_source.stop()

        def update_scan(frame):
            f = self.freq_slider.val + Config.SCAN_SPEED
            if f > Config.FREQ_RANGE[1]:
                f = Config.FREQ_RANGE[0]
            self.freq_slider.set_val(f)
            return (self.imshow_artist, self.scatter_artist)

        self.scan_ani = FuncAnimation(
            self.fig, update_scan, interval=50, blit=False, cache_frame_data=False)
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
