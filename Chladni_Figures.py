from typing import Callable, List, Tuple
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


class Config:
    FREQ_RANGE = (1.0, 20.0)  # Range for driving frequency slider (Hz)
    FREQ_STEP = 0.01  # Frequency slider step size
    INIT_FREQ = 5.0  # Initial driving frequency (Hz)
    # =========================================================
    GAMMA_RANGE = (0.001, 0.05)  # Range for damping coefficient γ
    GAMMA_STEP = 0.001  # Damping slider step size
    INIT_GAMMA = 0.01  # Default damping value (γ)
    # =========================================================
    MAX_MODE = 15  # Maximum mode indices m,n to compute
    RESONANCE_TOL = 0.02  # Frequency tolerance for resonance detection
    MODE_WEIGHT_THRESHOLD = 1.0  # Minimum % weight for mode to be listed
    MAX_DISPLAY_MODES = None  # Limit number of modes shown (None = all)
    EPS_FREQ_COMPARE = 1e-6  # Small epsilon for frequency equality check
    # =========================================================                                                                  
    RESOLUTION = 200  # Grid resolution for spatial mode shapes
    K = 1.0  # Frequency scaling factor for eigenmodes
    VISUAL_EXPONENT = 0.2  # Exponent for magnitude visualization (|Z|^exp)
    # =========================================================
    SCAN_SPEED = 0.02  # Frequency step per frame during Auto Scan
    SHOW_AXES = False  # Toggle for showing coordinate axes
    # =========================================================
    RESONANCE_CURVE_RANGE = 1  # Frequency range around resonance (Hz)
    RESONANCE_CURVE_SAMPLES = 20000  # Number of sampling points per Lorentzian
    # =========================================================
    SAND_EXP_SCALE = 0.05  # Scale for exponential probability decay
    NUM_GRAINS = 30000  # Number of sand grains to simulate
    SAND_NOISE_STD = 0.5  # Standard deviation for noise in sand positions
    SAND_SIZE = 0.1  # Size of sand grains in scatter plot
    SAND_COLOR = 'black'  # Color of sand grains


class ChladniSimulator:
    def __init__(self):
        self.resolution = Config.RESOLUTION
        self.max_mode = Config.MAX_MODE
        self._gamma = Config.INIT_GAMMA
        self._current_frequency = Config.INIT_FREQ
        self.k = Config.K

        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)
										 
        ms, ns = np.meshgrid(np.arange(1, self.max_mode + 1),
                             np.arange(1, self.max_mode + 1))
        self._modes = list(zip(ms.ravel(), ns.ravel()))
        self._mode_frequencies = self.k * \
            np.sqrt(np.array([m**2 + n**2 for m, n in self._modes]))

        self._mode_shapes = np.array([
            np.sin(m * np.pi * self.X) * np.sin(n * np.pi * self.Y)                        
            for m, n in self._modes
        ], dtype=np.float64)

        self._eigenfrequencies = [
            (m, n, f_mn) for (m, n), f_mn in zip(self._modes, self._mode_frequencies)
        ]                                                                                                                        
        self._eigenfrequencies.sort(key=lambda x: x[2])
        
        # Fast lookup: mode (m,n) → frequency
        self._mode_to_freq = {(m, n): f for m, n, f in self._eigenfrequencies}

        # Observer lists                                                                  
        self._freq_listeners: List[Callable[[float], None]] = []
        self._gamma_listeners: List[Callable[[float], None]] = []
        
    def get_eigenfrequency(self, m: int, n: int) -> float:
        """
        Return the eigenfrequency for mode (m, n) using precomputed values.
        
        Args:
            m: Mode index in x-direction
            n: Mode index in y-direction
            
        Returns:
            The eigenfrequency f_mn = k * sqrt(m² + n²)
            
        Raises:
            ValueError: If the mode is outside the computed range
        """                                  
        try:
            return self._mode_to_freq[(m, n)]
        except KeyError:
            raise ValueError(                    
                f"Mode ({m}, {n}) not in precomputed modes "
                f"(max_mode = {self.max_mode})"
            )

    def add_frequency_listener(self, callback: Callable[[float], None]):
        self._freq_listeners.append(callback)

    def add_gamma_listener(self, callback: Callable[[float], None]):
        self._gamma_listeners.append(callback)

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def current_frequency(self) -> float:
        return self._current_frequency

    def set_gamma(self, gamma: float) -> None:
        if abs(gamma - self._gamma) < 1e-9:
            return
        self._gamma = gamma
        for cb in self._gamma_listeners:
            cb(gamma)

    def set_current_frequency(self, f: float) -> None:
        if abs(f - self._current_frequency) < 1e-9:
            return
        self._current_frequency = f
        for cb in self._freq_listeners:
            cb(f)

    def compute_displacement(self, f: float) -> np.ndarray:
        weights = self.get_mode_weights_at_frequency(f)
        return np.tensordot(weights, self._mode_shapes, axes=(0, 0))

    def compute_lorentzian_weights(self, f_range: np.ndarray, target_freq: float) -> np.ndarray:
        return 1.0 / ((f_range - target_freq) ** 2 + self.gamma ** 2)

    def get_lorentzian_weight_at_freq(self, f: float, target_freq: float) -> float:
        return 1.0 / ((f - target_freq) ** 2 + self.gamma ** 2)

    def get_closest_resonance_info(self, current_f: float) -> Tuple[float, List[Tuple[int, int]]]:
        idx = np.argmin(np.abs(self._mode_frequencies - current_f))
        f_closest = self._mode_frequencies[idx]
        degenerate = [
            self._modes[i] for i in range(len(self._modes))
            if abs(self._mode_frequencies[i] - f_closest) < Config.EPS_FREQ_COMPARE
        ]
        return f_closest, degenerate

    @property
    def current_closest_resonance(self) -> Tuple[float, List[Tuple[int, int]]]:
        return self.get_closest_resonance_info(self.current_frequency)

    def get_mode_weights_at_frequency(self, f: float) -> np.ndarray:
        return 1.0 / ((f - self._mode_frequencies) ** 2 + self.gamma ** 2)

    def get_contributing_modes(self, f: float, threshold: float = Config.MODE_WEIGHT_THRESHOLD
                               ) -> List[Tuple[int, int, float, float]]:
        weights = self.get_mode_weights_at_frequency(f)
        total = np.sum(weights)
        if total == 0:
            return []
        percentages = (weights / total) * 100
        info = [
            (m, n, self._mode_frequencies[i], percentages[i])
            for i, (m, n) in enumerate(self._modes) if percentages[i] > threshold
        ]
        info.sort(key=lambda x: x[3], reverse=True)
        return info

    def get_sand_coordinates_from_Z(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_z = np.max(np.abs(Z)) or 1e-12
        p = np.exp(-np.abs(Z) / (max_z * Config.SAND_EXP_SCALE))
        p_flat = p.ravel()
        total = p_flat.sum()
        if total <= 0:
            p_flat.fill(1.0 / p_flat.size)
        else:
            p_flat /= total
        indices = np.random.choice(
            p_flat.size, size=Config.NUM_GRAINS, p=p_flat)
        y_idx, x_idx = np.unravel_index(
            indices, (self.resolution, self.resolution))
        x = x_idx + np.random.normal(0, Config.SAND_NOISE_STD, len(x_idx))
        y = y_idx + np.random.normal(0, Config.SAND_NOISE_STD, len(y_idx))
        return x, y

    def get_next_resonance_frequency(self, current_f: float) -> float:
        freqs = [f for _, _, f in self._eigenfrequencies if f > current_f]
        return min(freqs) if freqs else (self._eigenfrequencies[0][2] if self._eigenfrequencies else current_f)

    def get_previous_resonance_frequency(self, current_f: float) -> float:
        freqs = [f for _, _, f in self._eigenfrequencies if f < current_f]
        return max(freqs) if freqs else (self._eigenfrequencies[-1][2] if self._eigenfrequencies else current_f)

    def remove_frequency_listener(self, callback: Callable[[float], None]):
        if callback in self._freq_listeners:
            self._freq_listeners.remove(callback)

    def remove_gamma_listener(self, callback: Callable[[float], None]):
        if callback in self._gamma_listeners:
            self._gamma_listeners.remove(callback)


class ResonanceCurveWindow:
    def __init__(self, simulator: ChladniSimulator):
        self.simulator = simulator
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title('Lorentzian Resonance Curves')

        # Initial state
        self._last_resonance_f, self._last_modes = simulator.current_closest_resonance
        self.current_f_raw = simulator.current_frequency
        self.current_f_display = round(self.current_f_raw, 2)
        self.current_marker = None

        # Subscribe once
        simulator.add_frequency_listener(self._on_frequency_update)
        simulator.add_gamma_listener(self._on_gamma_update)

        self.setup_curve()

        # ── Cleanup when window is closed ───────────────────────────────
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, event):
        """Remove ourselves from simulator listeners when the figure closes"""
        self.simulator.remove_frequency_listener(self._on_frequency_update)
        self.simulator.remove_gamma_listener(self._on_gamma_update)

    def _needs_full_redraw(self) -> bool:
        curr_f, curr_modes = self.simulator.current_closest_resonance
        f_changed = abs(
            curr_f - self._last_resonance_f) > Config.EPS_FREQ_COMPARE
        modes_changed = set(curr_modes) != set(self._last_modes)
        return f_changed or modes_changed

    def _update_last_resonance(self):
        self._last_resonance_f, self._last_modes = self.simulator.current_closest_resonance

    def setup_curve(self) -> None:
        self.ax.clear()
        self._update_last_resonance()
        f_res, modes = self.simulator.current_closest_resonance
        f_min = max(Config.FREQ_RANGE[0], f_res - Config.RESONANCE_CURVE_RANGE)
        f_max = min(Config.FREQ_RANGE[1], f_res + Config.RESONANCE_CURVE_RANGE)
        self.f_range = np.linspace(
            f_min, f_max, Config.RESONANCE_CURVE_SAMPLES)
        colors = plt.cm.Set1(np.linspace(0, 1, len(modes)))
        for (m, n), c in zip(modes, colors):
            f_mn = self.simulator.get_eigenfrequency(m, n)          # ← this line
            w = self.simulator.compute_lorentzian_weights(self.f_range, f_mn)
            self.ax.plot(self.f_range, w, '-', color=c,
                         lw=2, label=f'Mode ({m},{n})')
        self.ax.axvline(f_res, color='red', ls='--', alpha=0.7,
                        label=f'Resonance: f={f_res:.2f}')
        max_w = 1.0 / (self.simulator.gamma ** 2)
        self.ax.axhline(max_w / 2, color='gray', ls=':',
                        alpha=0.5, label='Half Maximum')
        fwhm = 2 * self.simulator.gamma
        self.ax.text(0.02, 0.98, f'γ = {self.simulator.gamma:.3f}', va='top', transform=self.ax.transAxes,
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        self.ax.text(0.02, 0.88, f'FWHM = {fwhm:.3f}', va='top', transform=self.ax.transAxes,
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        modes_str = ', '.join(f"({m},{n})" for m, n in modes)
        self.ax.set_title(f'Lorentzian Resonance Curves – {modes_str}\n'
                          '(Normalized Lorentzian; peak rescaled, width ∝ γ)')
        self.ax.set_xlabel('Driving Frequency (f)')
        self.ax.set_ylabel('Resonance Weight')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(f_min, f_max)
        self.ax.set_ylim(0, max_w * 1.1)
        self.add_current_marker()
        self.fig.canvas.draw_idle()

    def add_current_marker(self) -> None:
        w = self.simulator.get_lorentzian_weight_at_freq(self.current_f_raw,
                                                         self.simulator.current_closest_resonance[0])
        self.current_marker, = self.ax.plot(self.current_f_raw, w, 'go', ms=8,
                                            label=f'Current: f={self.current_f_display:.2f}, w={w:.3f}')

    def _on_frequency_update(self, new_f: float):
        self.current_f_raw = new_f
        self.current_f_display = round(new_f, 2)

        if self._needs_full_redraw():
            self.setup_curve()
        else:
            self._update_marker_only()

    def _on_gamma_update(self, new_gamma: float):
        # Gamma change always requires full redraw (width changes)
        self.setup_curve()

    def _update_marker_only(self):
        if self.current_marker:
            self.current_marker.remove()
        w = self.simulator.get_lorentzian_weight_at_freq(
            self.current_f_raw, self.simulator.current_closest_resonance[0])
        self.current_marker, = self.ax.plot(self.current_f_raw, w, 'go', ms=8,
                                            label=f'Current: f={self.current_f_display:.2f}, w={w:.3f}')
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.fig.canvas.draw_idle()


class ChladniUI:

    # --------------------------------------------------------------
    # generic render helpers
    # --------------------------------------------------------------
    def _update_imshow(self, data: np.ndarray, cmap: str, clim: Tuple[float, float]) -> None:
        """Prepare the :class:`imshow` artist with *data*, *cmap* and
        color limits, hiding the scatter artist.
        """
        self.scatter_artist.set_visible(False)
        self.imshow_artist.set_visible(True)
        self.imshow_artist.set_data(data)
        self.imshow_artist.set_cmap(cmap)
        self.imshow_artist.set_clim(vmin=clim[0], vmax=clim[1])

    def _update_colorbar(self, label: str, visible: bool) -> None:
        if hasattr(self, 'cbar'):
            self.cbar.ax.set_visible(visible)
            if visible:
                self.cbar.set_label(label)

    def _render_magnitude(self, Z: np.ndarray) -> None:
        plot_data = np.abs(Z) ** Config.VISUAL_EXPONENT
        self._update_imshow(plot_data, 'plasma', (0, np.max(plot_data)))
        self._update_colorbar(
            f'Displacement (|Z|^{Config.VISUAL_EXPONENT})', True)

    def _render_phase(self, Z: np.ndarray) -> None:
        max_abs = np.max(np.abs(Z))
        self._update_imshow(Z, 'coolwarm', (-max_abs, max_abs))
        self._update_colorbar('Signed Displacement (Phase View)', True)

    def _render_sand(self, Z: np.ndarray) -> None:
        self.imshow_artist.set_visible(False)
        self.scatter_artist.set_visible(True)
        self._update_colorbar('', False)
        x, y = self.simulator.get_sand_coordinates_from_Z(Z)
        x_plot = x / (self.simulator.resolution - 1)
        y_plot = y / (self.simulator.resolution - 1)
        offsets = np.column_stack((x_plot, y_plot))
        self.scatter_artist.set_offsets(offsets)

    def __init__(self, simulator: ChladniSimulator):
        self.simulator = simulator
        self.simulator.set_gamma(Config.INIT_GAMMA)
        self.simulator.set_current_frequency(Config.INIT_FREQ)

        # Single source of truth for view modes
        self._views = {
            'magnitude': {'render': self._render_magnitude, 'title': 'Magnitude View', 'next_key': 'phase'},
            'phase':     {'render': self._render_phase,     'title': 'Phase View',     'next_key': 'sand'},
            'sand':      {'render': self._render_sand,      'title': 'Sand Simulation', 'next_key': 'magnitude'},
        }
        self.view_mode = 'magnitude'

        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.info_ax = self.fig.add_subplot(gs[1])
        self.info_ax.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.95)

        self.orig_xlim = (0, 1)
        self.orig_ylim = (0, 1)

        Z_init = simulator.compute_displacement(Config.INIT_FREQ)
        plot_data = np.abs(Z_init) ** Config.VISUAL_EXPONENT
        self.imshow_artist = self.ax.imshow(
            plot_data, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
        self.scatter_artist = self.ax.scatter(
            [], [], s=Config.SAND_SIZE, c=Config.SAND_COLOR, marker='.')
        self.scatter_artist.set_visible(False)

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

    # ------------------------------------------------------------------
    # widget factory helpers
    # ------------------------------------------------------------------
    def _make_slider(self, rect: Tuple[float, float, float, float], label: str,
                     valmin: float, valmax: float, valinit: float,
                     valstep: float, on_changed: Callable[[float], None]):
        """Create a :class:`matplotlib.widgets.Slider` at *rect* and hook the
        *on_changed* callback.  The rectangle is given in figure coordinates
        [x, y, width, height].
        """
        ax = plt.axes(rect)
        slider = Slider(ax, label, valmin, valmax,
                        valinit=valinit, valstep=valstep)
        slider.on_changed(on_changed)
        return slider

    def _make_button(self, rect: Tuple[float, float, float, float], label: str,
                     on_clicked: Callable[[object], None]):
        """Create a :class:`matplotlib.widgets.Button` and register a click
        callback.
        """
        ax = plt.axes(rect)
        btn = Button(ax, label)
        btn.on_clicked(on_clicked)
        return btn

    def _setup_widgets(self) -> None:
        # sliders
        self.freq_slider = self._make_slider(
            [0.05, 0.25, 0.8, 0.03], 'f.',
            *Config.FREQ_RANGE,
            valinit=Config.INIT_FREQ, valstep=Config.FREQ_STEP,
            on_changed=self.update_freq_slider
        )

        self.gamma_slider = self._make_slider(
            [0.05, 0.2, 0.8, 0.03], 'γ',
            *Config.GAMMA_RANGE,
            valinit=Config.INIT_GAMMA, valstep=Config.GAMMA_STEP,
            on_changed=self.update_gamma_slider
        )

        # resonance navigation buttons and label
        self.prev_button = self._make_button(
            [0.05, 0.1, 0.08, 0.04], '◀',
            self.jump_to_prev_resonance
        )
        self.next_button = self._make_button(
            [0.14, 0.1, 0.08, 0.04], '▶',
            self.jump_to_next_resonance
        )
        plt.text(0.135, 0.06, "Resonance Navigation",
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 transform=self.fig.transFigure)

        # scan controls
        self.scan_button = self._make_button(
            [0.25, 0.1, 0.1, 0.04], 'Auto Scan',
            self.start_scan
        )
        self.stop_button = self._make_button(
            [0.37, 0.1, 0.1, 0.04], 'Stop Scan',
            self.stop_scan
        )

        # auxiliary windows and toggles
        self.resonance_button = self._make_button(
            [0.69, 0.1, 0.15, 0.04], 'Show Resonance Curves',
            self.open_resonance_curve
        )
        self.toggle_button = self._make_button(
            [0.50, 0.1, 0.16, 0.04], 'Toggle to Phase View',
            self.toggle_view
        )

    def _refresh_toggle_label(self) -> None:
        """Update the text on *toggle_button* to indicate the next view."""
        next_key = self._views[self.view_mode]['next_key']
        next_title = self._views[next_key]['title']
        self.toggle_button.label.set_text(f"Toggle to {next_title}")

    def toggle_view(self, event) -> None:
        self.view_mode = self._views[self.view_mode]['next_key']
        self._refresh_toggle_label()
        self.update(self.freq_slider.val)

    def update(self, val: float) -> None:
        f_compute = self.simulator.current_frequency
        f_display = round(f_compute, 2)
        Z = self.simulator.compute_displacement(f_compute)

        self._views[self.view_mode]['render'](Z)

        self.fig.canvas.manager.set_window_title(
            f'Chladni Simulator — {self._views[self.view_mode]["title"]}')

        self.ax.set_xlim(self.orig_xlim)
        self.ax.set_ylim(self.orig_ylim)

        f_closest, deg_modes = self.simulator.current_closest_resonance
        title = f"f = {f_display:.2f}"
        if abs(f_compute - f_closest) < Config.RESONANCE_TOL:
            deg_str = ', '.join(f"({m},{n})" for m, n in deg_modes)
            title += f" ← Resonance: {deg_str} f_mn={f_closest:.2f}"
        self.ax.set_title(title)

        modes_info = self.simulator.get_contributing_modes(f_compute)
        max_m = Config.MAX_DISPLAY_MODES or len(modes_info)
        text = (
            "Contributing Modes (%)\n\n"
            f"{'Mode (m,n)':<10} {'f_mn':>5} {'Weight %':>12}\n"
            + "-"*32 + "\n"
        )
        for m, n, fmn, perc in modes_info[:max_m]:
            text += f"({m:>2},{n:<2}) {fmn:>8.2f} {perc:>10.1f}\n"
        self.mode_text.set_text(text)

        self.fig.canvas.draw_idle()

    def update_freq_slider(self, val: float) -> None:
        self.simulator.set_current_frequency(val)
        self.update(val)

    def update_gamma_slider(self, val: float) -> None:
        self.simulator.set_gamma(val)
        self.update(self.freq_slider.val)

    def jump_to_next_resonance(self, event) -> None:
        current_f = self.freq_slider.val
        next_f = self.simulator.get_next_resonance_frequency(current_f)
        self.simulator.set_current_frequency(next_f)
        self.freq_slider.set_val(next_f)

    def jump_to_prev_resonance(self, event) -> None:
        current_f = self.freq_slider.val
        prev_f = self.simulator.get_previous_resonance_frequency(current_f)
        self.simulator.set_current_frequency(prev_f)
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

    def open_resonance_curve(self, event) -> None:
        if self.resonance_window is None or not plt.fignum_exists(self.resonance_window.fig.number):
            self.resonance_window = ResonanceCurveWindow(self.simulator)
            self.resonance_window.fig.show()
        else:
            self.resonance_window.fig.canvas.manager.window.lift()

    def show(self) -> None:
        plt.show()


def main() -> None:
    sim = ChladniSimulator()
    ui = ChladniUI(sim)
    ui.show()


if __name__ == "__main__":
    main()
