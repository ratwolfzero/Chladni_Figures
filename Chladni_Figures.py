import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import TypeAlias

import matplotlib
print("Matplotlib version:", matplotlib.__version__)


class Config:
    # Frequency controls
    FREQ_RANGE = (1.0, 20.0)
    FREQ_STEP = 0.01
    INIT_FREQ = 5.0

    # Damping controls
    GAMMA_RANGE = (0.001, 0.05)
    GAMMA_STEP = 0.001
    INIT_GAMMA = 0.01

    # Mode / resonance parameters
    MAX_MODE = 15
    RESONANCE_TOL = 0.02
    MODE_WEIGHT_THRESHOLD = 1.0
    MAX_DISPLAY_MODES = None
    EPS_FREQ_COMPARE = 1e-6

    # Simulation grid & scaling
    RESOLUTION = 200
    K = 1.0
    VISUAL_EXPONENT = 0.2

    # UI & animation behavior
    SCAN_SPEED = 0.02
    SHOW_AXES = False

    # Resonance curve settings
    RESONANCE_CURVE_RANGE = 1
    RESONANCE_CURVE_SAMPLES = 20000


Mode: TypeAlias = tuple[int, int, float]


# ────────────────────────────────────────────────────────────────
#   Chladni Simulator – now also holds current driving frequency
# ────────────────────────────────────────────────────────────────
class ChladniSimulator:
    """Simulate Chladni figures for a square membrane."""

    def __init__(self):
        self.resolution = Config.RESOLUTION
        self.max_mode = Config.MAX_MODE
        self._gamma = Config.INIT_GAMMA
        self._current_frequency = Config.INIT_FREQ
        self.k = Config.K

        # Spatial grid
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

        # Precompute modes and frequencies
        ms, ns = np.meshgrid(np.arange(1, self.max_mode + 1),
                             np.arange(1, self.max_mode + 1))
        self._modes = list(zip(ms.ravel(), ns.ravel()))
        self._mode_frequencies = self.k * \
            np.sqrt(np.array([m**2 + n**2 for m, n in self._modes]))

        # Precompute mode shapes (modes × res × res)
        self._mode_shapes = np.array([
            np.sin(m * np.pi * self.X) * np.sin(n * np.pi * self.Y)
            for m, n in self._modes
        ], dtype=np.float64)

        # Sorted list of (m, n, frequency)
        self._eigenfrequencies = sorted(
            [(m, n, f) for (m, n), f in zip(self._modes, self._mode_frequencies)],
            key=lambda x: x[2]
        )

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def current_frequency(self) -> float:
        return self._current_frequency

    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma

    def set_current_frequency(self, f: float) -> None:
        self._current_frequency = f

    def compute_displacement(self, f: float) -> np.ndarray:
        weights = self.get_mode_weights_at_frequency(f)
        return np.tensordot(weights, self._mode_shapes, axes=(0, 0))

    def compute_lorentzian_weights(self, f_range: np.ndarray, target_freq: float) -> np.ndarray:
        return 1.0 / ((f_range - target_freq)**2 + self.gamma**2)

    def get_lorentzian_weight_at_freq(self, f: float, target_freq: float) -> float:
        return 1.0 / ((f - target_freq)**2 + self.gamma**2)

    def get_closest_resonance_info(self, current_f: float) -> tuple[float, list[Mode]]:
        idx = np.argmin(np.abs(self._mode_frequencies - current_f))
        f_closest = self._mode_frequencies[idx]
        degenerate = [
            self._modes[i] for i in range(len(self._modes))
            if abs(self._mode_frequencies[i] - f_closest) < Config.EPS_FREQ_COMPARE
        ]
        return f_closest, degenerate

    def get_mode_weights_at_frequency(self, f: float) -> np.ndarray:
        return 1.0 / ((f - self._mode_frequencies)**2 + self.gamma**2)

    def get_contributing_modes(self, f: float, threshold: float = Config.MODE_WEIGHT_THRESHOLD
                               ) -> list[tuple[int, int, float, float]]:
        weights = self.get_mode_weights_at_frequency(f)
        total = np.sum(weights)
        if total == 0:
            return []
        percentages = (weights / total) * 100
        modes_info = [
            (m, n, self._mode_frequencies[i], percentages[i])
            for i, (m, n) in enumerate(self._modes)
            if percentages[i] > threshold
        ]
        modes_info.sort(key=lambda x: x[3], reverse=True)
        return modes_info

    def get_next_resonance_frequency(self, current_f: float) -> float:
        freqs = [f for _, _, f in self._eigenfrequencies if f > current_f]
        return min(freqs) if freqs else self._eigenfrequencies[0][2]

    def get_previous_resonance_frequency(self, current_f: float) -> float:
        freqs = [f for _, _, f in self._eigenfrequencies if f < current_f]
        return max(freqs) if freqs else self._eigenfrequencies[-1][2]


# ────────────────────────────────────────────────────────────────
#   Resonance Curve Window – decoupled from main UI
# ────────────────────────────────────────────────────────────────
class ResonanceCurveWindow:
    def __init__(self, simulator: ChladniSimulator):
        self.simulator = simulator
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title('Lorentzian Resonance Curves')
        self.current_marker = None
        self._refresh_state()
        self.setup_curve()

    def _refresh_state(self):
        self.current_f_raw = self.simulator.current_frequency
        self.current_f_display = round(self.current_f_raw, 2)
        self.current_resonance_freq, self.current_modes = \
            self.simulator.get_closest_resonance_info(self.current_f_display)
        self.current_gamma = self.simulator.gamma

    def setup_curve(self) -> None:
        self.ax.clear()
        f_res = self.current_resonance_freq
        f_min = max(Config.FREQ_RANGE[0], f_res - Config.RESONANCE_CURVE_RANGE)
        f_max = min(Config.FREQ_RANGE[1], f_res + Config.RESONANCE_CURVE_RANGE)
        self.f_range = np.linspace(
            f_min, f_max, Config.RESONANCE_CURVE_SAMPLES)

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.current_modes)))
        for (m, n), color in zip(self.current_modes, colors):
            lor = self.simulator.compute_lorentzian_weights(
                self.f_range, f_res)
            self.ax.plot(self.f_range, lor, '-', color=color, lw=2,
                         label=f'Mode ({m},{n})')

        max_weight = self.simulator.get_lorentzian_weight_at_freq(f_res, f_res)

        self.ax.axvline(f_res, color='red', ls='--', alpha=0.7,
                        label=f'Resonance: f={f_res:.2f}')
        self.add_current_marker()

        fwhm = 2 * self.simulator.gamma
        half_max = max_weight / 2
        self.ax.axhline(half_max, color='gray', ls=':', alpha=0.5,
                        label='Half Maximum')

        self.ax.text(0.02, 0.98, f'γ = {self.simulator.gamma:.3f}',
                     transform=self.ax.transAxes, va='top',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        self.ax.text(0.02, 0.88, f'FWHM = {fwhm:.3f}',
                     transform=self.ax.transAxes, va='top',
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        modes_str = ', '.join(f"({m},{n})" for m, n in self.current_modes)
        self.ax.set_title(
            f'Lorentzian Resonance Curves – dominant mode(s): {modes_str}\n'
            '(height constant, width varies with γ)'
        )
        self.ax.set_xlabel('Driving Frequency (Hz)')
        self.ax.set_ylabel('Resonance Weight')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(f_min, f_max)
        self.ax.set_ylim(0, max_weight * 1.1)
        self.fig.canvas.draw_idle()

    def add_current_marker(self) -> None:
        w = self.simulator.get_lorentzian_weight_at_freq(
            self.current_f_raw, self.current_resonance_freq)
        self.current_marker, = self.ax.plot(
            self.current_f_raw, w, 'go', ms=8,
            label=f'Current: f={self.current_f_display:.2f}, w={w:.3f}'
        )

    def update_current_freq(self, _=None) -> None:
        if not plt.fignum_exists(self.fig.number):
            return
        old_res = self.current_resonance_freq
        old_modes_set = set(self.current_modes)

        self._refresh_state()

        if (abs(self.current_resonance_freq - old_res) > Config.EPS_FREQ_COMPARE or
            set(self.current_modes) != old_modes_set or
                self.current_f_raw < self.f_range[0] or self.current_f_raw > self.f_range[-1]):
            self.setup_curve()
        else:
            self.update_current_marker()

    def update_gamma(self, _=None) -> None:
        if plt.fignum_exists(self.fig.number):
            self.setup_curve()

    def update_current_marker(self) -> None:
        if self.current_marker:
            self.current_marker.remove()
        w = self.simulator.get_lorentzian_weight_at_freq(
            self.current_f_raw, self.current_resonance_freq)
        self.current_marker, = self.ax.plot(
            self.current_f_raw, w, 'go', ms=8,
            label=f'Current: f={self.current_f_display:.2f}, w={w:.3f}'
        )
        # Rebuild legend without duplicates
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.fig.canvas.draw_idle()


# ────────────────────────────────────────────────────────────────
#   Main Chladni UI
# ────────────────────────────────────────────────────────────────
class ChladniUI:
    def __init__(self, simulator: ChladniSimulator):
        self.simulator = simulator
        self.view_mode = 'magnitude'  # 'magnitude' | 'phase'

        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.info_ax = self.fig.add_subplot(gs[1])
        self.info_ax.axis('off')

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.95)

        self.orig_xlim = (0, 1)
        self.orig_ylim = (0, 1)

        # Initial plot
        Z = self.simulator.compute_displacement(Config.INIT_FREQ)
        plot_data = np.abs(Z) ** Config.VISUAL_EXPONENT
        self.plot_artist = self.ax.imshow(
            plot_data, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
        self.cbar = self.fig.colorbar(
            self.plot_artist, ax=self.ax,
            label=f'Displacement (|Z|^{Config.VISUAL_EXPONENT})')

        self._setup_axes()
        self._setup_widgets()

        self.mode_text = self.info_ax.text(
            0, 1, '', va='top', ha='left', fontsize=12,
            family='monospace')

        self.scan_ani = None
        self.resonance_window = None

        self.update(Config.INIT_FREQ)

    def _setup_axes(self):
        self.ax.set_facecolor('white')
        if not Config.SHOW_AXES:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_xlabel('')
            self.ax.set_ylabel('')
        else:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)

    def _setup_widgets(self):
        axf = plt.axes([0.05, 0.25, 0.8, 0.03])
        self.freq_slider = Slider(axf, 'f', *Config.FREQ_RANGE,
                                  valinit=Config.INIT_FREQ, valstep=Config.FREQ_STEP)
        self.freq_slider.on_changed(self.on_freq_changed)

        axg = plt.axes([0.05, 0.20, 0.8, 0.03])
        self.gamma_slider = Slider(axg, 'γ', *Config.GAMMA_RANGE,
                                   valinit=Config.INIT_GAMMA, valstep=Config.GAMMA_STEP)
        self.gamma_slider.on_changed(self.on_gamma_changed)

        ax_prev = plt.axes([0.05, 0.10, 0.08, 0.04])
        self.prev_btn = Button(ax_prev, '◀')
        self.prev_btn.on_clicked(self.jump_prev)

        ax_next = plt.axes([0.14, 0.10, 0.08, 0.04])
        self.next_btn = Button(ax_next, '▶')
        self.next_btn.on_clicked(self.jump_next)

        plt.text(0.135, 0.06, "Resonance Navigation", ha='center', va='center',
                 fontsize=10, fontweight='bold', transform=self.fig.transFigure)

        ax_scan = plt.axes([0.25, 0.10, 0.10, 0.04])
        self.scan_btn = Button(ax_scan, 'Auto Scan')
        self.scan_btn.on_clicked(self.start_scan)

        ax_stop = plt.axes([0.37, 0.10, 0.10, 0.04])
        self.stop_btn = Button(ax_stop, 'Stop Scan')
        self.stop_btn.on_clicked(self.stop_scan)

        ax_res = plt.axes([0.69, 0.10, 0.15, 0.04])
        self.res_btn = Button(ax_res, 'Show Resonance Curves')
        self.res_btn.on_clicked(self.open_resonance_window)

        ax_toggle = plt.axes([0.50, 0.10, 0.16, 0.04])
        self.toggle_btn = Button(ax_toggle, 'Toggle to Phase View')
        self.toggle_btn.on_clicked(self.toggle_view)

    def toggle_view(self, _):
        modes = ['magnitude', 'phase']
        labels = ['Magnitude View', 'Phase View']
        idx = modes.index(self.view_mode)
        self.view_mode = modes[1 - idx]
        self.toggle_btn.label.set_text(f'Toggle to {labels[idx]}')
        self.update(self.freq_slider.val)

    def open_resonance_window(self, _):
        if self.resonance_window is None or not plt.fignum_exists(self.resonance_window.fig.number):
            self.resonance_window = ResonanceCurveWindow(self.simulator)
        plt.figure(self.resonance_window.fig.number)
        plt.show()

    def on_freq_changed(self, val):
        self.simulator.set_current_frequency(val)
        self.update(val)
        if self.resonance_window and plt.fignum_exists(self.resonance_window.fig.number):
            self.resonance_window.update_current_freq()

    def on_gamma_changed(self, val):
        self.simulator.set_gamma(val)
        self.update(self.freq_slider.val)
        if self.resonance_window and plt.fignum_exists(self.resonance_window.fig.number):
            self.resonance_window.update_gamma()

    def update(self, _=None):
        f = self.simulator.current_frequency
        f_disp = round(f, 2)

        Z = self.simulator.compute_displacement(f)

        if self.view_mode == 'phase':
            data = Z
            cmap = 'coolwarm'
            vmin, vmax = -np.max(np.abs(Z)), np.max(np.abs(Z))
            label = 'Signed Displacement (Phase View)'
        else:
            data = np.abs(Z) ** Config.VISUAL_EXPONENT
            cmap = 'plasma'
            vmin, vmax = 0, np.max(data)
            label = f'Displacement (|Z|^{Config.VISUAL_EXPONENT})'

        self.plot_artist.set_data(data)
        self.plot_artist.set_cmap(cmap)
        self.plot_artist.set_clim(vmin, vmax)
        self.cbar.set_label(label)

        title_prefix = 'Phase View' if self.view_mode == 'phase' else 'Magnitude View'
        self.fig.canvas.manager.set_window_title(
            f'Chladni Simulator – {title_prefix}')

        self.ax.set_xlim(self.orig_xlim)
        self.ax.set_ylim(self.orig_ylim)

        f_close, deg_modes = self.simulator.get_closest_resonance_info(f_disp)
        title = f"f = {f_disp:.2f}"
        if abs(f_disp - f_close) < Config.RESONANCE_TOL:
            title += f"  ← Resonance: {', '.join(f'({m},{n})' for m, n in deg_modes)}  f={
                f_close:.2f}"
        self.ax.set_title(title)

        modes_info = self.simulator.get_contributing_modes(f)
        max_show = Config.MAX_DISPLAY_MODES or len(modes_info)
        text = "Contributing Modes (%)\n\n" + \
               f"{'Mode (m,n)':<10} {'f':>6} {'%':>10}\n" + "-"*32 + "\n"
        for m, n, fm, p in modes_info[:max_show]:
            text += f"({m:2d},{n:2d}) {fm:8.2f} {p:10.1f}\n"
        self.mode_text.set_text(text)

        self.fig.canvas.draw_idle()

    def jump_next(self, _):
        nxt = self.simulator.get_next_resonance_frequency(self.freq_slider.val)
        self.freq_slider.set_val(nxt)

    def jump_prev(self, _):
        prev = self.simulator.get_previous_resonance_frequency(
            self.freq_slider.val)
        self.freq_slider.set_val(prev)

    def start_scan(self, _):
        if self.scan_ani is not None:
            self.scan_ani.event_source.stop()

        def update_scan(frame):
            f = self.freq_slider.val + Config.SCAN_SPEED
            if f > Config.FREQ_RANGE[1]:
                f = Config.FREQ_RANGE[0]
            self.freq_slider.set_val(f)
            return (self.plot_artist)

        self.scan_ani = FuncAnimation(
            self.fig, update_scan, interval=50, blit=False, cache_frame_data=False)
        self.fig.canvas.draw_idle()

    def stop_scan(self, _):
        if self.scan_ani is not None:
            self.scan_ani.event_source.stop()
            self.scan_ani = None

    def show(self):
        plt.show()


def main():
    sim = ChladniSimulator()
    ui = ChladniUI(sim)
    ui.show()


if __name__ == "__main__":
    main()
