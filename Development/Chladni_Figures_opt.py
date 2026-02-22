import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import TypeAlias

# =========================================================
# ⚙️ Configuration
# =========================================================
class Config:
    # Frequency controls
    FREQ_RANGE = (1.0, 20.0)
    FREQ_STEP = 0.01
    INIT_FREQ = 5.0
    
    # Damping (gamma) controls
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
    
    # UI behavior
    SCAN_SPEED = 0.02
    SHOW_AXES = False
    
    # Resonance curve plot settings
    RESONANCE_CURVE_RANGE = 1.5
    RESONANCE_CURVE_SAMPLES = 1000

Mode: TypeAlias = tuple[int, int, float]

# =========================================================
# 🧮 Chladni Simulator (Logic Engine)
# =========================================================
class ChladniSimulator:
    """Simulate Chladni figures for a square membrane."""
    def __init__(self):
        self.resolution = Config.RESOLUTION
        self.max_mode = Config.MAX_MODE
        self.gamma = Config.INIT_GAMMA
        self.k = Config.K
        
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.mode_shapes = []
        self.mode_frequencies = []
        self._precompute_modes()
        
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
                mode_shape = np.sin(m * np.pi * self.X) * \
                             np.sin(n * np.pi * self.Y)
                self.mode_shapes.append(mode_shape)
        self.mode_frequencies = np.array(self.mode_frequencies)
        self.mode_shapes = np.array(self.mode_shapes, dtype=np.float64)

    def compute_displacement(self, f: float) -> np.ndarray:
        weights = 1.0 / ((f - self.mode_frequencies) ** 2 + self.gamma ** 2)
        return np.tensordot(weights, self.mode_shapes, axes=(0, 0))

    def get_closest_resonance_info(self, current_f: float) -> tuple[float, list[tuple[int, int]]]:
        idx_closest = np.argmin(np.abs(self.mode_frequencies - current_f))
        f_closest = self.mode_frequencies[idx_closest]
        mode_list = [(m, n) for m in range(1, self.max_mode + 1)
                     for n in range(1, self.max_mode + 1)]
        degenerate_modes = [(m, n) for idx, (m, n) in enumerate(mode_list)
                            if abs(self.mode_frequencies[idx] - f_closest) < Config.EPS_FREQ_COMPARE]
        return f_closest, degenerate_modes

# =========================================================
# 📈 Resonance Curve Window
# =========================================================
class ResonanceCurveWindow:
    """Displays Lorentzian resonance curves for the active modes."""
    def __init__(self, simulator: ChladniSimulator, main_ui):
        self.simulator = simulator
        self.main_ui = main_ui
        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.fig.canvas.manager.set_window_title('Lorentzian Analysis')
        
        self.current_marker = None
        self.setup_curve()
        
        self.main_ui.freq_slider.on_changed(self.on_freq_change)
        self.main_ui.gamma_slider.on_changed(lambda v: self.setup_curve())

    def setup_curve(self) -> None:
        self.ax.clear()
        f_val = self.main_ui.freq_slider.val
        f_res, modes = self.simulator.get_closest_resonance_info(f_val)
        
        f_min = max(Config.FREQ_RANGE[0], f_res - Config.RESONANCE_CURVE_RANGE)
        f_max = min(Config.FREQ_RANGE[1], f_res + Config.RESONANCE_CURVE_RANGE)
        f_range = np.linspace(f_min, f_max, Config.RESONANCE_CURVE_SAMPLES)
        
        # Calculate Lorentzian
        y = 1.0 / ((f_range - f_res) ** 2 + self.simulator.gamma ** 2)
        self.ax.plot(f_range, y, 'b-', lw=2, label=f"Resonance @ {f_res:.2f}Hz")
        self.ax.axvline(f_res, color='red', ls='--', alpha=0.5)
        
        # Optimized Marker
        curr_y = 1.0 / ((f_val - f_res) ** 2 + self.simulator.gamma ** 2)
        self.current_marker, = self.ax.plot(f_val, curr_y, 'ro', markersize=8)
        
        self.ax.set_title(f"Dominant Modes: {modes}")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.fig.canvas.draw_idle()

    def on_freq_change(self, val: float):
        if not plt.fignum_exists(self.fig.number): return
        # If we drift too far from the current resonance peak, rebuild the view
        f_res, _ = self.simulator.get_closest_resonance_info(val)
        if abs(val - f_res) > Config.RESONANCE_CURVE_RANGE * 0.8:
            self.setup_curve()
        else:
            curr_y = 1.0 / ((val - f_res) ** 2 + self.simulator.gamma ** 2)
            self.current_marker.set_data([val], [curr_y])
            self.fig.canvas.draw_idle()

# =========================================================
# 🖥️ Main Chladni UI (Optimized Display)
# =========================================================
class ChladniUI:
    def __init__(self, simulator: ChladniSimulator):
        self.simulator = simulator
        self.view_mode = 'magnitude'
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Chladni Simulator (Optimized)')
        
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.info_ax = self.fig.add_subplot(gs[1])
        self.info_ax.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.3, top=0.92)

        # 🚀 PERFORMANCE: Initialize plot elements ONCE
        Z_init = self.simulator.compute_displacement(Config.INIT_FREQ)
        init_data = np.abs(Z_init) ** Config.VISUAL_EXPONENT
        self.plot_artist = self.ax.imshow(
            init_data, cmap='plasma', origin='lower', extent=[0, 1, 0, 1], interpolation='bilinear'
        )
        self.cbar = self.fig.colorbar(self.plot_artist, ax=self.ax)
        
        self._setup_widgets()
        self.mode_text = self.info_ax.text(0, 1, '', va='top', fontfamily='Monospace', fontsize=10)
        
        self.scan_ani = None
        self.res_win = None
        self.update(Config.INIT_FREQ)

    def _setup_widgets(self):
        # UI controls setup
        self.freq_slider = Slider(plt.axes([0.15, 0.2, 0.7, 0.03]), 'Freq', *Config.FREQ_RANGE, valinit=Config.INIT_FREQ)
        self.gamma_slider = Slider(plt.axes([0.15, 0.15, 0.7, 0.03]), 'Gamma', *Config.GAMMA_RANGE, valinit=Config.INIT_GAMMA)
        
        self.freq_slider.on_changed(self.update)
        self.gamma_slider.on_changed(self.update_gamma)

        # Action Buttons
        self.btn_prev = Button(plt.axes([0.15, 0.05, 0.08, 0.04]), '◀')
        self.btn_next = Button(plt.axes([0.24, 0.05, 0.08, 0.04]), '▶')
        self.btn_toggle = Button(plt.axes([0.35, 0.05, 0.12, 0.04]), 'Toggle Phase')
        self.btn_scan = Button(plt.axes([0.48, 0.05, 0.12, 0.04]), 'Auto Scan')
        self.btn_res = Button(plt.axes([0.61, 0.05, 0.12, 0.04]), 'Curves')

        self.btn_prev.on_clicked(lambda e: self.jump_resonance(-1))
        self.btn_next.on_clicked(lambda e: self.jump_resonance(1))
        self.btn_toggle.on_clicked(self.toggle_view)
        self.btn_scan.on_clicked(self.toggle_scan)
        self.btn_res.on_clicked(self.open_resonance_window)

    def toggle_view(self, event):
        self.view_mode = 'phase' if self.view_mode == 'magnitude' else 'magnitude'
        self.btn_toggle.label.set_text('View: Phase' if self.view_mode == 'phase' else 'View: Mag')
        self.plot_artist.set_cmap('coolwarm' if self.view_mode == 'phase' else 'plasma')
        self.update(self.freq_slider.val)

    def update(self, val: float):
        Z = self.simulator.compute_displacement(val)
        
        if self.view_mode == 'phase':
            data = Z
            lim = np.max(np.abs(Z))
            vmin, vmax = -lim, lim
        else:
            data = np.abs(Z) ** Config.VISUAL_EXPONENT
            vmin, vmax = 0, np.max(data)

        # 🚀 PERFORMANCE: Update existing artist, don't re-create
        self.plot_artist.set_data(data)
        self.plot_artist.set_clim(vmin, vmax)
        
        # Update Resonance Info
        f_res, modes = self.simulator.get_closest_resonance_info(val)
        title = f"f = {val:.2f} Hz"
        if abs(val - f_res) < Config.RESONANCE_TOL:
            title += f" [RESONANCE: {modes}]"
        self.ax.set_title(title)

        # Update Sidebar Text
        weights = 1.0 / ((val - self.simulator.mode_frequencies)**2 + self.simulator.gamma**2)
        total = np.sum(weights)
        percentages = (weights / total) * 100
        
        msg = f"Active Weights (f={val:.2f})\n" + "-"*20 + "\n"
        indices = np.where(percentages > Config.MODE_WEIGHT_THRESHOLD)[0]
        for i in indices[np.argsort(percentages[indices])[::-1]]:
            m = (i // Config.MAX_MODE) + 1
            n = (i % Config.MAX_MODE) + 1
            msg += f"({m},{n}): {percentages[i]:>5.1f}%\n"
        self.mode_text.set_text(msg)
        
        self.fig.canvas.draw_idle()

    def update_gamma(self, val: float):
        self.simulator.gamma = val
        self.update(self.freq_slider.val)

    def jump_resonance(self, step):
        freqs = [f for _, _, f in self.simulator.eigenfrequencies]
        curr = self.freq_slider.val
        if step > 0:
            target = next((f for f in freqs if f > curr + 0.05), freqs[0])
        else:
            target = next((f for f in reversed(freqs) if f < curr - 0.05), freqs[-1])
        self.freq_slider.set_val(target)

    def toggle_scan(self, event):
        if self.scan_ani:
            self.scan_ani.event_source.stop()
            self.scan_ani = None
            self.btn_scan.label.set_text('Auto Scan')
        else:
            def scan_frame(i):
                new_f = (self.freq_slider.val + Config.SCAN_SPEED)
                if new_f > Config.FREQ_RANGE[1]: new_f = Config.FREQ_RANGE[0]
                self.freq_slider.set_val(new_f)
            self.scan_ani = FuncAnimation(self.fig, scan_frame, interval=20, cache_frame_data=False)
            self.btn_scan.label.set_text('Stop Scan')

    def open_resonance_window(self, event):
        if self.res_win is None or not plt.fignum_exists(self.res_win.fig.number):
            self.res_win = ResonanceCurveWindow(self.simulator, self)
        plt.figure(self.res_win.fig.number)
        plt.show()

if __name__ == "__main__":
    sim = ChladniSimulator()
    ui = ChladniUI(sim)
    plt.show()
