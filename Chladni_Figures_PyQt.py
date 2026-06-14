

import sys
from typing import List, Callable, Tuple
import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Switch to Qt backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QFontDatabase


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

    def get_previous_resonance_frequency(self, current_f: float) -> float:
        freqs = [f for _, _, f in self._eigenfrequencies 
                 if f < current_f - Config.EPS_FREQ_COMPARE]
        if freqs:
            return max(freqs)
        # Wrap around to highest
        return self._eigenfrequencies[-1][2]

    def get_next_resonance_frequency(self, current_f: float) -> float:
        freqs = [f for _, _, f in self._eigenfrequencies 
                 if f > current_f + Config.EPS_FREQ_COMPARE]
        if freqs:
            return min(freqs)
        # Wrap around to lowest
        return self._eigenfrequencies[0][2]

    def remove_frequency_listener(self, callback: Callable[[float], None]):
        if callback in self._freq_listeners:
            self._freq_listeners.remove(callback)

    def remove_gamma_listener(self, callback: Callable[[float], None]):
        if callback in self._gamma_listeners:
            self._gamma_listeners.remove(callback)



class ResonanceCurveWindow_PyQt(QMainWindow):
    def __init__(self, simulator: ChladniSimulator, parent=None):
        super().__init__(parent)
        self.simulator = simulator
        self.setWindowTitle('Lorentzian Resonance Curves')
        self.resize(800, 600)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Matplotlib Figure
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        # Initial state
        self._last_resonance_f, self._last_modes = simulator.current_closest_resonance
        self.current_f_raw = simulator.current_frequency
        self.current_marker = None

        # Subscribe to simulator
        self.simulator.add_frequency_listener(self._on_frequency_update)
        self.simulator.add_gamma_listener(self._on_gamma_update)

        self.setup_curve()

    def closeEvent(self, event):
        """Cleanup listeners when window closes."""
        self.simulator.remove_frequency_listener(self._on_frequency_update)
        self.simulator.remove_gamma_listener(self._on_gamma_update)
        event.accept()

    def _needs_full_redraw(self) -> bool:
        curr_f, curr_modes = self.simulator.current_closest_resonance
        f_changed = abs(curr_f - self._last_resonance_f) > Config.EPS_FREQ_COMPARE
        modes_changed = set(curr_modes) != set(self._last_modes)
        return f_changed or modes_changed

    def setup_curve(self) -> None:
        self.ax.clear()
        self._last_resonance_f, self._last_modes = self.simulator.current_closest_resonance
        f_res = self._last_resonance_f
        modes = self._last_modes

        f_min = max(Config.FREQ_RANGE[0], f_res - Config.RESONANCE_CURVE_RANGE)
        f_max = min(Config.FREQ_RANGE[1], f_res + Config.RESONANCE_CURVE_RANGE)
        self.f_range = np.linspace(f_min, f_max, Config.RESONANCE_CURVE_SAMPLES)
        
        # --- THE FIX: Use standard colormaps instead of internal properties ---
        import matplotlib.cm as cm
        colors = cm.Set1(np.linspace(0, 1, max(1, len(modes))))
        
        for (m, n), c in zip(modes, colors):
            f_mn = self.simulator.get_eigenfrequency(m, n)
            w = self.simulator.compute_lorentzian_weights(self.f_range, f_mn)
            self.ax.plot(self.f_range, w, '-', color=c, lw=2, label=f'Mode ({m},{n})')
        # ---------------------------------------------------------------------

        self.ax.axvline(f_res, color='red', ls='--', alpha=0.7, label=f'Resonance: f={f_res:.2f}')
        max_w = 1.0 / (self.simulator.gamma ** 2)
        self.ax.axhline(max_w / 2, color='gray', ls=':', alpha=0.5, label='Half Maximum')
        
        fwhm = 2 * self.simulator.gamma
        self.ax.text(0.02, 0.98, f'γ = {self.simulator.gamma:.3f}\nFWHM = {fwhm:.3f}', 
                     va='top', transform=self.ax.transAxes,
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        modes_str = ', '.join(f"({m},{n})" for m, n in modes)
        self.ax.set_title(f'Lorentzian Resonance Curves – {modes_str}')
        self.ax.set_xlabel('Driving Frequency (f)')
        self.ax.set_ylabel('Resonance Weight')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(f_min, f_max)
        self.ax.set_ylim(0, max_w * 1.1)

        self.add_current_marker()
        self.canvas.draw_idle()

    def add_current_marker(self) -> None:
        w = self.simulator.get_lorentzian_weight_at_freq(self.current_f_raw, self._last_resonance_f)
        self.current_marker, = self.ax.plot(self.current_f_raw, w, 'go', ms=8,
                                            label=f'Current: f={self.current_f_raw:.2f}, w={w:.3f}')

    def _on_frequency_update(self, new_f: float):
        self.current_f_raw = new_f
        if self._needs_full_redraw():
            self.setup_curve()
        else:
            self._update_marker_only()

    def _on_gamma_update(self, new_gamma: float):
        self.setup_curve()

    def _update_marker_only(self):
        if self.current_marker:
            self.current_marker.remove()
        w = self.simulator.get_lorentzian_weight_at_freq(self.current_f_raw, self._last_resonance_f)
        self.current_marker, = self.ax.plot(self.current_f_raw, w, 'go', ms=8,
                                            label=f'Current: f={self.current_f_raw:.2f}, w={w:.3f}')
        
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        self.canvas.draw_idle()


class ChladniUI_PyQt(QMainWindow):
    def __init__(self, simulator: ChladniSimulator):
        super().__init__()
        self.simulator = simulator
        self.setWindowTitle("Chladni Plate Simulator")
        self.resize(1200, 800)

        # Single source of truth for view modes
        self._views = {
            'magnitude': {'render': self._render_magnitude, 'title': 'Magnitude View', 'next_key': 'phase'},
            'phase':     {'render': self._render_phase,     'title': 'Phase View',     'next_key': 'sand'},
            'sand':      {'render': self._render_sand,      'title': 'Sand Simulation', 'next_key': 'magnitude'},
        }
        self.view_mode = 'magnitude'

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Matplotlib Canvas
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        main_layout.addWidget(self.canvas, stretch=3)

        # Right side: UI Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, stretch=1)

        self._setup_matplotlib_artists()
        self._setup_ui_controls(control_layout)

        # Resonance window reference
        self.resonance_window = None

        # Timer for Auto Scan
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self._scan_step)

        # Initialize simulation state
        self.simulator.set_gamma(Config.INIT_GAMMA)
        self.simulator.set_current_frequency(Config.INIT_FREQ)
        self.update_plot(Config.INIT_FREQ)

    def _setup_matplotlib_artists(self):
        self.ax.set_facecolor('white')
        if Config.SHOW_AXES:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        Z_init = self.simulator.compute_displacement(Config.INIT_FREQ)
        plot_data = np.abs(Z_init) ** Config.VISUAL_EXPONENT
        
        self.imshow_artist = self.ax.imshow(
            plot_data, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
        self.scatter_artist = self.ax.scatter(
            [], [], s=Config.SAND_SIZE, c=Config.SAND_COLOR, marker='.')
        self.scatter_artist.set_visible(False)

        self.cbar = self.fig.colorbar(
            self.imshow_artist, ax=self.ax, label=f'Displacement (|Z|^{Config.VISUAL_EXPONENT})')

    def _setup_ui_controls(self, layout: QVBoxLayout):
        # View Toggle
        self.btn_toggle_view = QPushButton(f"Toggle to {self._views[self._views[self.view_mode]['next_key']]['title']}")
        self.btn_toggle_view.clicked.connect(self.toggle_view)
        layout.addWidget(self.btn_toggle_view)

        # Sliders Group
        slider_group = QGroupBox("Parameters")
        slider_layout = QGridLayout()

        # Freq Slider (0.001 Hz precision)
        self.lbl_freq = QLabel(f"Frequency: {Config.INIT_FREQ:.3f} Hz")
        self.slider_freq = QSlider(Qt.Orientation.Horizontal)
        self.slider_freq.setRange(int(Config.FREQ_RANGE[0] * 1000), 
                                 int(Config.FREQ_RANGE[1] * 1000))
        self.slider_freq.setValue(int(Config.INIT_FREQ * 1000))
        self.slider_freq.valueChanged.connect(self.on_freq_slider_changed)
        
        slider_layout.addWidget(self.lbl_freq, 0, 0)
        slider_layout.addWidget(self.slider_freq, 1, 0)

        # Gamma Slider (Scale by 1000)
        self.lbl_gamma = QLabel(f"Damping (γ): {Config.INIT_GAMMA:.3f}")
        self.slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamma.setRange(int(Config.GAMMA_RANGE[0] * 1000), int(Config.GAMMA_RANGE[1] * 1000))
        self.slider_gamma.setValue(int(Config.INIT_GAMMA * 1000))
        self.slider_gamma.valueChanged.connect(self.on_gamma_slider_changed)

        slider_layout.addWidget(self.lbl_gamma, 2, 0)
        slider_layout.addWidget(self.slider_gamma, 3, 0)
        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)

        # Navigation Group
        nav_group = QGroupBox("Resonance Navigation")
        nav_layout = QHBoxLayout()
        btn_prev = QPushButton("◀ Prev")
        btn_next = QPushButton("Next ▶")
        btn_prev.clicked.connect(self.jump_prev)
        btn_next.clicked.connect(self.jump_next)
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)

        # Scan Group
        scan_group = QGroupBox("Auto Scan")
        scan_layout = QHBoxLayout()
        self.btn_scan = QPushButton("Start Scan")
        self.btn_stop = QPushButton("Stop Scan")
        self.btn_stop.setEnabled(False)
        self.btn_scan.clicked.connect(self.start_scan)
        self.btn_stop.clicked.connect(self.stop_scan)
        scan_layout.addWidget(self.btn_scan)
        scan_layout.addWidget(self.btn_stop)
        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)

        # Resonance Windows Button
        btn_res_win = QPushButton("Show Resonance Curves")
        btn_res_win.clicked.connect(self.show_resonance_window)
        layout.addWidget(btn_res_win)

        # Modes Text
        self.lbl_modes = QLabel("")
        
        # --- THE BULLETPROOF FONT FIX ---
        # Ask the system for its exact native monospace font (no searching required)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.lbl_modes.setFont(font)
        # --------------------------------
        
        self.lbl_modes.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.lbl_modes, stretch=1)

    # ---------------- UI Callbacks ----------------
    def toggle_view(self):
        self.view_mode = self._views[self.view_mode]['next_key']
        self.btn_toggle_view.setText(f"Toggle to {self._views[self._views[self.view_mode]['next_key']]['title']}")
        self.update_plot(self.simulator.current_frequency)

    def on_freq_slider_changed(self, val):
        f = val / 1000.0
        self.lbl_freq.setText(f"Frequency: {f:.3f} Hz")
        self.simulator.set_current_frequency(f)
        self.update_plot(f)

    def on_gamma_slider_changed(self, val):
        gamma = val / 1000.0
        self.lbl_gamma.setText(f"Damping (γ): {gamma:.3f}")
        self.simulator.set_gamma(gamma)
        self.update_plot(self.simulator.current_frequency)

    def jump_prev(self):
        """Jump to previous resonance exactly."""
        current = self.simulator.current_frequency
        f_exact = self.simulator.get_previous_resonance_frequency(current)
        
        self.simulator.set_current_frequency(f_exact)
        
        slider_value = round(f_exact * 1000)
        self.slider_freq.blockSignals(True)
        self.slider_freq.setValue(slider_value)
        self.slider_freq.blockSignals(False)
        
        self.lbl_freq.setText(f"Frequency: {f_exact:.3f} Hz")
        self.update_plot(f_exact)
        self.slider_freq.repaint()


    def jump_next(self):
        """Jump to next resonance exactly."""
        current = self.simulator.current_frequency
        f_exact = self.simulator.get_next_resonance_frequency(current)
        
        self.simulator.set_current_frequency(f_exact)
        
        slider_value = round(f_exact * 1000)
        self.slider_freq.blockSignals(True)
        self.slider_freq.setValue(slider_value)
        self.slider_freq.blockSignals(False)
        
        self.lbl_freq.setText(f"Frequency: {f_exact:.3f} Hz")
        self.update_plot(f_exact)
        self.slider_freq.repaint()

    def start_scan(self):
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.scan_timer.start(50)  # 50 ms

    def stop_scan(self):
        self.scan_timer.stop()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _scan_step(self):
        """Auto scan with clean updates."""
        f = self.simulator.current_frequency + Config.SCAN_SPEED
        if f > Config.FREQ_RANGE[1]:
            f = Config.FREQ_RANGE[0]
        
        self.simulator.set_current_frequency(f)
        
        self.slider_freq.blockSignals(True)
        self.slider_freq.setValue(round(f * 1000))
        self.slider_freq.blockSignals(False)
        
        self.lbl_freq.setText(f"Frequency: {f:.3f} Hz")
        self.update_plot(f)
        self.slider_freq.repaint()

    def show_resonance_window(self):
        if self.resonance_window is None or not self.resonance_window.isVisible():
            self.resonance_window = ResonanceCurveWindow_PyQt(self.simulator, self)
            self.resonance_window.show()
        else:
            self.resonance_window.raise_()
            self.resonance_window.activateWindow()

    # ---------------- Plot Updating ----------------
    def _update_imshow(self, data, cmap, clim):
        self.scatter_artist.set_visible(False)
        self.imshow_artist.set_visible(True)
        self.imshow_artist.set_data(data)
        self.imshow_artist.set_cmap(cmap)
        self.imshow_artist.set_clim(vmin=clim[0], vmax=clim[1])

    def _render_magnitude(self, Z):
        plot_data = np.abs(Z) ** Config.VISUAL_EXPONENT
        self._update_imshow(plot_data, 'plasma', (0, np.max(plot_data)))
        self.cbar.ax.set_visible(True)
        self.cbar.set_label(f'Displacement (|Z|^{Config.VISUAL_EXPONENT})')

    def _render_phase(self, Z):
        max_abs = np.max(np.abs(Z))
        self._update_imshow(Z, 'coolwarm', (-max_abs, max_abs))
        self.cbar.ax.set_visible(True)
        self.cbar.set_label('Signed Displacement (Phase View)')

    def _render_sand(self, Z):
        self.imshow_artist.set_visible(False)
        self.scatter_artist.set_visible(True)
        self.cbar.ax.set_visible(False)
        x, y = self.simulator.get_sand_coordinates_from_Z(Z)
        offsets = np.column_stack((x / (self.simulator.resolution - 1), 
                                   y / (self.simulator.resolution - 1)))
        self.scatter_artist.set_offsets(offsets)

    def update_plot(self, f: float):
        Z = self.simulator.compute_displacement(f)
        self._views[self.view_mode]['render'](Z)

        # Update Titles
        f_closest, deg_modes = self.simulator.current_closest_resonance
        title = f"f = {f:.2f}"
        if abs(f - f_closest) < Config.RESONANCE_TOL:
            deg_str = ', '.join(f"({m},{n})" for m, n in deg_modes)
            title += f" ← Resonance: {deg_str} f_mn={f_closest:.2f}"
        self.ax.set_title(title)
        self.setWindowTitle(f'Chladni Simulator — {self._views[self.view_mode]["title"]}')

        # Update Info Text
        modes_info = self.simulator.get_contributing_modes(f)
        max_m = Config.MAX_DISPLAY_MODES or len(modes_info)
        text = "Contributing Modes (%)\n\n"
        text += f"{'Mode':<10} {'f_mn':>6} {'Weight %':>10}\n"
        text += "-"*30 + "\n"
        for m, n, fmn, perc in modes_info[:max_m]:
            text += f"({m:>2},{n:<2}) {fmn:>8.2f} {perc:>9.1f}\n"
        self.lbl_modes.setText(text)

        self.canvas.draw_idle()


def main() -> None:
    # PyQt6 Application requires a QApplication instance
    app = QApplication(sys.argv)
    
    sim = ChladniSimulator()
    ui = ChladniUI_PyQt(sim)
    ui.show()
    
    # Execute the PyQt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
