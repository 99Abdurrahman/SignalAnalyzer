import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog, 
                           QGroupBox, QGridLayout, QTextEdit, QFrame, QSplitter, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
import os
from scipy.io import wavfile
from scipy import signal
from warnings import filterwarnings
filterwarnings('ignore')
# Import these from your existing modules
from signal_classification import SignalAnalyzer, sine_wave, decaying_exponential, gaussian_pulse, temperature_signal
from wavelet_analysis import WaveletAnalyzer


class MatplotlibCanvas(FigureCanvas):
    """Canvas for Matplotlib plots in PyQt5"""
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        """Initialize the canvas"""
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class SignalAnalysisGUI(QMainWindow):
    """
    Interactive PyQt5 GUI for signal analysis and classification
    """
    
    def __init__(self):
        """Initialize the GUI"""
        super().__init__()
        self.setWindowTitle("Signal Analysis & Classification Tool")
        self.resize(1200, 800)
        
        # Initialize variables
        self.signal_type = "sine"
        self.duration = 5.0
        self.amplitude = 2.0
        self.frequency = 2.0
        self.decay_rate = 0.5
        self.center = 2.5
        self.width = 0.5
        self.sampling_rate = 1000
        self.analyzer = None
        self.analysis_type = "basic"
        self.audio_path = None
        
        # Set up the GUI components
        self._create_widgets()
        
        # Set up initial state
        self._update_signal_controls()
        
    def _create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        splitter.addWidget(control_widget)
        
        # Right panel for plots
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        splitter.addWidget(plot_widget)
        
        # Set splitter sizes
        splitter.setSizes([300, 900])
        
        # === CONTROL PANEL ===
        # Signal controls group
        signal_group = QGroupBox("Signal Controls")
        signal_layout = QGridLayout()
        signal_group.setLayout(signal_layout)
        control_layout.addWidget(signal_group)
        
        # Signal type selection
        signal_layout.addWidget(QLabel("Signal Type:"), 0, 0)
        self.signal_combo = QComboBox()
        signal_types = ["sine", "exponential", "gaussian", "temperature", "chirp", "damped_oscillation", "load_audio"]
        self.signal_combo.addItems(signal_types)
        self.signal_combo.currentTextChanged.connect(self._update_signal_controls)
        signal_layout.addWidget(self.signal_combo, 0, 1)
        
        # Common parameters
        signal_layout.addWidget(QLabel("Duration (s):"), 1, 0)
        self.duration_edit = QLineEdit(str(self.duration))
        signal_layout.addWidget(self.duration_edit, 1, 1)
        
        signal_layout.addWidget(QLabel("Amplitude:"), 2, 0)
        self.amplitude_edit = QLineEdit(str(self.amplitude))
        signal_layout.addWidget(self.amplitude_edit, 2, 1)
        
        signal_layout.addWidget(QLabel("Sampling Rate (Hz):"), 3, 0)
        self.sampling_rate_edit = QLineEdit(str(self.sampling_rate))
        signal_layout.addWidget(self.sampling_rate_edit, 3, 1)
        
        # Signal-specific parameters group
        self.param_group = QGroupBox("Signal Parameters")
        self.param_layout = QGridLayout()
        self.param_group.setLayout(self.param_layout)
        signal_layout.addWidget(self.param_group, 4, 0, 1, 2)
        
        # Analysis options
        signal_layout.addWidget(QLabel("Analysis Type:"), 5, 0)
        self.analysis_combo = QComboBox()
        analysis_types = ["basic", "wavelet", "frequency", "energy_distribution"]
        self.analysis_combo.addItems(analysis_types)
        signal_layout.addWidget(self.analysis_combo, 5, 1)
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze Signal")
        self.analyze_btn.clicked.connect(self._analyze_signal)
        signal_layout.addWidget(self.analyze_btn, 6, 0, 1, 2)
        
        # Save results button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self._save_results)
        signal_layout.addWidget(self.save_btn, 7, 0, 1, 2)
        
        # Status display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        control_layout.addWidget(results_group)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        results_layout.addWidget(self.status_text)
        
        # Stretch to take up remaining space
        control_layout.addStretch(1)
        
        # === PLOT PANEL ===
        # Canvas for plots
        self.canvas = MatplotlibCanvas(self, width=10, height=8)
        plot_layout.addWidget(self.canvas)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
    
    def _update_signal_controls(self):
        """Update signal-specific parameter controls based on signal type"""
        # Clear existing widgets
        for i in reversed(range(self.param_layout.count())): 
            self.param_layout.itemAt(i).widget().setParent(None)
        
        signal_type = self.signal_combo.currentText()
        self.signal_type = signal_type
        
        if signal_type == "sine":
            self.param_layout.addWidget(QLabel("Frequency (Hz):"), 0, 0)
            self.freq_edit = QLineEdit(str(self.frequency))
            self.param_layout.addWidget(self.freq_edit, 0, 1)
            
        elif signal_type == "exponential":
            self.param_layout.addWidget(QLabel("Decay Rate:"), 0, 0)
            self.decay_edit = QLineEdit(str(self.decay_rate))
            self.param_layout.addWidget(self.decay_edit, 0, 1)
            
        elif signal_type == "gaussian":
            self.param_layout.addWidget(QLabel("Center:"), 0, 0)
            self.center_edit = QLineEdit(str(self.center))
            self.param_layout.addWidget(self.center_edit, 0, 1)
            
            self.param_layout.addWidget(QLabel("Width:"), 1, 0)
            self.width_edit = QLineEdit(str(self.width))
            self.param_layout.addWidget(self.width_edit, 1, 1)
            
        elif signal_type == "temperature":
            self.param_layout.addWidget(QLabel("Daily Variation (°C):"), 0, 0)
            self.daily_var_edit = QLineEdit("5.0")
            self.param_layout.addWidget(self.daily_var_edit, 0, 1)
            
            self.param_layout.addWidget(QLabel("Yearly Variation (°C):"), 1, 0)
            self.yearly_var_edit = QLineEdit("10.0")
            self.param_layout.addWidget(self.yearly_var_edit, 1, 1)
            
        elif signal_type == "chirp":
            self.param_layout.addWidget(QLabel("Start Frequency (Hz):"), 0, 0)
            self.start_freq_edit = QLineEdit("1.0")
            self.param_layout.addWidget(self.start_freq_edit, 0, 1)
            
            self.param_layout.addWidget(QLabel("End Frequency (Hz):"), 1, 0)
            self.end_freq_edit = QLineEdit("20.0")
            self.param_layout.addWidget(self.end_freq_edit, 1, 1)
            
        elif signal_type == "damped_oscillation":
            self.param_layout.addWidget(QLabel("Frequency (Hz):"), 0, 0)
            self.osc_freq_edit = QLineEdit("5.0")
            self.param_layout.addWidget(self.osc_freq_edit, 0, 1)
            
            self.param_layout.addWidget(QLabel("Decay Rate:"), 1, 0)
            self.osc_decay_edit = QLineEdit("0.5")
            self.param_layout.addWidget(self.osc_decay_edit, 1, 1)
            
        elif signal_type == "load_audio":
            browse_btn = QPushButton("Browse Audio File")
            browse_btn.clicked.connect(self._browse_audio)
            self.param_layout.addWidget(browse_btn, 0, 0, 1, 2)
            
            # Add label for showing selected file
            self.file_label = QLabel("No file selected")
            self.param_layout.addWidget(self.file_label, 1, 0, 1, 2)
    
    def _browse_audio(self):
        """Open file dialog to browse for audio files"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "WAV files (*.wav);;All files (*.*)"
        )
        if filename:
            self.audio_path = filename
            self.file_label.setText(f"Selected: {os.path.basename(filename)}")
            self.status_text.setText(f"Selected audio file: {os.path.basename(filename)}")
    
    def _get_signal_parameters(self):
        """Get all signal parameters from input fields"""
        try:
            self.duration = float(self.duration_edit.text())
            self.amplitude = float(self.amplitude_edit.text())
            self.sampling_rate = int(self.sampling_rate_edit.text())
            
            signal_type = self.signal_type
            
            if signal_type == "sine":
                self.frequency = float(self.freq_edit.text())
            elif signal_type == "exponential":
                self.decay_rate = float(self.decay_edit.text())
            elif signal_type == "gaussian":
                self.center = float(self.center_edit.text())
                self.width = float(self.width_edit.text())
            elif signal_type == "temperature":
                self.daily_var = float(self.daily_var_edit.text())
                self.yearly_var = float(self.yearly_var_edit.text())
            elif signal_type == "chirp":
                self.start_freq = float(self.start_freq_edit.text())
                self.end_freq = float(self.end_freq_edit.text())
            elif signal_type == "damped_oscillation":
                self.osc_freq = float(self.osc_freq_edit.text())
                self.osc_decay = float(self.osc_decay_edit.text())
            
            return True
        except ValueError as e:
            self.status_text.setText(f"Invalid input: {str(e)}")
            return False
    
    def _generate_signal(self):
        """Generate signal based on current settings"""
        if not self._get_signal_parameters():
            return False
            
        signal_type = self.signal_type
        sampling_rate = self.sampling_rate
        duration = self.duration
        amplitude = self.amplitude
        
        # Create a new analyzer instance
        self.analyzer = SignalAnalyzer(f"{signal_type.capitalize()} Signal", sampling_rate=sampling_rate)
        
        # Generate the appropriate signal
        if signal_type == "sine":
            frequency = self.frequency
            self.analyzer.create_signal(sine_wave, duration=duration, 
                                        frequency=frequency, amplitude=amplitude)
            
        elif signal_type == "exponential":
            decay_rate = self.decay_rate
            self.analyzer.create_signal(decaying_exponential, duration=duration, 
                                        amplitude=amplitude, decay_rate=decay_rate)
            
        elif signal_type == "gaussian":
            center = self.center
            width = self.width
            self.analyzer.create_signal(gaussian_pulse, duration=duration, 
                                        amplitude=amplitude, center=center, width=width)
            
        elif signal_type == "temperature":
            daily_var = getattr(self, 'daily_var', 5.0)
            yearly_var = getattr(self, 'yearly_var', 10.0)
            
            self.analyzer.create_signal(temperature_signal, duration=duration, 
                                        base_temp=amplitude, daily_variation=daily_var, 
                                        yearly_variation=yearly_var, noise_level=0.5)
            
        elif signal_type == "chirp":
            start_freq = getattr(self, 'start_freq', 1.0)
            end_freq = getattr(self, 'end_freq', 20.0)
            
            # Create a chirp signal
            def chirp_signal(t, start_freq=1.0, end_freq=20.0):
                from scipy import signal as sig
                return amplitude * sig.chirp(t, f0=start_freq, f1=end_freq, t1=duration, method='linear')
            
            self.analyzer.create_signal(chirp_signal, duration=duration, 
                                    start_freq=start_freq, end_freq=end_freq)
            
        elif signal_type == "damped_oscillation":
            freq = getattr(self, 'osc_freq', 5.0)
            decay = getattr(self, 'osc_decay', 0.5)
            
            # Create a damped oscillation signal
            def damped_oscillation(t, freq=5.0, decay=0.5):
                return amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)
            
            self.analyzer.create_signal(damped_oscillation, duration=duration, 
                                    freq=freq, decay=decay)
            
        elif signal_type == "load_audio":
            if hasattr(self, 'audio_path') and self.audio_path:
                self.analyzer.load_audio(self.audio_path)
            else:
                self.status_text.setText("No audio file selected!")
                return False
        
        return True
    
    @pyqtSlot()
    def _analyze_signal(self):
        """Analyze the signal with the currently selected method"""
        # Generate the signal based on current settings
        if not self._generate_signal():
            return
        
        # Clear previous plots
        self.canvas.fig.clear()
        
        # Get the selected analysis type
        analysis_type = self.analysis_combo.currentText()
        
        try:
            # Perform the appropriate analysis
            if analysis_type == "basic":
                # Basic signal classification
                self.analyzer.calculate_energy()
                self.analyzer.calculate_power()
                classification = self.analyzer.classify_signal()
                
                # Display results
                self.status_text.setText(str(self.analyzer))
                
                # Plot the signal with classification details
                plot_fig = self.analyzer.plot_signal()
                
                # Copy the plots to our existing figure
                for i, ax in enumerate(plot_fig.get_axes()):
                    if i >= len(self.canvas.fig.get_axes()):
                        self.canvas.fig.add_subplot(len(plot_fig.get_axes()), 1, i+1)
                    self.canvas.fig.get_axes()[i].clear()
                    for line in ax.get_lines():
                        self.canvas.fig.get_axes()[i].plot(line.get_xdata(), line.get_ydata())
                    self.canvas.fig.get_axes()[i].set_title(ax.get_title())
                    self.canvas.fig.get_axes()[i].set_xlabel(ax.get_xlabel())
                    self.canvas.fig.get_axes()[i].set_ylabel(ax.get_ylabel())
                    self.canvas.fig.get_axes()[i].grid(True)
                
            elif analysis_type == "wavelet":
                # Create a WaveletAnalyzer instance
                wavelet_analyzer = WaveletAnalyzer(f"{self.analyzer.name} - Wavelet", 
                                                  sampling_rate=self.analyzer.sampling_rate)
                wavelet_analyzer.signal = self.analyzer.signal
                wavelet_analyzer.time = self.analyzer.time
                
                # Perform wavelet transform with appropriate scales
                scales = np.arange(1, 128)  # Use a reasonable range of scales
                wavelet_analyzer.perform_wavelet_transform(wavelet='morl', scales=scales)
                
                # Display results
                self.status_text.setText("Wavelet Analysis Completed\n")
                self.status_text.append(f"Signal: {wavelet_analyzer.name}")
                
                # Plot wavelet analysis
                self.canvas.fig.clear()
                ax = self.canvas.fig.add_subplot(111)
                
                # Create the wavelet plot directly in our figure
                pcm = ax.pcolormesh(wavelet_analyzer.time, 
                                   wavelet_analyzer.frequencies, 
                                   np.abs(wavelet_analyzer.wavelet_results), 
                                   shading='gouraud', 
                                   cmap='viridis')
                self.canvas.fig.colorbar(pcm, ax=ax, label='Magnitude')
                ax.set_title(f'Wavelet Transform for {wavelet_analyzer.name}', fontsize=14)
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Time (s)')
                ax.set_yscale('log')  # Log scale for frequencies
                
            elif analysis_type == "frequency":
                # Create a WaveletAnalyzer instance for frequency analysis
                freq_analyzer = WaveletAnalyzer(f"{self.analyzer.name} - Frequency", 
                                            sampling_rate=self.analyzer.sampling_rate)
                freq_analyzer.signal = self.analyzer.signal
                freq_analyzer.time = self.analyzer.time
                
                # Perform frequency analysis
                freq_analyzer.compute_fft()
                
                # Display results
                self.status_text.setText("Frequency Analysis Completed\n")
                self.status_text.append(f"Signal: {freq_analyzer.name}")
                
                # Plot frequency analysis
                self.canvas.fig.clear()
                gs = self.canvas.fig.add_gridspec(2, 2)
                
                # Time domain
                ax1 = self.canvas.fig.add_subplot(gs[0, 0])
                ax1.plot(freq_analyzer.time, freq_analyzer.signal)
                ax1.set_title(f"{freq_analyzer.name} - Time Domain", fontsize=14)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Amplitude')
                ax1.grid(True)
                
                # Frequency domain
                ax2 = self.canvas.fig.add_subplot(gs[0, 1])
                ax2.plot(freq_analyzer.frequencies, freq_analyzer.spectrum)
                ax2.set_title('Frequency Domain (FFT)', fontsize=14)
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Magnitude')
                ax2.set_xlim([0, min(50, max(freq_analyzer.frequencies))])  # Limit to relevant frequencies
                ax2.grid(True)
                
                # Power spectral density (magnitude squared)
                ax3 = self.canvas.fig.add_subplot(gs[1, 0])
                ax3.plot(freq_analyzer.frequencies, freq_analyzer.spectrum**2)
                ax3.set_title('Power Spectral Density', fontsize=14)
                ax3.set_xlabel('Frequency (Hz)')
                ax3.set_ylabel('Power')
                ax3.set_xlim([0, min(50, max(freq_analyzer.frequencies))])  # Limit to relevant frequencies
                ax3.grid(True)
                
                # Compute and plot spectrogram
                ax4 = self.canvas.fig.add_subplot(gs[1, 1])
                f, t, Sxx = signal.spectrogram(freq_analyzer.signal, 
                                            1/(freq_analyzer.time[1]-freq_analyzer.time[0]))
                pcm = ax4.pcolormesh(t, f, 10 * np.log10(Sxx), 
                                    shading='gouraud', cmap='viridis')
                ax4.set_title('Spectrogram', fontsize=14)
                ax4.set_ylabel('Frequency (Hz)')
                ax4.set_xlabel('Time (s)')
                self.canvas.fig.colorbar(pcm, ax=ax4, label='Power/Frequency (dB/Hz)')
                
            elif analysis_type == "energy_distribution":
                # Create a WaveletAnalyzer instance for energy distribution analysis
                energy_analyzer = WaveletAnalyzer(f"{self.analyzer.name} - Energy", 
                                                sampling_rate=self.analyzer.sampling_rate)
                energy_analyzer.signal = self.analyzer.signal
                energy_analyzer.time = self.analyzer.time
                
                # Perform energy distribution analysis
                energy_analyzer.compute_fft()  # Required before energy distribution analysis
                
                # Calculate energy density across frequencies
                energy_density = energy_analyzer.spectrum**2
                
                # Calculate cumulative energy distribution
                cumulative_energy = np.cumsum(energy_density)
                cumulative_energy_normalized = (cumulative_energy / cumulative_energy[-1] 
                                               if cumulative_energy[-1] > 0 else cumulative_energy)
                
                # Display results
                self.status_text.setText("Energy Distribution Analysis Completed\n")
                self.status_text.append(f"Signal: {energy_analyzer.name}")
                
                # Plot energy distribution
                self.canvas.fig.clear()
                gs = self.canvas.fig.add_gridspec(2, 1)
                
                # Energy density
                ax1 = self.canvas.fig.add_subplot(gs[0])
                ax1.plot(energy_analyzer.frequencies, energy_density)
                ax1.set_title(f'Energy Density Spectrum for {energy_analyzer.name}', fontsize=14)
                ax1.set_xlabel('Frequency (Hz)')
                ax1.set_ylabel('Energy Density')
                ax1.set_xlim([0, min(50, max(energy_analyzer.frequencies))])  # Limit to relevant frequencies
                ax1.grid(True)
                
                # Cumulative energy distribution
                ax2 = self.canvas.fig.add_subplot(gs[1])
                ax2.plot(energy_analyzer.frequencies, cumulative_energy_normalized)
                ax2.set_title('Cumulative Energy Distribution', fontsize=14)
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Normalized Cumulative Energy')
                ax2.set_xlim([0, min(50, max(energy_analyzer.frequencies))])  # Limit to relevant frequencies
                ax2.grid(True)
                
                # Find frequency containing 90% of energy
                if cumulative_energy_normalized[-1] > 0:
                    idx_90 = np.where(cumulative_energy_normalized >= 0.9)[0]
                    if len(idx_90) > 0:
                        freq_90 = energy_analyzer.frequencies[idx_90[0]]
                        ax2.axhline(y=0.9, color='k', linestyle='--', alpha=0.7)
                        ax2.axvline(x=freq_90, color='k', linestyle='--', alpha=0.7)
                        ax2.text(freq_90*1.1, 0.85, f'90% Energy: {freq_90:.2f} Hz', 
                                bbox=dict(facecolor='white', alpha=0.7))
            
            # Adjust layout and refresh canvas
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.status_text.setText(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
    
    @pyqtSlot()
    def _save_results(self):
        """Save the current analysis results"""
        if not hasattr(self, 'analyzer') or self.analyzer is None:
            self.status_text.setText("No analysis to save! Please run an analysis first.")
            return
        
        # Save the figure
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Figure", "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg);;All files (*.*)"
        )
        
        if filename:
            try:
                self.canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.status_text.append(f"\nFigure saved to: {filename}")
            except Exception as e:
                self.status_text.append(f"\nError saving figure: {str(e)}")


def run_gui():
    """Run the signal analysis GUI application with PyQt5"""
    app = QApplication(sys.argv)
    window = SignalAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()