import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from matplotlib.gridspec import GridSpec
import pywt

class WaveletAnalyzer:
    """
    Class for wavelet transform and advanced frequency analysis of signals
    """
    
    def __init__(self, name, sampling_rate=1000):
        """Initialize the wavelet analyzer with a name and sampling rate"""
        self.name = name
        self.sampling_rate = sampling_rate
        self.signal = None
        self.time = None
        self.wavelet_results = None
        self.frequencies = None
        self.spectrum = None
        # For compatibility with SignalAnalyzer
        self.energy = None
        self.power = None
        self.classification = None
        self.energy_accumulation = None
        self.power_accumulation = None
        
    def create_signal(self, signal_func, duration=10, **kwargs):
        """
        Create a signal using the provided function
        
        Args:
            signal_func (function): Function that generates the signal
            duration (float): Duration of the signal in seconds
            **kwargs: Additional parameters to pass to the signal function
        """
        self.time = np.linspace(0, duration, int(duration * self.sampling_rate))
        self.signal = signal_func(self.time, **kwargs)
        return self
    
    def create_discrete_signal(self, signal_array, time_array=None):
        """
        Create a discrete signal from an array
        
        Args:
            signal_array (array): Array of signal values
            time_array (array, optional): Array of time values
        """
        self.signal = np.array(signal_array)
        if time_array is not None:
            self.time = np.array(time_array)
        else:
            self.time = np.arange(len(self.signal)) / self.sampling_rate
        return self
    
    def load_audio(self, filename):
        """
        Load an audio file as a signal
        
        Args:
            filename (str): Path to the audio file
        """
        from scipy.io import wavfile
        self.sampling_rate, audio_data = wavfile.read(filename)
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Normalize
        self.signal = audio_data / np.max(np.abs(audio_data))
        self.time = np.arange(len(self.signal)) / self.sampling_rate
        return self
        
    def perform_wavelet_transform(self, wavelet='morl', scales=None, num_scales=64):
        """
        Perform continuous wavelet transform on the signal
    
        Args:
            wavelet (str): Wavelet to use ('morl', 'cmor', 'gaus', etc.)
            scales (array, optional): Scales for wavelet transform
            num_scales (int): Number of scales if scales not provided
        
        Returns:
            tuple: (coefficients, frequencies)
        """
        if self.signal is None:
            raise ValueError("Signal not created yet")
        
        if scales is None:
            # Create logarithmically spaced scales
            scales = np.logspace(0, np.log10(len(self.time)//10), num_scales)
        
        # Calculate sampling period
        dt = self.time[1] - self.time[0] if len(self.time) > 1 else 1
        
        # Perform continuous wavelet transform using PyWavelets
        coefficients, frequencies = pywt.cwt(self.signal, scales, wavelet)
        
        # Convert scales to frequencies in Hz
        # Frequency calculation based on wavelet central frequency
        central_freq = pywt.central_frequency(wavelet)
        freq = central_freq / (scales * dt)
        
        self.wavelet_results = coefficients
        self.frequencies = freq
        
        return coefficients, freq
    
    def compute_fft(self):
        """
        Compute Fast Fourier Transform of the signal
        
        Returns:
            tuple: (frequencies, spectrum)
        """
        if self.signal is None:
            raise ValueError("Signal not created yet")
        
        # Calculate sampling period
        dt = self.time[1] - self.time[0] if len(self.time) > 1 else 1
        
        # Calculate FFT
        n = len(self.signal)
        yf = fft(self.signal)
        xf = fftfreq(n, dt)[:n//2]
        
        # Get only positive frequencies
        self.frequencies = xf
        self.spectrum = 2.0/n * np.abs(yf[0:n//2])
        
        return xf, 2.0/n * np.abs(yf[0:n//2])
    
    def calculate_energy(self):
        """
        Calculate the total energy of the signal
        
        Energy = ∫|x(t)|²dt for continuous-time signals
        Energy = ∑|x[n]|² for discrete-time signals
        
        Returns:
            float: Total energy of the signal
        """
        # Calculate squared amplitude
        squared_signal = np.abs(self.signal) ** 2
        
        # For discrete signals, we sum
        # For continuous signals approximated by discrete samples, we integrate (sum * dt)
        dt = self.time[1] - self.time[0] if len(self.time) > 1 else 1
        
        # Calculate cumulative energy over time
        self.energy_accumulation = np.cumsum(squared_signal) * dt
        self.energy = self.energy_accumulation[-1]
        
        return self.energy
    
    def calculate_power(self):
        """
        Calculate the average power of the signal
        
        Power = lim(T→∞) (1/2T) ∫_{-T}^{T} |x(t)|²dt for continuous-time signals
        Power = lim(N→∞) (1/(2N+1)) ∑_{n=-N}^{N} |x[n]|² for discrete-time signals
        
        Returns:
            float: Average power of the signal
        """
        # Calculate squared amplitude
        squared_signal = np.abs(self.signal) ** 2
        
        # Calculate cumulative power over time
        self.power_accumulation = np.cumsum(squared_signal) / np.arange(1, len(squared_signal) + 1)
        self.power = self.power_accumulation[-1]
        
        return self.power
    
    def classify_signal(self, energy_threshold=1e10, power_threshold=1e-10):
        """
        Classify the signal as energy signal, power signal, or neither
        
        Energy signal: Finite energy, zero average power
        Power signal: Infinite energy, finite non-zero average power
        Neither: Infinite energy, infinite average power
        
        Args:
            energy_threshold (float): Threshold for considering energy finite
            power_threshold (float): Threshold for considering power non-zero
        
        Returns:
            str: Classification as "Energy Signal", "Power Signal", or "Neither"
        """
        if self.energy is None:
            self.calculate_energy()
        if self.power is None:
            self.calculate_power()
        
        # Check energy convergence
        energy_convergence = self._check_convergence(self.energy_accumulation)
        
        # Check power convergence
        power_convergence = self._check_convergence(self.power_accumulation)
        
        # Determine classification based on energy and power characteristics
        if energy_convergence and self.energy < energy_threshold:
            self.classification = "Energy Signal"
        elif power_convergence and self.power > power_threshold:
            self.classification = "Power Signal"
        else:
            self.classification = "Neither"
            
        return self.classification
    
    def _check_convergence(self, sequence, window_size=100, threshold=0.01):
        """
        Check if a sequence has converged
        
        Args:
            sequence (array): Sequence to check for convergence
            window_size (int): Size of window to check for stability
            threshold (float): Maximum relative change for stability
        
        Returns:
            bool: True if the sequence has converged, False otherwise
        """
        if len(sequence) < window_size * 2:
            return False
        
        # Check if the sequence is stable in the last window
        last_values = sequence[-window_size:]
        if np.mean(last_values) == 0:
            return True
        
        relative_changes = np.abs(np.diff(last_values) / np.mean(last_values))
        return np.mean(relative_changes) < threshold
    
    def plot_time_frequency_analysis(self):
        """
        Plot the signal in time and frequency domains
        
        Returns:
            matplotlib.figure.Figure: Figure with time and frequency domain plots
        """
        if self.signal is None:
            raise ValueError("Signal not created yet")
        
        # Compute FFT if not already done
        if self.spectrum is None:
            self.compute_fft()
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot the signal in time domain
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.time, self.signal, 'b')
        ax1.set_title(f"{self.name} - Time Domain", fontsize=14)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Plot the magnitude spectrum
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.frequencies, self.spectrum, 'r')
        ax2.set_title('Frequency Domain (FFT)', fontsize=14)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_xlim([0, min(50, max(self.frequencies))])  # Limit to relevant frequencies
        ax2.grid(True)
        
        # Plot the power spectral density (magnitude squared)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.frequencies, self.spectrum**2, 'g')
        ax3.set_title('Power Spectral Density', fontsize=14)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.set_xlim([0, min(50, max(self.frequencies))])  # Limit to relevant frequencies
        ax3.grid(True)
        
        # Compute and plot spectrogram
        ax4 = fig.add_subplot(gs[1, 1])
        f, t, Sxx = signal.spectrogram(self.signal, 1/(self.time[1]-self.time[0]))
        pcm = ax4.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        ax4.set_title('Spectrogram', fontsize=14)
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xlabel('Time (s)')
        fig.colorbar(pcm, ax=ax4, label='Power/Frequency (dB/Hz)')
        
        plt.tight_layout()
        return fig
    
    def plot_wavelet_analysis(self, perform_transform=True):
        """
        Plot wavelet analysis of the signal
        
        Args:
            perform_transform (bool): Whether to perform wavelet transform if not already done
        
        Returns:
            matplotlib.figure.Figure: Figure with wavelet analysis plots
        """
        if self.signal is None:
            raise ValueError("Signal not created yet")
        
        # Perform wavelet transform if not already done
        if self.wavelet_results is None and perform_transform:
            self.perform_wavelet_transform()
        
        if self.wavelet_results is None:
            raise ValueError("Wavelet transform not performed yet")
        
        fig = plt.figure(figsize=(12, 8))
        
        # Plot the wavelet coefficients
        plt.pcolormesh(self.time, self.frequencies, np.abs(self.wavelet_results), 
                       shading='gouraud', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(f'Wavelet Transform for {self.name}', fontsize=14)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.yscale('log')  # Log scale for frequencies
        
        plt.tight_layout()
        return fig
    
    def analyze_energy_distribution(self):
        """
        Analyze the distribution of energy across frequencies
        
        Returns:
            matplotlib.figure.Figure: Figure with energy distribution analysis
        """
        if self.signal is None:
            raise ValueError("Signal not created yet")
        
        # Compute FFT if not already done
        if self.spectrum is None:
            self.compute_fft()
        
        # Calculate energy density across frequencies
        energy_density = self.spectrum**2
        
        # Calculate cumulative energy distribution
        cumulative_energy = np.cumsum(energy_density)
        cumulative_energy_normalized = cumulative_energy / cumulative_energy[-1] if cumulative_energy[-1] > 0 else cumulative_energy
        
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1])
        
        # Plot energy density
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.frequencies, energy_density, 'b')
        ax1.set_title(f'Energy Density Spectrum for {self.name}', fontsize=14)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Energy Density')
        ax1.set_xlim([0, min(50, max(self.frequencies))])  # Limit to relevant frequencies
        ax1.grid(True)
        
        # Plot cumulative energy distribution
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.frequencies, cumulative_energy_normalized, 'r')
        ax2.set_title('Cumulative Energy Distribution', fontsize=14)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Normalized Cumulative Energy')
        ax2.set_xlim([0, min(50, max(self.frequencies))])  # Limit to relevant frequencies
        ax2.grid(True)
        
        # Find frequency containing 90% of energy
        if cumulative_energy_normalized[-1] > 0:
            idx_90 = np.where(cumulative_energy_normalized >= 0.9)[0]
            if len(idx_90) > 0:
                freq_90 = self.frequencies[idx_90[0]]
                ax2.axhline(y=0.9, color='k', linestyle='--', alpha=0.7)
                ax2.axvline(x=freq_90, color='k', linestyle='--', alpha=0.7)
                ax2.text(freq_90*1.1, 0.85, f'90% Energy: {freq_90:.2f} Hz', 
                         bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def get_summary(self):
        """
        Get a summary of the signal analysis results
        
        Returns:
            dict: Summary of the signal analysis
        """
        if self.classification is None:
            self.classify_signal()
            
        return {
            'name': self.name,
            'energy': self.energy,
            'power': self.power,
            'classification': self.classification,
            'duration': self.time[-1] - self.time[0],
            'sampling_rate': self.sampling_rate
        }
    
    def get_wavelet_summary(self):
        """
        Get a comprehensive summary of wavelet analysis results
        
        Returns:
            dict: Summary of wavelet analysis 
        """
        summary = self.get_summary()  # Get basic summary from parent class
        
        # Add wavelet-specific information
        if self.spectrum is not None:
            # Find dominant frequency (peak in spectrum)
            dominant_freq_idx = np.argmax(self.spectrum)
            dominant_freq = self.frequencies[dominant_freq_idx]
            
            # Calculate bandwidth (frequency range containing 90% of energy)
            energy_density = self.spectrum**2
            cumulative_energy = np.cumsum(energy_density)
            cumulative_energy_normalized = cumulative_energy / cumulative_energy[-1] if cumulative_energy[-1] > 0 else cumulative_energy
            
            idx_90 = np.where(cumulative_energy_normalized >= 0.9)[0]
            freq_90 = self.frequencies[idx_90[0]] if len(idx_90) > 0 else None
            
            # Add to summary
            summary.update({
                'dominant_frequency': dominant_freq,
                'frequency_90percent_energy': freq_90,
                'has_wavelet_analysis': self.wavelet_results is not None,
                'has_frequency_analysis': True
            })
        
        return summary
    
    def __str__(self):
        """String representation of the analysis results"""
        if self.classification is None:
            self.classify_signal()
            
        return (f"Signal: {self.name}\n"
                f"Classification: {self.classification}\n"
                f"Total Energy: {self.energy:.4e}\n"
                f"Average Power: {self.power:.4e}")