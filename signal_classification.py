import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.io import wavfile
import time
#from matplotlib.ticker import ScientificFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)

class SignalAnalyzer:
    """
    A class for analyzing and classifying signals as energy signals, power signals, or neither
    """
    
    def __init__(self, name, sampling_rate=1000):
        """
        Initialize the signal analyzer with a name and sampling rate
        
        Args:
            name (str): Name of the signal
            sampling_rate (int): Sampling rate in Hz
        """
        self.name = name
        self.sampling_rate = sampling_rate
        self.signal = None
        self.time = None
        self.energy = None
        self.power = None
        self.classification = None
        
    def create_signal(self, signal_func, duration=10, **kwargs):
        """
        Create a signal using the provided function
        
        Args:
            signal_func (fuwnction): Function that generates the signal
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
        self.sampling_rate, audio_data = wavfile.read(filename)
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Normalize
        self.signal = audio_data / np.max(np.abs(audio_data))
        self.time = np.arange(len(self.signal)) / self.sampling_rate
        return self
    
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
    
    def plot_signal(self, show_classification=True):
        """
        Plot the signal and its energy/power characteristics
        
        Args:
            show_classification (bool): Whether to show the classification in the title
        """
        if self.classification is None:
            self.classify_signal()
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot the signal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.time, self.signal, 'b')
        ax1.set_title(f"{self.name}", fontsize=16)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Plot the energy accumulation
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.time, self.energy_accumulation, 'r')
        ax2.set_title('Energy Accumulation', fontsize=14)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy')
        ax2.grid(True)
        if self.energy > 1e5:
            ax2.yaxis.set_major_formatter(formatter)
        
        # Plot the power accumulation
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.time, self.power_accumulation, 'g')
        ax3.set_title('Power Accumulation', fontsize=14)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power')
        ax3.grid(True)
        
        # Plot the energy and power in detail for the last portion
        last_portion = max(1, int(len(self.time) * 0.8))
        
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.time[last_portion:], self.energy_accumulation[last_portion:], 'r')
        ax4.set_title('Energy Convergence Detail', fontsize=14)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Energy')
        ax4.grid(True)
        if self.energy > 1e5:
            ax4.yaxis.set_major_formatter(formatter)
        
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(self.time[last_portion:], self.power_accumulation[last_portion:], 'g')
        ax5.set_title('Power Convergence Detail', fontsize=14)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Power')
        ax5.grid(True)
        
        # Add classification text box
        if show_classification:
            classification_text = (
                f"Classification: {self.classification}\n"
                f"Total Energy: {self.energy:.4e}\n"
                f"Average Power: {self.power:.4e}"
            )
            fig.text(0.5, 0.02, classification_text, 
                     horizontalalignment='center',
                     bbox=dict(facecolor='wheat', alpha=0.5),
                     fontsize=14)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
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
    
    def __str__(self):
        """String representation of the analysis results"""
        if self.classification is None:
            self.classify_signal()
            
        return (f"Signal: {self.name}\n"
                f"Classification: {self.classification}\n"
                f"Total Energy: {self.energy:.4e}\n"
                f"Average Power: {self.power:.4e}")


# Signal generator functions
def sine_wave(t, frequency=1, amplitude=1, phase=0):
    """Generate a sine wave signal"""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

def decaying_exponential(t, amplitude=1, decay_rate=1):
    """Generate a decaying exponential signal"""
    return amplitude * np.exp(-decay_rate * t)

def gaussian_pulse(t, amplitude=1, center=0, width=1):
    """Generate a Gaussian pulse signal"""
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))

def temperature_signal(t, base_temp=20, daily_variation=5, yearly_variation=10, noise_level=1):
    """Generate a temperature signal with daily and yearly variations plus noise"""
    # Daily variation (period = 1 day = 24 hours)
    daily = daily_variation * np.sin(2 * np.pi * t / (24 * 3600))
    
    # Yearly variation (period = 1 year = 365.25 days)
    yearly = yearly_variation * np.sin(2 * np.pi * t / (365.25 * 24 * 3600))
    
    # Random noise
    noise = noise_level * np.random.randn(len(t))
    
    return base_temp + daily + yearly + noise