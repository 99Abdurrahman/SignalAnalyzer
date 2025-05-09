# Signal Analysis & Classification Tool

This project provides an interactive PyQt5-based GUI application for generating, analyzing, and classifying various types of signals. It supports time-domain, frequency-domain, wavelet, and energy distribution analysis of synthetic and audio signals.

## 📌 Features

- Generate common synthetic signals: sine, exponential, Gaussian pulse, chirp, damped oscillation, temperature simulation
- Load and analyze real audio signals (WAV format)
- Perform:
  - **Basic Signal Classification** (Energy vs. Power signal)
  - **Wavelet Transform Analysis**
  - **Frequency Domain Analysis** (FFT, PSD, Spectrogram)
  - **Energy Distribution Analysis**
- Save visualizations of the results
- Clean and modular code structure

## 🖼️ GUI Preview

The main GUI includes:
- Signal parameter inputs
- Type and method selection
- Live visualization panel with plots
- Result log output

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/99Abdurrahman/SignalAnalyzer.git
   cd signal-analysis-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Dependencies include**:
   - PyQt5
   - matplotlib
   - numpy
   - scipy
   - pywt (PyWavelets)

3. Run the application:
   ```bash
   python demo.py
   ```

## 📂 File Structure

- `demo.py` — Main GUI application using PyQt5.
- `signal_classification.py` — Logic for creating and classifying signals, calculating energy/power, and plotting.
- `wavelet_analysis.py` — Advanced analysis module with wavelet transforms, FFT, and energy spectrum utilities.

## 🔬 Signal Types Supported

- **Sine wave**
- **Decaying exponential**
- **Gaussian pulse**
- **Temperature simulation** (with daily/yearly cycles and noise)
- **Chirp signal**
- **Damped oscillation**
- **Audio file loader** (WAV)

## 📊 Analysis Modes

| Mode                | Description |
|---------------------|-------------|
| Basic               | Energy, Power, and classification plots |
| Wavelet             | Continuous Wavelet Transform (CWT) using `pywt` |
| Frequency           | FFT spectrum, power spectral density, and spectrogram |
| Energy Distribution | Cumulative energy spectrum and dominant frequency |

## 📤 Export Options

- Save current plot view in PNG, PDF, SVG formats using the GUI.

## 📄 License

This project is licensed under the MIT License.
