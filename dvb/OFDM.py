"""
DVB-T OFDM Modulation

OFDM (Orthogonal Frequency Division Multiplexing) converts frequency-domain
carriers to time-domain samples using the Inverse FFT.

DVB-T OFDM parameters:
- 2K mode: 2048-point FFT, 1705 active carriers
- 8K mode: 8192-point FFT, 6817 active carriers

The carriers are centered in the FFT bins with guard bands on each side.
DC (center carrier) is not used.

Reference: ETSI EN 300 744 Section 4.4
"""

import numpy as np
from typing import Tuple, Optional


class OFDMModulator:
    """
    DVB-T OFDM modulator.
    
    Converts frequency-domain carrier values to time-domain samples
    using IFFT.
    
    Attributes:
        mode: '2K' or '8K'
        fft_size: FFT size (2048 or 8192)
        active_carriers: Number of active carriers (1705 or 6817)
        
    Example:
        >>> mod = OFDMModulator('2K')
        >>> time_samples = mod.modulate(carrier_values)
    """
    
    # Mode parameters
    MODES = {
        '2K': {
            'fft_size': 2048,
            'active_carriers': 1705,
        },
        '8K': {
            'fft_size': 8192,
            'active_carriers': 6817,
        },
    }
    
    def __init__(self, mode: str = '2K', fast: bool = True):
        """
        Initialize OFDM modulator.
        
        Args:
            mode: '2K' or '8K'
            fast: Use numpy FFT (True) or educational DFT (False)
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.mode = mode
        self.fft_size = self.MODES[mode]['fft_size']
        self.active_carriers = self.MODES[mode]['active_carriers']
        self.fast = fast
        
        # Calculate carrier positions in FFT bins
        # Carriers are numbered 0 to Kmax, with K=0 being lowest frequency
        # Map to FFT bins: carrier 0 -> bin -Kmax/2, carrier Kmax -> bin +Kmax/2
        # In numpy FFT: positive frequencies in bins 0 to N/2-1
        #               negative frequencies in bins N/2 to N-1
        
        self._calc_carrier_mapping()
    
    def _calc_carrier_mapping(self) -> None:
        """Calculate mapping from carrier index to FFT bin."""
        N = self.fft_size
        K = self.active_carriers
        
        # Carrier k maps to frequency f_k = (k - (K-1)/2) * carrier_spacing
        # In FFT bins, negative frequencies wrap around
        
        # Center carrier index
        center = (K - 1) // 2
        
        # FFT bin for each carrier
        self._carrier_to_bin = np.zeros(K, dtype=np.int32)
        
        for k in range(K):
            # Frequency offset from DC
            freq_offset = k - center
            
            # Map to FFT bin
            if freq_offset >= 0:
                bin_idx = freq_offset
            else:
                bin_idx = N + freq_offset
            
            self._carrier_to_bin[k] = bin_idx
    
    def modulate(self, carriers: np.ndarray) -> np.ndarray:
        """
        Modulate carriers to time-domain samples.
        
        Args:
            carriers: Complex carrier values (length = active_carriers)
            
        Returns:
            Time-domain samples (length = fft_size)
        """
        if len(carriers) != self.active_carriers:
            raise ValueError(f"Expected {self.active_carriers} carriers, "
                           f"got {len(carriers)}")
        
        # Place carriers in FFT bins
        spectrum = np.zeros(self.fft_size, dtype=np.complex64)
        spectrum[self._carrier_to_bin] = carriers
        
        # IFFT to time domain
        if self.fast:
            # Note: numpy IFFT is normalized by 1/N, we want sqrt(N) for power
            samples = np.fft.ifft(spectrum) * np.sqrt(self.fft_size)
        else:
            samples = self._idft_slow(spectrum)
        
        return samples.astype(np.complex64)
    
    def _idft_slow(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Educational O(N²) IDFT implementation.
        
        IDFT: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(j*2*pi*k*n/N)
        
        Args:
            spectrum: Frequency domain samples
            
        Returns:
            Time domain samples
        """
        N = len(spectrum)
        samples = np.zeros(N, dtype=np.complex128)
        
        # Twiddle factors
        W = np.exp(2j * np.pi / N)
        
        for n in range(N):
            for k in range(N):
                samples[n] += spectrum[k] * (W ** (k * n))
        
        # Normalize for unit power
        return samples * np.sqrt(N) / N
    
    def modulate_symbol(self, carriers: np.ndarray) -> np.ndarray:
        """Alias for modulate()."""
        return self.modulate(carriers)


class OFDMDemodulator:
    """
    DVB-T OFDM demodulator.
    
    Converts time-domain samples back to frequency-domain carriers
    using FFT.
    
    Example:
        >>> demod = OFDMDemodulator('2K')
        >>> carriers = demod.demodulate(received_samples)
    """
    
    def __init__(self, mode: str = '2K', fast: bool = True):
        """
        Initialize OFDM demodulator.
        
        Args:
            mode: '2K' or '8K'
            fast: Use numpy FFT (True) or educational DFT (False)
        """
        if mode not in OFDMModulator.MODES:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.mode = mode
        self.fft_size = OFDMModulator.MODES[mode]['fft_size']
        self.active_carriers = OFDMModulator.MODES[mode]['active_carriers']
        self.fast = fast
        
        # Use same carrier mapping as modulator
        mod = OFDMModulator(mode)
        self._carrier_to_bin = mod._carrier_to_bin
    
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate time-domain samples to carriers.
        
        Args:
            samples: Time-domain samples (length = fft_size)
            
        Returns:
            Complex carrier values (length = active_carriers)
        """
        if len(samples) != self.fft_size:
            raise ValueError(f"Expected {self.fft_size} samples, "
                           f"got {len(samples)}")
        
        # FFT to frequency domain
        if self.fast:
            spectrum = np.fft.fft(samples) / np.sqrt(self.fft_size)
        else:
            spectrum = self._dft_slow(samples)
        
        # Extract active carriers
        carriers = spectrum[self._carrier_to_bin]
        
        return carriers.astype(np.complex64)
    
    def _dft_slow(self, samples: np.ndarray) -> np.ndarray:
        """
        Educational O(N²) DFT implementation.
        
        DFT: X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
        """
        N = len(samples)
        spectrum = np.zeros(N, dtype=np.complex128)
        
        W = np.exp(-2j * np.pi / N)
        
        for k in range(N):
            for n in range(N):
                spectrum[k] += samples[n] * (W ** (k * n))
        
        return spectrum / np.sqrt(N)
    
    def demodulate_symbol(self, samples: np.ndarray) -> np.ndarray:
        """Alias for demodulate()."""
        return self.demodulate(samples)


def get_carrier_frequencies(mode: str, bandwidth: str = '8MHz') -> np.ndarray:
    """
    Get carrier frequencies relative to center frequency.
    
    Args:
        mode: '2K' or '8K'
        bandwidth: '6MHz', '7MHz', or '8MHz'
        
    Returns:
        Array of carrier frequency offsets in Hz
    """
    params = OFDMModulator.MODES.get(mode)
    if not params:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Carrier spacing
    if bandwidth == '8MHz':
        carrier_spacing = 4464.2857  # Hz for 8MHz bandwidth
    elif bandwidth == '7MHz':
        carrier_spacing = 3906.25
    else:  # 6MHz
        carrier_spacing = 3348.2143
    
    K = params['active_carriers']
    center = (K - 1) / 2
    
    return (np.arange(K) - center) * carrier_spacing


def calculate_symbol_duration(mode: str, guard_interval: str,
                             bandwidth: str = '8MHz') -> Tuple[float, float]:
    """
    Calculate OFDM symbol timing.
    
    Args:
        mode: '2K' or '8K'
        guard_interval: '1/4', '1/8', '1/16', '1/32'
        bandwidth: Channel bandwidth
        
    Returns:
        Tuple of (useful_duration, guard_duration) in seconds
    """
    params = OFDMModulator.MODES.get(mode)
    if not params:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Sample rate
    sample_rates = {
        '8MHz': 9142857.142857143,
        '7MHz': 8000000.0,
        '6MHz': 6857142.857142857,
    }
    sample_rate = sample_rates.get(bandwidth, sample_rates['8MHz'])
    
    # Useful symbol duration
    fft_size = params['fft_size']
    useful_duration = fft_size / sample_rate
    
    # Guard interval duration
    guard_ratios = {'1/4': 4, '1/8': 8, '1/16': 16, '1/32': 32}
    guard_ratio = guard_ratios.get(guard_interval, 4)
    guard_duration = useful_duration / guard_ratio
    
    return useful_duration, guard_duration
