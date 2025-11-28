"""
DVB-T Channel Estimation and Equalization

Estimates the channel frequency response using pilot carriers and
equalizes data carriers to remove channel distortion.

DVB-T uses scattered and continual pilots for channel estimation:
- Scattered pilots: Rotating pattern every 12 carriers
- Continual pilots: Fixed positions for tracking

Channel estimation process:
1. Extract pilot carriers and compare to known values
2. Interpolate channel response to all data carrier positions
3. Apply equalization (ZF or MMSE)

Reference: ETSI EN 300 744
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum


class EqualizationMethod(Enum):
    """Equalization methods."""
    ZF = "zf"      # Zero-Forcing
    MMSE = "mmse"  # Minimum Mean Square Error


class ChannelEstimator:
    """
    DVB-T channel estimator using pilot carriers.
    
    Estimates the channel frequency response from scattered and continual
    pilot carriers, then interpolates to all carrier positions.
    
    Attributes:
        mode: '2K' or '8K'
        interpolation: 'linear', 'cubic', or 'dft'
        
    Example:
        >>> estimator = ChannelEstimator('2K')
        >>> channel = estimator.estimate(carriers, symbol_idx)
        >>> equalized = estimator.equalize(carriers, channel)
    """
    
    def __init__(self, mode: str = '2K', interpolation: str = 'linear'):
        """
        Initialize channel estimator.
        
        Args:
            mode: '2K', '8K', or 'audio'
            interpolation: Interpolation method ('linear', 'cubic', 'dft')
        """
        self.mode = mode
        self.interpolation = interpolation
        
        # Mode parameters
        self.fft_size = {'2K': 2048, '8K': 8192, 'audio': 64}[mode]
        self.active_carriers = {'2K': 1705, '8K': 6817, 'audio': 24}[mode]
        
        # Initialize pilot information
        self._init_pilots()
        
        # Channel state for time-domain filtering
        self._prev_channel = None
        self._time_filter_alpha = 0.3  # IIR filter coefficient
    
    def _init_pilots(self) -> None:
        """Initialize pilot positions and expected values."""
        from .Pilots import PilotGenerator, ContinualPilots, ScatteredPilots
        
        self.pilot_gen = PilotGenerator(self.mode)
        self.continual = ContinualPilots(self.mode)
        self.scattered = ScatteredPilots(self.mode)
        
        # Pre-compute continual pilot info
        self._continual_pos = self.continual.get_positions()
        self._continual_expected = self.pilot_gen.get_pilot_values(
            self._continual_pos, boost=True
        )
    
    def _get_pilot_channel(self, carriers: np.ndarray,
                           symbol_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get channel response at pilot positions.
        
        Args:
            carriers: Received carrier values
            symbol_index: Symbol index in frame
            
        Returns:
            Tuple of (pilot_positions, channel_at_pilots)
        """
        # Get scattered pilots for this symbol
        scattered_pos = self.scattered.get_positions(symbol_index)
        scattered_pos = scattered_pos[scattered_pos < len(carriers)]
        
        # Combine with continual pilots
        continual_pos = self._continual_pos[self._continual_pos < len(carriers)]
        
        # All pilot positions
        all_pos = np.union1d(continual_pos, scattered_pos)
        
        if len(all_pos) == 0:
            return np.array([]), np.array([])
        
        # Expected pilot values
        expected = self.pilot_gen.get_pilot_values(all_pos, boost=True)
        
        # Received pilot values
        received = carriers[all_pos]
        
        # Channel at pilots: H = received / expected
        channel_at_pilots = received / (expected + 1e-10)
        
        return all_pos, channel_at_pilots
    
    def estimate(self, carriers: np.ndarray, 
                 symbol_index: int) -> np.ndarray:
        """
        Estimate channel frequency response.
        
        Args:
            carriers: Received carrier values
            symbol_index: Symbol index in frame
            
        Returns:
            Complex channel response for all carriers
        """
        pilot_pos, pilot_channel = self._get_pilot_channel(carriers, symbol_index)
        
        if len(pilot_pos) < 2:
            # Not enough pilots, return unity channel
            return np.ones(len(carriers), dtype=np.complex64)
        
        # Interpolate to all carriers
        if self.interpolation == 'linear':
            channel = self._interpolate_linear(pilot_pos, pilot_channel, 
                                               len(carriers))
        elif self.interpolation == 'cubic':
            channel = self._interpolate_cubic(pilot_pos, pilot_channel,
                                              len(carriers))
        elif self.interpolation == 'dft':
            channel = self._interpolate_dft(pilot_pos, pilot_channel,
                                            len(carriers))
        else:
            channel = self._interpolate_linear(pilot_pos, pilot_channel,
                                               len(carriers))
        
        # Optional: time-domain filtering
        if self._prev_channel is not None:
            channel = (self._time_filter_alpha * channel + 
                      (1 - self._time_filter_alpha) * self._prev_channel)
        
        self._prev_channel = channel.copy()
        
        return channel
    
    def _interpolate_linear(self, positions: np.ndarray, 
                            values: np.ndarray,
                            num_carriers: int) -> np.ndarray:
        """
        Linear interpolation of channel response.
        
        Args:
            positions: Pilot carrier positions
            values: Channel values at pilot positions
            num_carriers: Total number of carriers
            
        Returns:
            Interpolated channel response
        """
        all_positions = np.arange(num_carriers)
        
        # Separate magnitude and phase for better interpolation
        mag = np.abs(values)
        phase = np.unwrap(np.angle(values))
        
        # Interpolate magnitude and phase separately
        interp_mag = np.interp(all_positions, positions, mag)
        interp_phase = np.interp(all_positions, positions, phase)
        
        return (interp_mag * np.exp(1j * interp_phase)).astype(np.complex64)
    
    def _interpolate_cubic(self, positions: np.ndarray,
                           values: np.ndarray,
                           num_carriers: int) -> np.ndarray:
        """
        Cubic spline interpolation of channel response.
        
        Args:
            positions: Pilot carrier positions
            values: Channel values at pilot positions
            num_carriers: Total number of carriers
            
        Returns:
            Interpolated channel response
        """
        try:
            from scipy.interpolate import CubicSpline
            
            all_positions = np.arange(num_carriers)
            
            # Separate real and imaginary
            cs_real = CubicSpline(positions, values.real)
            cs_imag = CubicSpline(positions, values.imag)
            
            interp_real = cs_real(all_positions)
            interp_imag = cs_imag(all_positions)
            
            return (interp_real + 1j * interp_imag).astype(np.complex64)
        except ImportError:
            # Fall back to linear if scipy not available
            return self._interpolate_linear(positions, values, num_carriers)
    
    def _interpolate_dft(self, positions: np.ndarray,
                         values: np.ndarray,
                         num_carriers: int) -> np.ndarray:
        """
        DFT-based interpolation (frequency-domain to time-domain and back).
        
        This method can provide better performance for channels with
        limited delay spread.
        
        Args:
            positions: Pilot carrier positions
            values: Channel values at pilot positions
            num_carriers: Total number of carriers
            
        Returns:
            Interpolated channel response
        """
        # Place pilots in full spectrum
        full_spectrum = np.zeros(num_carriers, dtype=np.complex64)
        full_spectrum[positions] = values
        
        # Create a mask for known positions
        mask = np.zeros(num_carriers)
        mask[positions] = 1
        
        # IFFT to time domain
        time_domain = np.fft.ifft(full_spectrum)
        mask_time = np.fft.ifft(mask)
        
        # Window in time domain (assume limited delay spread)
        window_len = min(num_carriers // 4, len(positions) * 2)
        window = np.zeros(num_carriers)
        window[:window_len // 2] = 1
        window[-window_len // 2:] = 1
        
        time_domain *= window
        
        # FFT back to frequency domain
        channel = np.fft.fft(time_domain)
        
        return channel.astype(np.complex64)
    
    def reset(self) -> None:
        """Reset estimator state."""
        self._prev_channel = None


class Equalizer:
    """
    DVB-T channel equalizer.
    
    Removes channel distortion from received carriers using
    estimated channel response.
    
    Attributes:
        method: Equalization method (ZF or MMSE)
        
    Example:
        >>> eq = Equalizer(method='zf')
        >>> equalized = eq.equalize(carriers, channel_estimate)
    """
    
    def __init__(self, method: str = 'zf', noise_variance: float = 0.1):
        """
        Initialize equalizer.
        
        Args:
            method: 'zf' (Zero-Forcing) or 'mmse' (Minimum Mean Square Error)
            noise_variance: Noise variance for MMSE (default 0.1)
        """
        self.method = EqualizationMethod(method.lower())
        self.noise_variance = noise_variance
    
    def equalize(self, carriers: np.ndarray, 
                 channel: np.ndarray,
                 noise_variance: Optional[float] = None) -> np.ndarray:
        """
        Equalize carriers using channel estimate.
        
        Args:
            carriers: Received carrier values
            channel: Channel frequency response estimate
            noise_variance: Optional noise variance override
            
        Returns:
            Equalized carrier values
        """
        if noise_variance is None:
            noise_variance = self.noise_variance
        
        if self.method == EqualizationMethod.ZF:
            return self._equalize_zf(carriers, channel)
        else:
            return self._equalize_mmse(carriers, channel, noise_variance)
    
    def _equalize_zf(self, carriers: np.ndarray,
                     channel: np.ndarray) -> np.ndarray:
        """
        Zero-forcing equalization.
        
        Simply divides by channel: Y/H
        
        Optimal for high SNR, can amplify noise at channel nulls.
        """
        # Avoid division by zero
        safe_channel = np.where(np.abs(channel) > 1e-6, channel, 1e-6)
        return (carriers / safe_channel).astype(np.complex64)
    
    def _equalize_mmse(self, carriers: np.ndarray,
                       channel: np.ndarray,
                       noise_variance: float) -> np.ndarray:
        """
        MMSE (Minimum Mean Square Error) equalization.
        
        W = H* / (|H|^2 + sigma^2)
        
        Better performance at low SNR, avoids noise amplification.
        """
        H = channel
        H_conj = np.conj(H)
        H_mag_sq = np.abs(H) ** 2
        
        # MMSE filter
        W = H_conj / (H_mag_sq + noise_variance)
        
        return (carriers * W).astype(np.complex64)
    
    def get_csi(self, channel: np.ndarray) -> np.ndarray:
        """
        Get Channel State Information (CSI) for soft demapping.
        
        CSI represents reliability of each carrier, used for
        weighted soft decision in Viterbi decoder.
        
        Args:
            channel: Channel frequency response
            
        Returns:
            CSI weights (higher = more reliable)
        """
        return np.abs(channel) ** 2


class ChannelEstimatorWithEqualization:
    """
    Combined channel estimation and equalization.
    
    Provides a simple interface for the complete channel
    compensation pipeline.
    
    Example:
        >>> ch_eq = ChannelEstimatorWithEqualization('2K', method='mmse')
        >>> equalized, csi = ch_eq.process(carriers, symbol_idx)
    """
    
    def __init__(self, mode: str = '2K', 
                 interpolation: str = 'linear',
                 method: str = 'zf',
                 noise_variance: float = 0.1):
        """
        Initialize combined estimator/equalizer.
        
        Args:
            mode: '2K' or '8K'
            interpolation: Channel interpolation method
            method: Equalization method ('zf' or 'mmse')
            noise_variance: Noise variance for MMSE
        """
        self.estimator = ChannelEstimator(mode, interpolation)
        self.equalizer = Equalizer(method, noise_variance)
    
    def process(self, carriers: np.ndarray,
                symbol_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate channel and equalize in one step.
        
        Args:
            carriers: Received carrier values
            symbol_index: Symbol index in frame
            
        Returns:
            Tuple of (equalized_carriers, channel_state_info)
        """
        # Estimate channel
        channel = self.estimator.estimate(carriers, symbol_index)
        
        # Equalize
        equalized = self.equalizer.equalize(carriers, channel)
        
        # Get CSI for soft demapping
        csi = self.equalizer.get_csi(channel)
        
        return equalized, csi
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.estimator.reset()


def estimate_snr_from_pilots(carriers: np.ndarray,
                             channel: np.ndarray,
                             pilot_positions: np.ndarray,
                             expected_pilots: np.ndarray) -> float:
    """
    Estimate SNR from pilot carrier error.
    
    Args:
        carriers: Received carriers
        channel: Estimated channel
        pilot_positions: Pilot carrier indices
        expected_pilots: Expected pilot values
        
    Returns:
        Estimated SNR in dB
    """
    # Equalize pilots
    equalized = carriers[pilot_positions] / (channel[pilot_positions] + 1e-10)
    
    # Error
    error = equalized - expected_pilots
    
    # SNR estimate
    signal_power = np.mean(np.abs(expected_pilots) ** 2)
    noise_power = np.mean(np.abs(error) ** 2)
    
    if noise_power < 1e-10:
        return 40.0  # Very high SNR
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return float(np.clip(snr_db, -10, 40))
