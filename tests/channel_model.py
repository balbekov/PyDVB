"""
DVB-T Channel Models

Simulates various channel impairments for receiver testing:
- AWGN (Additive White Gaussian Noise)
- CFO (Carrier Frequency Offset)
- SFO (Sample Frequency Offset)
- Multipath (echoes/reflections)
- Phase noise

Reference: Various DVB-T testing specifications
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Channel model configuration."""
    snr_db: float = 30.0          # Signal-to-noise ratio
    cfo_hz: float = 0.0           # Carrier frequency offset
    sfo_ppm: float = 0.0          # Sample frequency offset in ppm
    phase_noise_deg: float = 0.0  # Phase noise std deviation
    multipath: bool = False       # Enable multipath
    delays_samples: List[int] = None  # Echo delays
    gains_db: List[float] = None      # Echo gains


class ChannelModel:
    """
    DVB-T channel model for testing receivers.
    
    Simulates various real-world channel impairments.
    
    Example:
        >>> channel = ChannelModel()
        >>> impaired = channel.add_awgn(samples, snr_db=20)
        >>> 
        >>> # Full channel simulation
        >>> config = ChannelConfig(snr_db=25, cfo_hz=100)
        >>> impaired = channel.apply(samples, config)
    """
    
    def __init__(self, sample_rate: float = 9142857.142857143):
        """
        Initialize channel model.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    @staticmethod
    def add_awgn(samples: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add Additive White Gaussian Noise.
        
        Args:
            samples: Input I/Q samples
            snr_db: Target SNR in dB
            
        Returns:
            Noisy samples
        """
        signal_power = np.mean(np.abs(samples) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(samples)) + 
            1j * np.random.randn(len(samples))
        )
        
        return (samples + noise).astype(np.complex64)
    
    def add_cfo(self, samples: np.ndarray, cfo_hz: float) -> np.ndarray:
        """
        Add carrier frequency offset.
        
        Rotates signal phase continuously over time.
        
        Args:
            samples: Input I/Q samples
            cfo_hz: CFO in Hz
            
        Returns:
            CFO-impaired samples
        """
        t = np.arange(len(samples)) / self.sample_rate
        rotation = np.exp(2j * np.pi * cfo_hz * t)
        return (samples * rotation).astype(np.complex64)
    
    def add_sfo(self, samples: np.ndarray, sfo_ppm: float) -> np.ndarray:
        """
        Add sample frequency offset (resampling).
        
        Args:
            samples: Input I/Q samples
            sfo_ppm: Sample frequency offset in parts per million
            
        Returns:
            SFO-impaired samples
        """
        if abs(sfo_ppm) < 0.001:
            return samples
        
        # Resample ratio
        ratio = 1.0 + sfo_ppm / 1e6
        new_length = int(len(samples) * ratio)
        
        # Linear interpolation resampling
        old_indices = np.arange(len(samples))
        new_indices = np.linspace(0, len(samples) - 1, new_length)
        
        resampled = np.interp(new_indices, old_indices, samples.real) + \
                   1j * np.interp(new_indices, old_indices, samples.imag)
        
        return resampled.astype(np.complex64)
    
    @staticmethod
    def add_multipath(samples: np.ndarray,
                      delays_samples: List[int],
                      gains: List[float]) -> np.ndarray:
        """
        Add multipath propagation (echoes).
        
        Args:
            samples: Input I/Q samples
            delays_samples: Delay for each path in samples
            gains: Complex gain for each path (can include phase)
            
        Returns:
            Multipath-impaired samples
        """
        if not delays_samples:
            return samples
        
        # Convert dB gains to linear if needed
        gains = np.array(gains, dtype=np.complex64)
        
        max_delay = max(delays_samples)
        output = np.zeros(len(samples) + max_delay, dtype=np.complex64)
        
        for delay, gain in zip(delays_samples, gains):
            output[delay:delay + len(samples)] += gain * samples
        
        return output[:len(samples)]
    
    @staticmethod
    def add_phase_noise(samples: np.ndarray, 
                        std_deg: float) -> np.ndarray:
        """
        Add phase noise.
        
        Args:
            samples: Input I/Q samples
            std_deg: Standard deviation of phase noise in degrees
            
        Returns:
            Phase-noisy samples
        """
        if std_deg <= 0:
            return samples
        
        std_rad = np.deg2rad(std_deg)
        phase_noise = np.random.randn(len(samples)) * std_rad
        
        return (samples * np.exp(1j * phase_noise)).astype(np.complex64)
    
    @staticmethod
    def add_dc_offset(samples: np.ndarray, 
                      offset_i: float = 0.0,
                      offset_q: float = 0.0) -> np.ndarray:
        """
        Add DC offset.
        
        Args:
            samples: Input I/Q samples
            offset_i: I channel DC offset
            offset_q: Q channel DC offset
            
        Returns:
            DC-offset samples
        """
        return (samples + complex(offset_i, offset_q)).astype(np.complex64)
    
    @staticmethod
    def add_iq_imbalance(samples: np.ndarray,
                         amplitude_db: float = 0.0,
                         phase_deg: float = 0.0) -> np.ndarray:
        """
        Add I/Q imbalance.
        
        Args:
            samples: Input I/Q samples
            amplitude_db: Amplitude imbalance in dB
            phase_deg: Phase imbalance in degrees
            
        Returns:
            I/Q imbalanced samples
        """
        if amplitude_db == 0 and phase_deg == 0:
            return samples
        
        # Split into I and Q
        I = samples.real
        Q = samples.imag
        
        # Apply amplitude imbalance to Q
        amp_ratio = 10 ** (amplitude_db / 20)
        Q = Q * amp_ratio
        
        # Apply phase imbalance (rotation of Q)
        phase_rad = np.deg2rad(phase_deg)
        Q_new = Q * np.cos(phase_rad) - I * np.sin(phase_rad)
        
        return (I + 1j * Q_new).astype(np.complex64)
    
    def apply(self, samples: np.ndarray, 
              config: ChannelConfig) -> np.ndarray:
        """
        Apply full channel model.
        
        Args:
            samples: Input I/Q samples
            config: Channel configuration
            
        Returns:
            Impaired samples
        """
        output = samples.copy()
        
        # Apply impairments in realistic order
        
        # 1. Multipath (channel)
        if config.multipath and config.delays_samples:
            gains = [10 ** (g / 20) for g in (config.gains_db or [0.0])]
            output = self.add_multipath(output, config.delays_samples, gains)
        
        # 2. CFO (oscillator offset)
        if config.cfo_hz != 0:
            output = self.add_cfo(output, config.cfo_hz)
        
        # 3. SFO (clock offset)
        if config.sfo_ppm != 0:
            output = self.add_sfo(output, config.sfo_ppm)
        
        # 4. Phase noise
        if config.phase_noise_deg > 0:
            output = self.add_phase_noise(output, config.phase_noise_deg)
        
        # 5. AWGN (always last - thermal noise)
        output = self.add_awgn(output, config.snr_db)
        
        return output
    
    @staticmethod
    def ideal_channel(samples: np.ndarray) -> np.ndarray:
        """Return samples unchanged (ideal channel)."""
        return samples.copy()


# Standard channel profiles

def awgn_channel(snr_db: float) -> ChannelConfig:
    """Create AWGN-only channel config."""
    return ChannelConfig(snr_db=snr_db)


def rayleigh_channel(snr_db: float, 
                     num_taps: int = 4,
                     max_delay: int = 100) -> ChannelConfig:
    """
    Create Rayleigh fading channel config.
    
    Args:
        snr_db: SNR in dB
        num_taps: Number of multipath taps
        max_delay: Maximum delay in samples
        
    Returns:
        Channel configuration
    """
    delays = sorted(np.random.randint(0, max_delay, num_taps).tolist())
    delays[0] = 0  # Direct path
    
    # Exponential decay profile
    gains = [-3 * i for i in range(num_taps)]
    
    return ChannelConfig(
        snr_db=snr_db,
        multipath=True,
        delays_samples=delays,
        gains_db=gains
    )


def typical_urban_channel(snr_db: float) -> ChannelConfig:
    """
    Typical Urban (TU) channel model.
    
    Common DVB-T test profile with multipath.
    """
    # TU channel taps (approximate, scaled for DVB-T sample rate)
    delays = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30]
    gains_db = [0, -3, -6, -9, -12, -15, -18, -21, -24, -27]
    
    return ChannelConfig(
        snr_db=snr_db,
        multipath=True,
        delays_samples=delays,
        gains_db=gains_db
    )


def single_echo_channel(snr_db: float, 
                        delay_samples: int,
                        echo_gain_db: float = -6) -> ChannelConfig:
    """
    Single echo channel (simple multipath).
    
    Args:
        snr_db: SNR in dB
        delay_samples: Echo delay in samples
        echo_gain_db: Echo gain in dB (negative = attenuated)
    """
    return ChannelConfig(
        snr_db=snr_db,
        multipath=True,
        delays_samples=[0, delay_samples],
        gains_db=[0, echo_gain_db]
    )


def cfo_channel(snr_db: float, cfo_hz: float) -> ChannelConfig:
    """Create channel with CFO."""
    return ChannelConfig(snr_db=snr_db, cfo_hz=cfo_hz)


def measure_evm(transmitted: np.ndarray, 
                received: np.ndarray) -> float:
    """
    Measure Error Vector Magnitude.
    
    Args:
        transmitted: Reference symbols
        received: Received symbols (after equalization)
        
    Returns:
        EVM in percent
    """
    error = received - transmitted
    error_power = np.mean(np.abs(error) ** 2)
    ref_power = np.mean(np.abs(transmitted) ** 2)
    
    if ref_power < 1e-10:
        return 0.0
    
    return float(np.sqrt(error_power / ref_power) * 100)


def measure_ber(tx_bits: np.ndarray, 
                rx_bits: np.ndarray) -> float:
    """
    Measure Bit Error Rate.
    
    Args:
        tx_bits: Transmitted bits
        rx_bits: Received bits
        
    Returns:
        BER (0 to 1)
    """
    min_len = min(len(tx_bits), len(rx_bits))
    if min_len == 0:
        return 0.0
    
    errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    return float(errors / min_len)


def measure_per(tx_packets: List[bytes], 
                rx_packets: List[bytes]) -> float:
    """
    Measure Packet Error Rate.
    
    Args:
        tx_packets: List of transmitted packets
        rx_packets: List of received packets
        
    Returns:
        PER (0 to 1)
    """
    if len(tx_packets) == 0:
        return 0.0
    
    min_pkts = min(len(tx_packets), len(rx_packets))
    errors = sum(1 for i in range(min_pkts) if tx_packets[i] != rx_packets[i])
    
    return float(errors / len(tx_packets))
