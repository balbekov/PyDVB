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


# =============================================================================
# Room Channel Model - Real acoustic channel measurement
# =============================================================================

@dataclass
class RoomChannelResponse:
    """Measured room channel impulse response characteristics."""
    impulse_response: np.ndarray  # Complex impulse response
    sample_rate: int              # Sample rate in Hz
    delay_samples: int            # Propagation delay in samples
    delay_ms: float               # Propagation delay in ms
    delay_spread_samples: int     # Delay spread (multipath duration)
    delay_spread_ms: float        # Delay spread in ms
    snr_db: float                 # Estimated SNR from measurement
    peak_amplitude: float         # Peak of impulse response
    noise_floor: float            # Estimated noise floor


def generate_probe_signal(
    duration: float = 2.0,
    sample_rate: int = 48000,
    f_start: float = 200.0,
    f_end: float = 16000.0,
    amplitude: float = 0.8,
    silence_padding: float = 0.5
) -> Tuple[np.ndarray, dict]:
    """
    Generate a linear chirp probe signal for room impulse response measurement.
    
    The chirp sweeps from f_start to f_end over the duration. This provides
    excellent correlation properties for extracting the impulse response.
    
    Args:
        duration: Chirp duration in seconds
        sample_rate: Sample rate in Hz
        f_start: Start frequency in Hz
        f_end: End frequency in Hz
        amplitude: Signal amplitude (0.0 to 1.0)
        silence_padding: Silence before/after chirp in seconds
        
    Returns:
        Tuple of (audio signal, metadata dict)
    """
    from scipy import signal as scipy_signal
    
    # Generate time vector for chirp
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    
    # Generate linear chirp
    chirp = scipy_signal.chirp(t, f_start, duration, f_end, method='linear')
    
    # Apply amplitude and convert to float32
    chirp = (chirp * amplitude).astype(np.float32)
    
    # Add silence padding
    padding_samples = int(silence_padding * sample_rate)
    silence = np.zeros(padding_samples, dtype=np.float32)
    
    # Combine: silence + chirp + silence
    probe = np.concatenate([silence, chirp, silence])
    
    metadata = {
        'sample_rate': sample_rate,
        'duration': duration,
        'f_start': f_start,
        'f_end': f_end,
        'chirp_samples': num_samples,
        'total_samples': len(probe),
        'padding_samples': padding_samples,
    }
    
    return probe, metadata


def save_probe_wav(probe: np.ndarray, path: str, sample_rate: int = 48000) -> None:
    """
    Save probe signal to WAV file.
    
    Args:
        probe: Probe signal array
        path: Output WAV file path
        sample_rate: Sample rate in Hz
    """
    import wave
    
    # Normalize and convert to 16-bit PCM
    probe = np.clip(probe, -1.0, 1.0)
    audio_int16 = (probe * 32767).astype(np.int16)
    
    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    Load WAV file as float32 array.
    
    Args:
        path: WAV file path
        
    Returns:
        Tuple of (audio array, sample rate)
    """
    import wave
    
    with wave.open(path, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        
        raw_data = wav.readframes(n_frames)
        
        if sample_width == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32767.0
        elif sample_width == 4:
            audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 127.0 - 1.0
        
        # Take first channel if stereo
        if n_channels > 1:
            audio = audio[::n_channels]
    
    return audio, sample_rate


def extract_impulse_response(
    probe: np.ndarray,
    recorded: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0
) -> RoomChannelResponse:
    """
    Extract channel impulse response via cross-correlation.
    
    Cross-correlates the recorded signal with the original probe to find
    the impulse response, which captures the channel's multipath profile.
    
    Args:
        probe: Original probe signal
        recorded: Recorded signal (through speaker/mic)
        sample_rate: Sample rate in Hz
        threshold_db: Threshold below peak for delay spread calculation
        
    Returns:
        RoomChannelResponse with impulse response and statistics
    """
    from scipy import signal as scipy_signal
    
    # Normalize signals
    probe_norm = probe / (np.max(np.abs(probe)) + 1e-10)
    recorded_norm = recorded / (np.max(np.abs(recorded)) + 1e-10)
    
    # Cross-correlation to extract impulse response
    # Use 'full' mode and take the relevant portion
    correlation = scipy_signal.correlate(recorded_norm, probe_norm, mode='full')
    
    # The correlation peak indicates the delay
    # For mode='full', the zero-lag index is at len(probe) - 1
    zero_lag_idx = len(probe) - 1
    
    # Find the peak (should be at the propagation delay)
    peak_idx = np.argmax(np.abs(correlation))
    delay_samples = peak_idx - zero_lag_idx
    
    # Normalize correlation by probe energy for proper impulse response scaling
    probe_energy = np.sum(probe_norm ** 2)
    impulse_response = correlation / (probe_energy + 1e-10)
    
    # Extract the causal part of the impulse response (from delay onwards)
    # Include some samples before peak to catch any pre-cursor
    margin = int(0.01 * sample_rate)  # 10ms margin
    start_idx = max(0, peak_idx - margin)
    
    # Determine impulse response length based on delay spread
    peak_amplitude = np.abs(impulse_response[peak_idx])
    threshold_linear = peak_amplitude * (10 ** (threshold_db / 20))
    
    # Find where impulse response falls below threshold
    post_peak = np.abs(impulse_response[peak_idx:])
    below_threshold = np.where(post_peak < threshold_linear)[0]
    
    if len(below_threshold) > 0:
        # Find first sustained drop below threshold
        delay_spread_samples = below_threshold[0]
        for i in range(len(below_threshold) - 10):
            if all(post_peak[below_threshold[i]:below_threshold[i]+10] < threshold_linear):
                delay_spread_samples = below_threshold[i]
                break
    else:
        delay_spread_samples = len(post_peak)
    
    # Limit impulse response length (max 500ms for room acoustics)
    max_ir_samples = int(0.5 * sample_rate)
    end_idx = min(peak_idx + delay_spread_samples + margin, 
                  peak_idx + max_ir_samples,
                  len(impulse_response))
    
    # Extract the impulse response segment
    ir_segment = impulse_response[start_idx:end_idx].astype(np.complex64)
    
    # Estimate noise floor from tail of correlation
    tail_start = min(peak_idx + max_ir_samples, len(correlation) - 1000)
    if tail_start < len(correlation) - 100:
        noise_floor = float(np.std(np.abs(correlation[tail_start:])))
    else:
        noise_floor = float(np.min(np.abs(post_peak[:delay_spread_samples])))
    
    # Estimate SNR
    snr_db = 20 * np.log10(peak_amplitude / (noise_floor + 1e-10))
    
    return RoomChannelResponse(
        impulse_response=ir_segment,
        sample_rate=sample_rate,
        delay_samples=max(0, delay_samples),
        delay_ms=max(0, delay_samples) / sample_rate * 1000,
        delay_spread_samples=delay_spread_samples,
        delay_spread_ms=delay_spread_samples / sample_rate * 1000,
        snr_db=float(snr_db),
        peak_amplitude=float(peak_amplitude),
        noise_floor=float(noise_floor),
    )


class RoomChannelModel:
    """
    Real room acoustic channel model from measurement.
    
    This channel model applies a measured impulse response to simulate
    the acoustic path through a real room (speaker -> air -> microphone).
    
    The model captures:
    - Multipath reflections (room echoes)
    - Speaker/microphone frequency response
    - Propagation delay
    - Ambient noise characteristics
    
    Example:
        >>> # Calibrate from recorded probe signal
        >>> model = RoomChannelModel.calibrate('probe.wav', 'recorded.wav')
        >>> model.save('room_channel.npz')
        >>> 
        >>> # Load and use
        >>> model = RoomChannelModel.load('room_channel.npz')
        >>> impaired = model.apply(iq_samples)
    """
    
    def __init__(self, 
                 impulse_response: np.ndarray,
                 sample_rate: int,
                 snr_db: float = 30.0,
                 delay_samples: int = 0,
                 delay_spread_ms: float = 0.0,
                 add_noise: bool = True):
        """
        Initialize room channel model.
        
        Args:
            impulse_response: Complex impulse response array
            sample_rate: Sample rate in Hz
            snr_db: Measured/target SNR in dB
            delay_samples: Propagation delay in samples
            delay_spread_ms: Delay spread in ms (for info)
            add_noise: Whether to add AWGN based on measured SNR
        """
        self.impulse_response = np.asarray(impulse_response, dtype=np.complex64)
        self.sample_rate = sample_rate
        self.snr_db = snr_db
        self.delay_samples = delay_samples
        self.delay_spread_ms = delay_spread_ms
        self.add_noise = add_noise
        
        # Normalize impulse response to unity gain at DC
        ir_sum = np.sum(np.abs(self.impulse_response))
        if ir_sum > 1e-10:
            self.impulse_response = self.impulse_response / ir_sum
    
    def apply(self, samples: np.ndarray, 
              snr_override: Optional[float] = None) -> np.ndarray:
        """
        Apply room channel model to I/Q samples.
        
        Convolves the input with the measured impulse response and
        optionally adds AWGN at the measured SNR level.
        
        Args:
            samples: Input complex I/Q samples
            snr_override: Override SNR (None = use measured SNR)
            
        Returns:
            Channel-impaired samples
        """
        from scipy import signal as scipy_signal
        
        # Resample impulse response if sample rates differ
        ir = self._resample_ir_if_needed(samples)
        
        # Convolve with impulse response (this applies multipath + freq response)
        output = scipy_signal.convolve(samples, ir, mode='full')
        
        # Truncate to original length
        output = output[:len(samples)]
        
        # Add AWGN if enabled
        if self.add_noise:
            snr = snr_override if snr_override is not None else self.snr_db
            output = ChannelModel.add_awgn(output, snr)
        
        return output.astype(np.complex64)
    
    def _resample_ir_if_needed(self, samples: np.ndarray) -> np.ndarray:
        """
        Resample impulse response if input sample rate differs.
        
        For now, we assume sample rates match. In practice, the IR
        should be resampled to match the I/Q sample rate.
        """
        # TODO: Add resampling support if needed
        return self.impulse_response
    
    def get_frequency_response(self, n_points: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the channel frequency response.
        
        Args:
            n_points: Number of frequency points
            
        Returns:
            Tuple of (frequencies in Hz, magnitude response in dB)
        """
        # Zero-pad impulse response for better frequency resolution
        ir_padded = np.zeros(n_points, dtype=np.complex64)
        ir_len = min(len(self.impulse_response), n_points)
        ir_padded[:ir_len] = self.impulse_response[:ir_len]
        
        # FFT to get frequency response
        freq_response = np.fft.fft(ir_padded)
        freqs = np.fft.fftfreq(n_points, 1.0 / self.sample_rate)
        
        # Take positive frequencies
        positive = freqs >= 0
        freqs = freqs[positive]
        magnitude_db = 20 * np.log10(np.abs(freq_response[positive]) + 1e-10)
        
        return freqs, magnitude_db
    
    def save(self, path: str) -> None:
        """
        Save channel model to NPZ file.
        
        Args:
            path: Output file path
        """
        np.savez(
            path,
            impulse_response=self.impulse_response,
            sample_rate=self.sample_rate,
            snr_db=self.snr_db,
            delay_samples=self.delay_samples,
            delay_spread_ms=self.delay_spread_ms,
        )
    
    @classmethod
    def load(cls, path: str) -> 'RoomChannelModel':
        """
        Load channel model from NPZ file.
        
        Args:
            path: Input file path
            
        Returns:
            RoomChannelModel instance
        """
        data = np.load(path)
        return cls(
            impulse_response=data['impulse_response'],
            sample_rate=int(data['sample_rate']),
            snr_db=float(data['snr_db']),
            delay_samples=int(data['delay_samples']),
            delay_spread_ms=float(data['delay_spread_ms']),
        )
    
    @classmethod
    def calibrate(cls, 
                  probe_wav: str, 
                  recorded_wav: str,
                  verbose: bool = True) -> 'RoomChannelModel':
        """
        Create channel model from probe/recording pair.
        
        This is the main calibration method. Play probe.wav through your
        speaker, record with your microphone, then call this method to
        extract the room channel model.
        
        Args:
            probe_wav: Path to original probe signal WAV
            recorded_wav: Path to recorded signal WAV
            verbose: Print calibration statistics
            
        Returns:
            Calibrated RoomChannelModel
        """
        # Load probe and recording
        probe, probe_rate = load_wav(probe_wav)
        recorded, rec_rate = load_wav(recorded_wav)
        
        if probe_rate != rec_rate:
            raise ValueError(f"Sample rate mismatch: probe={probe_rate}, recorded={rec_rate}")
        
        # Extract impulse response
        response = extract_impulse_response(probe, recorded, probe_rate)
        
        if verbose:
            print(f"Room Channel Calibration Results:")
            print(f"  Propagation delay: {response.delay_ms:.1f} ms ({response.delay_samples} samples)")
            print(f"  Delay spread: {response.delay_spread_ms:.1f} ms ({response.delay_spread_samples} samples)")
            print(f"  Estimated SNR: {response.snr_db:.1f} dB")
            print(f"  Peak amplitude: {response.peak_amplitude:.4f}")
            print(f"  Noise floor: {response.noise_floor:.6f}")
            print(f"  Impulse response length: {len(response.impulse_response)} samples")
        
        return cls(
            impulse_response=response.impulse_response,
            sample_rate=response.sample_rate,
            snr_db=response.snr_db,
            delay_samples=response.delay_samples,
            delay_spread_ms=response.delay_spread_ms,
        )
    
    def __repr__(self) -> str:
        return (f"RoomChannelModel(sample_rate={self.sample_rate}, "
                f"snr_db={self.snr_db:.1f}, delay_ms={self.delay_samples/self.sample_rate*1000:.1f}, "
                f"delay_spread_ms={self.delay_spread_ms:.1f})")
