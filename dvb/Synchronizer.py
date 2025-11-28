"""
DVB-T Synchronization

Implements time, frequency, and frame synchronization for DVB-T receiver.

The synchronization process includes:
1. Coarse time sync: Guard interval correlation to find symbol boundaries
2. Coarse frequency sync: CFO estimation from guard interval phase rotation
3. Fine frequency sync: Pilot-based CFO estimation and tracking
4. Frame sync: TPS-based frame/superframe detection

Reference: ETSI EN 300 744
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SyncResult:
    """Synchronization results."""
    symbol_start: int
    coarse_cfo: float  # Hz
    fine_cfo: float    # Hz
    snr_estimate: float  # dB
    confidence: float  # 0-1


class CoarseSync:
    """
    Coarse synchronization using guard interval correlation.
    
    The cyclic prefix (guard interval) is a copy of the end of the symbol.
    Correlating the signal with a delayed version reveals peaks at symbol
    boundaries, and the phase of correlation gives coarse CFO estimate.
    
    Attributes:
        fft_size: FFT size (2048, 8192, or 128 for audio)
        guard_length: Guard interval length in samples
        
    Example:
        >>> sync = CoarseSync(mode='2K', guard_interval='1/4')
        >>> start, cfo = sync.find_symbol_start(samples)
    """
    
    def __init__(self, mode: str = '2K', guard_interval: str = '1/4',
                 sample_rate: float = 9142857.142857143):
        """
        Initialize coarse synchronizer.
        
        Args:
            mode: '2K', '8K', or 'audio'
            guard_interval: '1/4', '1/8', '1/16', '1/32'
            sample_rate: Sample rate in Hz
        """
        self.mode = mode
        self.guard_interval = guard_interval
        self.sample_rate = sample_rate
        
        # FFT parameters
        self.fft_size = {'2K': 2048, '8K': 8192, 'audio': 64}[mode]
        
        # Guard interval length
        gi_fractions = {'1/4': 4, '1/8': 8, '1/16': 16, '1/32': 32, 'acoustic': 0.4}
        divisor = gi_fractions[guard_interval]
        if divisor < 1:
            # Fractional divisor for acoustic mode (guard > fft_size)
            self.guard_length = int(self.fft_size / divisor)
        else:
            self.guard_length = self.fft_size // int(divisor)
        
        # Total symbol length
        self.symbol_length = self.fft_size + self.guard_length
        
        # Carrier spacing
        self.carrier_spacing = sample_rate / self.fft_size
    
    def _guard_correlation(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute guard interval correlation.
        
        Correlates samples with their delayed version, where the delay
        is the FFT size. The guard interval creates correlation peaks.
        
        Args:
            samples: Input I/Q samples
            
        Returns:
            Complex correlation values
        """
        # Correlation: sum of samples[n] * conj(samples[n + FFT_size])
        # over the guard interval length
        
        delay = self.fft_size
        corr_len = len(samples) - delay
        
        if corr_len <= 0:
            return np.array([])
        
        # Vectorized correlation over sliding window
        corr = np.zeros(corr_len, dtype=np.complex64)
        
        for i in range(corr_len):
            window_end = min(i + self.guard_length, corr_len)
            window_len = window_end - i
            
            if window_len > 0:
                segment1 = samples[i:i + window_len]
                segment2 = samples[i + delay:i + delay + window_len]
                corr[i] = np.sum(segment1 * np.conj(segment2))
        
        return corr
    
    def _guard_correlation_fast(self, samples: np.ndarray) -> np.ndarray:
        """
        Fast guard interval correlation using moving sum.
        
        Args:
            samples: Input I/Q samples
            
        Returns:
            Complex correlation values
        """
        delay = self.fft_size
        if len(samples) < delay + self.guard_length:
            return np.array([])
        
        # Point-wise product of delayed samples
        product = samples[:-delay] * np.conj(samples[delay:])
        
        # Moving sum over guard length using cumsum
        cumsum = np.cumsum(product)
        
        # Moving sum: cumsum[i+G] - cumsum[i]
        result = np.zeros(len(cumsum) - self.guard_length + 1, dtype=np.complex64)
        result[0] = cumsum[self.guard_length - 1]
        result[1:] = cumsum[self.guard_length:] - cumsum[:-self.guard_length]
        
        return result
    
    def find_symbol_start(self, samples: np.ndarray, 
                          search_range: Optional[int] = None) -> Tuple[int, float]:
        """
        Find symbol start position using guard interval correlation.
        
        Args:
            samples: Input I/Q samples (at least 2 symbol lengths)
            search_range: Number of samples to search (default: 1 symbol)
            
        Returns:
            Tuple of (symbol_start_index, correlation_peak_value)
        """
        if search_range is None:
            search_range = self.symbol_length
        
        # Compute correlation
        corr = self._guard_correlation_fast(samples[:search_range + self.fft_size + self.guard_length])
        
        if len(corr) == 0:
            return 0, 0.0
        
        # Find peak
        corr_mag = np.abs(corr)
        peak_idx = np.argmax(corr_mag)
        peak_val = corr_mag[peak_idx]
        
        return int(peak_idx), float(peak_val)
    
    def estimate_coarse_cfo(self, samples: np.ndarray,
                            symbol_start: Optional[int] = None) -> float:
        """
        Estimate coarse carrier frequency offset.
        
        The phase rotation of the guard interval correlation gives
        the CFO as: cfo = phase / (2 * pi * FFT_size / sample_rate)
        
        Args:
            samples: Input I/Q samples
            symbol_start: Known symbol start (or None to find it)
            
        Returns:
            Coarse CFO estimate in Hz
        """
        if symbol_start is None:
            symbol_start, _ = self.find_symbol_start(samples)
        
        # Get samples at symbol boundary
        if symbol_start + self.fft_size + self.guard_length > len(samples):
            return 0.0
        
        # Correlation at known symbol start
        segment1 = samples[symbol_start:symbol_start + self.guard_length]
        segment2 = samples[symbol_start + self.fft_size:
                          symbol_start + self.fft_size + self.guard_length]
        
        correlation = np.sum(segment1 * np.conj(segment2))
        
        # Phase gives CFO
        phase = np.angle(correlation)
        
        # CFO = -phase / (2 * pi * Tu) where Tu = FFT_size / sample_rate
        # The negative sign is because correlation of s1*conj(s2) where s2 
        # is delayed by FFT_size samples gives phase = -2*pi*cfo*FFT_size/fs
        cfo = -phase * self.sample_rate / (2 * np.pi * self.fft_size)
        
        return float(cfo)
    
    def correct_cfo(self, samples: np.ndarray, cfo_hz: float) -> np.ndarray:
        """
        Apply CFO correction to samples.
        
        Args:
            samples: Input samples
            cfo_hz: CFO in Hz to correct
            
        Returns:
            CFO-corrected samples
        """
        t = np.arange(len(samples)) / self.sample_rate
        correction = np.exp(-2j * np.pi * cfo_hz * t)
        return samples * correction.astype(np.complex64)
    
    def find_multiple_symbols(self, samples: np.ndarray,
                              num_symbols: int = 4) -> List[int]:
        """
        Find multiple symbol start positions.
        
        Args:
            samples: Input I/Q samples
            num_symbols: Number of symbols to find
            
        Returns:
            List of symbol start indices
        """
        # Compute full correlation
        corr = self._guard_correlation_fast(samples)
        corr_mag = np.abs(corr)
        
        # Find peaks at symbol intervals
        symbol_starts = []
        
        # Initial peak
        first_start, _ = self.find_symbol_start(samples)
        symbol_starts.append(first_start)
        
        # Find subsequent symbols
        for i in range(1, num_symbols):
            expected = first_start + i * self.symbol_length
            
            # Search around expected position
            search_start = max(0, expected - self.guard_length // 2)
            search_end = min(len(corr_mag), expected + self.guard_length // 2)
            
            if search_start >= search_end:
                break
            
            local_peak = search_start + np.argmax(corr_mag[search_start:search_end])
            symbol_starts.append(int(local_peak))
        
        return symbol_starts


class FineSync:
    """
    Fine synchronization using pilot carriers.
    
    After OFDM demodulation, the known pilot positions allow for:
    - Fine CFO estimation (residual after coarse correction)
    - Phase tracking for coherent demodulation
    - Integer CFO estimation (subcarrier shift)
    
    Attributes:
        mode: '2K', '8K', or 'audio'
        
    Example:
        >>> sync = FineSync('2K')
        >>> fine_cfo = sync.estimate_fine_cfo(carriers, symbol_idx)
    """
    
    def __init__(self, mode: str = '2K', 
                 sample_rate: float = 9142857.142857143):
        """
        Initialize fine synchronizer.
        
        Args:
            mode: '2K', '8K', or 'audio'
            sample_rate: Sample rate in Hz
        """
        self.mode = mode
        self.sample_rate = sample_rate
        self.fft_size = {'2K': 2048, '8K': 8192, 'audio': 64}[mode]
        self.carrier_spacing = sample_rate / self.fft_size
        self.active_carriers = {'2K': 1705, '8K': 6817, 'audio': 24}[mode]
        
        # Generate expected pilot values
        self._generate_pilots()
        
        # Phase tracking state
        self._last_phase = 0.0
    
    def _generate_pilots(self) -> None:
        """Generate expected pilot values (PRBS-based BPSK)."""
        from .Pilots import PilotGenerator, ContinualPilots, ScatteredPilots
        
        self.pilot_gen = PilotGenerator(self.mode)
        self.continual = ContinualPilots(self.mode)
        self.scattered = ScatteredPilots(self.mode)
        
        # Pre-compute expected pilot values for continual pilots
        self._continual_positions = self.continual.get_positions()
        self._continual_expected = self.pilot_gen.get_pilot_values(
            self._continual_positions, boost=True
        )
    
    def estimate_fine_cfo(self, carriers: np.ndarray,
                          symbol_index: int) -> float:
        """
        Estimate fine CFO from pilot carriers.
        
        Compares received pilot phases with expected phases.
        
        Args:
            carriers: Demodulated carrier values
            symbol_index: Symbol number in frame
            
        Returns:
            Fine CFO estimate in Hz
        """
        # Get scattered pilot positions for this symbol
        scattered_pos = self.scattered.get_positions(symbol_index)
        
        # Combine with continual pilots
        all_pilot_pos = np.union1d(self._continual_positions, scattered_pos)
        all_pilot_pos = all_pilot_pos[all_pilot_pos < len(carriers)]
        
        if len(all_pilot_pos) < 2:
            return 0.0
        
        # Get expected values
        expected = self.pilot_gen.get_pilot_values(all_pilot_pos, boost=True)
        
        # Get received values
        received = carriers[all_pilot_pos]
        
        # Phase difference
        phase_diff = np.angle(received * np.conj(expected))
        
        # Linear fit to phase vs carrier index gives CFO
        # phase = 2*pi*cfo*k/N where k is carrier index
        if len(all_pilot_pos) > 1:
            # Least squares fit
            A = np.vstack([all_pilot_pos, np.ones(len(all_pilot_pos))]).T
            try:
                slope, _ = np.linalg.lstsq(A, phase_diff, rcond=None)[0]
                cfo = slope * self.carrier_spacing / (2 * np.pi)
            except:
                cfo = 0.0
        else:
            cfo = 0.0
        
        return float(cfo)
    
    def estimate_phase_offset(self, carriers: np.ndarray,
                              symbol_index: int) -> float:
        """
        Estimate common phase offset from pilots.
        
        Args:
            carriers: Demodulated carrier values
            symbol_index: Symbol number in frame
            
        Returns:
            Phase offset in radians
        """
        scattered_pos = self.scattered.get_positions(symbol_index)
        all_pilot_pos = np.union1d(self._continual_positions, scattered_pos)
        all_pilot_pos = all_pilot_pos[all_pilot_pos < len(carriers)]
        
        if len(all_pilot_pos) == 0:
            return 0.0
        
        expected = self.pilot_gen.get_pilot_values(all_pilot_pos, boost=True)
        received = carriers[all_pilot_pos]
        
        # Average phase difference
        phase_offset = np.angle(np.sum(received * np.conj(expected)))
        
        return float(phase_offset)
    
    def correct_phase(self, carriers: np.ndarray, 
                      phase_offset: float) -> np.ndarray:
        """
        Apply phase correction to carriers.
        
        Args:
            carriers: Input carriers
            phase_offset: Phase to correct (radians)
            
        Returns:
            Phase-corrected carriers
        """
        return carriers * np.exp(-1j * phase_offset)
    
    def track_phase(self, carriers: np.ndarray,
                    symbol_index: int) -> Tuple[np.ndarray, float]:
        """
        Track and correct phase using pilots.
        
        Maintains phase continuity across symbols.
        
        Args:
            carriers: Demodulated carriers
            symbol_index: Symbol number
            
        Returns:
            Tuple of (corrected carriers, tracked phase)
        """
        phase = self.estimate_phase_offset(carriers, symbol_index)
        
        # Unwrap phase relative to last
        while phase - self._last_phase > np.pi:
            phase -= 2 * np.pi
        while phase - self._last_phase < -np.pi:
            phase += 2 * np.pi
        
        self._last_phase = phase
        
        corrected = self.correct_phase(carriers, phase)
        return corrected, phase
    
    def estimate_integer_cfo(self, carriers: np.ndarray,
                             symbol_index: int) -> int:
        """
        Estimate integer CFO (subcarrier offset).
        
        Large CFO can shift the signal by integer subcarriers.
        This detects such shifts by finding pilot correlation peaks.
        
        Args:
            carriers: Demodulated carriers
            symbol_index: Symbol number
            
        Returns:
            Integer carrier offset
        """
        # Get continual pilot positions
        expected = self._continual_expected
        positions = self._continual_positions
        
        # Try different offsets
        max_offset = 5  # Search range
        best_corr = 0
        best_offset = 0
        
        for offset in range(-max_offset, max_offset + 1):
            shifted_pos = positions + offset
            valid = (shifted_pos >= 0) & (shifted_pos < len(carriers))
            
            if np.sum(valid) < 3:
                continue
            
            received = carriers[shifted_pos[valid]]
            exp_valid = expected[valid]
            
            corr = np.abs(np.sum(received * np.conj(exp_valid)))
            
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
        
        return best_offset


class FrameSync:
    """
    Frame and superframe synchronization using TPS.
    
    TPS (Transmission Parameter Signaling) carriers carry frame
    synchronization words that identify frame boundaries.
    
    Attributes:
        mode: '2K', '8K', or 'audio'
        
    Example:
        >>> sync = FrameSync('2K')
        >>> frame_start = sync.find_frame_start(tps_bits)
    """
    
    # TPS sync words
    SYNC_EVEN = np.array([0,0,1,1,0,1,0,1,1,1,1,0,1,1,1,0], dtype=np.uint8)
    SYNC_ODD = np.array([1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], dtype=np.uint8)
    
    # Audio mode sync words (shorter)
    SYNC_AUDIO_EVEN = np.array([0,1,0,1], dtype=np.uint8)
    SYNC_AUDIO_ODD = np.array([1,0,1,0], dtype=np.uint8)
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize frame synchronizer.
        
        Args:
            mode: '2K', '8K', or 'audio'
        """
        self.mode = mode
        self._frame_number = 0
        self._symbol_in_frame = 0
        self._synced = False
        self.symbols_per_frame = 16 if mode == 'audio' else 68
    
    def detect_sync_word(self, tps_bits: np.ndarray) -> Tuple[bool, bool]:
        """
        Detect TPS sync word in received bits.
        
        Args:
            tps_bits: TPS bits from symbol 1 (16 bits for 2K/8K, 4 for audio)
            
        Returns:
            Tuple of (is_sync_detected, is_even_frame)
        """
        if self.mode == 'audio':
            if len(tps_bits) < 4:
                return False, False
            sync_bits = tps_bits[:4]
            even_corr = np.sum(sync_bits == self.SYNC_AUDIO_EVEN)
            odd_corr = np.sum(sync_bits == self.SYNC_AUDIO_ODD)
            threshold = 3  # Allow 1 bit error for audio
        else:
            if len(tps_bits) < 16:
                return False, False
            sync_bits = tps_bits[:16]
            even_corr = np.sum(sync_bits == self.SYNC_EVEN)
            odd_corr = np.sum(sync_bits == self.SYNC_ODD)
            threshold = 12  # Allow some bit errors
        
        if even_corr >= threshold:
            return True, True
        elif odd_corr >= threshold:
            return True, False
        
        return False, False
    
    def update(self, symbol_index: int) -> None:
        """
        Update frame sync state.
        
        Args:
            symbol_index: Current symbol in frame (0-67 for 2K/8K, 0-15 for audio)
        """
        self._symbol_in_frame = symbol_index
        
        if symbol_index == self.symbols_per_frame - 1:
            self._frame_number = (self._frame_number + 1) % 4
    
    def get_frame_position(self) -> Tuple[int, int]:
        """
        Get current position in superframe.
        
        Returns:
            Tuple of (frame_number, symbol_in_frame)
        """
        return self._frame_number, self._symbol_in_frame
    
    def is_synced(self) -> bool:
        """Check if frame sync is acquired."""
        return self._synced
    
    def acquire_sync(self, tps_sequence: List[np.ndarray]) -> bool:
        """
        Attempt to acquire frame synchronization.
        
        Looks for sync word pattern across multiple symbols.
        
        Args:
            tps_sequence: List of TPS bit arrays from consecutive symbols
            
        Returns:
            True if sync acquired
        """
        if len(tps_sequence) < self.symbols_per_frame:
            return False
        
        # Look for sync word at symbol 1
        for i in range(len(tps_sequence) - (self.symbols_per_frame - 1)):
            is_sync, is_even = self.detect_sync_word(tps_sequence[i + 1])
            
            if is_sync:
                self._symbol_in_frame = 1
                self._frame_number = 0 if is_even else 1
                self._synced = True
                return True
        
        return False


class DVBTSynchronizer:
    """
    Complete DVB-T synchronization system.
    
    Combines coarse, fine, and frame synchronization.
    
    Example:
        >>> sync = DVBTSynchronizer(mode='2K', guard_interval='1/4')
        >>> result = sync.synchronize(samples)
        >>> aligned = samples[result.symbol_start:]
        
        >>> # Audio mode
        >>> sync = DVBTSynchronizer(mode='audio', guard_interval='1/4', sample_rate=48000)
        >>> result = sync.synchronize(samples)
    """
    
    def __init__(self, mode: str = '2K', guard_interval: str = '1/4',
                 sample_rate: float = 9142857.142857143):
        """
        Initialize complete synchronizer.
        
        Args:
            mode: '2K', '8K', or 'audio'
            guard_interval: Guard interval ratio
            sample_rate: Sample rate in Hz
        """
        self.mode = mode
        self.guard_interval = guard_interval
        self.sample_rate = sample_rate
        
        # Component synchronizers
        self.coarse = CoarseSync(mode, guard_interval, sample_rate)
        self.fine = FineSync(mode, sample_rate)
        self.frame = FrameSync(mode)
        
        # State
        self._cfo_estimate = 0.0
        self._synced = False
    
    def synchronize(self, samples: np.ndarray) -> SyncResult:
        """
        Perform full synchronization.
        
        Args:
            samples: Input I/Q samples
            
        Returns:
            SyncResult with timing and frequency offsets
        """
        # Step 1: Coarse time sync
        symbol_start, corr_peak = self.coarse.find_symbol_start(samples)
        
        # Estimate SNR from correlation peak
        signal_power = np.mean(np.abs(samples[symbol_start:symbol_start + 
                                             self.coarse.symbol_length]) ** 2)
        noise_power = max(1e-10, signal_power - corr_peak / self.coarse.guard_length)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 30.0
        
        # Step 2: Coarse CFO estimation
        coarse_cfo = self.coarse.estimate_coarse_cfo(samples, symbol_start)
        
        # Step 3: Apply coarse correction
        corrected = self.coarse.correct_cfo(samples, coarse_cfo)
        
        # Confidence based on correlation peak
        confidence = min(1.0, corr_peak / (signal_power * self.coarse.guard_length))
        
        return SyncResult(
            symbol_start=symbol_start,
            coarse_cfo=coarse_cfo,
            fine_cfo=0.0,  # Computed after OFDM demod
            snr_estimate=float(snr_db),
            confidence=float(confidence)
        )
    
    def refine_sync(self, carriers: np.ndarray, 
                    symbol_index: int) -> Tuple[float, float]:
        """
        Refine synchronization after OFDM demodulation.
        
        Args:
            carriers: Demodulated carriers
            symbol_index: Symbol index in frame
            
        Returns:
            Tuple of (fine_cfo, phase_offset)
        """
        fine_cfo = self.fine.estimate_fine_cfo(carriers, symbol_index)
        phase = self.fine.estimate_phase_offset(carriers, symbol_index)
        
        self._cfo_estimate = fine_cfo
        
        return fine_cfo, phase
    
    def get_total_cfo(self) -> float:
        """Get total estimated CFO (coarse + fine)."""
        return self._cfo_estimate
