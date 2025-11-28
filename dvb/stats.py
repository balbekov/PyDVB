"""
DVB-T Real-Time Statistics Collector

Thread-safe statistics collection for the debug dashboard.
Uses ring buffers to maintain rolling windows of metrics.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, List, Tuple
import numpy as np


@dataclass
class AudioStats:
    """Audio input statistics."""
    sample_rate: int = 48000
    carrier_freq: float = 5000.0
    audio_library: str = "none"
    buffer_size: int = 0
    buffer_capacity: int = 10
    is_streaming: bool = False
    peak_level: float = 0.0
    rms_level: float = 0.0


@dataclass
class DVBTParams:
    """DVB-T modulation parameters."""
    mode: str = "audio"
    constellation: str = "QPSK"
    code_rate: str = "1/2"
    guard_interval: str = "1/4"
    bandwidth: str = "audio"
    fft_size: int = 128
    data_carriers: int = 96
    sample_rate: float = 48000.0
    data_rate: float = 0.0


@dataclass
class SignalQuality:
    """Signal quality metrics."""
    snr_db: float = 0.0
    cfo_hz: float = 0.0
    evm_percent: float = 0.0
    mer_db: float = 0.0
    phase_deg: float = 0.0
    signal_present: bool = False


@dataclass
class FECStats:
    """Forward Error Correction statistics."""
    rs_corrected: int = 0
    rs_uncorrectable: int = 0
    ber_pre_fec: float = 0.0
    ber_post_fec: float = 0.0
    viterbi_errors: int = 0


@dataclass
class TransportStats:
    """Transport stream statistics."""
    packets_received: int = 0
    bytes_received: int = 0
    throughput_bps: float = 0.0
    duration_sec: float = 0.0
    symbols_processed: int = 0
    frames_processed: int = 0


@dataclass 
class ImageStats:
    """Image reception statistics."""
    found_header: bool = False
    image_format: str = ""
    width: int = 0
    height: int = 0
    expected_size: int = 0
    received_size: int = 0
    progress_percent: float = 0.0
    crc_valid: bool = False
    image_data: Optional[bytes] = None


class StatsCollector:
    """
    Thread-safe statistics collector for DVB-T debug dashboard.
    
    Maintains rolling windows of metrics for visualization and
    provides atomic access to current statistics.
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize stats collector.
        
        Args:
            history_size: Number of samples to keep in rolling windows
        """
        self._lock = threading.Lock()
        self._history_size = history_size
        
        # Current stats
        self.audio = AudioStats()
        self.dvbt = DVBTParams()
        self.signal = SignalQuality()
        self.fec = FECStats()
        self.transport = TransportStats()
        self.image = ImageStats()
        
        # Rolling windows
        self._snr_history: Deque[float] = deque(maxlen=history_size)
        self._throughput_history: Deque[float] = deque(maxlen=50)
        self._level_history: Deque[float] = deque(maxlen=60)
        self._symbol_power: Deque[float] = deque(maxlen=68)
        self._iq_samples: Deque[complex] = deque(maxlen=512)
        self._evm_history: Deque[float] = deque(maxlen=50)
        
        # Timing
        self._start_time: Optional[float] = None
        self._last_update: float = 0.0
        self._last_bytes: int = 0
        
        # Log messages
        self._log_messages: Deque[Tuple[float, str, str]] = deque(maxlen=100)
    
    def start(self) -> None:
        """Start timing."""
        with self._lock:
            self._start_time = time.time()
            self._last_update = self._start_time
            self._last_bytes = 0
    
    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.audio = AudioStats()
            self.dvbt = DVBTParams()
            self.signal = SignalQuality()
            self.fec = FECStats()
            self.transport = TransportStats()
            self.image = ImageStats()
            
            self._snr_history.clear()
            self._throughput_history.clear()
            self._level_history.clear()
            self._symbol_power.clear()
            self._iq_samples.clear()
            self._evm_history.clear()
            self._log_messages.clear()
            
            self._start_time = None
            self._last_update = 0.0
            self._last_bytes = 0
    
    def update_audio(self, 
                     sample_rate: int = None,
                     carrier_freq: float = None,
                     audio_library: str = None,
                     buffer_size: int = None,
                     buffer_capacity: int = None,
                     is_streaming: bool = None,
                     iq_chunk: np.ndarray = None,
                     rms_level: float = None,
                     peak_level: float = None) -> None:
        """Update audio input statistics."""
        with self._lock:
            if sample_rate is not None:
                self.audio.sample_rate = sample_rate
            if carrier_freq is not None:
                self.audio.carrier_freq = carrier_freq
            if audio_library is not None:
                self.audio.audio_library = audio_library
            if buffer_size is not None:
                self.audio.buffer_size = buffer_size
            if buffer_capacity is not None:
                self.audio.buffer_capacity = buffer_capacity
            if is_streaming is not None:
                self.audio.is_streaming = is_streaming
            if rms_level is not None:
                self.audio.rms_level = rms_level
                self._level_history.append(rms_level)
            if peak_level is not None:
                self.audio.peak_level = peak_level
            
            if iq_chunk is not None and len(iq_chunk) > 0:
                # Calculate levels from I/Q
                abs_vals = np.abs(iq_chunk)
                self.audio.peak_level = float(np.max(abs_vals))
                self.audio.rms_level = float(np.sqrt(np.mean(abs_vals ** 2)))
                self._level_history.append(self.audio.rms_level)
                
                # Store I/Q samples for constellation
                self._iq_samples.extend(iq_chunk[-256:])
    
    def update_dvbt_params(self,
                           mode: str = None,
                           constellation: str = None,
                           code_rate: str = None,
                           guard_interval: str = None,
                           bandwidth: str = None,
                           fft_size: int = None,
                           data_carriers: int = None,
                           sample_rate: float = None,
                           data_rate: float = None) -> None:
        """Update DVB-T parameters."""
        with self._lock:
            if mode is not None:
                self.dvbt.mode = mode
            if constellation is not None:
                self.dvbt.constellation = constellation
            if code_rate is not None:
                self.dvbt.code_rate = code_rate
            if guard_interval is not None:
                self.dvbt.guard_interval = guard_interval
            if bandwidth is not None:
                self.dvbt.bandwidth = bandwidth
            if fft_size is not None:
                self.dvbt.fft_size = fft_size
            if data_carriers is not None:
                self.dvbt.data_carriers = data_carriers
            if sample_rate is not None:
                self.dvbt.sample_rate = sample_rate
            if data_rate is not None:
                self.dvbt.data_rate = data_rate
    
    def update_signal(self,
                      snr_db: float = None,
                      cfo_hz: float = None,
                      evm_percent: float = None,
                      mer_db: float = None,
                      phase_deg: float = None,
                      signal_present: bool = None) -> None:
        """Update signal quality metrics."""
        with self._lock:
            if snr_db is not None:
                self.signal.snr_db = snr_db
                self._snr_history.append(snr_db)
                self.signal.signal_present = snr_db > 3.0
            if cfo_hz is not None:
                self.signal.cfo_hz = cfo_hz
            if evm_percent is not None:
                self.signal.evm_percent = evm_percent
                self._evm_history.append(evm_percent)
            if mer_db is not None:
                self.signal.mer_db = mer_db
            if phase_deg is not None:
                self.signal.phase_deg = phase_deg
            if signal_present is not None:
                self.signal.signal_present = signal_present
    
    def update_fec(self,
                   rs_corrected: int = None,
                   rs_uncorrectable: int = None,
                   ber_pre_fec: float = None,
                   ber_post_fec: float = None,
                   viterbi_errors: int = None,
                   increment: bool = False) -> None:
        """
        Update FEC statistics.
        
        Args:
            increment: If True, add to existing counts instead of replacing
        """
        with self._lock:
            if rs_corrected is not None:
                if increment:
                    self.fec.rs_corrected += rs_corrected
                else:
                    self.fec.rs_corrected = rs_corrected
            if rs_uncorrectable is not None:
                if increment:
                    self.fec.rs_uncorrectable += rs_uncorrectable
                else:
                    self.fec.rs_uncorrectable = rs_uncorrectable
            if ber_pre_fec is not None:
                self.fec.ber_pre_fec = ber_pre_fec
            if ber_post_fec is not None:
                self.fec.ber_post_fec = ber_post_fec
            if viterbi_errors is not None:
                if increment:
                    self.fec.viterbi_errors += viterbi_errors
                else:
                    self.fec.viterbi_errors = viterbi_errors
    
    def update_transport(self,
                         packets: int = None,
                         bytes_received: int = None,
                         symbols: int = None,
                         frames: int = None,
                         duration_sec: float = None,
                         increment: bool = False) -> None:
        """
        Update transport stream statistics.
        
        Args:
            increment: If True, add to existing counts instead of replacing
        """
        now = time.time()
        
        with self._lock:
            if packets is not None:
                if increment:
                    self.transport.packets_received += packets
                else:
                    self.transport.packets_received = packets
            
            if bytes_received is not None:
                if increment:
                    self.transport.bytes_received += bytes_received
                else:
                    self.transport.bytes_received = bytes_received
            
            if symbols is not None:
                if increment:
                    self.transport.symbols_processed += symbols
                else:
                    self.transport.symbols_processed = symbols
            
            if frames is not None:
                if increment:
                    self.transport.frames_processed += frames
                else:
                    self.transport.frames_processed = frames
            
            # Calculate throughput / duration
            if duration_sec is not None:
                self.transport.duration_sec = duration_sec
            elif self._start_time is not None:
                self.transport.duration_sec = now - self._start_time
                
                # Update throughput every 0.5 seconds
                dt = now - self._last_update
                if dt >= 0.5:
                    bytes_delta = self.transport.bytes_received - self._last_bytes
                    self.transport.throughput_bps = (bytes_delta * 8) / dt
                    self._throughput_history.append(self.transport.throughput_bps)
                    self._last_update = now
                    self._last_bytes = self.transport.bytes_received
    
    def update_symbol_power(self, power: float) -> None:
        """Add symbol power sample for sparkline."""
        with self._lock:
            self._symbol_power.append(power)
    
    def update_image(self,
                     found_header: bool = None,
                     image_format: str = None,
                     width: int = None,
                     height: int = None,
                     expected_size: int = None,
                     received_size: int = None,
                     crc_valid: bool = None,
                     image_data: bytes = None) -> None:
        """Update image reception statistics."""
        with self._lock:
            if found_header is not None:
                self.image.found_header = found_header
            if image_format is not None:
                self.image.image_format = image_format
            if width is not None:
                self.image.width = width
            if height is not None:
                self.image.height = height
            if expected_size is not None:
                self.image.expected_size = expected_size
            if received_size is not None:
                self.image.received_size = received_size
                if self.image.expected_size > 0:
                    self.image.progress_percent = min(100.0, 
                        100.0 * received_size / self.image.expected_size)
            if crc_valid is not None:
                self.image.crc_valid = crc_valid
            if image_data is not None:
                self.image.image_data = image_data
    
    def log(self, level: str, message: str) -> None:
        """Add log message."""
        with self._lock:
            self._log_messages.append((time.time(), level, message))
    
    # Getters for history data (return copies for thread safety)
    
    def get_snr_history(self) -> List[float]:
        """Get SNR history."""
        with self._lock:
            return list(self._snr_history)
    
    def get_throughput_history(self) -> List[float]:
        """Get throughput history."""
        with self._lock:
            return list(self._throughput_history)
    
    def get_level_history(self) -> List[float]:
        """Get audio level history."""
        with self._lock:
            return list(self._level_history)
    
    def get_symbol_power(self) -> List[float]:
        """Get symbol power history."""
        with self._lock:
            return list(self._symbol_power)
    
    def get_iq_samples(self) -> np.ndarray:
        """Get recent I/Q samples for constellation."""
        with self._lock:
            return np.array(list(self._iq_samples), dtype=np.complex64)
    
    def get_log_messages(self, count: int = 10) -> List[Tuple[float, str, str]]:
        """Get recent log messages."""
        with self._lock:
            return list(self._log_messages)[-count:]
    
    def get_snapshot(self) -> dict:
        """Get complete snapshot of all stats."""
        with self._lock:
            return {
                'audio': AudioStats(
                    sample_rate=self.audio.sample_rate,
                    carrier_freq=self.audio.carrier_freq,
                    audio_library=self.audio.audio_library,
                    buffer_size=self.audio.buffer_size,
                    buffer_capacity=self.audio.buffer_capacity,
                    is_streaming=self.audio.is_streaming,
                    peak_level=self.audio.peak_level,
                    rms_level=self.audio.rms_level,
                ),
                'dvbt': DVBTParams(
                    mode=self.dvbt.mode,
                    constellation=self.dvbt.constellation,
                    code_rate=self.dvbt.code_rate,
                    guard_interval=self.dvbt.guard_interval,
                    bandwidth=self.dvbt.bandwidth,
                    fft_size=self.dvbt.fft_size,
                    data_carriers=self.dvbt.data_carriers,
                    sample_rate=self.dvbt.sample_rate,
                    data_rate=self.dvbt.data_rate,
                ),
                'signal': SignalQuality(
                    snr_db=self.signal.snr_db,
                    cfo_hz=self.signal.cfo_hz,
                    evm_percent=self.signal.evm_percent,
                    mer_db=self.signal.mer_db,
                    phase_deg=self.signal.phase_deg,
                    signal_present=self.signal.signal_present,
                ),
                'fec': FECStats(
                    rs_corrected=self.fec.rs_corrected,
                    rs_uncorrectable=self.fec.rs_uncorrectable,
                    ber_pre_fec=self.fec.ber_pre_fec,
                    ber_post_fec=self.fec.ber_post_fec,
                    viterbi_errors=self.fec.viterbi_errors,
                ),
                'transport': TransportStats(
                    packets_received=self.transport.packets_received,
                    bytes_received=self.transport.bytes_received,
                    throughput_bps=self.transport.throughput_bps,
                    duration_sec=self.transport.duration_sec,
                    symbols_processed=self.transport.symbols_processed,
                    frames_processed=self.transport.frames_processed,
                ),
                'image': ImageStats(
                    found_header=self.image.found_header,
                    image_format=self.image.image_format,
                    width=self.image.width,
                    height=self.image.height,
                    expected_size=self.image.expected_size,
                    received_size=self.image.received_size,
                    progress_percent=self.image.progress_percent,
                    crc_valid=self.image.crc_valid,
                    image_data=self.image.image_data,
                ),
            }

