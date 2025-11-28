"""
DVB-T Audio Input - Microphone to I/Q Reception

Receives audio from a microphone and demodulates to I/Q samples.
This enables acoustic-channel DVB-T reception for educational purposes.

The audio signal is demodulated by mixing with carrier and low-pass filtering:
    I(t) = LPF{ audio(t) * cos(2π * f_c * t) } * 2
    Q(t) = LPF{ audio(t) * -sin(2π * f_c * t) } * 2

Reference: Quadrature demodulation from real-valued signal
"""

import numpy as np
from typing import Optional, Union, Iterator, Callable, Tuple, TYPE_CHECKING
import time
from pathlib import Path

if TYPE_CHECKING:
    from .stats import StatsCollector
import wave
import threading
import queue
import time


class AudioInput:
    """
    Audio input for I/Q sample reception.
    
    Receives audio from microphone and demodulates to complex I/Q samples.
    The audio carrier is removed to recover the baseband I/Q signal.
    
    Attributes:
        sample_rate: Audio sample rate (44100 or 48000 Hz typical)
        carrier_freq: Expected carrier frequency for I/Q demodulation
        
    Example:
        >>> audio_in = AudioInput(sample_rate=48000, carrier_freq=8000)
        >>> iq_samples = audio_in.read_wav('recording.wav')
        >>> 
        >>> # Or for live reception:
        >>> audio_in.start_stream()
        >>> while receiving:
        ...     iq_chunk = audio_in.read()
        >>> audio_in.stop_stream()
    """
    
    def __init__(self,
                 sample_rate: int = 48000,
                 carrier_freq: float = 8000.0,
                 buffer_size: int = 1024,
                 lpf_bandwidth: Optional[float] = None):
        """
        Initialize audio input.
        
        Args:
            sample_rate: Audio sample rate in Hz
            carrier_freq: Expected carrier frequency for demodulation (Hz)
            buffer_size: Audio buffer size in samples
            lpf_bandwidth: Low-pass filter bandwidth (default: carrier_freq)
        """
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.buffer_size = buffer_size
        self.lpf_bandwidth = lpf_bandwidth or carrier_freq
        
        # Streaming state
        self._stream = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._phase = 0.0
        self._stats_callback = None
        
        # Design low-pass filter for I/Q recovery
        self._design_lpf()
        
        # Try to import audio library
        self._audio_lib = None
        self._init_audio_library()
    
    def _init_audio_library(self) -> None:
        """Initialize audio input library."""
        try:
            import sounddevice as sd
            self._audio_lib = 'sounddevice'
            self._sd = sd
            return
        except ImportError:
            pass
        
        try:
            import pyaudio
            self._audio_lib = 'pyaudio'
            self._pyaudio = pyaudio.PyAudio()
            return
        except ImportError:
            pass
        
        self._audio_lib = None
    
    def _design_lpf(self) -> None:
        """Design low-pass filter for I/Q demodulation."""
        from scipy import signal
        
        # Normalized cutoff frequency
        nyquist = self.sample_rate / 2
        cutoff = min(self.lpf_bandwidth, nyquist * 0.9) / nyquist
        
        # Design 6th order Butterworth low-pass
        self._lpf_b, self._lpf_a = signal.butter(6, cutoff, btype='low')
        
        # Initialize filter state for streaming
        self._lpf_zi_i = signal.lfilter_zi(self._lpf_b, self._lpf_a)
        self._lpf_zi_q = signal.lfilter_zi(self._lpf_b, self._lpf_a)
    
    def audio_to_iq(self, audio: np.ndarray,
                    reset_phase: bool = False,
                    stereo: bool = True) -> np.ndarray:
        """
        Demodulate audio signal to I/Q samples.
        
        Args:
            audio: Audio samples - shape (N,2) for stereo, (N,) for mono
            reset_phase: Reset carrier phase (for non-streaming)
            stereo: If True, expect stereo (I=left, Q=right).
                   If False, demodulate from carrier.
            
        Returns:
            Complex I/Q samples
        """
        from scipy import signal
        
        audio = np.asarray(audio, dtype=np.float32)
        
        # Check if input is stereo
        if audio.ndim == 2 and audio.shape[1] == 2:
            # Stereo input: left=I, right=Q
            i_component = audio[:, 0]
            q_component = audio[:, 1]
            iq = (i_component + 1j * q_component).astype(np.complex64)
            return iq
        
        # Mono input: demodulate from carrier
        if reset_phase:
            self._phase = 0.0
            self._lpf_zi_i = signal.lfilter_zi(self._lpf_b, self._lpf_a)
            self._lpf_zi_q = signal.lfilter_zi(self._lpf_b, self._lpf_a)
        
        # Flatten if needed
        if audio.ndim == 2:
            audio = audio[:, 0]  # Take first channel
        
        # Generate time vector
        n = len(audio)
        t = np.arange(n) / self.sample_rate
        
        # Generate local oscillator signals
        carrier_phase = 2.0 * np.pi * self.carrier_freq * t + self._phase
        cos_carrier = np.cos(carrier_phase)
        sin_carrier = -np.sin(carrier_phase)  # -sin for Q
        
        # Mix down to baseband
        i_mixed = audio * cos_carrier * 2.0
        q_mixed = audio * sin_carrier * 2.0
        
        # Low-pass filter to remove 2*fc component
        # Use filtfilt for zero-phase filtering (better for file-based processing)
        if reset_phase:
            # For file-based processing, use filtfilt (zero-phase)
            i_filtered = signal.filtfilt(self._lpf_b, self._lpf_a, i_mixed)
            q_filtered = signal.filtfilt(self._lpf_b, self._lpf_a, q_mixed)
        else:
            # For streaming, use regular lfilter with state
            i_filtered, self._lpf_zi_i = signal.lfilter(
                self._lpf_b, self._lpf_a, i_mixed, zi=self._lpf_zi_i * i_mixed[0]
            )
            q_filtered, self._lpf_zi_q = signal.lfilter(
                self._lpf_b, self._lpf_a, q_mixed, zi=self._lpf_zi_q * q_mixed[0]
            )
        
        # Update phase for continuity
        self._phase = (carrier_phase[-1] + 
                       2.0 * np.pi * self.carrier_freq / self.sample_rate) % (2.0 * np.pi)
        
        # Combine to complex I/Q
        iq = (i_filtered + 1j * q_filtered).astype(np.complex64)
        
        return iq
    
    def preview_iq(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate I/Q preview without disturbing streaming state.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) == 0:
            return np.array([], dtype=np.complex64)
        
        # Backup state
        phase_backup = self._phase
        zi_i_backup = self._lpf_zi_i.copy()
        zi_q_backup = self._lpf_zi_q.copy()
        
        try:
            iq = self.audio_to_iq(audio, reset_phase=False, stereo=False)
        finally:
            # Restore state
            self._phase = phase_backup
            self._lpf_zi_i = zi_i_backup
            self._lpf_zi_q = zi_q_backup
        
        return iq
    
    def record(self, duration: float, save_raw_audio: Optional[str] = None) -> np.ndarray:
        """
        Record audio and return I/Q samples.
        
        Args:
            duration: Recording duration in seconds
            save_raw_audio: Optional path to save raw audio as WAV
            
        Returns:
            Complex I/Q samples
        """
        if self._audio_lib is None:
            raise RuntimeError(
                "No audio library available. Install sounddevice or pyaudio:\n"
                "  pip install sounddevice\n"
                "  or: pip install pyaudio"
            )
        
        num_samples = int(duration * self.sample_rate)
        
        if self._audio_lib == 'sounddevice':
            # Query device to get supported channels
            try:
                dev_info = self._sd.query_devices(kind='input')
                max_channels = dev_info.get('max_input_channels', 1)
            except Exception:
                max_channels = 1
            
            channels = min(2, max_channels)  # Use stereo if available, else mono
            
            # Use streaming recording for real-time level updates
            recorded_chunks = []
            samples_recorded = [0]
            
            def audio_callback(indata, frames, time_info, status):
                recorded_chunks.append(indata.copy())
                samples_recorded[0] += frames
                
                # Update stats for real-time display
                if hasattr(self, '_stats_callback') and self._stats_callback:
                    # Calculate audio levels
                    chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    peak = float(np.max(np.abs(chunk)))
                    progress = samples_recorded[0] / num_samples
                    self._stats_callback(rms, peak, progress, chunk)
            
            with self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=channels,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=2048
            ):
                # Wait for recording to complete
                import time as time_module
                start_time = time_module.time()
                while samples_recorded[0] < num_samples:
                    time_module.sleep(0.01)
                    # Timeout safety
                    if time_module.time() - start_time > duration + 2:
                        break
            
            # Concatenate recorded chunks
            if recorded_chunks:
                raw_audio = np.concatenate(recorded_chunks, axis=0)[:num_samples]
            else:
                raw_audio = np.zeros((num_samples, channels), dtype=np.float32)
            
            # Save raw audio if requested
            if save_raw_audio:
                self._save_raw_audio(raw_audio, channels, save_raw_audio)
            
            # Take first channel if stereo, flatten if mono
            if channels == 2:
                audio = raw_audio[:, 0]
            else:
                audio = raw_audio.flatten()
        
        elif self._audio_lib == 'pyaudio':
            # Try stereo first (some macOS devices don't support mono)
            try:
                stream = self._pyaudio.open(
                    format=self._pyaudio.get_format_from_width(2),  # 16-bit
                    channels=2,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.buffer_size
                )
                channels = 2
            except Exception:
                stream = self._pyaudio.open(
                    format=self._pyaudio.get_format_from_width(2),  # 16-bit
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.buffer_size
                )
                channels = 1
            
            frames = []
            samples_recorded = 0
            for _ in range(0, num_samples, self.buffer_size):
                chunk_size = min(self.buffer_size, num_samples - samples_recorded)
                data = stream.read(chunk_size)
                frames.append(data)
                samples_recorded += chunk_size
                
                if hasattr(self, '_stats_callback') and self._stats_callback:
                    chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
                    if channels == 2:
                        chunk = chunk[::2]
                    rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                    peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
                    progress = samples_recorded / num_samples
                    self._stats_callback(rms, peak, progress, chunk)
            
            stream.stop_stream()
            stream.close()
            
            # Convert bytes to float
            audio_bytes = b''.join(frames)
            raw_audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Save raw audio if requested
            if save_raw_audio:
                if channels == 2:
                    raw_reshaped = raw_audio.reshape(-1, 2)
                else:
                    raw_reshaped = raw_audio.reshape(-1, 1)
                self._save_raw_audio(raw_reshaped, channels, save_raw_audio)
            
            # Take first channel if stereo
            if channels == 2:
                audio = raw_audio[::2]
            else:
                audio = raw_audio
        
        # Convert to I/Q
        return self.audio_to_iq(audio, reset_phase=True)
    
    def _save_raw_audio(self, audio: np.ndarray, channels: int, path: str) -> None:
        """
        Save raw audio to WAV file.
        
        Args:
            audio: Audio samples as float32, shape (N,) or (N, channels)
            channels: Number of channels
            path: Output WAV file path
        """
        # Normalize and convert to int16
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        
        # Clip and scale to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(path, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_int16.tobytes())
    
    def start_stream(self, 
                     callback: Optional[Callable[[np.ndarray], None]] = None) -> None:
        """
        Start audio input stream for real-time reception.
        
        Args:
            callback: Optional callback for each I/Q chunk received
        """
        if self._audio_lib is None:
            raise RuntimeError("No audio library available")
        
        self._running = True
        self._audio_queue = queue.Queue()
        self._callback = callback
        
        # Reset phase for new stream
        from scipy import signal
        self._phase = 0.0
        self._lpf_zi_i = signal.lfilter_zi(self._lpf_b, self._lpf_a)
        self._lpf_zi_q = signal.lfilter_zi(self._lpf_b, self._lpf_a)
        
        if self._audio_lib == 'sounddevice':
            # Query device to get supported channels
            try:
                dev_info = self._sd.query_devices(kind='input')
                max_channels = dev_info.get('max_input_channels', 1)
            except Exception:
                max_channels = 1
            
            self._sd_channels = min(2, max_channels)
            
            def sd_callback(indata, frames, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                
                # Take first channel (works for both mono and stereo)
                if indata.ndim == 2:
                    audio = indata[:, 0].astype(np.float32)
                else:
                    audio = indata.flatten().astype(np.float32)
                iq = self.audio_to_iq(audio)
                
                if self._stats_callback:
                    chunk = audio.copy()
                    rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                    peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
                    self._stats_callback(rms, peak, None, chunk)
                
                self._audio_queue.put(iq)
                
                if self._callback:
                    self._callback(iq)
            
            self._stream = self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=self._sd_channels,
                dtype=np.float32,
                blocksize=self.buffer_size,
                callback=sd_callback
            )
            self._stream.start()
        
        elif self._audio_lib == 'pyaudio':
            self._pa_channels = 2  # Try stereo first
            
            def pa_callback(in_data, frame_count, time_info, status):
                audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32767.0
                # Take first channel if stereo
                if self._pa_channels == 2:
                    audio = audio[::2]
                iq = self.audio_to_iq(audio)
                
                if self._stats_callback:
                    chunk = audio.copy()
                    rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                    peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
                    self._stats_callback(rms, peak, None, chunk)
                
                self._audio_queue.put(iq)
                
                if self._callback:
                    self._callback(iq)
                
                return (None, self._pyaudio.paContinue)
            
            self._stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(2),
                channels=2,  # Stereo - some macOS devices don't support mono
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=pa_callback
            )
            self._stream.start_stream()
    
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read I/Q samples from stream buffer.
        
        Args:
            timeout: Timeout in seconds (None = blocking)
            
        Returns:
            I/Q samples or None if timeout
        """
        if not self._running:
            raise RuntimeError("Stream not started. Call start_stream() first.")
        
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def read_all(self) -> np.ndarray:
        """
        Read all available I/Q samples from buffer.
        
        Returns:
            Concatenated I/Q samples
        """
        chunks = []
        while not self._audio_queue.empty():
            try:
                chunks.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break
        
        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.complex64)
    
    def stop_stream(self) -> None:
        """Stop audio input stream."""
        self._running = False
        
        if self._stream is not None:
            if self._audio_lib == 'sounddevice':
                self._stream.stop()
                self._stream.close()
            elif self._audio_lib == 'pyaudio':
                self._stream.stop_stream()
                self._stream.close()
            
            self._stream = None
    
    def read_wav(self, path: Union[str, Path], 
                 stereo: Optional[bool] = None) -> np.ndarray:
        """
        Read WAV file and demodulate to I/Q.
        
        Args:
            path: Input WAV file path
            stereo: If True, expect stereo WAV (I=left, Q=right).
                   If False, demodulate from carrier.
                   If None (default), auto-detect based on channel count and correlation.
            
        Returns:
            Complex I/Q samples
        """
        path = Path(path)
        
        with wave.open(str(path), 'rb') as wav:
            n_channels = wav.getnchannels()
            wav_rate = wav.getframerate()
            
            if wav_rate != self.sample_rate:
                print(f"Warning: WAV sample rate {wav_rate} differs from expected {self.sample_rate}")
            
            # Read all frames
            n_frames = wav.getnframes()
            raw_data = wav.readframes(n_frames)
            
            # Convert to numpy array
            sample_width = wav.getsampwidth()
            if sample_width == 1:
                audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
                audio = (audio - 128) / 127.0
            elif sample_width == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                audio = audio / 32767.0
            elif sample_width == 4:
                audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32)
                audio = audio / 2147483647.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Handle stereo vs mono
            if n_channels == 2:
                # Reshape to (N, 2)
                audio = audio.reshape(-1, 2)
                
                # Auto-detect if stereo is None
                if stereo is None:
                    # Check if L and R are highly correlated (mono recording)
                    left = audio[:min(10000, len(audio)), 0]
                    right = audio[:min(10000, len(audio)), 1]
                    if len(left) > 100:
                        corr = np.corrcoef(left, right)[0, 1]
                        # If correlation > 0.95, it's effectively mono
                        stereo = corr < 0.95
                        if not stereo:
                            print(f"Auto-detected: mono (L/R correlation {corr:.3f})")
                        else:
                            print(f"Auto-detected: stereo I/Q (L/R correlation {corr:.3f})")
                    else:
                        stereo = True  # Default to stereo for short files
                
                if not stereo:
                    # Mono: take first channel for carrier demodulation
                    audio = audio[:, 0]
            elif n_channels == 1:
                stereo = False  # Mono file, must use carrier demodulation
            else:
                # Multi-channel: take first channel
                audio = audio[::n_channels]
                stereo = False
        
        # Demodulate to I/Q
        return self.audio_to_iq(audio, reset_phase=True, stereo=stereo if stereo is not None else True)
    
    def iter_wav(self, path: Union[str, Path],
                 chunk_samples: int = 8192) -> Iterator[np.ndarray]:
        """
        Iterate over WAV file in chunks.
        
        Args:
            path: Input WAV file path
            chunk_samples: Samples per chunk
            
        Yields:
            I/Q sample chunks
        """
        path = Path(path)
        
        with wave.open(str(path), 'rb') as wav:
            wav_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            
            # Reset phase for new file
            from scipy import signal
            self._phase = 0.0
            self._lpf_zi_i = signal.lfilter_zi(self._lpf_b, self._lpf_a)
            self._lpf_zi_q = signal.lfilter_zi(self._lpf_b, self._lpf_a)
            
            while True:
                raw_data = wav.readframes(chunk_samples)
                if not raw_data:
                    break
                
                # Convert to float
                if sample_width == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                    audio = audio / 32767.0
                else:
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Take first channel if stereo
                if n_channels > 1:
                    audio = audio[::n_channels]
                
                yield self.audio_to_iq(audio)
    
    def close(self) -> None:
        """Clean up audio resources."""
        self.stop_stream()
        if self._audio_lib == 'pyaudio' and hasattr(self, '_pyaudio'):
            self._pyaudio.terminate()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class AcousticDVBTReceiver:
    """
    Acoustic DVB-T receiver.
    
    Receives audio from microphone, demodulates from carrier to I/Q, 
    and decodes the DVB-T signal to recover the transport stream.
    
    Uses narrowband 'audio' mode: 16 kHz OFDM (64-pt FFT, 24 carriers, 6 kHz bandwidth)
    modulated onto a 13 kHz carrier for acoustic transmission.
    
    Example:
        >>> rx = AcousticDVBTReceiver()
        >>> ts_data, stats = rx.receive(duration=5.0)
        >>> 
        >>> # Or from WAV file:
        >>> ts_data, stats = rx.receive_file('recording.wav')
        >>>
        >>> # With debug dashboard:
        >>> from dvb.dashboard import DVBDashboard
        >>> with DVBDashboard(rx) as dash:
        ...     ts_data, stats = rx.receive(30.0)
    """
    
    def __init__(self,
                 audio_sample_rate: int = 48000,
                 carrier_freq: float = 5000,
                 mode: str = '2K',
                 constellation: str = 'QPSK',
                 code_rate: str = '1/2',
                 guard_interval: str = 'acoustic',
                 stats_collector: Optional['StatsCollector'] = None):
        """
        Initialize acoustic DVB-T receiver.
        
        Args:
            audio_sample_rate: Audio input sample rate
            carrier_freq: Expected audio carrier frequency (default 5kHz for acoustic)
            mode: DVB-T mode ('2K' or '8K')
            constellation: 'QPSK', '16QAM', or '64QAM'
            code_rate: FEC rate
            guard_interval: Guard interval ratio ('acoustic' = 10ms for speaker/mic)
            stats_collector: Optional StatsCollector for dashboard integration
        """
        self.audio_sample_rate = audio_sample_rate
        self.carrier_freq = carrier_freq
        self._stats = stats_collector
        
        # Create audio input with LPF bandwidth matching OFDM signal (~6 kHz)
        self.audio_input = AudioInput(
            sample_rate=audio_sample_rate,
            carrier_freq=carrier_freq,
            lpf_bandwidth=8000  # 8 kHz to capture 6 kHz OFDM signal with margin
        )
        
        # Set up real-time audio level callback for dashboard
        if stats_collector is not None:
            def audio_stats_callback(rms, peak, progress, chunk):
                stats_collector.update_audio(
                    rms_level=rms,
                    peak_level=peak,
                    is_streaming=True,
                )
                if chunk is not None and len(chunk) > 0:
                    iq_preview = self.audio_input.preview_iq(chunk)
                    if len(iq_preview) > 0:
                        stats_collector.update_audio(iq_chunk=iq_preview)
                        mag = np.abs(iq_preview)
                        signal = float(np.mean(mag))
                        noise = float(np.std(mag - signal)) + 1e-6
                        snr = 20 * np.log10(max(signal, 1e-6) / noise)
                        evm = (noise / (signal + 1e-6)) * 100
                        phase = float(np.angle(np.mean(iq_preview)))
                        stats_collector.update_signal(
                            snr_db=snr,
                            evm_percent=evm,
                            phase_deg=np.degrees(phase),
                            signal_present=True,
                        )
                if stats_collector._start_time:
                    stats_collector.update_transport(duration_sec=time.time() - stats_collector._start_time)
            
            self.audio_input._stats_callback = audio_stats_callback
        else:
            self.audio_input._stats_callback = None
        
        # Create demodulator with 'audio' bandwidth
        # OFDM runs at 16 kHz, audio input is 48 kHz
        from .DVB import DVBTDemodulator
        
        # Progress callback for real-time dashboard updates
        def progress_cb(done, total, phase):
            if self._stats is not None:
                self._stats.update_transport(symbols=done)
                if phase == 'symbols':
                    self._stats.update_signal(signal_present=True)
        
        self.demodulator = DVBTDemodulator(
            mode=mode,
            constellation=constellation,
            code_rate=code_rate,
            guard_interval=guard_interval,
            bandwidth='audio',
            progress_callback=progress_cb if stats_collector else None,
        )
        
        # DVB-T OFDM sample rate (16 kHz for audio mode)
        self.dvbt_sample_rate = self.demodulator.get_sample_rate()
    
    def _resample_iq(self, iq_samples: np.ndarray,
                     from_rate: float,
                     to_rate: float) -> np.ndarray:
        """Resample I/Q samples."""
        # No resampling needed if rates match
        if abs(from_rate - to_rate) < 1.0:
            return iq_samples.astype(np.complex64)
        
        from scipy import signal
        
        num_samples = int(len(iq_samples) * to_rate / from_rate)
        return signal.resample(iq_samples, num_samples).astype(np.complex64)
    
    def receive(self, duration: float) -> Tuple[bytes, dict]:
        """
        Receive and decode DVB-T from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (transport stream data, statistics)
        """
        # Record audio and convert to I/Q
        iq_samples = self.audio_input.record(duration)
        
        # Resample to DVB-T sample rate
        iq_resampled = self._resample_iq(
            iq_samples,
            self.audio_sample_rate,
            self.dvbt_sample_rate
        )
        
        # Decode
        return self.demodulator.demodulate(iq_resampled)
    
    def receive_file(self, path: Union[str, Path]) -> Tuple[bytes, dict]:
        """
        Receive and decode DVB-T from WAV file.
        
        Args:
            path: Input WAV file path
            
        Returns:
            Tuple of (transport stream data, statistics)
        """
        # Read WAV and convert to I/Q
        iq_samples = self.audio_input.read_wav(path)
        
        # Resample to DVB-T sample rate if needed
        iq_resampled = self._resample_iq(
            iq_samples,
            self.audio_sample_rate,
            self.dvbt_sample_rate
        )
        
        # Update stats with I/Q samples for real-time constellation display  
        if self._stats is not None:
            self._stats.update_audio(iq_chunk=iq_resampled[:1024], is_streaming=True)
            self._stats.log("INFO", f"Processing {len(iq_resampled):,} I/Q samples...")
        
        # Use the full DVBTDemodulator (batch processing required for correct FEC)
        ts_data, rx_stats = self.demodulator.demodulate(iq_resampled)
        
        # Update stats from demodulator results
        if self._stats is not None:
            self._stats.update_signal(
                snr_db=rx_stats.get('snr_db', 0),
                cfo_hz=rx_stats.get('cfo_hz', 0),
                signal_present=rx_stats.get('packets_recovered', 0) > 0,
            )
            self._stats.update_fec(
                rs_corrected=rx_stats.get('rs_errors', 0),
                rs_uncorrectable=rx_stats.get('rs_uncorrectable', 0),
            )
            self._stats.update_transport(
                packets=rx_stats.get('packets_recovered', 0),
                bytes_received=len(ts_data),
                symbols=rx_stats.get('symbols', 0),
            )
            self._stats.log("INFO", f"Decoded {rx_stats.get('packets_recovered', 0)} packets")
            
            # Try to extract image info
            self._update_image_preview(ts_data, final=True)
        
        return ts_data, rx_stats
    
    def _demodulate_with_stats(self, iq_samples: np.ndarray) -> Tuple[bytes, dict]:
        """Demodulate with real-time stats updates."""
        from .OFDM import OFDMDemodulator
        from .GuardInterval import GuardIntervalRemover
        from .Pilots import PilotExtractor
        from .ChannelEstimator import ChannelEstimatorWithEqualization
        from .InnerInterleaver import SymbolInterleaver, BitInterleaver
        from .Puncturing import Depuncturer
        from .Convolutional import ConvolutionalDecoder
        from .OuterInterleaver import OuterDeinterleaver
        from .ReedSolomon import ReedSolomon
        from .Scrambler import Scrambler
        from .QAM import QAMDemapper
        from .Synchronizer import DVBTSynchronizer
        
        mode = self.demodulator.mode
        constellation = self.demodulator.constellation
        code_rate = self.demodulator.code_rate
        guard_interval = self.demodulator.guard_interval
        sample_rate = self.demodulator.sample_rate
        
        stats = {
            'symbols': 0,
            'rs_errors': 0,
            'rs_uncorrectable': 0,
            'snr_db': 0.0,
            'cfo_hz': 0.0,
            'packets_recovered': 0,
        }
        
        # FFT sizes depend on mode
        fft_sizes = {'2K': 2048, '8K': 8192, 'audio': 64}
        fft_size = fft_sizes.get(mode, 2048)
        guard_fracs = {'1/4': 4, '1/8': 8, '1/16': 16, '1/32': 32, 'acoustic': 0.4}
        divisor = guard_fracs[guard_interval]
        guard_length = int(fft_size / divisor) if divisor < 1 else fft_size // int(divisor)
        symbol_len = fft_size + guard_length
        symbols_per_frame = 16 if mode == 'audio' else 68
        
        # Initialize stages
        synchronizer = DVBTSynchronizer(mode, guard_interval, sample_rate)
        guard_remover = GuardIntervalRemover(guard_interval, fft_size)
        ofdm_demod = OFDMDemodulator(mode)
        channel_eq = ChannelEstimatorWithEqualization(mode)
        pilot_extractor = PilotExtractor(mode)
        symbol_deinterleaver = SymbolInterleaver(mode)
        bit_deinterleaver = BitInterleaver(constellation, mode)
        depuncturer = Depuncturer(code_rate)
        conv_decoder = ConvolutionalDecoder()
        outer_deinterleaver = OuterDeinterleaver()
        rs_decoder = ReedSolomon()
        descrambler = Scrambler()
        qam_demapper = QAMDemapper(constellation)
        
        # Step 1: Synchronization
        sync_result = synchronizer.synchronize(iq_samples)
        stats['cfo_hz'] = sync_result.coarse_cfo
        stats['snr_db'] = sync_result.snr_estimate
        
        if self._stats is not None:
            self._stats.update_signal(
                snr_db=stats['snr_db'],
                cfo_hz=stats['cfo_hz'],
                signal_present=True,
            )
            self._stats.log("INFO", f"Signal found: SNR={stats['snr_db']:.1f}dB, CFO={stats['cfo_hz']:.1f}Hz")
        
        # Apply CFO correction
        corrected_samples = synchronizer.coarse.correct_cfo(iq_samples, sync_result.coarse_cfo)
        aligned_samples = corrected_samples[sync_result.symbol_start:]
        
        num_symbols = len(aligned_samples) // symbol_len
        stats['symbols'] = num_symbols
        
        if num_symbols == 0:
            return b'', stats
        
        if self._stats is not None:
            self._stats.log("INFO", f"Processing {num_symbols} OFDM symbols...")
        
        # Step 2: Process symbols with real-time updates
        all_data = []
        channel_eq.reset()
        
        for sym_idx in range(num_symbols):
            start = sym_idx * symbol_len
            symbol_samples = aligned_samples[start:start + symbol_len]
            
            useful = guard_remover.remove(symbol_samples)
            carriers = ofdm_demod.demodulate(useful)
            equalized, csi = channel_eq.process(carriers, sym_idx % symbols_per_frame)
            
            fine_cfo, phase = synchronizer.refine_sync(equalized, sym_idx % symbols_per_frame)
            equalized = synchronizer.fine.correct_phase(equalized, phase)
            
            data_carriers, _, _ = pilot_extractor.extract(equalized, sym_idx % symbols_per_frame)
            all_data.append(data_carriers)
            
            # Update stats every 8 symbols for smooth dashboard updates
            if self._stats is not None and sym_idx % 8 == 0:
                symbol_power = float(np.mean(np.abs(data_carriers) ** 2))
                self._stats.update_symbol_power(symbol_power)
                self._stats.update_transport(
                    symbols=sym_idx + 1,
                    frames=(sym_idx + 1) // symbols_per_frame,
                )
                # Update constellation with recent equalized carriers
                self._stats.update_audio(iq_chunk=data_carriers)
                
                # Calculate EVM from constellation
                if constellation == 'QPSK':
                    ideal_amp = 1.0 / np.sqrt(2)
                    evm = np.sqrt(np.mean(np.abs(np.abs(data_carriers) - ideal_amp) ** 2)) * 100
                    self._stats.update_signal(evm_percent=float(evm), phase_deg=float(phase * 180 / np.pi))
        
        if self._stats is not None:
            self._stats.log("INFO", "Symbol processing complete, decoding FEC...")
        
        # Step 3-7: FEC decoding
        data_carriers = np.concatenate(all_data)
        deinterleaved = symbol_deinterleaver.deinterleave(data_carriers)
        bits = qam_demapper.demap(deinterleaved)
        bits = bit_deinterleaver.deinterleave(bits)
        depunctured = depuncturer.depuncture(bits.astype(np.float32))
        decoded = conv_decoder.decode((depunctured > 0.5).astype(np.uint8), terminated=False)
        
        byte_data = np.packbits(decoded).tobytes()
        deinterleaved_bytes = outer_deinterleaver.process(byte_data)
        
        # Skip interleaver fill for audio mode
        if mode == 'audio':
            interleaver_fill = 11 * 204
            deinterleaved_bytes = deinterleaved_bytes[interleaver_fill:]
        
        # Step 9: RS decoding with progress updates
        recovered = bytearray()
        num_codewords = len(deinterleaved_bytes) // 204
        
        for i in range(num_codewords):
            codeword = deinterleaved_bytes[i * 204:(i + 1) * 204]
            decoded_pkt, errors = rs_decoder.decode(codeword)
            
            if errors < 0:
                stats['rs_uncorrectable'] += 1
                if self._stats is not None:
                    self._stats.update_fec(rs_uncorrectable=1, increment=True)
            else:
                if errors > 0:
                    stats['rs_errors'] += errors
                    if self._stats is not None:
                        self._stats.update_fec(rs_corrected=errors, increment=True)
                recovered.extend(decoded_pkt)
            
            # Update transport stats and check for image data every 10 codewords
            if self._stats is not None and i % 10 == 0:
                self._stats.update_transport(
                    packets=len(recovered) // 188,
                    bytes_received=len(recovered),
                )
                # Try to parse partial image data for preview
                self._update_image_preview(bytes(recovered))
        
        # Step 10: Descrambling
        ts_data = descrambler.descramble(bytes(recovered))
        stats['packets_recovered'] = len(ts_data) // 188
        
        # Final stats update
        if self._stats is not None:
            self._stats.update_transport(
                packets=stats['packets_recovered'],
                bytes_received=len(ts_data),
                symbols=num_symbols,
                frames=num_symbols // symbols_per_frame,
            )
            self._stats.log("INFO", f"Decoded {stats['packets_recovered']} TS packets")
            
            # Final image extraction
            self._update_image_preview(ts_data, final=True)
        
        return ts_data, stats
    
    def _update_image_preview(self, ts_data: bytes, final: bool = False) -> None:
        """Update image preview in stats collector."""
        if self._stats is None or len(ts_data) < 188:
            return
        
        try:
            from .ImageTransport import ts_to_image, IMAGE_MAGIC, HEADER_PID, IMAGE_PID
            from .TransportStream import TransportStream
            
            # Quick check for image header in first few packets
            ts = TransportStream.from_bytes(ts_data[:188 * 20])  # Check first 20 packets
            
            header_data = None
            for pkt in ts:
                if pkt.pid == HEADER_PID and pkt.payload_unit_start:
                    header_data = pkt.payload
                    break
            
            if header_data is None or header_data[:4] != IMAGE_MAGIC:
                return
            
            # Parse header
            import struct
            img_format = header_data[5:9].rstrip(b'\x00').decode('ascii', errors='ignore')
            width = struct.unpack('>H', header_data[9:11])[0]
            height = struct.unpack('>H', header_data[11:13])[0]
            expected_size = struct.unpack('>I', header_data[13:17])[0]
            
            # Count received image data
            ts_full = TransportStream.from_bytes(ts_data)
            received_bytes = 0
            image_chunks = []
            for pkt in ts_full:
                if pkt.pid == IMAGE_PID:
                    image_chunks.append(pkt.payload)
                    received_bytes += 184
            
            received_bytes = min(received_bytes, expected_size)
            
            self._stats.update_image(
                found_header=True,
                image_format=img_format,
                width=width,
                height=height,
                expected_size=expected_size,
                received_size=received_bytes,
            )
            
            # For final update, include actual image data
            if final and image_chunks:
                image_data = b''.join(image_chunks)[:expected_size]
                if len(image_data) == expected_size:
                    self._stats.update_image(image_data=image_data, crc_valid=True)
                    self._stats.log("INFO", f"Image received: {width}x{height} {img_format.upper()}")
        except Exception:
            pass  # Ignore parsing errors during streaming
    
    def start_receive(self, 
                      callback: Optional[Callable[[bytes, dict], None]] = None) -> None:
        """
        Start continuous reception.
        
        Args:
            callback: Callback for each decoded chunk (ts_data, stats)
        """
        self._receive_callback = callback
        self._receive_buffer = np.array([], dtype=np.complex64)
        
        # Need enough samples for at least one OFDM frame
        # 2K mode: 2048 + 512 (1/4 guard) = 2560 samples per symbol
        # 68 symbols per frame = ~174080 samples at DVB-T rate
        # At 48kHz audio, that's ~913 audio samples per DVB-T sample
        
        def on_audio_chunk(iq_chunk):
            self._receive_buffer = np.concatenate([self._receive_buffer, iq_chunk])
            
            # Process when we have enough samples
            min_samples = 10000  # Minimum for processing
            if len(self._receive_buffer) >= min_samples:
                # Resample accumulated buffer
                iq_resampled = self._resample_iq(
                    self._receive_buffer,
                    self.audio_sample_rate,
                    self.dvbt_sample_rate
                )
                
                # Try to decode
                try:
                    ts_data, stats = self.demodulator.demodulate(iq_resampled)
                    if callback and len(ts_data) > 0:
                        callback(ts_data, stats)
                except Exception as e:
                    pass  # Not enough data or sync lost
                
                # Keep overlap for continuity
                overlap = min_samples // 2
                self._receive_buffer = self._receive_buffer[-overlap:]
        
        self.audio_input.start_stream(callback=on_audio_chunk)
    
    def stop_receive(self) -> None:
        """Stop continuous reception."""
        self.audio_input.stop_stream()
    
    def set_stats_collector(self, stats: 'StatsCollector') -> None:
        """
        Set stats collector for dashboard integration.
        
        Args:
            stats: StatsCollector instance to receive real-time stats
        """
        self._stats = stats
        
        # Initialize stats with receiver parameters
        if stats is not None:
            stats.update_audio(
                sample_rate=self.audio_sample_rate,
                carrier_freq=self.carrier_freq,
                audio_library=self.audio_input._audio_lib or "none",
            )
            stats.update_dvbt_params(
                mode=self.demodulator.mode,
                constellation=self.demodulator.constellation,
                code_rate=self.demodulator.code_rate,
                guard_interval=self.demodulator.guard_interval,
                bandwidth=self.demodulator.bandwidth,
                fft_size=getattr(self.demodulator, 'fft_size', 128),
                sample_rate=self.dvbt_sample_rate,
            )
    
    def close(self) -> None:
        """Clean up resources."""
        self.audio_input.close()


def wav_to_iq(input_path: Union[str, Path],
              audio_sample_rate: int = 48000,
              carrier_freq: float = 5000) -> np.ndarray:
    """
    Convenience function to convert WAV file to I/Q samples.
    
    Args:
        input_path: Input WAV file path
        audio_sample_rate: Expected audio sample rate
        carrier_freq: Carrier frequency for demodulation
        
    Returns:
        Complex I/Q samples
    """
    audio_in = AudioInput(
        sample_rate=audio_sample_rate,
        carrier_freq=carrier_freq
    )
    return audio_in.read_wav(input_path)

