"""
DVB-T Audio Output - I/Q to Speaker Transmission

Converts baseband I/Q samples to audio for transmission through a speaker.
This enables acoustic-channel DVB-T transmission for educational purposes.

The complex I/Q signal is modulated onto an audio carrier:
    audio(t) = I(t) * cos(2π * f_c * t) - Q(t) * sin(2π * f_c * t)

This preserves both I and Q components in a single real-valued audio signal.

Reference: Quadrature Amplitude Modulation over acoustic channel
"""

import numpy as np
from typing import Optional, Union, Callable
from pathlib import Path
import wave
import struct
import threading
import queue
import time


class AudioOutput:
    """
    Audio output for I/Q sample transmission.
    
    Converts complex baseband I/Q samples to audio for playback through
    speakers. The I/Q signal is upconverted to an audio carrier frequency.
    
    Attributes:
        sample_rate: Audio sample rate (44100 or 48000 Hz typical)
        carrier_freq: Center frequency for I/Q modulation in audio band
        bandwidth: Effective bandwidth of the I/Q signal
        
    Example:
        >>> audio_out = AudioOutput(sample_rate=48000, carrier_freq=8000)
        >>> audio_out.play(iq_samples)
        >>> 
        >>> # Or for streaming:
        >>> audio_out.start_stream()
        >>> audio_out.write(iq_chunk)
        >>> audio_out.stop_stream()
    """
    
    def __init__(self, 
                 sample_rate: int = 48000,
                 carrier_freq: float = 8000.0,
                 amplitude: float = 0.8,
                 buffer_size: int = 1024):
        """
        Initialize audio output.
        
        Args:
            sample_rate: Audio sample rate in Hz (44100 or 48000 typical)
            carrier_freq: Carrier frequency for I/Q modulation (Hz)
            amplitude: Output amplitude (0.0 to 1.0)
            buffer_size: Audio buffer size in samples
        """
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.amplitude = amplitude
        self.buffer_size = buffer_size
        
        # Maximum I/Q bandwidth is limited by carrier and Nyquist
        self.max_iq_bandwidth = min(carrier_freq, sample_rate / 2 - carrier_freq)
        
        # Streaming state
        self._stream = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._phase = 0.0
        
        # Try to import audio library
        self._audio_lib = None
        self._init_audio_library()
    
    def _init_audio_library(self) -> None:
        """Initialize audio output library (sounddevice or pyaudio)."""
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
        
        # No audio library available - will only support file output
        self._audio_lib = None
    
    def iq_to_audio(self, iq_samples: np.ndarray, 
                    iq_sample_rate: Optional[float] = None,
                    stereo: bool = True) -> np.ndarray:
        """
        Convert I/Q samples to audio signal.
        
        Args:
            iq_samples: Complex baseband I/Q samples
            iq_sample_rate: Sample rate of I/Q (for resampling). 
                           If None, assumes same as audio sample rate.
            stereo: If True, output stereo (I=left, Q=right) for full bandwidth.
                   If False, modulate onto carrier (limited bandwidth).
                           
        Returns:
            Audio samples - shape (N,2) if stereo, (N,) if mono
        """
        # Resample I/Q if needed
        if iq_sample_rate is not None and abs(iq_sample_rate - self.sample_rate) > 1.0:
            iq_samples = self._resample_iq(iq_samples, iq_sample_rate)
        
        # Normalize I/Q amplitude
        max_amp = np.max(np.abs(iq_samples))
        if max_amp > 0:
            iq_samples = iq_samples / max_amp
        
        i_component = np.real(iq_samples)
        q_component = np.imag(iq_samples)
        
        if stereo:
            # Stereo mode: I on left channel, Q on right channel
            # This preserves full bandwidth (up to Nyquist)
            audio = np.column_stack([
                i_component * self.amplitude,
                q_component * self.amplitude
            ]).astype(np.float32)
        else:
            # Mono mode: modulate onto carrier (limited bandwidth)
            n = len(iq_samples)
            t = np.arange(n) / self.sample_rate
            carrier_phase = 2.0 * np.pi * self.carrier_freq * t + self._phase
            
            audio = (i_component * np.cos(carrier_phase) - 
                     q_component * np.sin(carrier_phase))
            
            self._phase = (carrier_phase[-1] + 2.0 * np.pi * self.carrier_freq / 
                           self.sample_rate) % (2.0 * np.pi)
            
            audio = (audio * self.amplitude).astype(np.float32)
        
        return audio
    
    def _resample_iq(self, iq_samples: np.ndarray, 
                     from_rate: float) -> np.ndarray:
        """
        Resample I/Q to audio sample rate.
        
        Args:
            iq_samples: Input I/Q samples
            from_rate: Input sample rate
            
        Returns:
            Resampled I/Q samples
        """
        from scipy import signal
        
        # Calculate resampling ratio
        ratio = self.sample_rate / from_rate
        
        if ratio < 1:
            # Downsampling - need to apply anti-aliasing filter first
            # Design low-pass filter at new Nyquist frequency
            nyquist_new = self.sample_rate / 2
            nyquist_old = from_rate / 2
            cutoff = nyquist_new / nyquist_old * 0.9  # 90% of new Nyquist
            
            # Apply filter
            b, a = signal.butter(8, cutoff)
            iq_filtered = signal.filtfilt(b, a, iq_samples)
        else:
            iq_filtered = iq_samples
        
        # Resample
        num_samples = int(len(iq_samples) * ratio)
        resampled = signal.resample(iq_filtered, num_samples)
        
        return resampled.astype(np.complex64)
    
    def play(self, iq_samples: np.ndarray,
             iq_sample_rate: Optional[float] = None,
             blocking: bool = True) -> None:
        """
        Play I/Q samples through speaker.
        
        Args:
            iq_samples: Complex I/Q samples to play
            iq_sample_rate: I/Q sample rate (for resampling)
            blocking: Wait for playback to complete
        """
        if self._audio_lib is None:
            raise RuntimeError(
                "No audio library available. Install sounddevice or pyaudio:\n"
                "  pip install sounddevice\n"
                "  or: pip install pyaudio"
            )
        
        # Convert to audio
        audio = self.iq_to_audio(iq_samples, iq_sample_rate)
        
        if self._audio_lib == 'sounddevice':
            self._sd.play(audio, self.sample_rate)
            if blocking:
                self._sd.wait()
        
        elif self._audio_lib == 'pyaudio':
            stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(4),  # float32
                channels=1,
                rate=self.sample_rate,
                output=True
            )
            
            # Convert to bytes
            audio_bytes = audio.tobytes()
            stream.write(audio_bytes)
            
            if blocking:
                time.sleep(len(audio) / self.sample_rate)
            
            stream.stop_stream()
            stream.close()
    
    def start_stream(self, callback: Optional[Callable] = None) -> None:
        """
        Start audio output stream for real-time playback.
        
        Args:
            callback: Optional callback when buffer needs data
        """
        if self._audio_lib is None:
            raise RuntimeError("No audio library available")
        
        self._running = True
        self._audio_queue = queue.Queue()
        
        if self._audio_lib == 'sounddevice':
            def sd_callback(outdata, frames, time_info, status):
                try:
                    data = self._audio_queue.get_nowait()
                    if len(data) < frames:
                        outdata[:len(data), 0] = data
                        outdata[len(data):, 0] = 0
                    else:
                        outdata[:, 0] = data[:frames]
                except queue.Empty:
                    outdata[:] = 0
                    if callback:
                        callback()
            
            self._stream = self._sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.buffer_size,
                callback=sd_callback
            )
            self._stream.start()
        
        elif self._audio_lib == 'pyaudio':
            def pa_callback(in_data, frame_count, time_info, status):
                try:
                    data = self._audio_queue.get_nowait()
                    if len(data) < frame_count:
                        audio = np.zeros(frame_count, dtype=np.float32)
                        audio[:len(data)] = data
                    else:
                        audio = data[:frame_count]
                    return (audio.tobytes(), self._pyaudio.paContinue)
                except queue.Empty:
                    if callback:
                        callback()
                    return (np.zeros(frame_count, dtype=np.float32).tobytes(),
                            self._pyaudio.paContinue)
            
            self._stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(4),
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=pa_callback
            )
            self._stream.start_stream()
    
    def write(self, iq_samples: np.ndarray,
              iq_sample_rate: Optional[float] = None) -> None:
        """
        Write I/Q samples to stream buffer.
        
        Args:
            iq_samples: I/Q samples to queue for playback
            iq_sample_rate: Sample rate for resampling
        """
        if not self._running:
            raise RuntimeError("Stream not started. Call start_stream() first.")
        
        audio = self.iq_to_audio(iq_samples, iq_sample_rate)
        
        # Split into buffer-sized chunks
        for i in range(0, len(audio), self.buffer_size):
            chunk = audio[i:i + self.buffer_size]
            self._audio_queue.put(chunk)
    
    def stop_stream(self) -> None:
        """Stop audio output stream."""
        self._running = False
        
        if self._stream is not None:
            if self._audio_lib == 'sounddevice':
                self._stream.stop()
                self._stream.close()
            elif self._audio_lib == 'pyaudio':
                self._stream.stop_stream()
                self._stream.close()
            
            self._stream = None
    
    def write_wav(self, path: Union[str, Path], 
                  iq_samples: np.ndarray,
                  iq_sample_rate: Optional[float] = None,
                  stereo: bool = True) -> None:
        """
        Write I/Q samples to WAV file.
        
        Args:
            path: Output WAV file path
            iq_samples: I/Q samples to convert
            iq_sample_rate: I/Q sample rate for resampling
            stereo: If True, write stereo WAV (I=left, Q=right)
        """
        path = Path(path)
        
        # Convert to audio
        audio = self.iq_to_audio(iq_samples, iq_sample_rate, stereo=stereo)
        
        # Scale to 16-bit PCM, handling any NaN/inf values
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(2 if stereo else 1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_int16.tobytes())
    
    def get_effective_bandwidth(self) -> float:
        """
        Get effective I/Q bandwidth in audio mode.
        
        Returns:
            Maximum I/Q bandwidth in Hz
        """
        return self.max_iq_bandwidth
    
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


class AcousticDVBT:
    """
    Acoustic DVB-T transmitter.
    
    Wraps DVBTModulator to create audio-compatible I/Q output.
    Uses 'audio' bandwidth mode which runs DVB-T natively at 48 kHz.
    
    Example:
        >>> tx = AcousticDVBT()
        >>> tx.transmit(ts_data)  # Plays through speaker
        >>> tx.transmit_to_file(ts_data, 'output.wav')
    """
    
    def __init__(self, 
                 audio_sample_rate: int = 48000,
                 carrier_freq: float = 5000,
                 amplitude: float = 0.8,
                 mode: str = '2K',
                 constellation: str = 'QPSK',
                 code_rate: str = '1/2',
                 guard_interval: str = 'acoustic'):
        """
        Initialize acoustic DVB-T transmitter.
        
        Args:
            audio_sample_rate: Audio output sample rate
            carrier_freq: Audio carrier frequency (default 5kHz for acoustic mode)
            amplitude: Output amplitude (0-1)
            mode: DVB-T mode ('2K' or '8K')
            constellation: 'QPSK', '16QAM', or '64QAM'
            code_rate: FEC rate
            guard_interval: Guard interval ratio ('acoustic' = 10ms for speaker/mic)
        """
        self.audio_sample_rate = audio_sample_rate
        self.carrier_freq = carrier_freq
        
        # Create audio output
        self.audio_output = AudioOutput(
            sample_rate=audio_sample_rate,
            carrier_freq=carrier_freq,
            amplitude=amplitude
        )
        
        # Create modulator with 'audio' bandwidth
        # OFDM runs at 16 kHz, upsampled to 48 kHz for audio output
        from .DVB import DVBTModulator
        self.modulator = DVBTModulator(
            mode=mode,
            constellation=constellation,
            code_rate=code_rate,
            guard_interval=guard_interval,
            bandwidth='audio',
        )
        
        # DVB-T OFDM sample rate (16 kHz for audio mode)
        self.dvbt_sample_rate = self.modulator.get_sample_rate()
    
    def modulate(self, ts_data: bytes) -> np.ndarray:
        """
        Modulate transport stream to I/Q samples.
        
        Args:
            ts_data: Transport stream data
            
        Returns:
            Complex I/Q samples
        """
        return self.modulator.modulate(ts_data)
    
    def transmit(self, ts_data: bytes, blocking: bool = True) -> None:
        """
        Transmit transport stream through audio.
        
        Args:
            ts_data: Transport stream data
            blocking: Wait for playback to complete
        """
        iq_samples = self.modulate(ts_data)
        self.audio_output.play(
            iq_samples, 
            iq_sample_rate=self.dvbt_sample_rate,
            blocking=blocking
        )
    
    def transmit_to_file(self, ts_data: bytes, 
                         path: Union[str, Path],
                         stereo: bool = False) -> None:
        """
        Transmit to WAV file instead of speaker.
        
        Args:
            ts_data: Transport stream data
            path: Output WAV file path
            stereo: If True, write stereo I/Q (for virtual audio loopback).
                   If False (default), modulate onto carrier (for acoustic/speaker transmission).
        """
        iq_samples = self.modulate(ts_data)
        self.audio_output.write_wav(
            path, iq_samples,
            iq_sample_rate=self.dvbt_sample_rate,
            stereo=stereo
        )
    
    def close(self) -> None:
        """Clean up resources."""
        self.audio_output.close()


def iq_to_wav(iq_samples: np.ndarray,
              output_path: Union[str, Path],
              iq_sample_rate: float = 9142857.0,
              audio_sample_rate: int = 48000,
              carrier_freq: float = 5000) -> None:
    """
    Convenience function to convert I/Q samples to WAV file.
    
    Args:
        iq_samples: Complex I/Q samples
        output_path: Output WAV file path
        iq_sample_rate: Input I/Q sample rate
        audio_sample_rate: Output audio sample rate
        carrier_freq: Audio carrier frequency
    """
    audio_out = AudioOutput(
        sample_rate=audio_sample_rate,
        carrier_freq=carrier_freq
    )
    audio_out.write_wav(output_path, iq_samples, iq_sample_rate)

