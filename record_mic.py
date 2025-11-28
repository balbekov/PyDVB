#!/usr/bin/env python3
"""
Record microphone input to a WAV file.

Usage:
    python record_mic.py [output.wav] [duration_seconds]
    
Examples:
    python record_mic.py                    # Record to recording.wav for 5 seconds
    python record_mic.py my_audio.wav       # Record to my_audio.wav for 5 seconds
    python record_mic.py my_audio.wav 10    # Record to my_audio.wav for 10 seconds
"""

import sys
import wave
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Install with: pip install sounddevice")
    sys.exit(1)


def record_to_wav(output_path: str = "recording.wav", 
                  duration: float = 5.0,
                  sample_rate: int = 48000,
                  channels: int = 1) -> None:
    """
    Record audio from microphone and save to WAV file.
    
    Args:
        output_path: Output WAV file path
        duration: Recording duration in seconds
        sample_rate: Audio sample rate in Hz
        channels: Number of channels (1=mono, 2=stereo)
    """
    print(f"Recording {duration}s of audio to '{output_path}'...")
    print(f"Sample rate: {sample_rate} Hz, Channels: {channels}")
    print("Recording... ", end="", flush=True)
    
    # Record audio
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype=np.int16
    )
    sd.wait()
    print("done!")
    
    # Save to WAV file
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    
    print(f"Saved {len(audio)} samples ({duration:.1f}s) to '{output_path}'")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "recording.wav"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    
    record_to_wav(output, duration)

