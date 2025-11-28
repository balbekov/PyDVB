#!/usr/bin/env python3
"""
Room Channel Calibration CLI

This script helps you measure and create a channel model of your room's
acoustic path (speaker -> air -> microphone) for use in DVB-T testing.

Workflow:
    1. Generate probe signal:
       python scripts/calibrate_room.py generate -o probe.wav
    
    2. Play probe.wav through your speaker while recording with your mic
       Save the recording as recorded.wav
    
    3. Analyze and create channel model:
       python scripts/calibrate_room.py analyze probe.wav recorded.wav -o room_channel.npz
    
    4. (Optional) View channel model info:
       python scripts/calibrate_room.py info room_channel.npz

Usage with tests:
    pytest tests/test_loopback.py::TestChannelModelLoopback -v
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def cmd_generate(args):
    """Generate probe signal for room calibration."""
    from tests.channel_model import generate_probe_signal, save_probe_wav
    
    print(f"Generating probe signal...")
    print(f"  Duration: {args.duration}s")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Frequency range: {args.f_start} - {args.f_end} Hz")
    
    probe, metadata = generate_probe_signal(
        duration=args.duration,
        sample_rate=args.sample_rate,
        f_start=args.f_start,
        f_end=args.f_end,
        amplitude=args.amplitude,
        silence_padding=args.padding,
    )
    
    save_probe_wav(probe, args.output, args.sample_rate)
    
    total_duration = len(probe) / args.sample_rate
    print(f"\nProbe signal saved to: {args.output}")
    print(f"  Total duration: {total_duration:.2f}s (including {args.padding}s padding)")
    print(f"  Samples: {len(probe)}")
    print(f"\nNext steps:")
    print(f"  1. Play {args.output} through your speaker")
    print(f"  2. Record with your microphone (save as WAV)")
    print(f"  3. Run: python scripts/calibrate_room.py analyze {args.output} <recording.wav>")


def cmd_analyze(args):
    """Analyze recorded signal and create channel model."""
    from tests.channel_model import RoomChannelModel
    
    print(f"Analyzing room channel...")
    print(f"  Probe: {args.probe}")
    print(f"  Recording: {args.recording}")
    
    # Calibrate channel model
    model = RoomChannelModel.calibrate(
        probe_wav=args.probe,
        recorded_wav=args.recording,
        verbose=True,
    )
    
    # Save model
    output_path = args.output or "room_channel.npz"
    model.save(output_path)
    print(f"\nChannel model saved to: {output_path}")
    
    # Show frequency response if matplotlib available
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Impulse response
            t_ms = np.arange(len(model.impulse_response)) / model.sample_rate * 1000
            axes[0].plot(t_ms, np.abs(model.impulse_response))
            axes[0].set_xlabel('Time (ms)')
            axes[0].set_ylabel('Magnitude')
            axes[0].set_title('Room Impulse Response')
            axes[0].grid(True, alpha=0.3)
            
            # Frequency response
            freqs, mag_db = model.get_frequency_response(2048)
            axes[1].plot(freqs / 1000, mag_db)
            axes[1].set_xlabel('Frequency (kHz)')
            axes[1].set_ylabel('Magnitude (dB)')
            axes[1].set_title('Channel Frequency Response')
            axes[1].set_xlim(0, model.sample_rate / 2000)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = Path(output_path).with_suffix('.png')
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to: {plot_path}")
            
            if not args.no_show:
                plt.show()
        except ImportError:
            print("(matplotlib not available - skipping plot)")
    
    print(f"\nTo use in tests:")
    print(f"  pytest tests/test_loopback.py::TestChannelModelLoopback -v")


def cmd_info(args):
    """Display information about a saved channel model."""
    from tests.channel_model import RoomChannelModel
    
    model = RoomChannelModel.load(args.model)
    
    print(f"Room Channel Model: {args.model}")
    print(f"  Sample rate: {model.sample_rate} Hz")
    print(f"  Propagation delay: {model.delay_samples / model.sample_rate * 1000:.2f} ms")
    print(f"  Delay spread: {model.delay_spread_ms:.2f} ms")
    print(f"  Estimated SNR: {model.snr_db:.1f} dB")
    print(f"  Impulse response length: {len(model.impulse_response)} samples "
          f"({len(model.impulse_response) / model.sample_rate * 1000:.1f} ms)")
    
    # Show frequency response stats
    freqs, mag_db = model.get_frequency_response(2048)
    print(f"\nFrequency response:")
    print(f"  DC gain: {mag_db[0]:.1f} dB")
    print(f"  Max gain: {np.max(mag_db):.1f} dB at {freqs[np.argmax(mag_db)]:.0f} Hz")
    print(f"  Min gain: {np.min(mag_db):.1f} dB at {freqs[np.argmin(mag_db)]:.0f} Hz")
    print(f"  Variation: {np.max(mag_db) - np.min(mag_db):.1f} dB")


def cmd_test(args):
    """Test the channel model with a simple signal."""
    from tests.channel_model import RoomChannelModel, ChannelModel
    
    model = RoomChannelModel.load(args.model)
    print(f"Testing channel model: {args.model}")
    
    # Generate test signal (tone)
    duration = 0.1  # 100ms
    sample_rate = model.sample_rate
    t = np.arange(int(duration * sample_rate)) / sample_rate
    test_signal = np.exp(2j * np.pi * 1000 * t).astype(np.complex64)  # 1 kHz tone
    
    print(f"\nInput signal:")
    print(f"  Duration: {duration*1000:.0f} ms")
    print(f"  Frequency: 1000 Hz")
    print(f"  Power: {10*np.log10(np.mean(np.abs(test_signal)**2)):.1f} dB")
    
    # Apply channel
    output = model.apply(test_signal)
    
    print(f"\nOutput signal:")
    print(f"  Power: {10*np.log10(np.mean(np.abs(output)**2)):.1f} dB")
    print(f"  Gain: {10*np.log10(np.mean(np.abs(output)**2) / np.mean(np.abs(test_signal)**2)):.1f} dB")
    
    # Compare with simulated channel
    sim_channel = ChannelModel(sample_rate)
    sim_output = sim_channel.add_awgn(test_signal, model.snr_db)
    
    print(f"\nComparison with simulated AWGN channel at {model.snr_db:.1f} dB:")
    print(f"  Simulated output power: {10*np.log10(np.mean(np.abs(sim_output)**2)):.1f} dB")


def main():
    parser = argparse.ArgumentParser(
        description='Room Channel Calibration for DVB-T Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate probe signal')
    gen_parser.add_argument('-o', '--output', default='probe.wav',
                           help='Output WAV file path (default: probe.wav)')
    gen_parser.add_argument('-d', '--duration', type=float, default=2.0,
                           help='Chirp duration in seconds (default: 2.0)')
    gen_parser.add_argument('-r', '--sample-rate', type=int, default=48000,
                           help='Sample rate in Hz (default: 48000)')
    gen_parser.add_argument('--f-start', type=float, default=200.0,
                           help='Start frequency in Hz (default: 200)')
    gen_parser.add_argument('--f-end', type=float, default=16000.0,
                           help='End frequency in Hz (default: 16000)')
    gen_parser.add_argument('-a', '--amplitude', type=float, default=0.8,
                           help='Signal amplitude 0-1 (default: 0.8)')
    gen_parser.add_argument('-p', '--padding', type=float, default=0.5,
                           help='Silence padding in seconds (default: 0.5)')
    gen_parser.set_defaults(func=cmd_generate)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze recording and create model')
    analyze_parser.add_argument('probe', help='Path to probe WAV file')
    analyze_parser.add_argument('recording', help='Path to recorded WAV file')
    analyze_parser.add_argument('-o', '--output', help='Output model file (default: room_channel.npz)')
    analyze_parser.add_argument('--plot', action='store_true',
                               help='Generate impulse/frequency response plot')
    analyze_parser.add_argument('--no-show', action='store_true',
                               help='Save plot but do not display')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show channel model information')
    info_parser.add_argument('model', help='Path to channel model NPZ file')
    info_parser.set_defaults(func=cmd_info)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test channel model with signal')
    test_parser.add_argument('model', help='Path to channel model NPZ file')
    test_parser.set_defaults(func=cmd_test)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()

