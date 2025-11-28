"""
DVB-T Command Line Interface

Usage:
    python -m dvb info <file.ts>           # Show TS file information
    python -m dvb demux <file.ts> -o dir/  # Extract elementary streams
    python -m dvb modulate <file.ts> -o output.cf32  # Modulate to I/Q
    python -m dvb audio-tx <file.ts>       # Transmit via audio speaker
    python -m dvb audio-rx -d 10           # Receive from microphone for 10s
    python -m dvb send-image photo.jpg     # Send image via audio
    python -m dvb receive-image -d 10      # Receive image from audio
    
Examples:
    # Show transport stream information
    python -m dvb info sample.ts
    
    # Modulate with 64-QAM, code rate 2/3
    python -m dvb modulate sample.ts -o output.cf32 \\
        --mode 2K --constellation 64QAM --code-rate 2/3 --guard 1/4
    
    # Play on HackRF (requires hackrf_transfer)
    hackrf_transfer -t output.cs8 -f 474e6 -s 9142857 -x 40
    
    # Transmit transport stream via audio
    python -m dvb audio-tx sample.ts --output audio.wav  # Save to file
    python -m dvb audio-tx sample.ts --play              # Play through speaker
    
    # Receive from microphone and decode
    python -m dvb audio-rx -d 10 -o received.ts          # Record 10 seconds
    python -m dvb audio-rx -i recording.wav -o out.ts    # Decode WAV file
    
    # Send image via acoustic channel
    python -m dvb send-image photo.jpg --output audio.wav  # Save to WAV
    python -m dvb send-image photo.jpg --play              # Play through speaker
    
    # Receive image via acoustic channel
    python -m dvb receive-image -d 15 -o received.jpg      # Record 15 seconds
    python -m dvb receive-image -i audio.wav -o out.jpg    # Decode WAV file
    
    # Test image loopback (TX -> WAV -> RX)
    python -m dvb image-loopback photo.jpg -o recovered.jpg
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__


def cmd_info(args: argparse.Namespace) -> int:
    """Show transport stream information."""
    from .TransportStream import TransportStream
    
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1
    
    print(f"Analyzing: {path}")
    print(f"File size: {path.stat().st_size:,} bytes")
    print()
    
    try:
        ts = TransportStream.from_file(path)
        ts.parse_tables()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1
    
    analysis = ts.analyze()
    
    print(f"Total packets: {analysis['total_packets']:,}")
    print(f"Total bytes: {analysis['total_bytes']:,}")
    print()
    
    # Show PAT info
    if ts.pat:
        print("Program Association Table (PAT):")
        print(f"  Transport Stream ID: {ts.pat.transport_stream_id}")
        print(f"  Programs: {list(ts.pat.programs.keys())}")
        print()
    
    # Show PMT info
    if ts.pmts:
        print("Program Map Tables (PMT):")
        for prog_num, pmt in ts.pmts.items():
            print(f"  Program {prog_num}:")
            print(f"    PCR PID: 0x{pmt.pcr_pid:04X}")
            for es in pmt.streams:
                print(f"    Stream: PID=0x{es.pid:04X}, Type={es.type_description}")
        print()
    
    # Show PID summary
    print("PID Summary:")
    print(f"  {'PID':>6}  {'Packets':>10}  {'%':>6}  {'Description':<30}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*30}")
    
    for pid, info in sorted(analysis['pids'].items()):
        desc = info['description'] or ""
        if pid == 0x1FFF:
            desc = "Null (stuffing)"
        elif pid == 0x0000:
            desc = "PAT"
        
        print(f"  0x{pid:04X}  {info['packets']:10,}  {info['percent']:5.1f}%  {desc}")
    
    return 0


def cmd_demux(args: argparse.Namespace) -> int:
    """Demultiplex transport stream."""
    from .TransportStream import TransportStream
    
    input_path = Path(args.file)
    output_dir = Path(args.output) if args.output else input_path.parent
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ts = TransportStream.from_file(input_path)
    ts.parse_tables()
    
    # Extract each PID
    pid_data = {}
    for pkt in ts:
        pid = pkt.pid
        if pid == 0x1FFF:  # Skip null packets
            continue
        
        if pid not in pid_data:
            pid_data[pid] = bytearray()
        
        pid_data[pid].extend(pkt.payload)
    
    # Write files
    for pid, data in pid_data.items():
        info = ts.get_pid_info(pid)
        desc = info.description if info else ""
        
        # Determine extension
        if "Video" in desc:
            ext = ".264" if "H.264" in desc else ".mpv"
        elif "Audio" in desc:
            ext = ".mpa"
        else:
            ext = ".bin"
        
        output_file = output_dir / f"pid_{pid:04x}{ext}"
        
        with open(output_file, 'wb') as f:
            f.write(data)
        
        print(f"Wrote: {output_file} ({len(data):,} bytes)")
    
    return 0


def cmd_modulate(args: argparse.Namespace) -> int:
    """Modulate transport stream to I/Q samples."""
    from .DVB import DVBTModulator
    from .IQWriter import detect_format
    
    input_path = Path(args.file)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1
    
    # Read input
    print(f"Reading: {input_path}")
    with open(input_path, 'rb') as f:
        ts_data = f.read()
    
    print(f"Input size: {len(ts_data):,} bytes")
    print(f"Packets: {len(ts_data) // 188:,}")
    print()
    
    # Configure modulator
    print("DVB-T Parameters:")
    print(f"  Mode: {args.mode}")
    print(f"  Constellation: {args.constellation}")
    print(f"  Code rate: {args.code_rate}")
    print(f"  Guard interval: {args.guard}")
    print(f"  Bandwidth: {args.bandwidth}")
    print()
    
    modulator = DVBTModulator(
        mode=args.mode,
        constellation=args.constellation,
        code_rate=args.code_rate,
        guard_interval=args.guard,
        bandwidth=args.bandwidth,
    )
    
    print(f"Sample rate: {modulator.get_sample_rate():,.0f} Hz")
    print(f"Net data rate: {modulator.get_data_rate():,.0f} bps")
    print()
    
    # Modulate
    print("Modulating...")
    iq_samples = modulator.modulate(ts_data)
    
    print(f"Generated {len(iq_samples):,} I/Q samples")
    print(f"Duration: {len(iq_samples) / modulator.get_sample_rate():.3f} seconds")
    print()
    
    # Write output
    output_format = detect_format(output_path)
    print(f"Writing: {output_path} (format: {output_format})")
    modulator.write_iq(output_path, iq_samples, output_format)
    
    print("Done!")
    
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate test transport stream."""
    from .TransportStream import TransportStream
    from .Packet import Packet
    from .PAT import PAT
    from .PMT import PMT, StreamType
    
    output_path = Path(args.output)
    num_packets = args.packets
    
    ts = TransportStream()
    
    # Create PAT
    pat = PAT(transport_stream_id=1)
    pat.add_program(1, 0x100)  # Program 1, PMT on PID 0x100
    
    # Create PMT
    pmt = PMT(program_number=1, pcr_pid=0x101)
    pmt.add_stream(StreamType.H264_VIDEO, 0x101)
    pmt.add_stream(StreamType.AAC_AUDIO, 0x102)
    
    # Add PAT packet
    pat_data = pat.to_bytes()
    ts.append(Packet(
        pid=0x0000,
        payload_unit_start=True,
        payload=bytes([0]) + pat_data,  # Pointer field + PAT
        continuity_counter=0,
    ))
    
    # Add PMT packet
    pmt_data = pmt.to_bytes()
    ts.append(Packet(
        pid=0x100,
        payload_unit_start=True,
        payload=bytes([0]) + pmt_data,
        continuity_counter=0,
    ))
    
    # Add dummy data packets
    video_cc = 0
    audio_cc = 0
    
    for i in range(num_packets - 2):
        if i % 10 == 0:  # Audio every 10th packet
            ts.append(Packet(
                pid=0x102,
                payload=bytes([i % 256] * 184),
                continuity_counter=audio_cc,
            ))
            audio_cc = (audio_cc + 1) % 16
        else:
            ts.append(Packet(
                pid=0x101,
                payload=bytes([i % 256] * 184),
                continuity_counter=video_cc,
            ))
            video_cc = (video_cc + 1) % 16
    
    # Write
    ts.to_file(output_path)
    print(f"Wrote: {output_path} ({len(ts)} packets)")
    
    return 0


def cmd_audio_tx(args: argparse.Namespace) -> int:
    """Transmit transport stream via audio."""
    from .AudioOutput import AudioOutput, AcousticDVBT
    
    input_path = Path(args.file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1
    
    # Read input
    print(f"Reading: {input_path}")
    with open(input_path, 'rb') as f:
        ts_data = f.read()
    
    print(f"Input size: {len(ts_data):,} bytes")
    print(f"Packets: {len(ts_data) // 188:,}")
    print()
    
    # Configure acoustic transmitter
    print("Acoustic DVB-T Parameters:")
    print(f"  Audio sample rate: {args.sample_rate} Hz")
    print(f"  Carrier frequency: {args.carrier} Hz")
    print(f"  Mode: QPSK, Code rate 1/2 (robust)")
    print()
    
    # Create transmitter
    tx = AcousticDVBT(
        audio_sample_rate=args.sample_rate,
        carrier_freq=args.carrier,
        amplitude=args.amplitude
    )
    
    if args.output:
        # Write to WAV file
        output_path = Path(args.output)
        print(f"Modulating to audio file: {output_path}")
        tx.transmit_to_file(ts_data, output_path)
        print(f"Wrote: {output_path}")
    
    if args.play:
        # Play through speaker
        print("Playing through speaker...")
        try:
            tx.transmit(ts_data, blocking=True)
            print("Playback complete.")
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Install sounddevice: pip install sounddevice", file=sys.stderr)
            return 1
    
    if not args.output and not args.play:
        print("No output specified. Use --output FILE.wav or --play")
        return 1
    
    tx.close()
    return 0


def cmd_audio_rx(args: argparse.Namespace) -> int:
    """Receive DVB-T from audio input."""
    from .AudioInput import AudioInput, AcousticDVBTReceiver
    
    # Create receiver
    rx = AcousticDVBTReceiver(
        audio_sample_rate=args.sample_rate,
        carrier_freq=args.carrier
    )
    
    print("Acoustic DVB-T Receiver")
    print(f"  Audio sample rate: {args.sample_rate} Hz")
    print(f"  Carrier frequency: {args.carrier} Hz")
    print()
    
    if args.input:
        # Decode from WAV file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            return 1
        
        print(f"Decoding from: {input_path}")
        ts_data, stats = rx.receive_file(input_path)
    
    else:
        # Record from microphone
        duration = args.duration
        if duration <= 0:
            print("Error: Duration must be positive", file=sys.stderr)
            return 1
        
        print(f"Recording for {duration} seconds...")
        try:
            ts_data, stats = rx.receive(duration)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Install sounddevice: pip install sounddevice", file=sys.stderr)
            return 1
    
    # Show stats
    print()
    print("Reception Statistics:")
    print(f"  Symbols processed: {stats.get('symbols', 0):,}")
    print(f"  RS errors corrected: {stats.get('rs_errors', 0)}")
    print(f"  RS uncorrectable: {stats.get('rs_uncorrectable', 0)}")
    print(f"  SNR estimate: {stats.get('snr_db', 0):.1f} dB")
    print(f"  Packets recovered: {stats.get('packets_recovered', 0)}")
    print(f"  Data recovered: {len(ts_data):,} bytes")
    print()
    
    # Write output
    if args.output and len(ts_data) > 0:
        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            f.write(ts_data)
        print(f"Wrote: {output_path}")
    elif len(ts_data) == 0:
        print("No data recovered. Signal may be too weak or corrupted.")
    
    rx.close()
    return 0


def cmd_audio_loopback(args: argparse.Namespace) -> int:
    """Test audio loopback (TX -> file -> RX)."""
    from .AudioOutput import AcousticDVBT
    from .AudioInput import AcousticDVBTReceiver
    import tempfile
    
    # Generate test data
    print("Generating test transport stream...")
    from .TransportStream import TransportStream
    from .Packet import Packet
    
    ts = TransportStream()
    for i in range(args.packets):
        ts.append(Packet(
            pid=0x100,
            payload=bytes([(i * j) % 256 for j in range(184)]),
            continuity_counter=i % 16,
        ))
    
    ts_data = ts.to_bytes()
    print(f"Generated {len(ts_data)} bytes ({args.packets} packets)")
    
    # Create temp WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = Path(tmp.name)
    
    try:
        # Transmit to WAV
        print(f"\nTransmitting to: {wav_path}")
        tx = AcousticDVBT(
            audio_sample_rate=args.sample_rate,
            carrier_freq=args.carrier
        )
        tx.transmit_to_file(ts_data, wav_path)
        tx.close()
        
        print(f"WAV file size: {wav_path.stat().st_size:,} bytes")
        
        # Receive from WAV
        print("\nReceiving from WAV file...")
        rx = AcousticDVBTReceiver(
            audio_sample_rate=args.sample_rate,
            carrier_freq=args.carrier
        )
        recovered_data, stats = rx.receive_file(wav_path)
        rx.close()
        
        # Compare
        print("\nResults:")
        print(f"  Original:  {len(ts_data):,} bytes")
        print(f"  Recovered: {len(recovered_data):,} bytes")
        print(f"  RS errors: {stats.get('rs_errors', 0)}")
        print(f"  RS uncorrectable: {stats.get('rs_uncorrectable', 0)}")
        
        if len(recovered_data) > 0:
            # Check how much matches
            match_len = min(len(ts_data), len(recovered_data))
            matches = sum(1 for i in range(match_len) 
                         if ts_data[i] == recovered_data[i])
            print(f"  Byte match: {matches}/{match_len} ({100*matches/match_len:.1f}%)")
        
    finally:
        # Clean up
        if wav_path.exists():
            wav_path.unlink()
    
    return 0


def cmd_send_image(args: argparse.Namespace) -> int:
    """Send image via audio using DVB-T audio mode."""
    from .ImageTransport import send_image_audio
    
    image_path = Path(args.file)
    
    if not image_path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        return 1
    
    print(f"Image: {image_path}")
    print(f"Size: {image_path.stat().st_size:,} bytes")
    print()
    
    print("DVB-T Audio Mode Parameters:")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Mode: audio (64-pt FFT, 24 carriers, 6kHz BW)")
    print(f"  Modulation: QPSK with FEC")
    print()
    
    if args.output:
        output_path = Path(args.output)
        print(f"Writing to: {output_path}")
        info = send_image_audio(image_path, output_path)
        print(f"WAV file written: {output_path}")
        print(f"Image data: {info.get('image_size', 0):,} bytes")
    
    if args.play:
        print("Playing through speaker...")
        try:
            # Read image and transmit
            from .ImageTransport import image_to_ts
            from .AudioOutput import AcousticDVBT
            import sounddevice as sd
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Convert to transport stream
            ts_data = image_to_ts(image_data)
            
            # Create DVB-T audio transmitter
            tx = AcousticDVBT(audio_sample_rate=args.sample_rate)
            iq, _ = tx.modulate(ts_data)
            audio = tx.audio_output._convert_iq_to_stereo_audio(iq)
            
            # Play via sounddevice
            sd.play(audio, args.sample_rate)
            sd.wait()
            print("Playback complete.")
            tx.close()
        except ImportError:
            print("Error: sounddevice not installed. Use --output instead.", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    if not args.output and not args.play:
        print("No output specified. Use --output FILE.wav or --play")
        return 1
    
    return 0


def cmd_receive_image(args: argparse.Namespace) -> int:
    """Receive image via audio using DVB-T audio mode."""
    from .ImageTransport import receive_image_audio
    
    print("DVB-T Audio Mode Receiver")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Carrier: {args.carrier} Hz")
    print(f"  Mode: audio (64-pt FFT, 6kHz BW)")
    print()
    
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            return 1
        print(f"Decoding from: {input_path}")
        image_data, stats = receive_image_audio(input_path, carrier_freq=args.carrier)
    else:
        duration = args.duration
        print(f"Recording for {duration} seconds...")
        try:
            from .AudioInput import AcousticDVBTReceiver
            from .ImageTransport import ts_to_image
            
            # Use AcousticDVBTReceiver which handles channel detection
            rx = AcousticDVBTReceiver(audio_sample_rate=args.sample_rate, carrier_freq=args.carrier)
            ts_data, stats = rx.receive(duration)
            rx.close()
            
            # Extract image from TS
            if len(ts_data) > 0:
                result = ts_to_image(ts_data)
                image_data = result[0] if isinstance(result, tuple) else result
            else:
                image_data = None
        except ImportError:
            print("Error: sounddevice not installed. Use -i FILE.wav instead.", file=sys.stderr)
            return 1
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Show results
    print()
    print("Reception Results:")
    print(f"  OFDM symbols: {stats.get('symbols', 0):,}")
    print(f"  RS errors corrected: {stats.get('rs_errors', 0)}")
    print(f"  RS uncorrectable: {stats.get('rs_uncorrectable', 0)}")
    print(f"  SNR estimate: {stats.get('snr_db', 0):.1f} dB")
    print(f"  Packets recovered: {stats.get('packets_recovered', 0)}")
    
    # Save if requested
    if args.output and image_data:
        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print()
        print(f"Saved to: {output_path} ({len(image_data):,} bytes)")
    elif image_data is None:
        print()
        print("No image data recovered. Check audio quality and sync.")
    
    return 0


def cmd_audio_rx_debug(args: argparse.Namespace) -> int:
    """Receive DVB-T from audio with real-time debug dashboard."""
    try:
        from .dashboard import run_debug_dashboard
    except ImportError as e:
        print(f"Error: Rich library required for dashboard: {e}", file=sys.stderr)
        print("Install with: pip install rich", file=sys.stderr)
        return 1
    
    print("Starting DVB-T Audio Debug Dashboard...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        ts_data, stats = run_debug_dashboard(
            input_file=args.input,
            duration=args.duration,
            sample_rate=args.sample_rate,
            carrier_freq=args.carrier,
            output_file=args.output,
            output_image=getattr(args, 'output_image', None),
            save_audio=getattr(args, 'save_audio', None),
        )
        
        # Print final summary
        print()
        print("Final Statistics:")
        print(f"  Symbols processed: {stats.get('symbols', 0):,}")
        print(f"  RS errors corrected: {stats.get('rs_errors', 0)}")
        print(f"  RS uncorrectable: {stats.get('rs_uncorrectable', 0)}")
        print(f"  Packets recovered: {stats.get('packets_recovered', 0)}")
        print(f"  Data recovered: {len(ts_data):,} bytes")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nStopped by user")
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_image_loopback(args: argparse.Namespace) -> int:
    """Test image send/receive loopback using DVB-T audio mode."""
    from .ImageTransport import send_image_audio, receive_image_audio
    import tempfile
    
    image_path = Path(args.file)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}", file=sys.stderr)
        return 1
    
    print(f"Image: {image_path}")
    print(f"Size: {image_path.stat().st_size:,} bytes")
    print()
    print("DVB-T Audio Mode Loopback Test")
    print("  Mode: audio (64-pt FFT, 24 carriers, 6kHz BW)")
    print("  Modulation: QPSK with full DVB-T FEC chain")
    
    # Read original image
    with open(image_path, 'rb') as f:
        original_data = f.read()
    
    # Create temp WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = Path(tmp.name)
    
    try:
        # Transmit to WAV
        print(f"\nTransmitting via DVB-T audio mode...")
        info = send_image_audio(image_path, wav_path)
        print(f"WAV file: {wav_path.stat().st_size:,} bytes")
        
        # Receive from WAV
        print("\nReceiving from audio...")
        recovered_data, stats = receive_image_audio(wav_path)
        
        print("\nDVB-T Reception Statistics:")
        print(f"  OFDM symbols: {stats.get('symbols', 0)}")
        print(f"  RS errors corrected: {stats.get('rs_errors', 0)}")
        print(f"  RS uncorrectable: {stats.get('rs_uncorrectable', 0)}")
        print(f"  SNR estimate: {stats.get('snr_db', 0):.1f} dB")
        print(f"  Packets recovered: {stats.get('packets_recovered', 0)}")
        
        if recovered_data:
            match = recovered_data == original_data
            print(f"  Data match: {match}")
            
            if match:
                print("\n  SUCCESS: Image transmitted via DVB-T audio and received correctly!")
            else:
                # Show byte match rate
                match_len = min(len(original_data), len(recovered_data))
                matches = sum(1 for i in range(match_len)
                             if original_data[i] == recovered_data[i])
                print(f"  Byte match: {matches}/{match_len} ({100*matches/match_len:.1f}%)")
            
            # Save recovered image
            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'wb') as f:
                    f.write(recovered_data)
                print(f"\nSaved recovered image to: {output_path}")
        else:
            print("\n  FAILED: Could not recover image data")
        
    finally:
        if wav_path.exists():
            wav_path.unlink()
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='dvb',
        description='PyDVB - Educational DVB-T Implementation',
    )
    parser.add_argument('--version', action='version', 
                       version=f'PyDVB {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show TS file information')
    info_parser.add_argument('file', help='Transport stream file')
    
    # Demux command
    demux_parser = subparsers.add_parser('demux', help='Demultiplex TS file')
    demux_parser.add_argument('file', help='Transport stream file')
    demux_parser.add_argument('-o', '--output', help='Output directory')
    
    # Modulate command
    mod_parser = subparsers.add_parser('modulate', help='Modulate TS to I/Q')
    mod_parser.add_argument('file', help='Transport stream file')
    mod_parser.add_argument('-o', '--output', required=True, help='Output I/Q file')
    mod_parser.add_argument('--mode', choices=['2K', '8K'], default='2K',
                           help='OFDM mode (default: 2K)')
    mod_parser.add_argument('--constellation', choices=['QPSK', '16QAM', '64QAM'],
                           default='QPSK', help='Constellation (default: QPSK)')
    mod_parser.add_argument('--code-rate', choices=['1/2', '2/3', '3/4', '5/6', '7/8'],
                           default='1/2', help='Code rate (default: 1/2)')
    mod_parser.add_argument('--guard', choices=['1/4', '1/8', '1/16', '1/32'],
                           default='1/4', help='Guard interval (default: 1/4)')
    mod_parser.add_argument('--bandwidth', choices=['6MHz', '7MHz', '8MHz'],
                           default='8MHz', help='Bandwidth (default: 8MHz)')
    
    # Generate command (for testing)
    gen_parser = subparsers.add_parser('generate', help='Generate test TS file')
    gen_parser.add_argument('-o', '--output', required=True, help='Output TS file')
    gen_parser.add_argument('-n', '--packets', type=int, default=1000,
                           help='Number of packets (default: 1000)')
    
    # Audio TX command
    audio_tx_parser = subparsers.add_parser('audio-tx', 
                                            help='Transmit TS via audio speaker')
    audio_tx_parser.add_argument('file', help='Transport stream file')
    audio_tx_parser.add_argument('-o', '--output', help='Output WAV file')
    audio_tx_parser.add_argument('--play', action='store_true',
                                help='Play through speaker')
    audio_tx_parser.add_argument('--sample-rate', type=int, default=48000,
                                help='Audio sample rate (default: 48000)')
    audio_tx_parser.add_argument('--carrier', type=float, default=5000,
                                help='Carrier frequency (default: 5000)')
    audio_tx_parser.add_argument('--amplitude', type=float, default=0.8,
                                help='Output amplitude 0-1 (default: 0.8)')
    
    # Audio RX command
    audio_rx_parser = subparsers.add_parser('audio-rx',
                                            help='Receive DVB-T from microphone')
    audio_rx_parser.add_argument('-i', '--input', help='Input WAV file (instead of mic)')
    audio_rx_parser.add_argument('-o', '--output', help='Output TS file')
    audio_rx_parser.add_argument('-d', '--duration', type=float, default=5.0,
                                help='Recording duration in seconds (default: 5)')
    audio_rx_parser.add_argument('--sample-rate', type=int, default=48000,
                                help='Audio sample rate (default: 48000)')
    audio_rx_parser.add_argument('--carrier', type=float, default=5000,
                                help='Carrier frequency (default: 5000)')
    
    # Audio loopback test
    loopback_parser = subparsers.add_parser('audio-loopback',
                                            help='Test audio TX->WAV->RX loopback')
    loopback_parser.add_argument('-n', '--packets', type=int, default=100,
                                help='Number of test packets (default: 100)')
    loopback_parser.add_argument('--sample-rate', type=int, default=48000,
                                help='Audio sample rate (default: 48000)')
    loopback_parser.add_argument('--carrier', type=float, default=5000,
                                help='Carrier frequency (default: 5000)')
    
    # Send image via audio
    send_img_parser = subparsers.add_parser('send-image',
                                            help='Send image via audio')
    send_img_parser.add_argument('file', help='Image file (JPG, PNG, etc.)')
    send_img_parser.add_argument('-o', '--output', help='Output WAV file')
    send_img_parser.add_argument('--play', action='store_true',
                                help='Play through speaker')
    send_img_parser.add_argument('--sample-rate', type=int, default=48000,
                                help='Audio sample rate (default: 48000)')
    send_img_parser.add_argument('--carrier', type=float, default=5000,
                                help='Carrier frequency (default: 5000)')
    
    # Receive image via audio
    recv_img_parser = subparsers.add_parser('receive-image',
                                            help='Receive image via audio')
    recv_img_parser.add_argument('-i', '--input', help='Input WAV file')
    recv_img_parser.add_argument('-o', '--output', help='Output image file')
    recv_img_parser.add_argument('-d', '--duration', type=float, default=10.0,
                                help='Recording duration (default: 10)')
    recv_img_parser.add_argument('--sample-rate', type=int, default=48000,
                                help='Audio sample rate (default: 48000)')
    recv_img_parser.add_argument('--carrier', type=float, default=5000,
                                help='Carrier frequency (default: 5000 for laptop speakers)')
    
    # Image loopback test
    img_loopback_parser = subparsers.add_parser('image-loopback',
                                                help='Test image TX->WAV->RX')
    img_loopback_parser.add_argument('file', help='Image file to test')
    img_loopback_parser.add_argument('-o', '--output', help='Save recovered image')
    img_loopback_parser.add_argument('--sample-rate', type=int, default=48000,
                                    help='Audio sample rate (default: 48000)')
    img_loopback_parser.add_argument('--carrier', type=float, default=5000,
                                    help='Carrier frequency (default: 5000 for laptop speakers)')
    
    # Audio RX with debug dashboard
    audio_rx_debug_parser = subparsers.add_parser('audio-rx-debug',
                                                  help='Receive DVB-T with debug dashboard')
    audio_rx_debug_parser.add_argument('-i', '--input', help='Input WAV file (instead of mic)')
    audio_rx_debug_parser.add_argument('-o', '--output', help='Output TS file')
    audio_rx_debug_parser.add_argument('--output-image', help='Output image file (auto-saves if not specified)')
    audio_rx_debug_parser.add_argument('--save-audio', help='Save input audio to WAV file')
    audio_rx_debug_parser.add_argument('-d', '--duration', type=float, default=30.0,
                                      help='Recording duration in seconds (default: 30)')
    audio_rx_debug_parser.add_argument('--sample-rate', type=int, default=48000,
                                      help='Audio sample rate (default: 48000)')
    audio_rx_debug_parser.add_argument('--carrier', type=float, default=5000,
                                      help='Carrier frequency (default: 5000)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        'info': cmd_info,
        'demux': cmd_demux,
        'modulate': cmd_modulate,
        'generate': cmd_generate,
        'audio-tx': cmd_audio_tx,
        'audio-rx': cmd_audio_rx,
        'audio-loopback': cmd_audio_loopback,
        'audio-rx-debug': cmd_audio_rx_debug,
        'send-image': cmd_send_image,
        'receive-image': cmd_receive_image,
        'image-loopback': cmd_image_loopback,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
