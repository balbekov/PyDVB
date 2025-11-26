"""
DVB-T Command Line Interface

Usage:
    python -m dvb info <file.ts>           # Show TS file information
    python -m dvb demux <file.ts> -o dir/  # Extract elementary streams
    python -m dvb modulate <file.ts> -o output.cf32  # Modulate to I/Q
    
Examples:
    # Show transport stream information
    python -m dvb info sample.ts
    
    # Modulate with 64-QAM, code rate 2/3
    python -m dvb modulate sample.ts -o output.cf32 \\
        --mode 2K --constellation 64QAM --code-rate 2/3 --guard 1/4
    
    # Play on HackRF (requires hackrf_transfer)
    hackrf_transfer -t output.cs8 -f 474e6 -s 9142857 -x 40
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
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        'info': cmd_info,
        'demux': cmd_demux,
        'modulate': cmd_modulate,
        'generate': cmd_generate,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
