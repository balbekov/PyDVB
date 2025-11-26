"""
MPEG-2 Transport Stream Container

A Transport Stream is a container format for multiplexing audio, video,
and data streams. It consists of 188-byte packets with Program Specific
Information (PSI) tables that describe the stream structure.

Features:
- Multiple programs (services) per stream
- Error resilience (each packet is independent)
- Constant bit rate support with null packet stuffing
- Timing via PCR (Program Clock Reference)

Reference: ISO/IEC 13818-1 (MPEG-2 Systems)
"""

import numpy as np
from typing import Iterator, List, Dict, Optional, BinaryIO, Union
from dataclasses import dataclass, field
from pathlib import Path

from .Packet import Packet, PACKET_SIZE, PID_PAT, PID_NULL
from .PAT import PAT
from .PMT import PMT


@dataclass
class PIDInfo:
    """Information about a single PID in the stream."""
    pid: int
    packet_count: int = 0
    continuity_errors: int = 0
    last_cc: int = -1
    stream_type: Optional[int] = None
    description: str = ""


class TransportStream:
    """
    MPEG-2 Transport Stream container.
    
    This class represents a collection of TS packets and provides
    methods for reading, writing, and analyzing transport streams.
    
    Attributes:
        packets: List of Packet objects
        pat: Program Association Table (if parsed)
        pmts: Dict of Program Map Tables keyed by program number
        
    Example:
        >>> ts = TransportStream.from_file('input.ts')
        >>> print(f"Packets: {len(ts)}")
        >>> for pkt in ts.iter_pid(0x100):
        ...     print(pkt)
    """
    
    def __init__(self, packets: Optional[List[Packet]] = None):
        """
        Initialize TransportStream.
        
        Args:
            packets: Initial list of packets (optional)
        """
        self.packets: List[Packet] = packets or []
        self.pat: Optional[PAT] = None
        self.pmts: Dict[int, PMT] = {}
        self._pid_info: Dict[int, PIDInfo] = {}
    
    def __len__(self) -> int:
        """Return number of packets."""
        return len(self.packets)
    
    def __iter__(self) -> Iterator[Packet]:
        """Iterate over all packets."""
        return iter(self.packets)
    
    def __getitem__(self, index: int) -> Packet:
        """Get packet by index."""
        return self.packets[index]
    
    def append(self, packet: Packet) -> None:
        """Add a packet to the stream."""
        self.packets.append(packet)
        self._update_pid_info(packet)
    
    def extend(self, packets: List[Packet]) -> None:
        """Add multiple packets to the stream."""
        for pkt in packets:
            self.append(pkt)
    
    def _update_pid_info(self, packet: Packet) -> None:
        """Update PID statistics for a packet."""
        pid = packet.pid
        
        if pid not in self._pid_info:
            self._pid_info[pid] = PIDInfo(pid=pid)
        
        info = self._pid_info[pid]
        info.packet_count += 1
        
        # Check continuity counter (should increment by 1 mod 16)
        if info.last_cc >= 0:
            expected_cc = (info.last_cc + 1) % 16
            if packet.continuity_counter != expected_cc:
                # Don't count errors for null packets
                if pid != PID_NULL:
                    info.continuity_errors += 1
        
        info.last_cc = packet.continuity_counter
    
    def iter_pid(self, pid: int) -> Iterator[Packet]:
        """Iterate over packets with specific PID."""
        for pkt in self.packets:
            if pkt.pid == pid:
                yield pkt
    
    def get_pids(self) -> List[int]:
        """Get list of all PIDs in the stream."""
        return list(self._pid_info.keys())
    
    def get_pid_info(self, pid: int) -> Optional[PIDInfo]:
        """Get statistics for a specific PID."""
        return self._pid_info.get(pid)
    
    def analyze(self) -> Dict:
        """
        Analyze stream and return summary information.
        
        Returns:
            Dict with stream statistics
        """
        total_packets = len(self.packets)
        
        return {
            'total_packets': total_packets,
            'total_bytes': total_packets * PACKET_SIZE,
            'pids': {
                pid: {
                    'packets': info.packet_count,
                    'percent': 100 * info.packet_count / total_packets if total_packets else 0,
                    'cc_errors': info.continuity_errors,
                    'description': info.description,
                }
                for pid, info in sorted(self._pid_info.items())
            },
        }
    
    def parse_tables(self) -> None:
        """Parse PSI tables (PAT and PMTs) from the stream."""
        # Collect PAT sections
        pat_data = bytearray()
        for pkt in self.iter_pid(PID_PAT):
            if pkt.payload_unit_start:
                # Pointer field
                pointer = pkt.payload[0]
                pat_data = bytearray(pkt.payload[1 + pointer:])
            else:
                pat_data.extend(pkt.payload)
        
        if pat_data:
            try:
                self.pat = PAT.from_bytes(bytes(pat_data))
                
                # Label PAT PID
                if PID_PAT in self._pid_info:
                    self._pid_info[PID_PAT].description = "PAT"
                
                # Parse PMTs
                for program_num, pmt_pid in self.pat.programs.items():
                    if program_num == 0:
                        continue  # Skip NIT reference
                    
                    pmt_data = bytearray()
                    for pkt in self.iter_pid(pmt_pid):
                        if pkt.payload_unit_start:
                            pointer = pkt.payload[0]
                            pmt_data = bytearray(pkt.payload[1 + pointer:])
                        else:
                            pmt_data.extend(pkt.payload)
                    
                    if pmt_data:
                        try:
                            pmt = PMT.from_bytes(bytes(pmt_data))
                            self.pmts[program_num] = pmt
                            
                            # Label PMT PID
                            if pmt_pid in self._pid_info:
                                self._pid_info[pmt_pid].description = f"PMT(prog={program_num})"
                            
                            # Label elementary stream PIDs
                            for es in pmt.streams:
                                if es.pid in self._pid_info:
                                    self._pid_info[es.pid].stream_type = es.stream_type
                                    self._pid_info[es.pid].description = es.type_description
                        except Exception:
                            pass  # Skip malformed PMT
            except Exception:
                pass  # Skip malformed PAT
    
    def to_bytes(self) -> bytes:
        """
        Serialize entire stream to bytes.
        
        Returns:
            Concatenated packet data
        """
        return b''.join(pkt.to_bytes() for pkt in self.packets)
    
    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray]) -> 'TransportStream':
        """
        Parse transport stream from bytes.
        
        Args:
            data: Raw TS data (must be multiple of 188 bytes)
            
        Returns:
            Parsed TransportStream object
        """
        if len(data) % PACKET_SIZE != 0:
            raise ValueError(f"Data length {len(data)} not multiple of {PACKET_SIZE}")
        
        ts = cls()
        
        for i in range(0, len(data), PACKET_SIZE):
            packet_data = data[i:i + PACKET_SIZE]
            try:
                packet = Packet.from_bytes(packet_data)
                ts.append(packet)
            except ValueError as e:
                # Try to resync on sync byte
                pass
        
        return ts
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'TransportStream':
        """
        Read transport stream from file.
        
        Args:
            path: Path to .ts file
            
        Returns:
            Parsed TransportStream object
        """
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = f.read()
        
        ts = cls.from_bytes(data)
        ts.parse_tables()
        
        return ts
    
    def to_file(self, path: Union[str, Path]) -> None:
        """
        Write transport stream to file.
        
        Args:
            path: Output path for .ts file
        """
        path = Path(path)
        
        with open(path, 'wb') as f:
            f.write(self.to_bytes())
    
    @classmethod
    def from_stream(cls, stream: BinaryIO, max_packets: Optional[int] = None) -> 'TransportStream':
        """
        Read transport stream from file-like object.
        
        Args:
            stream: Binary file-like object
            max_packets: Maximum packets to read (None for all)
            
        Returns:
            Parsed TransportStream object
        """
        ts = cls()
        count = 0
        
        while True:
            if max_packets is not None and count >= max_packets:
                break
            
            data = stream.read(PACKET_SIZE)
            if len(data) < PACKET_SIZE:
                break
            
            try:
                packet = Packet.from_bytes(data)
                ts.append(packet)
                count += 1
            except ValueError:
                pass
        
        return ts
    
    def pad_to_bitrate(self, target_bitrate_bps: float, duration_seconds: float) -> None:
        """
        Add null packets to achieve target bitrate.
        
        Args:
            target_bitrate_bps: Target bitrate in bits per second
            duration_seconds: Stream duration in seconds
        """
        current_bytes = len(self.packets) * PACKET_SIZE
        target_bytes = int(target_bitrate_bps * duration_seconds / 8)
        
        if target_bytes > current_bytes:
            null_packets_needed = (target_bytes - current_bytes) // PACKET_SIZE
            
            # Insert null packets evenly throughout the stream
            if self.packets:
                interval = max(1, len(self.packets) // max(1, null_packets_needed))
                
                new_packets = []
                null_count = 0
                
                for i, pkt in enumerate(self.packets):
                    new_packets.append(pkt)
                    if null_count < null_packets_needed and (i + 1) % interval == 0:
                        new_packets.append(Packet.null_packet())
                        null_count += 1
                
                # Add remaining null packets at end
                while null_count < null_packets_needed:
                    new_packets.append(Packet.null_packet())
                    null_count += 1
                
                self.packets = new_packets
    
    def __repr__(self) -> str:
        return f"TransportStream({len(self.packets)} packets, {len(self._pid_info)} PIDs)"
