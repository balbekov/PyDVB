"""
Packetized Elementary Stream (PES)

PES packets carry the actual audio/video elementary stream data.
They can span multiple TS packets and include timing information
(PTS/DTS) for synchronization.

Structure:
    packet_start_code_prefix    (24 bits) - 0x000001
    stream_id                   (8 bits)  - identifies stream type
    PES_packet_length          (16 bits) - 0 = unbounded (video)
    [optional PES header]
    [payload data]

Reference: ISO/IEC 13818-1 Section 2.4.3.6
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import struct


# PES start code prefix
PES_START_CODE = b'\x00\x00\x01'

# Stream ID values
class StreamID:
    """Standard stream ID values."""
    PROGRAM_STREAM_MAP = 0xBC
    PRIVATE_STREAM_1 = 0xBD
    PADDING_STREAM = 0xBE
    PRIVATE_STREAM_2 = 0xBF
    AUDIO_STREAM_BASE = 0xC0  # 0xC0-0xDF = MPEG audio
    VIDEO_STREAM_BASE = 0xE0  # 0xE0-0xEF = MPEG video
    ECM_STREAM = 0xF0
    EMM_STREAM = 0xF1
    DSMCC_STREAM = 0xF2
    H222_TYPE_E = 0xF8
    PROGRAM_STREAM_DIR = 0xFF


# PTS/DTS clock frequency: 90 kHz
PTS_CLOCK_FREQ = 90_000


@dataclass
class PESPacket:
    """
    Packetized Elementary Stream Packet.
    
    PES packets encapsulate elementary stream data (video frames, audio
    samples) with timing information for synchronization.
    
    Attributes:
        stream_id: Stream identifier (e.g., 0xE0 for video)
        payload: Elementary stream data
        pts: Presentation Time Stamp (90kHz ticks)
        dts: Decode Time Stamp (90kHz ticks, usually for B-frames)
        data_alignment: Data alignment indicator
        copyright: Copyright flag
        original: Original/copy flag
        
    Example:
        >>> pes = PESPacket(stream_id=0xE0, payload=video_frame)
        >>> pes.pts = int(time_seconds * 90000)
        >>> data = pes.to_bytes()
    """
    
    stream_id: int
    payload: bytes = field(default_factory=bytes)
    pts: Optional[int] = None  # 33-bit PTS (90kHz ticks)
    dts: Optional[int] = None  # 33-bit DTS (90kHz ticks)
    data_alignment: bool = False
    copyright: bool = False
    original: bool = True
    priority: bool = False
    scrambling_control: int = 0
    
    @staticmethod
    def pts_from_seconds(seconds: float) -> int:
        """Convert time in seconds to 90kHz PTS value."""
        return int(seconds * PTS_CLOCK_FREQ)
    
    @staticmethod
    def pts_to_seconds(pts: int) -> float:
        """Convert 90kHz PTS value to seconds."""
        return pts / PTS_CLOCK_FREQ
    
    @property
    def has_pts(self) -> bool:
        """Check if PTS is present."""
        return self.pts is not None
    
    @property
    def has_dts(self) -> bool:
        """Check if DTS is present."""
        return self.dts is not None
    
    def _encode_timestamp(self, ts: int, marker_bits: int) -> bytes:
        """
        Encode 33-bit timestamp to 5 bytes.
        
        Format: '00XX' TS[32:30] '1' TS[29:15] '1' TS[14:0] '1'
        where XX = marker_bits (0010 for PTS only, 0011/0001 for PTS+DTS)
        """
        result = bytearray(5)
        
        # Byte 0: marker[3:0] | ts[32:30] | 1
        result[0] = ((marker_bits & 0x0F) << 4) | ((ts >> 29) & 0x0E) | 0x01
        
        # Bytes 1-2: ts[29:15] | 1
        result[1] = (ts >> 22) & 0xFF
        result[2] = ((ts >> 14) & 0xFE) | 0x01
        
        # Bytes 3-4: ts[14:0] | 1
        result[3] = (ts >> 7) & 0xFF
        result[4] = ((ts << 1) & 0xFE) | 0x01
        
        return bytes(result)
    
    @staticmethod
    def _decode_timestamp(data: bytes) -> int:
        """Decode 5-byte timestamp to 33-bit value."""
        ts = 0
        ts |= (data[0] & 0x0E) << 29
        ts |= data[1] << 22
        ts |= (data[2] & 0xFE) << 14
        ts |= data[3] << 7
        ts |= (data[4] & 0xFE) >> 1
        return ts
    
    def to_bytes(self) -> bytes:
        """
        Serialize PES packet.
        
        Returns:
            Complete PES packet bytes
        """
        # Check if this stream type has optional header
        has_optional_header = self.stream_id not in (
            StreamID.PROGRAM_STREAM_MAP,
            StreamID.PADDING_STREAM,
            StreamID.PRIVATE_STREAM_2,
            StreamID.ECM_STREAM,
            StreamID.EMM_STREAM,
            StreamID.PROGRAM_STREAM_DIR,
            StreamID.DSMCC_STREAM,
            StreamID.H222_TYPE_E,
        )
        
        if has_optional_header:
            return self._encode_with_header()
        else:
            return self._encode_simple()
    
    def _encode_simple(self) -> bytes:
        """Encode PES packet without optional header."""
        packet = bytearray(PES_START_CODE)
        packet.append(self.stream_id)
        
        # Length (0 for unbounded video)
        length = len(self.payload)
        if length > 65535:
            length = 0
        packet.extend(struct.pack('>H', length))
        
        packet.extend(self.payload)
        return bytes(packet)
    
    def _encode_with_header(self) -> bytes:
        """Encode PES packet with optional header."""
        # Build optional header
        header = bytearray()
        
        # First 2 bytes of optional header
        flags1 = (0b10 << 6 |  # Fixed '10'
                  (self.scrambling_control & 0x3) << 4 |
                  (self.priority & 1) << 3 |
                  (self.data_alignment & 1) << 2 |
                  (self.copyright & 1) << 1 |
                  (self.original & 1))
        
        pts_dts_flags = 0
        if self.pts is not None and self.dts is not None:
            pts_dts_flags = 0b11
        elif self.pts is not None:
            pts_dts_flags = 0b10
        
        flags2 = ((pts_dts_flags & 0x3) << 6 |
                  0)  # Other flags = 0
        
        header.append(flags1)
        header.append(flags2)
        
        # PES header data
        header_data = bytearray()
        
        if self.pts is not None and self.dts is not None:
            header_data.extend(self._encode_timestamp(self.pts, 0b0011))
            header_data.extend(self._encode_timestamp(self.dts, 0b0001))
        elif self.pts is not None:
            header_data.extend(self._encode_timestamp(self.pts, 0b0010))
        
        # Header data length
        header.append(len(header_data))
        header.extend(header_data)
        
        # Build full packet
        packet = bytearray(PES_START_CODE)
        packet.append(self.stream_id)
        
        # Total length (0 for unbounded)
        total_length = len(header) + len(self.payload)
        if total_length > 65535:
            total_length = 0
        packet.extend(struct.pack('>H', total_length))
        
        packet.extend(header)
        packet.extend(self.payload)
        
        return bytes(packet)
    
    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray]) -> 'PESPacket':
        """
        Parse PES packet from bytes.
        
        Args:
            data: PES packet bytes (starting with 0x000001)
            
        Returns:
            Parsed PESPacket object
        """
        if len(data) < 6:
            raise ValueError("PES packet too short")
        
        if data[0:3] != PES_START_CODE:
            raise ValueError(f"Invalid PES start code: {data[0:3].hex()}")
        
        stream_id = data[3]
        pes_length = struct.unpack('>H', data[4:6])[0]
        
        # Check if has optional header
        has_optional_header = stream_id not in (
            StreamID.PROGRAM_STREAM_MAP,
            StreamID.PADDING_STREAM,
            StreamID.PRIVATE_STREAM_2,
            StreamID.ECM_STREAM,
            StreamID.EMM_STREAM,
            StreamID.PROGRAM_STREAM_DIR,
            StreamID.DSMCC_STREAM,
            StreamID.H222_TYPE_E,
        )
        
        if not has_optional_header:
            payload = data[6:]
            return cls(stream_id=stream_id, payload=bytes(payload))
        
        # Parse optional header
        if len(data) < 9:
            raise ValueError("PES optional header too short")
        
        flags1 = data[6]
        flags2 = data[7]
        header_length = data[8]
        
        scrambling_control = (flags1 >> 4) & 0x03
        priority = bool(flags1 & 0x08)
        data_alignment = bool(flags1 & 0x04)
        copyright = bool(flags1 & 0x02)
        original = bool(flags1 & 0x01)
        
        pts_dts_flags = (flags2 >> 6) & 0x03
        
        pos = 9
        pts = None
        dts = None
        
        if pts_dts_flags == 0b10:
            pts = cls._decode_timestamp(data[pos:pos + 5])
            pos += 5
        elif pts_dts_flags == 0b11:
            pts = cls._decode_timestamp(data[pos:pos + 5])
            pos += 5
            dts = cls._decode_timestamp(data[pos:pos + 5])
            pos += 5
        
        # Skip to payload
        payload_start = 9 + header_length
        payload = data[payload_start:]
        
        return cls(
            stream_id=stream_id,
            payload=bytes(payload),
            pts=pts,
            dts=dts,
            data_alignment=data_alignment,
            copyright=copyright,
            original=original,
            priority=priority,
            scrambling_control=scrambling_control,
        )
    
    def __repr__(self) -> str:
        pts_str = f", PTS={self.pts_to_seconds(self.pts):.3f}s" if self.pts else ""
        dts_str = f", DTS={self.pts_to_seconds(self.dts):.3f}s" if self.dts else ""
        return f"PESPacket(stream_id=0x{self.stream_id:02X}, {len(self.payload)}B{pts_str}{dts_str})"


# Convenience alias
PES = PESPacket
