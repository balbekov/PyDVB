"""
Program Map Table (PMT)

The PMT describes the structure of a single program (service).
It lists all elementary streams (video, audio, etc.) and their PIDs.

Structure:
    PSI header (table_id = 0x02)
    program_number              (16 bits) = table_id_extension
    reserved                   (3 bits)
    PCR_PID                    (13 bits)
    reserved                   (4 bits)
    program_info_length        (12 bits)
    [program descriptors]
    [for each elementary stream:]
        stream_type            (8 bits)
        reserved              (3 bits)
        elementary_PID        (13 bits)
        reserved              (4 bits)
        ES_info_length        (12 bits)
        [ES descriptors]
    CRC_32                     (32 bits)

Reference: ISO/IEC 13818-1 Section 2.4.4.8
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional
import struct

from .PSI import PSI, TableID


# Stream type codes (ISO/IEC 13818-1 Table 2-29)
class StreamType:
    """Standard stream type values."""
    RESERVED = 0x00
    MPEG1_VIDEO = 0x01
    MPEG2_VIDEO = 0x02
    MPEG1_AUDIO = 0x03
    MPEG2_AUDIO = 0x04
    PRIVATE_SECTIONS = 0x05
    PRIVATE_PES = 0x06
    MHEG = 0x07
    DSM_CC = 0x08
    H222_ATM = 0x09
    DSM_CC_A = 0x0A
    DSM_CC_B = 0x0B
    DSM_CC_C = 0x0C
    DSM_CC_D = 0x0D
    MPEG2_AUX = 0x0E
    AAC_AUDIO = 0x0F
    MPEG4_VIDEO = 0x10
    MPEG4_AUDIO = 0x11  # LATM
    MPEG4_PES = 0x12
    METADATA_PES = 0x15
    H264_VIDEO = 0x1B
    H265_VIDEO = 0x24
    
    # DVB specific
    AC3_AUDIO = 0x81
    DTS_AUDIO = 0x82
    
    @classmethod
    def get_description(cls, stream_type: int) -> str:
        """Get human-readable description of stream type."""
        descriptions = {
            cls.MPEG1_VIDEO: "MPEG-1 Video",
            cls.MPEG2_VIDEO: "MPEG-2 Video",
            cls.MPEG1_AUDIO: "MPEG-1 Audio",
            cls.MPEG2_AUDIO: "MPEG-2 Audio",
            cls.AAC_AUDIO: "AAC Audio",
            cls.MPEG4_VIDEO: "MPEG-4 Video",
            cls.MPEG4_AUDIO: "MPEG-4 Audio",
            cls.H264_VIDEO: "H.264/AVC Video",
            cls.H265_VIDEO: "H.265/HEVC Video",
            cls.AC3_AUDIO: "AC-3 Audio",
            cls.DTS_AUDIO: "DTS Audio",
            cls.PRIVATE_PES: "Private Data",
        }
        return descriptions.get(stream_type, f"Unknown (0x{stream_type:02X})")


@dataclass
class ElementaryStream:
    """
    Elementary stream entry in PMT.
    
    Attributes:
        stream_type: Type of stream (video, audio, etc.)
        pid: PID carrying this elementary stream
        descriptors: Stream-specific descriptors
    """
    stream_type: int
    pid: int
    descriptors: bytes = field(default_factory=bytes)
    
    @property
    def type_description(self) -> str:
        """Get human-readable stream type description."""
        return StreamType.get_description(self.stream_type)
    
    def to_bytes(self) -> bytes:
        """Serialize elementary stream entry."""
        data = bytearray()
        
        # Stream type
        data.append(self.stream_type)
        
        # Reserved + elementary PID
        data.extend(struct.pack('>H', 0xE000 | (self.pid & 0x1FFF)))
        
        # Reserved + ES info length
        data.extend(struct.pack('>H', 0xF000 | (len(self.descriptors) & 0x0FFF)))
        
        # Descriptors
        data.extend(self.descriptors)
        
        return bytes(data)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ElementaryStream':
        """Parse elementary stream entry."""
        stream_type = data[0]
        pid = struct.unpack('>H', data[1:3])[0] & 0x1FFF
        es_info_length = struct.unpack('>H', data[3:5])[0] & 0x0FFF
        descriptors = data[5:5 + es_info_length]
        
        return cls(stream_type=stream_type, pid=pid, descriptors=descriptors)
    
    def __repr__(self) -> str:
        return f"ES(type={self.type_description}, pid=0x{self.pid:04X})"


@dataclass
class PMT(PSI):
    """
    Program Map Table.
    
    The PMT describes one program's structure, listing all elementary
    streams and their PIDs.
    
    Attributes:
        program_number: Program number (from PAT)
        pcr_pid: PID carrying PCR for this program
        program_descriptors: Program-level descriptors
        streams: List of elementary streams
        
    Example:
        >>> pmt = PMT(program_number=1, pcr_pid=0x100)
        >>> pmt.add_stream(StreamType.H264_VIDEO, 0x100)
        >>> pmt.add_stream(StreamType.AAC_AUDIO, 0x101)
        >>> data = pmt.to_bytes()
    """
    
    table_id: int = field(default=TableID.PMT, init=False)
    program_number: int = 1
    pcr_pid: int = 0x1FFF
    program_descriptors: bytes = field(default_factory=bytes)
    streams: List[ElementaryStream] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize base PSI fields."""
        self.table_id = TableID.PMT
        self.table_id_extension = self.program_number
        self.section_syntax_indicator = True
    
    def add_stream(self, stream_type: int, pid: int, 
                   descriptors: bytes = b'') -> ElementaryStream:
        """
        Add an elementary stream to the PMT.
        
        Args:
            stream_type: Stream type code
            pid: PID for this stream
            descriptors: Optional stream descriptors
            
        Returns:
            Created ElementaryStream object
        """
        es = ElementaryStream(stream_type=stream_type, pid=pid, 
                              descriptors=descriptors)
        self.streams.append(es)
        return es
    
    def to_bytes(self) -> bytes:
        """
        Serialize PMT to bytes.
        
        Returns:
            Complete PMT section including CRC
        """
        # Build table data
        table_data = bytearray()
        
        # PCR PID
        table_data.extend(struct.pack('>H', 0xE000 | (self.pcr_pid & 0x1FFF)))
        
        # Program info length + descriptors
        table_data.extend(struct.pack('>H', 
            0xF000 | (len(self.program_descriptors) & 0x0FFF)))
        table_data.extend(self.program_descriptors)
        
        # Elementary streams
        for es in self.streams:
            table_data.extend(es.to_bytes())
        
        return self.encode_section(bytes(table_data))
    
    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray]) -> 'PMT':
        """
        Parse PMT from bytes.
        
        Args:
            data: Complete PMT section including CRC
            
        Returns:
            Parsed PMT object
        """
        # Parse header
        header = cls.parse_header(data)
        
        if header['table_id'] != TableID.PMT:
            raise ValueError(f"Not a PMT: table_id=0x{header['table_id']:02X}")
        
        section_length = header['section_length']
        
        # PCR PID (after extended header, offset 8)
        pcr_pid = struct.unpack('>H', data[8:10])[0] & 0x1FFF
        
        # Program info length
        program_info_length = struct.unpack('>H', data[10:12])[0] & 0x0FFF
        
        # Program descriptors
        program_descriptors = data[12:12 + program_info_length]
        
        pmt = cls(
            program_number=header.get('table_id_extension', 1),
            pcr_pid=pcr_pid,
            program_descriptors=bytes(program_descriptors),
        )
        pmt.version = header.get('version', 0)
        
        # Parse elementary streams
        pos = 12 + program_info_length
        end = 3 + section_length - 4  # Before CRC
        
        while pos + 5 <= end:
            stream_type = data[pos]
            es_pid = struct.unpack('>H', data[pos + 1:pos + 3])[0] & 0x1FFF
            es_info_length = struct.unpack('>H', data[pos + 3:pos + 5])[0] & 0x0FFF
            
            es_descriptors = data[pos + 5:pos + 5 + es_info_length]
            
            pmt.streams.append(ElementaryStream(
                stream_type=stream_type,
                pid=es_pid,
                descriptors=bytes(es_descriptors),
            ))
            
            pos += 5 + es_info_length
        
        return pmt
    
    def get_video_pid(self) -> Optional[int]:
        """Get PID of first video stream."""
        video_types = {
            StreamType.MPEG1_VIDEO, StreamType.MPEG2_VIDEO,
            StreamType.MPEG4_VIDEO, StreamType.H264_VIDEO, StreamType.H265_VIDEO
        }
        for es in self.streams:
            if es.stream_type in video_types:
                return es.pid
        return None
    
    def get_audio_pids(self) -> List[int]:
        """Get PIDs of all audio streams."""
        audio_types = {
            StreamType.MPEG1_AUDIO, StreamType.MPEG2_AUDIO,
            StreamType.AAC_AUDIO, StreamType.MPEG4_AUDIO,
            StreamType.AC3_AUDIO, StreamType.DTS_AUDIO
        }
        return [es.pid for es in self.streams if es.stream_type in audio_types]
    
    def __repr__(self) -> str:
        return (f"PMT(program={self.program_number}, "
                f"pcr_pid=0x{self.pcr_pid:04X}, "
                f"streams={len(self.streams)})")
