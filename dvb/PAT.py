"""
Program Association Table (PAT)

The PAT lists all programs (services) in the transport stream and
the PID of each program's PMT. It's always on PID 0x0000.

Structure:
    PSI header (table_id = 0x00)
    transport_stream_id         (16 bits) = table_id_extension
    [for each program:]
        program_number          (16 bits)
        reserved               (3 bits)
        program_map_PID        (13 bits)
    CRC_32                     (32 bits)

Program number 0 refers to the Network Information Table (NIT).

Reference: ISO/IEC 13818-1 Section 2.4.4.3
"""

from dataclasses import dataclass, field
from typing import Dict, Union, List
import struct

from .PSI import PSI, TableID
from .CRC import CRC32


@dataclass
class PAT(PSI):
    """
    Program Association Table.
    
    The PAT is the root of the PSI hierarchy. It maps program numbers
    to PMT PIDs, allowing a decoder to find all services in the stream.
    
    Attributes:
        transport_stream_id: Unique identifier for this TS
        programs: Dict mapping program_number -> PMT PID
        
    Example:
        >>> pat = PAT(transport_stream_id=1)
        >>> pat.programs[1] = 0x100  # Program 1's PMT is on PID 0x100
        >>> data = pat.to_bytes()
    """
    
    table_id: int = field(default=TableID.PAT, init=False)
    transport_stream_id: int = 1
    programs: Dict[int, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize base PSI fields."""
        self.table_id = TableID.PAT
        self.table_id_extension = self.transport_stream_id
        self.section_syntax_indicator = True
    
    def add_program(self, program_number: int, pmt_pid: int) -> None:
        """
        Add a program to the PAT.
        
        Args:
            program_number: Program number (1-65535, 0 reserved for NIT)
            pmt_pid: PID where PMT for this program is found
        """
        if not 0 <= program_number <= 65535:
            raise ValueError(f"Invalid program number: {program_number}")
        if not 0 <= pmt_pid <= 0x1FFF:
            raise ValueError(f"Invalid PID: {pmt_pid}")
        
        self.programs[program_number] = pmt_pid
    
    def to_bytes(self) -> bytes:
        """
        Serialize PAT to bytes.
        
        Returns:
            Complete PAT section including CRC
        """
        # Build program loop
        program_data = bytearray()
        
        for program_num, pmt_pid in sorted(self.programs.items()):
            program_data.extend(struct.pack('>H', program_num))
            program_data.extend(struct.pack('>H', 0xE000 | (pmt_pid & 0x1FFF)))
        
        return self.encode_section(bytes(program_data))
    
    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray]) -> 'PAT':
        """
        Parse PAT from bytes.
        
        Args:
            data: Complete PAT section including CRC
            
        Returns:
            Parsed PAT object
        """
        # Parse header
        header = cls.parse_header(data)
        
        if header['table_id'] != TableID.PAT:
            raise ValueError(f"Not a PAT: table_id=0x{header['table_id']:02X}")
        
        pat = cls(
            transport_stream_id=header.get('table_id_extension', 0),
        )
        pat.version = header.get('version', 0)
        pat.section_number = header.get('section_number', 0)
        pat.last_section_number = header.get('last_section_number', 0)
        
        # Parse program loop
        section_length = header['section_length']
        
        # Data starts after header (3 bytes), extended header (5 bytes)
        # Ends before CRC (4 bytes)
        start = 8
        end = 3 + section_length - 4  # section_length includes from table_id_extension to CRC
        
        pos = start
        while pos + 4 <= end:
            program_num = struct.unpack('>H', data[pos:pos + 2])[0]
            pmt_pid = struct.unpack('>H', data[pos + 2:pos + 4])[0] & 0x1FFF
            
            pat.programs[program_num] = pmt_pid
            pos += 4
        
        return pat
    
    def get_program_pids(self) -> List[int]:
        """Get list of all PMT PIDs."""
        return list(self.programs.values())
    
    def get_nit_pid(self) -> int:
        """Get NIT PID (program number 0), default 0x0010."""
        return self.programs.get(0, 0x0010)
    
    def __repr__(self) -> str:
        return (f"PAT(tsid={self.transport_stream_id}, "
                f"programs={list(self.programs.keys())})")
