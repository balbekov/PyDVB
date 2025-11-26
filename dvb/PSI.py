"""
Program Specific Information (PSI) Base

PSI tables provide the structure information for transport streams.
This module provides the base class for all PSI section types.

Common PSI structure:
    table_id                    (8 bits)
    section_syntax_indicator    (1 bit)
    '0'                        (1 bit)
    reserved                   (2 bits)
    section_length             (12 bits)
    [table-specific data]
    CRC_32                     (32 bits)

Reference: ISO/IEC 13818-1 Section 2.4.4
"""

from dataclasses import dataclass, field
from typing import Union, Optional
import struct

from .CRC import CRC32


# Table IDs
class TableID:
    """Standard PSI table IDs."""
    PAT = 0x00  # Program Association Table
    CAT = 0x01  # Conditional Access Table
    PMT = 0x02  # Program Map Table
    TSDT = 0x03  # Transport Stream Description Table
    
    # DVB SI tables
    NIT_ACTUAL = 0x40  # Network Information Table (actual)
    NIT_OTHER = 0x41   # Network Information Table (other)
    SDT_ACTUAL = 0x42  # Service Description Table (actual)
    SDT_OTHER = 0x46   # Service Description Table (other)
    BAT = 0x4A         # Bouquet Association Table
    EIT_PF_ACTUAL = 0x4E  # Event Information Table (present/following, actual)
    EIT_PF_OTHER = 0x4F   # Event Information Table (present/following, other)
    TDT = 0x70         # Time and Date Table
    RST = 0x71         # Running Status Table
    ST = 0x72          # Stuffing Table
    TOT = 0x73         # Time Offset Table
    DIT = 0x7E         # Discontinuity Information Table
    SIT = 0x7F         # Selection Information Table


@dataclass
class PSI:
    """
    Base class for Program Specific Information sections.
    
    All PSI tables share a common header structure and CRC protection.
    Subclasses implement specific table parsing/generation.
    
    Attributes:
        table_id: Identifies the table type
        section_syntax_indicator: If True, extended header is present
        section_length: Length of data following this field
        table_id_extension: Table-specific extension (program_number for PAT)
        version: Version number (0-31)
        current_next: True if currently applicable, False if next
        section_number: Section number within table
        last_section_number: Last section number
        
    Example:
        >>> psi = PSI(table_id=0x00, table_id_extension=1)
        >>> header = psi.encode_header()
    """
    
    table_id: int
    section_syntax_indicator: bool = True
    table_id_extension: int = 0
    version: int = 0
    current_next: bool = True
    section_number: int = 0
    last_section_number: int = 0
    private_indicator: bool = False
    
    _crc = CRC32()
    
    def encode_header(self) -> bytes:
        """
        Encode PSI section header (without section_length).
        
        Returns:
            Header bytes up to but not including section_length
        """
        header = bytearray()
        header.append(self.table_id)
        return bytes(header)
    
    def encode_extended_header(self) -> bytes:
        """
        Encode extended header (after section_length).
        
        Returns:
            5 bytes: table_id_extension (2) + version/current_next (1) + 
                    section_number (1) + last_section_number (1)
        """
        header = bytearray()
        
        # Table ID extension (16 bits)
        header.extend(struct.pack('>H', self.table_id_extension))
        
        # Version and current/next
        header.append((0b11 << 6) |  # Reserved
                      ((self.version & 0x1F) << 1) |
                      (self.current_next & 0x01))
        
        # Section numbers
        header.append(self.section_number)
        header.append(self.last_section_number)
        
        return bytes(header)
    
    @classmethod
    def parse_header(cls, data: Union[bytes, bytearray]) -> dict:
        """
        Parse PSI section header.
        
        Args:
            data: Section data starting with table_id
            
        Returns:
            Dict with parsed header fields
        """
        if len(data) < 3:
            raise ValueError("PSI header too short")
        
        table_id = data[0]
        section_syntax_indicator = bool(data[1] & 0x80)
        private_indicator = bool(data[1] & 0x40)
        section_length = ((data[1] & 0x0F) << 8) | data[2]
        
        result = {
            'table_id': table_id,
            'section_syntax_indicator': section_syntax_indicator,
            'private_indicator': private_indicator,
            'section_length': section_length,
        }
        
        # Parse extended header if present
        if section_syntax_indicator and len(data) >= 8:
            result['table_id_extension'] = struct.unpack('>H', data[3:5])[0]
            result['version'] = (data[5] >> 1) & 0x1F
            result['current_next'] = bool(data[5] & 0x01)
            result['section_number'] = data[6]
            result['last_section_number'] = data[7]
        
        return result
    
    def encode_section(self, table_data: bytes) -> bytes:
        """
        Encode complete PSI section with CRC.
        
        Args:
            table_data: Table-specific payload data
            
        Returns:
            Complete section bytes including header and CRC
        """
        # Build section without CRC
        section = bytearray()
        section.append(self.table_id)
        
        # Calculate section length
        if self.section_syntax_indicator:
            # Extended header (5) + table_data + CRC (4)
            section_length = 5 + len(table_data) + 4
        else:
            # Just table_data + CRC (4)
            section_length = len(table_data) + 4
        
        # Section syntax indicator + private + reserved + section_length
        section.append((self.section_syntax_indicator << 7) |
                       (self.private_indicator << 6) |
                       0b00110000 |  # Reserved bits
                       ((section_length >> 8) & 0x0F))
        section.append(section_length & 0xFF)
        
        # Extended header
        if self.section_syntax_indicator:
            section.extend(self.encode_extended_header())
        
        # Table data
        section.extend(table_data)
        
        # CRC-32
        crc = self._crc.calculate(bytes(section))
        section.extend(struct.pack('>I', crc))
        
        return bytes(section)
    
    @classmethod
    def verify_crc(cls, data: Union[bytes, bytearray]) -> bool:
        """
        Verify CRC-32 of PSI section.
        
        Args:
            data: Complete section including CRC
            
        Returns:
            True if CRC is valid
        """
        if len(data) < 4:
            return False
        
        # CRC should result in 0 when calculated over entire section
        crc = cls._crc.calculate(data)
        return crc == 0
    
    def __repr__(self) -> str:
        return (f"PSI(table_id=0x{self.table_id:02X}, "
                f"version={self.version}, "
                f"section={self.section_number}/{self.last_section_number})")
