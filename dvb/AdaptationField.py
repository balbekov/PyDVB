"""
MPEG-2 Transport Stream Adaptation Field

The adaptation field carries timing information (PCR/OPCR), splice countdowns,
private data, and stuffing bytes. It's used when the payload doesn't fill
the full 184 bytes, or when timing/signaling info is needed.

Structure:
    adaptation_field_length     (8 bits)
    discontinuity_indicator     (1 bit)
    random_access_indicator     (1 bit)
    es_priority_indicator       (1 bit)
    PCR_flag                    (1 bit)
    OPCR_flag                   (1 bit)
    splicing_point_flag         (1 bit)
    transport_private_data_flag (1 bit)
    adaptation_field_extension_flag (1 bit)
    [optional fields based on flags]
    [stuffing bytes]

Reference: ISO/IEC 13818-1 Section 2.4.3.4
"""

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class AdaptationField:
    """
    MPEG-2 Transport Stream Adaptation Field.
    
    The adaptation field provides timing information and allows packets
    to be padded to exactly 188 bytes.
    
    Attributes:
        length: Total length of adaptation field (not including length byte)
        discontinuity: Discontinuity indicator
        random_access: Random access point (e.g., I-frame)
        es_priority: Elementary stream priority
        pcr: Program Clock Reference (42-bit base + 9-bit extension)
        opcr: Original PCR (for stream copying)
        splice_countdown: Countdown to splice point
        private_data: Transport private data
        
    Example:
        >>> af = AdaptationField(length=7, pcr=12345678)
        >>> data = af.to_bytes()
        >>> assert data[0] == 7  # Length byte
    """
    
    length: int = 0
    discontinuity: bool = False
    random_access: bool = False
    es_priority: bool = False
    pcr: Optional[int] = None  # 33-bit base + 9-bit extension = 42 bits
    opcr: Optional[int] = None
    splice_countdown: Optional[int] = None
    private_data: bytes = field(default_factory=bytes)
    
    # PCR system clock frequency: 27 MHz
    PCR_CLOCK_FREQ = 27_000_000
    # PCR base frequency: 90 kHz  
    PCR_BASE_FREQ = 90_000
    
    def __len__(self) -> int:
        """Return total adaptation field length including length byte."""
        return 1 + self.length
    
    @classmethod
    def pcr_from_timestamp(cls, timestamp_seconds: float) -> int:
        """
        Convert timestamp in seconds to PCR value.
        
        PCR = (base * 300) + extension
        where base is 90kHz ticks and extension is 27MHz remainder
        
        Args:
            timestamp_seconds: Time in seconds
            
        Returns:
            42-bit PCR value (33-bit base << 9 | 9-bit extension)
        """
        total_27mhz = int(timestamp_seconds * cls.PCR_CLOCK_FREQ)
        pcr_base = total_27mhz // 300  # 90kHz ticks
        pcr_ext = total_27mhz % 300    # Remainder at 27MHz
        
        # Pack as 33-bit base + 9-bit extension
        return (pcr_base << 9) | pcr_ext
    
    @classmethod
    def pcr_to_timestamp(cls, pcr: int) -> float:
        """
        Convert PCR value to timestamp in seconds.
        
        Args:
            pcr: 42-bit PCR value
            
        Returns:
            Time in seconds
        """
        pcr_base = pcr >> 9
        pcr_ext = pcr & 0x1FF
        total_27mhz = pcr_base * 300 + pcr_ext
        return total_27mhz / cls.PCR_CLOCK_FREQ
    
    def _calculate_minimum_length(self) -> int:
        """Calculate minimum length needed for current fields."""
        length = 1  # Flags byte
        
        if self.pcr is not None:
            length += 6  # PCR is 6 bytes
        if self.opcr is not None:
            length += 6  # OPCR is 6 bytes
        if self.splice_countdown is not None:
            length += 1
        if self.private_data:
            length += 1 + len(self.private_data)  # Length byte + data
            
        return length
    
    def to_bytes(self) -> bytes:
        """
        Serialize adaptation field.
        
        Returns:
            Bytes including length byte
        """
        if self.length == 0:
            return b'\x00'  # Just length byte
        
        data = bytearray()
        
        # Length byte
        data.append(self.length)
        
        # Flags byte
        flags = ((self.discontinuity & 1) << 7 |
                 (self.random_access & 1) << 6 |
                 (self.es_priority & 1) << 5 |
                 (1 if self.pcr is not None else 0) << 4 |
                 (1 if self.opcr is not None else 0) << 3 |
                 (1 if self.splice_countdown is not None else 0) << 2 |
                 (1 if self.private_data else 0) << 1 |
                 0)  # Extension flag
        data.append(flags)
        
        # Optional fields
        if self.pcr is not None:
            data.extend(self._encode_pcr(self.pcr))
        
        if self.opcr is not None:
            data.extend(self._encode_pcr(self.opcr))
        
        if self.splice_countdown is not None:
            data.append(self.splice_countdown & 0xFF)
        
        if self.private_data:
            data.append(len(self.private_data))
            data.extend(self.private_data)
        
        # Stuffing bytes to reach declared length
        while len(data) < self.length + 1:
            data.append(0xFF)
        
        return bytes(data)
    
    def _encode_pcr(self, pcr: int) -> bytes:
        """
        Encode PCR value to 6 bytes.
        
        Format: [base 33 bits] [reserved 6 bits] [extension 9 bits]
        """
        pcr_base = pcr >> 9
        pcr_ext = pcr & 0x1FF
        
        # Pack into 48 bits (6 bytes)
        result = bytearray(6)
        result[0] = (pcr_base >> 25) & 0xFF
        result[1] = (pcr_base >> 17) & 0xFF
        result[2] = (pcr_base >> 9) & 0xFF
        result[3] = (pcr_base >> 1) & 0xFF
        result[4] = ((pcr_base & 1) << 7) | 0x7E | ((pcr_ext >> 8) & 1)
        result[5] = pcr_ext & 0xFF
        
        return bytes(result)
    
    @classmethod
    def _decode_pcr(cls, data: bytes) -> int:
        """Decode 6-byte PCR field."""
        pcr_base = ((data[0] << 25) |
                    (data[1] << 17) |
                    (data[2] << 9) |
                    (data[3] << 1) |
                    ((data[4] >> 7) & 1))
        pcr_ext = ((data[4] & 1) << 8) | data[5]
        
        return (pcr_base << 9) | pcr_ext
    
    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray]) -> 'AdaptationField':
        """
        Parse adaptation field from bytes.
        
        Args:
            data: Bytes starting with length byte
            
        Returns:
            Parsed AdaptationField object
        """
        if len(data) < 1:
            raise ValueError("Adaptation field data too short")
        
        length = data[0]
        
        if length == 0:
            return cls(length=0)
        
        if len(data) < length + 1:
            raise ValueError(f"Adaptation field data too short: need {length + 1}, got {len(data)}")
        
        # Parse flags
        flags = data[1]
        discontinuity = bool(flags & 0x80)
        random_access = bool(flags & 0x40)
        es_priority = bool(flags & 0x20)
        pcr_flag = bool(flags & 0x10)
        opcr_flag = bool(flags & 0x08)
        splice_flag = bool(flags & 0x04)
        private_flag = bool(flags & 0x02)
        
        pos = 2
        pcr = None
        opcr = None
        splice_countdown = None
        private_data = b''
        
        if pcr_flag:
            pcr = cls._decode_pcr(data[pos:pos + 6])
            pos += 6
        
        if opcr_flag:
            opcr = cls._decode_pcr(data[pos:pos + 6])
            pos += 6
        
        if splice_flag:
            splice_countdown = data[pos]
            pos += 1
        
        if private_flag:
            private_length = data[pos]
            pos += 1
            private_data = bytes(data[pos:pos + private_length])
            pos += private_length
        
        return cls(
            length=length,
            discontinuity=discontinuity,
            random_access=random_access,
            es_priority=es_priority,
            pcr=pcr,
            opcr=opcr,
            splice_countdown=splice_countdown,
            private_data=private_data,
        )
    
    @classmethod
    def stuffing(cls, length: int) -> 'AdaptationField':
        """
        Create adaptation field for stuffing (padding).
        
        Args:
            length: Total length of adaptation field (not including length byte)
            
        Returns:
            Stuffing adaptation field
        """
        return cls(length=length)
    
    def __repr__(self) -> str:
        parts = [f"AdaptationField(len={self.length}"]
        if self.discontinuity:
            parts.append("disc")
        if self.random_access:
            parts.append("RA")
        if self.pcr is not None:
            parts.append(f"PCR={self.pcr_to_timestamp(self.pcr):.6f}s")
        if self.opcr is not None:
            parts.append(f"OPCR")
        if self.splice_countdown is not None:
            parts.append(f"splice={self.splice_countdown}")
        return ", ".join(parts) + ")"
