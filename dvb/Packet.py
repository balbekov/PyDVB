"""
MPEG-2 Transport Stream Packet

A Transport Stream packet is exactly 188 bytes:
- 4-byte header (sync byte + flags + PID + continuity counter)
- 0-183 bytes adaptation field (optional)
- 0-184 bytes payload

Header structure:
    Byte 0:     Sync byte (0x47)
    Byte 1-2:   Transport Error | Payload Unit Start | Priority | PID (13 bits)
    Byte 3:     Scrambling | Adaptation Field | Continuity Counter

Reference: ISO/IEC 13818-1 (MPEG-2 Systems)
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field

from .AdaptationField import AdaptationField


# Sync byte that starts every TS packet
SYNC_BYTE = 0x47

# Packet size
PACKET_SIZE = 188
HEADER_SIZE = 4
MAX_PAYLOAD_SIZE = 184

# Special PIDs
PID_PAT = 0x0000      # Program Association Table
PID_CAT = 0x0001      # Conditional Access Table  
PID_TSDT = 0x0002     # Transport Stream Description Table
PID_NIT = 0x0010      # Network Information Table
PID_SDT = 0x0011      # Service Description Table
PID_EIT = 0x0012      # Event Information Table
PID_RST = 0x0013      # Running Status Table
PID_TDT = 0x0014      # Time and Date Table
PID_NULL = 0x1FFF     # Null packet (stuffing)


@dataclass
class Packet:
    """
    MPEG-2 Transport Stream Packet (188 bytes).
    
    Attributes:
        pid: Packet Identifier (13 bits, 0-8191)
        payload: Packet payload data (up to 184 bytes)
        transport_error: Transport error indicator
        payload_unit_start: Payload unit start indicator (for PES/PSI)
        transport_priority: Transport priority
        scrambling_control: Scrambling control (00 = not scrambled)
        continuity_counter: 4-bit counter for packet sequence
        adaptation_field: Optional adaptation field
        
    Example:
        >>> pkt = Packet(pid=0x100, payload=b'\\x00' * 184)
        >>> data = pkt.to_bytes()
        >>> assert len(data) == 188
    """
    
    pid: int
    payload: bytes = field(default_factory=bytes)
    transport_error: bool = False
    payload_unit_start: bool = False
    transport_priority: bool = False
    scrambling_control: int = 0
    continuity_counter: int = 0
    adaptation_field: Optional[AdaptationField] = None
    
    def __post_init__(self):
        """Validate packet parameters."""
        if not 0 <= self.pid <= 0x1FFF:
            raise ValueError(f"PID must be 0-8191, got {self.pid}")
        if not 0 <= self.scrambling_control <= 3:
            raise ValueError(f"Scrambling control must be 0-3, got {self.scrambling_control}")
        if not 0 <= self.continuity_counter <= 15:
            raise ValueError(f"Continuity counter must be 0-15, got {self.continuity_counter}")
        
        # Calculate maximum payload size given adaptation field
        max_payload = MAX_PAYLOAD_SIZE
        if self.adaptation_field is not None:
            max_payload -= len(self.adaptation_field)
        
        if len(self.payload) > max_payload:
            raise ValueError(f"Payload too large: {len(self.payload)} > {max_payload}")
    
    @property
    def has_adaptation_field(self) -> bool:
        """Check if packet has adaptation field."""
        return self.adaptation_field is not None
    
    @property
    def has_payload(self) -> bool:
        """Check if packet has payload."""
        return len(self.payload) > 0
    
    @property
    def adaptation_field_control(self) -> int:
        """
        Get adaptation field control value (2 bits).
        
        00 = Reserved
        01 = Payload only
        10 = Adaptation field only
        11 = Adaptation field + payload
        """
        has_af = self.has_adaptation_field
        has_pl = self.has_payload
        
        if has_af and has_pl:
            return 0b11
        elif has_af:
            return 0b10
        elif has_pl:
            return 0b01
        else:
            return 0b01  # Default to payload-only for empty packets
    
    def to_bytes(self) -> bytes:
        """
        Serialize packet to 188 bytes.
        
        Returns:
            188-byte packet data
        """
        # Build header
        header = bytearray(HEADER_SIZE)
        
        # Byte 0: Sync
        header[0] = SYNC_BYTE
        
        # Bytes 1-2: Error | PUSI | Priority | PID
        header[1] = ((self.transport_error & 1) << 7 |
                     (self.payload_unit_start & 1) << 6 |
                     (self.transport_priority & 1) << 5 |
                     (self.pid >> 8) & 0x1F)
        header[2] = self.pid & 0xFF
        
        # Byte 3: Scrambling | Adaptation | CC
        header[3] = ((self.scrambling_control & 0x3) << 6 |
                     (self.adaptation_field_control & 0x3) << 4 |
                     (self.continuity_counter & 0xF))
        
        # Build packet
        packet = bytearray(header)
        
        # Add adaptation field if present
        if self.adaptation_field is not None:
            packet.extend(self.adaptation_field.to_bytes())
        
        # Add payload
        packet.extend(self.payload)
        
        # Pad with stuffing bytes (0xFF) to 188 bytes
        if len(packet) < PACKET_SIZE:
            packet.extend(b'\xFF' * (PACKET_SIZE - len(packet)))
        
        return bytes(packet)
    
    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray]) -> 'Packet':
        """
        Parse packet from 188 bytes.
        
        Args:
            data: 188 bytes of packet data
            
        Returns:
            Parsed Packet object
            
        Raises:
            ValueError: If data is not valid TS packet
        """
        if len(data) < PACKET_SIZE:
            raise ValueError(f"Packet must be {PACKET_SIZE} bytes, got {len(data)}")
        
        # Check sync byte
        if data[0] != SYNC_BYTE:
            raise ValueError(f"Invalid sync byte: 0x{data[0]:02X}, expected 0x47")
        
        # Parse header
        transport_error = bool(data[1] & 0x80)
        payload_unit_start = bool(data[1] & 0x40)
        transport_priority = bool(data[1] & 0x20)
        pid = ((data[1] & 0x1F) << 8) | data[2]
        
        scrambling_control = (data[3] >> 6) & 0x03
        adaptation_field_control = (data[3] >> 4) & 0x03
        continuity_counter = data[3] & 0x0F
        
        # Parse adaptation field and payload
        pos = HEADER_SIZE
        adaptation_field = None
        payload = b''
        
        has_adaptation = adaptation_field_control in (0b10, 0b11)
        has_payload = adaptation_field_control in (0b01, 0b11)
        
        if has_adaptation:
            adaptation_length = data[pos]
            if adaptation_length > 0:
                adaptation_field = AdaptationField.from_bytes(
                    data[pos:pos + 1 + adaptation_length]
                )
            pos += 1 + adaptation_length
        
        if has_payload:
            payload = bytes(data[pos:PACKET_SIZE])
            # Strip trailing stuffing bytes for cleaner payload
            # (only if this is a section, indicated by PUSI)
            
        return cls(
            pid=pid,
            payload=payload,
            transport_error=transport_error,
            payload_unit_start=payload_unit_start,
            transport_priority=transport_priority,
            scrambling_control=scrambling_control,
            continuity_counter=continuity_counter,
            adaptation_field=adaptation_field,
        )
    
    @classmethod
    def null_packet(cls) -> 'Packet':
        """Create a null (stuffing) packet."""
        return cls(pid=PID_NULL, payload=b'\xFF' * MAX_PAYLOAD_SIZE)
    
    def __repr__(self) -> str:
        return (f"Packet(pid=0x{self.pid:04X}, "
                f"cc={self.continuity_counter}, "
                f"payload={len(self.payload)}B"
                f"{', PUSI' if self.payload_unit_start else ''}"
                f"{', AF' if self.has_adaptation_field else ''})")
