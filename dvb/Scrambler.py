"""
DVB Energy Dispersal Scrambler

Energy dispersal ensures a uniform distribution of '0' and '1' bits in the
transmitted signal, preventing long runs of identical bits which could
cause synchronization problems or DC buildup.

DVB uses a PRBS (Pseudo-Random Binary Sequence) generator:
    Polynomial: x^15 + x^14 + 1
    Initial state: 100101010000000 (after SYNC byte initialization)

The scrambler is reset at the start of each 8-packet superframe.
The sync byte (0x47) of the first packet is inverted to 0xB8 to mark
the superframe boundary.

Reference: ETSI EN 300 744 Section 4.3.1
"""

import numpy as np
from typing import Union


class Scrambler:
    """
    DVB PRBS scrambler for energy dispersal.
    
    Uses the polynomial x^15 + x^14 + 1 to generate a pseudo-random
    sequence that is XORed with the transport stream data.
    
    Attributes:
        polynomial: LFSR polynomial (default x^15 + x^14 + 1)
        init_state: Initial LFSR state
        
    Example:
        >>> scrambler = Scrambler()
        >>> scrambled = scrambler.scramble(ts_packets)
        >>> original = scrambler.descramble(scrambled)
    """
    
    # PRBS initialization sequence (after processing SYNC)
    # Binary: 100101010000000 = 0x4A80
    INIT_STATE = 0b100101010000000
    
    # Number of packets per superframe
    SUPERFRAME_PACKETS = 8
    
    def __init__(self, use_fast: bool = True):
        """
        Initialize scrambler.
        
        Args:
            use_fast: Use numpy vectorized operations (True) or 
                      bit-by-bit for education (False)
        """
        self.use_fast = use_fast
        self._state = self.INIT_STATE
    
    def reset(self) -> None:
        """Reset LFSR to initial state."""
        self._state = self.INIT_STATE
    
    def _lfsr_step(self) -> int:
        """
        Single step of LFSR.
        
        Returns:
            Output bit (0 or 1)
        """
        # Output bit is bit 14 (MSB of 15-bit register)
        output = (self._state >> 14) & 1
        
        # Feedback = bit14 XOR bit13
        feedback = ((self._state >> 14) ^ (self._state >> 13)) & 1
        
        # Shift left and insert feedback
        self._state = ((self._state << 1) | feedback) & 0x7FFF
        
        return output
    
    def _generate_prbs_byte(self) -> int:
        """Generate one byte of PRBS sequence."""
        byte = 0
        for i in range(8):
            bit = self._lfsr_step()
            byte = (byte << 1) | bit
        return byte
    
    def _generate_prbs_sequence(self, length: int) -> bytes:
        """
        Generate PRBS sequence of given length.
        
        Args:
            length: Number of bytes to generate
            
        Returns:
            PRBS byte sequence
        """
        return bytes(self._generate_prbs_byte() for _ in range(length))
    
    def scramble(self, data: Union[bytes, bytearray, np.ndarray], 
                 is_superframe_start: bool = True) -> bytes:
        """
        Scramble data using PRBS.
        
        For DVB-T, this is applied to transport stream packets.
        The first byte of each superframe (8 packets) has its SYNC
        byte inverted (0x47 -> 0xB8) to mark the boundary.
        
        Args:
            data: Input data (should be multiple of 188 bytes)
            is_superframe_start: Reset LFSR at start
            
        Returns:
            Scrambled data
        """
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        data = bytearray(data)
        
        if is_superframe_start:
            self.reset()
        
        result = bytearray()
        packet_count = len(data) // 188
        
        for pkt_idx in range(packet_count):
            offset = pkt_idx * 188
            packet = data[offset:offset + 188]
            
            # Check if this is start of superframe
            is_sf_start = (pkt_idx % self.SUPERFRAME_PACKETS == 0)
            
            if is_sf_start:
                self.reset()
                # Invert sync byte to mark superframe
                scrambled_pkt = bytearray([packet[0] ^ 0xFF])  # 0x47 -> 0xB8
            else:
                scrambled_pkt = bytearray([packet[0]])  # Keep sync byte
            
            # Scramble remaining 187 bytes
            if self.use_fast:
                prbs = self._generate_prbs_sequence(187)
                scrambled_pkt.extend(bytes(p ^ d for p, d in zip(prbs, packet[1:])))
            else:
                for byte in packet[1:]:
                    prbs_byte = self._generate_prbs_byte()
                    scrambled_pkt.append(byte ^ prbs_byte)
            
            result.extend(scrambled_pkt)
        
        return bytes(result)
    
    def descramble(self, data: Union[bytes, bytearray, np.ndarray],
                   is_superframe_start: bool = True) -> bytes:
        """
        Descramble data using PRBS.
        
        Since PRBS XOR is self-inverse, this uses the same logic as scramble
        but restores the sync bytes.
        
        Args:
            data: Scrambled data
            is_superframe_start: Reset LFSR at start
            
        Returns:
            Original data
        """
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        data = bytearray(data)
        
        if is_superframe_start:
            self.reset()
        
        result = bytearray()
        packet_count = len(data) // 188
        
        for pkt_idx in range(packet_count):
            offset = pkt_idx * 188
            packet = data[offset:offset + 188]
            
            # Check if this is start of superframe
            is_sf_start = (pkt_idx % self.SUPERFRAME_PACKETS == 0)
            
            if is_sf_start:
                self.reset()
                # Restore sync byte
                descrambled_pkt = bytearray([packet[0] ^ 0xFF])  # 0xB8 -> 0x47
            else:
                descrambled_pkt = bytearray([packet[0]])  # Keep sync byte
            
            # Descramble remaining 187 bytes
            prbs = self._generate_prbs_sequence(187)
            descrambled_pkt.extend(bytes(p ^ d for p, d in zip(prbs, packet[1:])))
            
            result.extend(descrambled_pkt)
        
        return bytes(result)
    
    def scramble_bytes(self, data: Union[bytes, bytearray], reset: bool = True) -> bytes:
        """
        Simple byte scrambling without packet structure.
        
        Useful for testing or non-TS data.
        
        Args:
            data: Input bytes
            reset: Reset LFSR before scrambling
            
        Returns:
            Scrambled bytes
        """
        if reset:
            self.reset()
        
        prbs = self._generate_prbs_sequence(len(data))
        return bytes(p ^ d for p, d in zip(prbs, data))
    
    # Descrambling is identical to scrambling for XOR cipher
    descramble_bytes = scramble_bytes
    
    @staticmethod
    def get_prbs_table(length: int = 1503) -> np.ndarray:
        """
        Generate PRBS lookup table.
        
        Returns 1503 bytes (187 * 8 + 7) covering one superframe.
        
        Args:
            length: Table length in bytes
            
        Returns:
            numpy array of PRBS bytes
        """
        scr = Scrambler(use_fast=False)
        scr.reset()
        return np.array([scr._generate_prbs_byte() for _ in range(length)], 
                        dtype=np.uint8)


# Generate lookup table for fast scrambling
_PRBS_TABLE = None


def get_prbs_table() -> np.ndarray:
    """Get cached PRBS table."""
    global _PRBS_TABLE
    if _PRBS_TABLE is None:
        _PRBS_TABLE = Scrambler.get_prbs_table()
    return _PRBS_TABLE
