"""
CRC-32 Calculation for DVB/MPEG-2 Transport Stream

DVB uses CRC-32/MPEG-2 polynomial for PSI table integrity checking.
Polynomial: x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
In hex: 0x04C11DB7
"""

import numpy as np
from typing import Union


class CRC32:
    """
    CRC-32 calculator using MPEG-2 polynomial.
    
    The DVB/MPEG-2 CRC uses:
    - Polynomial: 0x04C11DB7
    - Initial value: 0xFFFFFFFF
    - No final XOR (unlike CRC-32/IEEE)
    - MSB-first processing
    
    Example:
        >>> crc = CRC32()
        >>> result = crc.calculate(b'\\x00\\x01\\x02\\x03')
        >>> print(hex(result))
    """
    
    # MPEG-2 polynomial (bit-reversed from 0x04C11DB7)
    POLYNOMIAL = 0x04C11DB7
    INITIAL = 0xFFFFFFFF
    
    def __init__(self, use_table: bool = True):
        """
        Initialize CRC-32 calculator.
        
        Args:
            use_table: Use lookup table for speed (True) or compute directly (False)
        """
        self.use_table = use_table
        if use_table:
            self._table = self._generate_table()
    
    def _generate_table(self) -> np.ndarray:
        """Generate 256-entry lookup table for fast CRC computation."""
        table = np.zeros(256, dtype=np.uint32)
        
        for i in range(256):
            crc = i << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ self.POLYNOMIAL
                else:
                    crc <<= 1
                crc &= 0xFFFFFFFF
            table[i] = crc
        
        return table
    
    def calculate(self, data: Union[bytes, bytearray, np.ndarray]) -> int:
        """
        Calculate CRC-32 of data.
        
        Args:
            data: Input bytes to calculate CRC over
            
        Returns:
            32-bit CRC value
        """
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        if self.use_table:
            return self._calculate_table(data)
        else:
            return self._calculate_slow(data)
    
    def _calculate_table(self, data: bytes) -> int:
        """Fast CRC calculation using lookup table."""
        crc = self.INITIAL
        
        for byte in data:
            table_index = ((crc >> 24) ^ byte) & 0xFF
            crc = ((crc << 8) ^ int(self._table[table_index])) & 0xFFFFFFFF
        
        return crc
    
    def _calculate_slow(self, data: bytes) -> int:
        """
        Slow but educational bit-by-bit CRC calculation.
        
        This directly implements the polynomial division, useful for
        understanding how CRC works.
        """
        crc = self.INITIAL
        
        for byte in data:
            crc ^= (byte << 24)
            
            for _ in range(8):
                if crc & 0x80000000:
                    crc = ((crc << 1) ^ self.POLYNOMIAL) & 0xFFFFFFFF
                else:
                    crc = (crc << 1) & 0xFFFFFFFF
        
        return crc
    
    def verify(self, data: Union[bytes, bytearray], expected_crc: int) -> bool:
        """
        Verify CRC-32 of data matches expected value.
        
        Args:
            data: Input bytes (not including the CRC bytes)
            expected_crc: Expected CRC value
            
        Returns:
            True if CRC matches
        """
        return self.calculate(data) == expected_crc
    
    def append(self, data: Union[bytes, bytearray]) -> bytes:
        """
        Calculate CRC and append it to data (big-endian).
        
        Args:
            data: Input bytes
            
        Returns:
            Data with 4-byte CRC appended
        """
        crc = self.calculate(data)
        return bytes(data) + crc.to_bytes(4, byteorder='big')
    
    @staticmethod
    def to_bytes(crc: int) -> bytes:
        """Convert CRC value to 4 bytes (big-endian)."""
        return crc.to_bytes(4, byteorder='big')
    
    @staticmethod
    def from_bytes(data: bytes) -> int:
        """Extract CRC value from 4 bytes (big-endian)."""
        return int.from_bytes(data[:4], byteorder='big')


# Module-level instance for convenience
_default_crc = CRC32()


def crc32(data: Union[bytes, bytearray, np.ndarray]) -> int:
    """Calculate CRC-32 of data using default calculator."""
    return _default_crc.calculate(data)


def crc32_verify(data: Union[bytes, bytearray], expected: int) -> bool:
    """Verify CRC-32 of data matches expected value."""
    return _default_crc.verify(data, expected)
