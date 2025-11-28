"""
DVB QAM Constellation Mapping

DVB-T supports three constellation types:
- QPSK (4-QAM): 2 bits per symbol
- 16-QAM: 4 bits per symbol
- 64-QAM: 6 bits per symbol

All constellations use Gray coding for minimal bit errors when
decoding to adjacent symbol positions.

The constellations are normalized for unit average power.

Reference: ETSI EN 300 744 Section 4.3.5
"""

import numpy as np
from typing import Union, Tuple, Dict
from math import sqrt


class QAMMapper:
    """
    DVB QAM constellation mapper.
    
    Maps groups of bits to complex constellation symbols using
    Gray coding.
    
    Attributes:
        constellation: Type ('QPSK', '16QAM', '64QAM')
        bits_per_symbol: Number of bits mapped to each symbol
        
    Example:
        >>> mapper = QAMMapper('16QAM')
        >>> symbols = mapper.map(bits)
        >>> assert symbols.dtype == np.complex64
    """
    
    def __init__(self, constellation: str = 'QPSK'):
        """
        Initialize QAM mapper.
        
        Args:
            constellation: 'QPSK', '16QAM', or '64QAM'
        """
        if constellation not in ('QPSK', '16QAM', '64QAM'):
            raise ValueError(f"Unsupported constellation: {constellation}")
        
        self.constellation = constellation
        self.bits_per_symbol = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[constellation]
        
        # Build lookup table
        self._build_table()
    
    def _build_table(self) -> None:
        """Build mapping lookup table."""
        if self.constellation == 'QPSK':
            self._table = self._build_qpsk()
        elif self.constellation == '16QAM':
            self._table = self._build_16qam()
        else:
            self._table = self._build_64qam()
    
    def _build_qpsk(self) -> np.ndarray:
        """
        Build QPSK constellation table.
        
        QPSK mapping (Gray coded):
            b0 b1 -> symbol
            0  0  -> (-1-1j)/√2 (third quadrant)
            0  1  -> (-1+1j)/√2 (second quadrant)
            1  0  -> (+1-1j)/√2 (fourth quadrant)
            1  1  -> (+1+1j)/√2 (first quadrant)
            
        b0 determines I sign, b1 determines Q sign.
        """
        norm = 1.0 / sqrt(2)
        table = np.zeros(4, dtype=np.complex64)
        
        for bits in range(4):
            b0 = (bits >> 1) & 1
            b1 = bits & 1
            
            # Map bits to constellation points
            I = (2 * b0 - 1) * norm
            Q = (2 * b1 - 1) * norm
            
            table[bits] = complex(I, Q)
        
        return table
    
    def _build_16qam(self) -> np.ndarray:
        """
        Build 16-QAM constellation table.
        
        16-QAM mapping (Gray coded):
            I values: -3, -1, +1, +3 (normalized by √10)
            Q values: -3, -1, +1, +3 (normalized by √10)
            
        Bits b0,b1 -> I; bits b2,b3 -> Q
        """
        norm = 1.0 / sqrt(10)
        table = np.zeros(16, dtype=np.complex64)
        
        # Gray code mapping for 2 bits -> amplitude
        # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
        amp_map = {
            0b00: -3,
            0b01: -1,
            0b11: +1,
            0b10: +3,
        }
        
        for bits in range(16):
            b_I = (bits >> 2) & 0b11
            b_Q = bits & 0b11
            
            I = amp_map[b_I] * norm
            Q = amp_map[b_Q] * norm
            
            table[bits] = complex(I, Q)
        
        return table
    
    def _build_64qam(self) -> np.ndarray:
        """
        Build 64-QAM constellation table.
        
        64-QAM mapping (Gray coded):
            I values: -7, -5, -3, -1, +1, +3, +5, +7 (normalized by √42)
            Q values: -7, -5, -3, -1, +1, +3, +5, +7 (normalized by √42)
            
        Bits b0,b1,b2 -> I; bits b3,b4,b5 -> Q
        """
        norm = 1.0 / sqrt(42)
        table = np.zeros(64, dtype=np.complex64)
        
        # Gray code mapping for 3 bits -> amplitude
        # 000 -> -7, 001 -> -5, 011 -> -3, 010 -> -1
        # 110 -> +1, 111 -> +3, 101 -> +5, 100 -> +7
        amp_map = {
            0b000: -7,
            0b001: -5,
            0b011: -3,
            0b010: -1,
            0b110: +1,
            0b111: +3,
            0b101: +5,
            0b100: +7,
        }
        
        for bits in range(64):
            b_I = (bits >> 3) & 0b111
            b_Q = bits & 0b111
            
            I = amp_map[b_I] * norm
            Q = amp_map[b_Q] * norm
            
            table[bits] = complex(I, Q)
        
        return table
    
    def map(self, bits: np.ndarray) -> np.ndarray:
        """
        Map bits to QAM symbols.
        
        Args:
            bits: Input bits (length must be multiple of bits_per_symbol)
            
        Returns:
            Complex QAM symbols
        """
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(f"Bit length must be multiple of {self.bits_per_symbol}")
        
        num_symbols = len(bits) // self.bits_per_symbol
        
        # Pack bits into symbol indices
        indices = np.zeros(num_symbols, dtype=np.int32)
        
        for i in range(self.bits_per_symbol):
            indices |= bits[i::self.bits_per_symbol].astype(np.int32) << (self.bits_per_symbol - 1 - i)
        
        return self._table[indices]
    
    def map_single(self, bits: Union[int, Tuple[int, ...]]) -> complex:
        """
        Map a single bit group to symbol.
        
        Args:
            bits: Integer (packed bits) or tuple of bits
            
        Returns:
            Complex symbol
        """
        if isinstance(bits, (tuple, list)):
            index = 0
            for b in bits:
                index = (index << 1) | (b & 1)
        else:
            index = int(bits) & ((1 << self.bits_per_symbol) - 1)
        
        return complex(self._table[index])


class QAMDemapper:
    """
    DVB QAM constellation demapper.
    
    Converts received complex symbols back to bits using
    hard or soft decision decoding.
    
    Attributes:
        constellation: Type ('QPSK', '16QAM', '64QAM')
        soft_output: Use soft decision output
        
    Example:
        >>> demapper = QAMDemapper('16QAM')
        >>> bits = demapper.demap(received_symbols)
    """
    
    def __init__(self, constellation: str = 'QPSK', soft_output: bool = False):
        """
        Initialize QAM demapper.
        
        Args:
            constellation: Constellation type
            soft_output: Return soft decisions (LLRs) instead of hard bits
        """
        self.constellation = constellation
        self.soft_output = soft_output
        self.bits_per_symbol = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[constellation]
        
        # Get mapper table for reference
        mapper = QAMMapper(constellation)
        self._table = mapper._table
        
        # Build inverse table for hard decision
        self._build_inverse_table()
    
    def _build_inverse_table(self) -> None:
        """Build lookup for nearest symbol (for hard decision)."""
        # For each symbol in table, store its bit pattern
        self._inverse = {}
        for i, symbol in enumerate(self._table):
            self._inverse[symbol] = i
    
    def demap(self, symbols: np.ndarray, 
              noise_variance: float = 1.0) -> np.ndarray:
        """
        Demap symbols to bits.
        
        Args:
            symbols: Received complex symbols
            noise_variance: Noise variance for soft decision (sigma^2)
            
        Returns:
            Bits (hard decision) or LLRs (soft decision)
        """
        if self.soft_output:
            return self._demap_soft(symbols, noise_variance)
        else:
            return self._demap_hard(symbols)
    
    def _demap_hard(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard decision demapping (vectorized).
        
        Finds nearest constellation point for each received symbol.
        """
        num_symbols = len(symbols)
        
        # Compute distances from all symbols to all constellation points
        # symbols: (N,), table: (M,) -> distances: (N, M)
        distances = np.abs(symbols[:, np.newaxis] - self._table[np.newaxis, :])
        
        # Find nearest constellation point for each symbol
        nearest_idx = np.argmin(distances, axis=1)
        
        # Extract bits from indices (vectorized)
        bits = np.zeros(num_symbols * self.bits_per_symbol, dtype=np.uint8)
        for b in range(self.bits_per_symbol):
            shift = self.bits_per_symbol - 1 - b
            bits[b::self.bits_per_symbol] = (nearest_idx >> shift) & 1
        
        return bits
    
    def _demap_soft(self, symbols: np.ndarray, 
                    noise_variance: float) -> np.ndarray:
        """
        Soft decision demapping.
        
        Computes log-likelihood ratios (LLRs) for each bit.
        LLR > 0 means bit is more likely 0
        LLR < 0 means bit is more likely 1
        """
        num_symbols = len(symbols)
        llrs = np.zeros(num_symbols * self.bits_per_symbol, dtype=np.float32)
        
        num_points = len(self._table)
        
        for i, s in enumerate(symbols):
            # Compute squared distances to all constellation points
            distances_sq = np.abs(s - self._table) ** 2
            
            # For each bit position
            for b in range(self.bits_per_symbol):
                bit_mask = 1 << (self.bits_per_symbol - 1 - b)
                
                # Points where this bit is 0 vs 1
                zero_mask = np.array([(idx & bit_mask) == 0 for idx in range(num_points)])
                one_mask = ~zero_mask
                
                # Min distance to 0-points and 1-points
                min_d0 = np.min(distances_sq[zero_mask])
                min_d1 = np.min(distances_sq[one_mask])
                
                # LLR = (d1² - d0²) / (2σ²)
                # Approximation: use min distance instead of full sum
                llrs[i * self.bits_per_symbol + b] = (min_d1 - min_d0) / (2 * noise_variance)
        
        return llrs
    
    def demap_single(self, symbol: complex) -> int:
        """
        Demap single symbol to bit pattern.
        
        Args:
            symbol: Complex received symbol
            
        Returns:
            Integer with packed bits
        """
        distances = np.abs(symbol - self._table)
        return int(np.argmin(distances))


def get_constellation_points(constellation: str) -> np.ndarray:
    """
    Get all constellation points for a modulation type.
    
    Args:
        constellation: 'QPSK', '16QAM', or '64QAM'
        
    Returns:
        Complex array of constellation points
    """
    mapper = QAMMapper(constellation)
    return mapper._table.copy()


def get_normalization_factor(constellation: str) -> float:
    """
    Get normalization factor for constellation.
    
    The factor ensures unit average power.
    
    Args:
        constellation: Modulation type
        
    Returns:
        Normalization factor
    """
    factors = {
        'QPSK': 1.0 / sqrt(2),
        '16QAM': 1.0 / sqrt(10),
        '64QAM': 1.0 / sqrt(42),
    }
    return factors.get(constellation, 1.0)
