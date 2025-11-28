"""
DVB Inner Interleaver (Bit and Symbol Interleaving)

The inner interleaver operates after puncturing and before QAM mapping.
It consists of two stages:

1. Bit Interleaving: Demultiplexes bits into v sub-streams (v = bits per symbol)
   and applies different permutations to each sub-stream.

2. Symbol Interleaving: Permutes symbols within each OFDM symbol to spread
   errors across carriers.

For 2K mode with 1512 data carriers:
- v = 2 for QPSK, 4 for 16-QAM, 6 for 64-QAM
- Symbol interleaver has depth 1512 (one OFDM symbol)

Reference: ETSI EN 300 744 Section 4.3.5
"""

import numpy as np
from typing import List


class BitInterleaver:
    """
    DVB bit interleaver for inner interleaving.
    
    Demultiplexes input bits into v parallel streams and applies
    different permutations to each stream based on constellation.
    
    Attributes:
        constellation: Modulation type ('QPSK', '16QAM', '64QAM')
        v: Number of bits per symbol
        
    Example:
        >>> interleaver = BitInterleaver('64QAM')
        >>> interleaved = interleaver.interleave(punctured_bits)
    """
    
    # Permutation polynomials for each bit stream (DVB-T specification)
    # H(w) for each stream, where w is bit position within 126-bit block
    PERMUTATIONS = {
        0: lambda w: w,  # I0: no permutation
        1: lambda w: (w + 63) % 126,  # I1
        2: lambda w: (w + 105) % 126,  # I2  
        3: lambda w: (w + 42) % 126,  # I3
        4: lambda w: (w + 21) % 126,  # I4
        5: lambda w: (w + 84) % 126,  # I5
    }
    
    # Block size for bit interleaving
    BLOCK_SIZE = 126
    
    def __init__(self, constellation: str = 'QPSK', mode: str = '2K'):
        """
        Initialize bit interleaver.
        
        Args:
            constellation: 'QPSK', '16QAM', or '64QAM'
            mode: '2K' or '8K'
        """
        self.constellation = constellation
        self.mode = mode
        
        # Bits per symbol
        self.v = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[constellation]
        
        # Build permutation tables
        self._build_permutation_tables()
    
    def _build_permutation_tables(self) -> None:
        """Pre-compute permutation lookup tables."""
        self._perm_tables = []
        
        for i in range(self.v):
            table = np.array([self.PERMUTATIONS[i](w) 
                             for w in range(self.BLOCK_SIZE)], dtype=np.int32)
            self._perm_tables.append(table)
    
    def interleave(self, data: np.ndarray) -> np.ndarray:
        """
        Interleave bits.
        
        Input bits are demultiplexed into v streams, each stream is
        permuted, then bits are re-multiplexed for symbol mapping.
        
        Args:
            data: Input bits (must be multiple of v * 126)
            
        Returns:
            Interleaved bits
        """
        # Ensure length is multiple of block size * v
        block_bits = self.v * self.BLOCK_SIZE
        if len(data) % block_bits != 0:
            pad_len = block_bits - (len(data) % block_bits)
            data = np.concatenate([data, np.zeros(pad_len, dtype=data.dtype)])
        
        num_blocks = len(data) // block_bits
        output = np.zeros_like(data)
        
        for b in range(num_blocks):
            block_start = b * block_bits
            
            # Demultiplex into v streams
            streams = [data[block_start + i::self.v][:self.BLOCK_SIZE] 
                       for i in range(self.v)]
            
            # Apply permutation to each stream
            permuted = []
            for i, stream in enumerate(streams):
                perm_stream = np.zeros(self.BLOCK_SIZE, dtype=stream.dtype)
                for w in range(self.BLOCK_SIZE):
                    perm_stream[self._perm_tables[i][w]] = stream[w]
                permuted.append(perm_stream)
            
            # Re-multiplex
            for i in range(self.v):
                output[block_start + i::self.v][:self.BLOCK_SIZE] = permuted[i]
        
        return output
    
    def deinterleave(self, data: np.ndarray) -> np.ndarray:
        """
        Deinterleave bits (inverse operation).
        
        Args:
            data: Interleaved bits
            
        Returns:
            Original bit order
        """
        block_bits = self.v * self.BLOCK_SIZE
        if len(data) % block_bits != 0:
            pad_len = block_bits - (len(data) % block_bits)
            data = np.concatenate([data, np.zeros(pad_len, dtype=data.dtype)])
        
        num_blocks = len(data) // block_bits
        output = np.zeros_like(data)
        
        for b in range(num_blocks):
            block_start = b * block_bits
            
            # Demultiplex
            streams = [data[block_start + i::self.v][:self.BLOCK_SIZE]
                       for i in range(self.v)]
            
            # Apply inverse permutation
            depermuted = []
            for i, stream in enumerate(streams):
                deperm_stream = np.zeros(self.BLOCK_SIZE, dtype=stream.dtype)
                for w in range(self.BLOCK_SIZE):
                    deperm_stream[w] = stream[self._perm_tables[i][w]]
                depermuted.append(deperm_stream)
            
            # Re-multiplex
            for i in range(self.v):
                output[block_start + i::self.v][:self.BLOCK_SIZE] = depermuted[i]
        
        return output


class SymbolInterleaver:
    """
    DVB symbol interleaver.
    
    Permutes data carriers within each OFDM symbol to spread
    errors from frequency-selective fading.
    
    Attributes:
        mode: '2K', '8K', or 'audio'
        num_carriers: Number of data carriers per symbol
        
    Example:
        >>> interleaver = SymbolInterleaver('2K')
        >>> interleaved = interleaver.interleave(qam_symbols)
    """
    
    # Number of data carriers per mode
    DATA_CARRIERS = {
        '2K': 1512,
        '8K': 6048,
        'audio': 18,  # 24 carriers - 6 pilots = 18 data carriers
    }
    
    # Maximum carrier index for permutation
    MAX_CARRIER = {
        '2K': 2048,  # 2^11
        '8K': 8192,  # 2^13
        'audio': 32,  # 2^5 (enough for 24 carriers)
    }
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize symbol interleaver.
        
        Args:
            mode: '2K', '8K', or 'audio'
        """
        self.mode = mode
        self.num_carriers = self.DATA_CARRIERS[mode]
        self._build_permutation()
    
    def _build_permutation(self) -> None:
        """Build symbol permutation table using R'(q) generator."""
        if self.mode == 'audio':
            # For audio mode with only 18 carriers, use a simple bit-reversal permutation
            # This provides good spreading without the complexity of LFSR
            Nr = 5  # 2^5 = 32 > 18
            perm_seq = []
            for q in range(32):
                # Bit-reverse the 5-bit value
                H = 0
                for i in range(Nr):
                    if q & (1 << i):
                        H |= 1 << (Nr - 1 - i)
                if H < self.num_carriers:
                    perm_seq.append(H)
            self._permutation = np.array(perm_seq[:self.num_carriers], dtype=np.int32)
        else:
            Nmax = self.MAX_CARRIER[self.mode]
            
            # Register size based on mode
            if self.mode == '2K':
                Nr = 11
            else:  # 8K
                Nr = 13
            
            # R'(q) is generated by:
            # R'_i(q+1) = R'_i+1(q) for i=0..Nr-2
            # R'_Nr-1(q+1) = R'_0(q) XOR R'_3(q) (for 2K)
            # R'_Nr-1(q+1) = R'_0(q) XOR R'_1(q) XOR R'_4(q) XOR R'_6(q) (for 8K)
            
            # Generate permutation sequence
            perm_seq = []
            R = [0] * Nr
            R[0] = 1  # Initial state
            
            for q in range(Nmax):
                # Calculate H(q) from R'(q)
                H = 0
                for i in range(Nr):
                    H |= R[Nr - 1 - i] << i
                
                if H < self.num_carriers:
                    perm_seq.append(H)
                
                # Update R for next iteration
                if self.mode == '2K':
                    new_bit = R[0] ^ R[3]
                else:  # 8K
                    new_bit = R[0] ^ R[1] ^ R[4] ^ R[6]
                
                R = R[1:] + [new_bit]
            
            self._permutation = np.array(perm_seq[:self.num_carriers], dtype=np.int32)
        
        # Build inverse permutation
        self._inv_permutation = np.zeros(self.num_carriers, dtype=np.int32)
        for i, p in enumerate(self._permutation):
            self._inv_permutation[p] = i
    
    def interleave(self, data: np.ndarray) -> np.ndarray:
        """
        Interleave symbols within OFDM symbol.
        
        Args:
            data: QAM symbols (complex or real values)
                  Length should be multiple of num_carriers
            
        Returns:
            Interleaved symbols
        """
        if len(data) % self.num_carriers != 0:
            raise ValueError(f"Data length must be multiple of {self.num_carriers}")
        
        num_symbols = len(data) // self.num_carriers
        output = np.zeros_like(data)
        
        for s in range(num_symbols):
            start = s * self.num_carriers
            end = start + self.num_carriers
            
            # y(H(q)) = x(q)
            output[start:end][self._permutation] = data[start:end]
        
        return output
    
    def deinterleave(self, data: np.ndarray) -> np.ndarray:
        """
        Deinterleave symbols (inverse operation).
        
        Args:
            data: Interleaved symbols
            
        Returns:
            Original symbol order
        """
        if len(data) % self.num_carriers != 0:
            raise ValueError(f"Data length must be multiple of {self.num_carriers}")
        
        num_symbols = len(data) // self.num_carriers
        output = np.zeros_like(data)
        
        for s in range(num_symbols):
            start = s * self.num_carriers
            end = start + self.num_carriers
            
            # x(q) = y(H(q))
            output[start:end] = data[start:end][self._permutation]
        
        return output


class InnerInterleaver:
    """
    Combined bit and symbol interleaver.
    
    Convenience class that chains bit and symbol interleaving.
    
    Example:
        >>> interleaver = InnerInterleaver('64QAM', '2K')
        >>> interleaved = interleaver.interleave(bits)
    """
    
    def __init__(self, constellation: str = 'QPSK', mode: str = '2K'):
        """
        Initialize combined interleaver.
        
        Args:
            constellation: 'QPSK', '16QAM', or '64QAM'
            mode: '2K' or '8K'
        """
        self.bit_interleaver = BitInterleaver(constellation, mode)
        self.symbol_interleaver = SymbolInterleaver(mode)
        self.v = self.bit_interleaver.v
    
    def interleave_bits(self, bits: np.ndarray) -> np.ndarray:
        """Apply bit interleaving only."""
        return self.bit_interleaver.interleave(bits)
    
    def interleave_symbols(self, symbols: np.ndarray) -> np.ndarray:
        """Apply symbol interleaving only."""
        return self.symbol_interleaver.interleave(symbols)
    
    def deinterleave_bits(self, bits: np.ndarray) -> np.ndarray:
        """Apply bit deinterleaving."""
        return self.bit_interleaver.deinterleave(bits)
    
    def deinterleave_symbols(self, symbols: np.ndarray) -> np.ndarray:
        """Apply symbol deinterleaving."""
        return self.symbol_interleaver.deinterleave(symbols)
