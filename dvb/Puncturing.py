"""
DVB Puncturing for Rate Matching

The convolutional encoder produces rate 1/2 code. To achieve
higher spectral efficiency, some bits are "punctured" (not transmitted).

Supported code rates and their puncturing patterns:
- 1/2: No puncturing (transmit all bits)
- 2/3: Puncture pattern [1 1; 1 0] - transmit 3 of 4 bits
- 3/4: Puncture pattern [1 1 0; 1 0 1] - transmit 4 of 6 bits
- 5/6: Puncture pattern [1 1 0 1 0; 1 0 1 0 1] - transmit 6 of 10 bits
- 7/8: Puncture pattern [1 1 0 1 0 1 0; 1 0 1 0 1 0 1] - transmit 8 of 14 bits

Reference: ETSI EN 300 744 Section 4.3.4
"""

import numpy as np
from typing import Union, Tuple


# Puncturing patterns for each code rate
# Each row is for one encoder output (G1 and G2)
# 1 = transmit, 0 = puncture
PUNCTURE_PATTERNS = {
    '1/2': np.array([[1], [1]], dtype=np.uint8),
    '2/3': np.array([[1, 1], [1, 0]], dtype=np.uint8),
    '3/4': np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8),
    '5/6': np.array([[1, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.uint8),
    '7/8': np.array([[1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8),
}


class Puncturer:
    """
    DVB puncturer for rate matching.
    
    Takes rate 1/2 encoded bits and removes (punctures) some bits
    according to the selected code rate pattern.
    
    Attributes:
        rate: Code rate string ('1/2', '2/3', etc.)
        pattern: Puncturing pattern matrix
        
    Example:
        >>> punct = Puncturer('2/3')
        >>> punctured = punct.puncture(encoded_bits)
        >>> # 4 input bits -> 3 output bits
    """
    
    def __init__(self, rate: str = '1/2'):
        """
        Initialize puncturer.
        
        Args:
            rate: Code rate ('1/2', '2/3', '3/4', '5/6', '7/8')
        """
        if rate not in PUNCTURE_PATTERNS:
            raise ValueError(f"Unsupported rate: {rate}")
        
        self.rate = rate
        self.pattern = PUNCTURE_PATTERNS[rate]
        
        # Calculate input/output ratio
        self.pattern_length = self.pattern.shape[1]
        self.input_per_period = 2 * self.pattern_length  # 2 bits per input bit
        self.output_per_period = int(np.sum(self.pattern))
    
    def puncture(self, data: np.ndarray) -> np.ndarray:
        """
        Puncture encoded bits.
        
        Args:
            data: Rate 1/2 encoded bits (from convolutional encoder)
                  Format: [G1_0, G2_0, G1_1, G2_1, ...]
            
        Returns:
            Punctured bits (fewer bits)
        """
        if len(data) % self.input_per_period != 0:
            # Pad to multiple of pattern period
            pad_len = self.input_per_period - (len(data) % self.input_per_period)
            data = np.concatenate([data, np.zeros(pad_len, dtype=np.uint8)])
        
        num_periods = len(data) // self.input_per_period
        output = []
        
        for p in range(num_periods):
            start = p * self.input_per_period
            
            for i in range(self.pattern_length):
                # G1 output at position 2*i
                if self.pattern[0, i]:
                    output.append(data[start + 2*i])
                
                # G2 output at position 2*i + 1
                if self.pattern[1, i]:
                    output.append(data[start + 2*i + 1])
        
        return np.array(output, dtype=np.uint8)
    
    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output length for given input length.
        
        Args:
            input_length: Number of rate 1/2 encoded bits
            
        Returns:
            Number of punctured bits
        """
        periods = (input_length + self.input_per_period - 1) // self.input_per_period
        return periods * self.output_per_period
    
    @property
    def rate_numerator(self) -> int:
        """Get numerator of code rate fraction."""
        parts = self.rate.split('/')
        return int(parts[0])
    
    @property
    def rate_denominator(self) -> int:
        """Get denominator of code rate fraction."""
        parts = self.rate.split('/')
        return int(parts[1])


class Depuncturer:
    """
    DVB depuncturer for receiver.
    
    Inserts erasure markers where bits were punctured,
    allowing the Viterbi decoder to handle missing bits.
    
    Attributes:
        rate: Code rate string
        pattern: Puncturing pattern
        
    Example:
        >>> depunct = Depuncturer('2/3')
        >>> depunctured = depunct.depuncture(received_bits)
        >>> # 3 input bits -> 4 output bits (with erasures)
    """
    
    # Value to insert for punctured (erased) positions
    # For hard decision: use 0.5 or special marker
    # For soft decision: use 0 (neutral)
    ERASURE = 0
    
    def __init__(self, rate: str = '1/2', soft_decision: bool = False):
        """
        Initialize depuncturer.
        
        Args:
            rate: Code rate
            soft_decision: Use soft decision values
        """
        if rate not in PUNCTURE_PATTERNS:
            raise ValueError(f"Unsupported rate: {rate}")
        
        self.rate = rate
        self.pattern = PUNCTURE_PATTERNS[rate]
        self.soft_decision = soft_decision
        
        self.pattern_length = self.pattern.shape[1]
        self.input_per_period = int(np.sum(self.pattern))
        self.output_per_period = 2 * self.pattern_length
    
    def depuncture(self, data: np.ndarray, 
                   erasure_value: float = None) -> np.ndarray:
        """
        Depuncture received bits.
        
        Inserts erasure markers at punctured positions.
        
        Args:
            data: Received punctured bits
            erasure_value: Value to insert for erasures (default: 0 or 0.5)
            
        Returns:
            Depunctured bits with erasures
        """
        if erasure_value is None:
            erasure_value = 0.0 if self.soft_decision else 0.5
        
        if len(data) % self.input_per_period != 0:
            # Pad to multiple of pattern period
            pad_len = self.input_per_period - (len(data) % self.input_per_period)
            data = np.concatenate([data, 
                                   np.full(pad_len, erasure_value, dtype=data.dtype)])
        
        num_periods = len(data) // self.input_per_period
        
        dtype = data.dtype if self.soft_decision else np.float32
        output = np.full(num_periods * self.output_per_period, erasure_value, dtype=dtype)
        
        in_idx = 0
        for p in range(num_periods):
            out_start = p * self.output_per_period
            
            for i in range(self.pattern_length):
                # G1 position
                if self.pattern[0, i]:
                    output[out_start + 2*i] = data[in_idx]
                    in_idx += 1
                
                # G2 position
                if self.pattern[1, i]:
                    output[out_start + 2*i + 1] = data[in_idx]
                    in_idx += 1
        
        return output
    
    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output length for given input length.
        
        Args:
            input_length: Number of punctured bits
            
        Returns:
            Number of depunctured bits
        """
        periods = (input_length + self.input_per_period - 1) // self.input_per_period
        return periods * self.output_per_period


def get_puncture_pattern(rate: str) -> np.ndarray:
    """
    Get puncturing pattern for a code rate.
    
    Args:
        rate: Code rate string
        
    Returns:
        Puncturing pattern matrix (2 x N)
    """
    return PUNCTURE_PATTERNS.get(rate, PUNCTURE_PATTERNS['1/2']).copy()


def calculate_bits_per_symbol(constellation: str, code_rate: str) -> float:
    """
    Calculate effective bits per OFDM symbol.
    
    Args:
        constellation: 'QPSK', '16QAM', or '64QAM'
        code_rate: '1/2', '2/3', '3/4', '5/6', or '7/8'
        
    Returns:
        Bits per constellation symbol after coding
    """
    bits_per_symbol = {'QPSK': 2, '16QAM': 4, '64QAM': 6}
    rates = {'1/2': 0.5, '2/3': 2/3, '3/4': 0.75, '5/6': 5/6, '7/8': 7/8}
    
    return bits_per_symbol[constellation] * rates[code_rate]
