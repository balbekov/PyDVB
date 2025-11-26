"""
DVB Convolutional Code (Inner Code)

DVB uses a rate 1/2, constraint length K=7 convolutional code.

Generator polynomials (octal):
- G1 = 171 (0b1111001) = x^6 + x^5 + x^4 + x^3 + 1
- G2 = 133 (0b1011011) = x^6 + x^4 + x^3 + x + 1

The encoder takes 1 input bit and produces 2 output bits.
The mother code rate is 1/2, which is then punctured to achieve
higher rates (2/3, 3/4, 5/6, 7/8).

Reference: ETSI EN 300 744 Section 4.3.4
"""

import numpy as np
from typing import Union, Tuple, List


class ConvolutionalEncoder:
    """
    DVB convolutional encoder (K=7, rate 1/2).
    
    This encoder uses two generator polynomials to produce
    two output bits for each input bit.
    
    Attributes:
        K: Constraint length (7)
        G1: First generator polynomial (octal 171)
        G2: Second generator polynomial (octal 133)
        
    Example:
        >>> encoder = ConvolutionalEncoder()
        >>> encoded = encoder.encode(input_bits)
        >>> assert len(encoded) == 2 * len(input_bits)
    """
    
    # Constraint length
    K = 7
    
    # Generator polynomials (binary representation)
    G1 = 0b1111001  # Octal 171
    G2 = 0b1011011  # Octal 133
    
    def __init__(self, use_fast: bool = True):
        """
        Initialize encoder.
        
        Args:
            use_fast: Use lookup table (True) or bit-by-bit (False)
        """
        self.use_fast = use_fast
        self._state = 0  # 6-bit shift register
        
        if use_fast:
            self._build_tables()
    
    def _build_tables(self) -> None:
        """Build state transition and output tables."""
        # For each state (64) and input bit (2), compute:
        # - next_state
        # - output (2 bits)
        
        self._next_state = np.zeros((64, 2), dtype=np.uint8)
        self._output = np.zeros((64, 2), dtype=np.uint8)
        
        for state in range(64):
            for input_bit in range(2):
                # Shift register: input enters at MSB
                reg = (input_bit << 6) | state
                
                # Compute outputs using generator polynomials
                out1 = self._parity(reg & self.G1)
                out2 = self._parity(reg & self.G2)
                
                # Next state: drop LSB
                next_state = reg >> 1
                
                self._next_state[state, input_bit] = next_state
                self._output[state, input_bit] = (out1 << 1) | out2
    
    @staticmethod
    def _parity(x: int) -> int:
        """Compute parity (XOR of all bits) of x."""
        p = 0
        while x:
            p ^= (x & 1)
            x >>= 1
        return p
    
    def reset(self) -> None:
        """Reset encoder state to zero."""
        self._state = 0
    
    def encode(self, data: Union[bytes, bytearray, np.ndarray],
               terminate: bool = True) -> np.ndarray:
        """
        Encode data using convolutional code.
        
        Args:
            data: Input bytes
            terminate: Add tail bits to return to zero state
            
        Returns:
            Encoded bits as numpy array (rate 1/2)
        """
        # Convert bytes to bits
        if isinstance(data, (bytes, bytearray)):
            data = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        elif isinstance(data, np.ndarray) and data.dtype != np.uint8:
            data = data.astype(np.uint8)
        
        if self.use_fast:
            return self._encode_fast(data, terminate)
        else:
            return self._encode_slow(data, terminate)
    
    def _encode_fast(self, bits: np.ndarray, terminate: bool) -> np.ndarray:
        """
        Fast encoding using lookup tables.
        
        Args:
            bits: Input bit array
            terminate: Add termination bits
            
        Returns:
            Encoded bits (2x input length + 12 if terminated)
        """
        if terminate:
            # Add K-1 = 6 zero bits for trellis termination
            bits = np.concatenate([bits, np.zeros(self.K - 1, dtype=np.uint8)])
        
        output = np.zeros(len(bits) * 2, dtype=np.uint8)
        
        state = self._state
        for i, bit in enumerate(bits):
            out = self._output[state, bit]
            output[2*i] = (out >> 1) & 1
            output[2*i + 1] = out & 1
            state = self._next_state[state, bit]
        
        self._state = state
        return output
    
    def _encode_slow(self, bits: np.ndarray, terminate: bool) -> np.ndarray:
        """
        Bit-by-bit encoding (educational).
        
        Shows how the shift register works.
        """
        if terminate:
            bits = np.concatenate([bits, np.zeros(self.K - 1, dtype=np.uint8)])
        
        output = []
        
        for bit in bits:
            # Shift register: [b6 b5 b4 b3 b2 b1 b0] where b0 is oldest
            # New bit enters at b6 position
            reg = (int(bit) << 6) | self._state
            
            # Output 1: taps at G1 positions, XOR together
            out1 = 0
            temp = reg & self.G1
            while temp:
                out1 ^= (temp & 1)
                temp >>= 1
            
            # Output 2: taps at G2 positions
            out2 = 0
            temp = reg & self.G2
            while temp:
                out2 ^= (temp & 1)
                temp >>= 1
            
            output.extend([out1, out2])
            
            # Update state: shift right, dropping oldest bit
            self._state = reg >> 1
        
        return np.array(output, dtype=np.uint8)
    
    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        """
        Encode bits without termination.
        
        Useful for continuous encoding.
        
        Args:
            bits: Input bits
            
        Returns:
            Encoded bits
        """
        return self.encode(bits, terminate=False)


class ConvolutionalDecoder:
    """
    Viterbi decoder for DVB convolutional code.
    
    This decoder finds the maximum likelihood path through the
    trellis using the Viterbi algorithm.
    
    Attributes:
        K: Constraint length (7)
        num_states: Number of trellis states (64)
        
    Example:
        >>> decoder = ConvolutionalDecoder()
        >>> decoded = decoder.decode(received_bits)
    """
    
    K = 7
    G1 = 0b1111001
    G2 = 0b1011011
    
    def __init__(self, soft_decision: bool = False):
        """
        Initialize decoder.
        
        Args:
            soft_decision: Use soft decision decoding
        """
        self.soft_decision = soft_decision
        self.num_states = 2 ** (self.K - 1)  # 64 states
        self._build_trellis()
    
    def _build_trellis(self) -> None:
        """Build trellis structure for Viterbi decoding."""
        # For each state, store:
        # - previous states that can lead here
        # - input bits that cause those transitions
        # - expected output for those transitions
        
        self._prev_states = np.zeros((self.num_states, 2), dtype=np.int32)
        self._prev_bits = np.zeros((self.num_states, 2), dtype=np.uint8)
        self._expected_output = np.zeros((self.num_states, 2, 2), dtype=np.uint8)
        
        # Build forward transition table first
        next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        output = np.zeros((self.num_states, 2, 2), dtype=np.uint8)
        
        for state in range(self.num_states):
            for inp in range(2):
                reg = (inp << 6) | state
                
                out1 = bin(reg & self.G1).count('1') % 2
                out2 = bin(reg & self.G2).count('1') % 2
                
                ns = reg >> 1
                next_state[state, inp] = ns
                output[state, inp] = [out1, out2]
        
        # Build reverse lookup
        counts = np.zeros(self.num_states, dtype=np.int32)
        
        for state in range(self.num_states):
            for inp in range(2):
                ns = next_state[state, inp]
                idx = counts[ns]
                self._prev_states[ns, idx] = state
                self._prev_bits[ns, idx] = inp
                self._expected_output[ns, idx] = output[state, inp]
                counts[ns] += 1
        
        self._forward_output = output
        self._forward_next = next_state
    
    def decode(self, received: np.ndarray, terminated: bool = True) -> np.ndarray:
        """
        Decode using Viterbi algorithm.
        
        Args:
            received: Received bits (hard decision) or soft values
            terminated: Expect terminated trellis (ending at state 0)
            
        Returns:
            Decoded bits
        """
        if len(received) % 2 != 0:
            raise ValueError("Received sequence must have even length")
        
        num_symbols = len(received) // 2
        
        # Path metrics (costs to reach each state)
        pm = np.full(self.num_states, np.inf)
        pm[0] = 0  # Start at state 0
        
        # Survivor paths
        survivors = np.zeros((num_symbols, self.num_states), dtype=np.int32)
        
        # Process each received symbol
        for t in range(num_symbols):
            rx = received[2*t:2*t+2]
            
            # New path metrics
            new_pm = np.full(self.num_states, np.inf)
            
            for state in range(self.num_states):
                for i in range(2):  # Two paths into each state
                    prev_state = self._prev_states[state, i]
                    expected = self._expected_output[state, i]
                    
                    # Branch metric (Hamming distance for hard decision)
                    if self.soft_decision:
                        # For soft: use Euclidean-like distance
                        bm = np.sum(np.abs(rx - expected))
                    else:
                        bm = np.sum(rx != expected)
                    
                    # Total path metric
                    total = pm[prev_state] + bm
                    
                    if total < new_pm[state]:
                        new_pm[state] = total
                        survivors[t, state] = prev_state
            
            pm = new_pm
        
        # Traceback
        if terminated:
            state = 0  # Must end at state 0
        else:
            state = np.argmin(pm)  # Best final state
        
        decoded = np.zeros(num_symbols, dtype=np.uint8)
        
        for t in range(num_symbols - 1, -1, -1):
            prev_state = survivors[t, state]
            
            # Find which input bit caused this transition
            for inp in range(2):
                if self._forward_next[prev_state, inp] == state:
                    decoded[t] = inp
                    break
            
            state = prev_state
        
        # Remove tail bits if terminated
        if terminated:
            decoded = decoded[:-(self.K - 1)]
        
        return decoded
    
    def decode_to_bytes(self, received: np.ndarray, 
                        terminated: bool = True) -> bytes:
        """
        Decode and convert to bytes.
        
        Args:
            received: Received bits
            terminated: Expect terminated trellis
            
        Returns:
            Decoded bytes
        """
        bits = self.decode(received, terminated)
        
        # Pad to multiple of 8 bits
        pad_len = (8 - len(bits) % 8) % 8
        if pad_len:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
        
        return np.packbits(bits).tobytes()
