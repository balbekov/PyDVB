"""
DVB Outer Interleaver (Forney Convolutional Interleaver)

The outer interleaver disperses burst errors across multiple RS codewords
so that they appear as random errors to the RS decoder.

DVB uses a Forney convolutional interleaver with parameters:
- I = 12 branches
- M = 17 bytes per cell
- Depth = I * M = 204 bytes (one RS codeword)

Each branch i has a delay of i * M bytes. Byte n of input goes to
branch (n mod I), experiences delay (n mod I) * M bytes, and outputs
to position corresponding to branch (n mod I) at the output.

The interleaver effectively rotates bytes cyclically while adding
different delays to spread out burst errors.

Reference: ETSI EN 300 744 Section 4.3.3
"""

import numpy as np
from typing import Union


class OuterInterleaver:
    """
    DVB Forney convolutional interleaver.
    
    This interleaver spreads errors across RS codewords by routing
    bytes through different delay branches.
    
    Attributes:
        I: Number of branches (12)
        M: Cells per branch delay unit (17)
        
    Example:
        >>> interleaver = OuterInterleaver()
        >>> interleaved = interleaver.interleave(rs_codewords)
        >>> original = interleaver.deinterleave(interleaved)
    """
    
    # DVB outer interleaver parameters
    I = 12  # Number of branches
    M = 17  # Delay unit in bytes
    
    def __init__(self, use_fast: bool = True):
        """
        Initialize outer interleaver.
        
        Args:
            use_fast: Use numpy operations (True) or step-by-step (False)
        """
        self.use_fast = use_fast
        self._init_buffers()
    
    def _init_buffers(self) -> None:
        """Initialize delay line buffers."""
        # Each branch i has delay of i * M bytes
        # Branch 0 has no delay
        # Branch 1 has M bytes delay
        # ...
        # Branch I-1 has (I-1)*M bytes delay
        
        self._buffers = [
            np.zeros(i * self.M, dtype=np.uint8)
            for i in range(self.I)
        ]
        self._positions = [0] * self.I
    
    def reset(self) -> None:
        """Reset interleaver state (clear all delays)."""
        self._init_buffers()
    
    def _process_byte(self, byte: int, branch: int) -> int:
        """
        Process one byte through a branch.
        
        Args:
            byte: Input byte
            branch: Branch number (0 to I-1)
            
        Returns:
            Output byte (delayed input)
        """
        if branch == 0:
            return byte  # No delay for branch 0
        
        buf = self._buffers[branch]
        pos = self._positions[branch]
        
        # Get delayed byte
        output = int(buf[pos])
        
        # Store input byte
        buf[pos] = byte
        
        # Advance position
        self._positions[branch] = (pos + 1) % len(buf)
        
        return output
    
    def interleave(self, data: Union[bytes, bytearray, np.ndarray], 
                   sync: bool = False) -> bytes:
        """
        Interleave data bytes.
        
        Bytes are distributed across I branches, each with different
        delay. This spreads burst errors across multiple RS codewords.
        
        Args:
            data: Input bytes (should be multiple of 204 for RS codewords)
            sync: Reset interleaver state before processing
            
        Returns:
            Interleaved bytes (same length as input)
        """
        if sync:
            self.reset()
        
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        if self.use_fast:
            return self._interleave_fast(data)
        else:
            return self._interleave_slow(data)
    
    def _interleave_slow(self, data: bytes) -> bytes:
        """
        Byte-by-byte interleaving (educational).
        
        Each byte goes to branch (byte_index mod I), gets delayed
        by branch_number * M bytes.
        """
        result = bytearray(len(data))
        
        for i, byte in enumerate(data):
            branch = i % self.I
            result[i] = self._process_byte(byte, branch)
        
        return bytes(result)
    
    def _interleave_fast(self, data: bytes) -> bytes:
        """
        Optimized interleaving using numpy.
        
        Still processes branch-by-branch but uses array operations.
        """
        data_arr = np.frombuffer(data, dtype=np.uint8).copy()
        result = np.zeros_like(data_arr)
        
        for branch in range(self.I):
            # Extract bytes for this branch
            branch_indices = np.arange(branch, len(data), self.I)
            branch_bytes = data_arr[branch_indices]
            
            if branch == 0:
                # No delay
                result[branch_indices] = branch_bytes
            else:
                # Apply delay using buffer
                delay = branch * self.M
                buf = self._buffers[branch]
                
                # Output is: [buffer contents, then input (minus delay)]
                if len(branch_bytes) <= delay:
                    # All outputs come from buffer
                    result[branch_indices] = buf[:len(branch_bytes)]
                    # Update buffer: shift and add new bytes
                    new_buf = np.concatenate([
                        buf[len(branch_bytes):],
                        branch_bytes
                    ])
                    self._buffers[branch] = new_buf
                else:
                    # Some from buffer, rest from input
                    result[branch_indices[:delay]] = buf
                    result[branch_indices[delay:]] = branch_bytes[:-delay]
                    
                    # Update buffer with last 'delay' input bytes
                    self._buffers[branch] = branch_bytes[-delay:].copy()
        
        return bytes(result)
    
    def deinterleave(self, data: Union[bytes, bytearray, np.ndarray],
                     sync: bool = False) -> bytes:
        """
        Deinterleave data bytes.
        
        The deinterleaver has complementary delays:
        - Interleaver branch i has delay i * M
        - Deinterleaver branch i has delay (I - 1 - i) * M
        
        Total delay for each path is (I-1) * M = 11 * 17 = 187 bytes.
        
        Args:
            data: Interleaved bytes
            sync: Reset deinterleaver state before processing
            
        Returns:
            Deinterleaved bytes
        """
        if sync:
            self.reset()
        
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        # Use same buffers but with complementary branch mapping
        result = bytearray(len(data))
        
        for i, byte in enumerate(data):
            # Complementary branch delay
            branch = i % self.I
            complement_branch = (self.I - 1 - branch) % self.I
            
            if complement_branch == 0:
                result[i] = byte
            else:
                # Use buffer for complementary delay
                buf = self._buffers[complement_branch]
                pos = self._positions[complement_branch]
                
                result[i] = int(buf[pos])
                buf[pos] = byte
                self._positions[complement_branch] = (pos + 1) % len(buf)
        
        return bytes(result)
    
    @property
    def total_delay(self) -> int:
        """
        Total latency through interleaver + deinterleaver.
        
        Returns:
            Total delay in bytes
        """
        return (self.I - 1) * self.M  # 11 * 17 = 187 bytes


class OuterDeinterleaver(OuterInterleaver):
    """
    Standalone deinterleaver with separate state.
    
    Use this when interleaver and deinterleaver are in different
    parts of the system.
    """
    
    def __init__(self, use_fast: bool = True):
        super().__init__(use_fast)
        # Initialize with complementary delays
        self._buffers = [
            np.zeros((self.I - 1 - i) * self.M, dtype=np.uint8)
            for i in range(self.I)
        ]
    
    def process(self, data: Union[bytes, bytearray, np.ndarray],
                sync: bool = False) -> bytes:
        """
        Deinterleave data.
        
        Args:
            data: Interleaved bytes
            sync: Reset state before processing
            
        Returns:
            Deinterleaved bytes
        """
        if sync:
            self._buffers = [
                np.zeros((self.I - 1 - i) * self.M, dtype=np.uint8)
                for i in range(self.I)
            ]
            self._positions = [0] * self.I
        
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        result = bytearray(len(data))
        
        for i, byte in enumerate(data):
            branch = i % self.I
            
            if branch == self.I - 1:
                # Branch I-1 has zero delay in deinterleaver
                result[i] = byte
            else:
                buf = self._buffers[branch]
                if len(buf) == 0:
                    result[i] = byte
                else:
                    pos = self._positions[branch]
                    result[i] = int(buf[pos])
                    buf[pos] = byte
                    self._positions[branch] = (pos + 1) % len(buf)
        
        return bytes(result)
