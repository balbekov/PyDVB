"""
DVB-T Guard Interval (Cyclic Prefix)

The guard interval protects against inter-symbol interference (ISI)
caused by multipath propagation. It is implemented as a cyclic prefix:
the last portion of the OFDM symbol is copied and prepended.

DVB-T guard interval options:
- 1/4: Guard = 1/4 of useful symbol (most protection, lowest throughput)
- 1/8: Guard = 1/8 of useful symbol
- 1/16: Guard = 1/16 of useful symbol
- 1/32: Guard = 1/32 of useful symbol (least protection, highest throughput)

Reference: ETSI EN 300 744 Section 4.4
"""

import numpy as np
from typing import Tuple


# Guard interval fractions (divisor: guard = fft_size / divisor)
# For 'acoustic', we use a special value that means guard = 2.5 * fft_size
GUARD_FRACTIONS = {
    '1/4': 4,
    '1/8': 8,
    '1/16': 16,
    '1/32': 32,
    'acoustic': 0.4,  # Special: guard = fft_size / 0.2 = 5 * fft_size (~20ms at 16kHz with 64-pt FFT)
}


class GuardIntervalInserter:
    """
    DVB-T guard interval inserter.
    
    Adds cyclic prefix to OFDM symbols by copying the end
    of each symbol to the beginning.
    
    Attributes:
        ratio: Guard interval ratio ('1/4', '1/8', etc.)
        fft_size: FFT size (determines useful symbol length)
        guard_length: Number of guard samples
        
    Example:
        >>> gi = GuardIntervalInserter('1/4', 2048)
        >>> symbol_with_guard = gi.add(symbol)
        >>> assert len(symbol_with_guard) == 2048 + 512
    """
    
    def __init__(self, ratio: str = '1/4', fft_size: int = 2048):
        """
        Initialize guard interval inserter.
        
        Args:
            ratio: Guard interval ratio ('1/4', '1/8', '1/16', '1/32', 'acoustic')
            fft_size: FFT size (useful symbol length)
        """
        if ratio not in GUARD_FRACTIONS:
            raise ValueError(f"Invalid guard interval: {ratio}")
        
        self.ratio = ratio
        self.fft_size = fft_size
        
        # Calculate guard length - handle fractional divisors for acoustic mode
        divisor = GUARD_FRACTIONS[ratio]
        if divisor < 1:
            # Fractional divisor means guard > fft_size (e.g., 0.4 -> guard = 2.5 * fft)
            self.guard_length = int(fft_size / divisor)
        else:
            self.guard_length = fft_size // int(divisor)
        
        self.symbol_length = fft_size + self.guard_length
    
    def add(self, symbol: np.ndarray) -> np.ndarray:
        """
        Add guard interval (cyclic prefix) to OFDM symbol.
        
        Copies the last guard_length samples to the beginning.
        For acoustic mode where guard > fft_size, uses cyclic extension
        (tiles the symbol to create longer guard).
        
        Args:
            symbol: Time-domain OFDM symbol (length = fft_size)
            
        Returns:
            Symbol with guard interval (length = fft_size + guard_length)
        """
        if len(symbol) != self.fft_size:
            raise ValueError(f"Expected {self.fft_size} samples, "
                           f"got {len(symbol)}")
        
        # Cyclic prefix: copy end to beginning
        # For acoustic mode, guard may be longer than fft_size
        if self.guard_length <= self.fft_size:
            guard = symbol[-self.guard_length:]
        else:
            # Cyclic extension: tile the symbol to fill guard
            # E.g., for 160-sample guard with 64-sample symbol:
            # guard = [symbol, symbol, symbol][-160:] = last 160 samples of tiled symbol
            tiles_needed = (self.guard_length + self.fft_size - 1) // self.fft_size + 1
            tiled = np.tile(symbol, tiles_needed)
            guard = tiled[-self.guard_length:]
        return np.concatenate([guard, symbol])
    
    def add_multiple(self, symbols: np.ndarray) -> np.ndarray:
        """
        Add guard intervals to multiple symbols.
        
        Args:
            symbols: Array of symbols (shape: num_symbols × fft_size)
            
        Returns:
            Symbols with guard intervals
        """
        if symbols.ndim == 1:
            return self.add(symbols)
        
        num_symbols = symbols.shape[0]
        result = np.zeros((num_symbols, self.symbol_length), 
                         dtype=symbols.dtype)
        
        for i in range(num_symbols):
            result[i] = self.add(symbols[i])
        
        return result


class GuardIntervalRemover:
    """
    DVB-T guard interval remover.
    
    Removes the cyclic prefix from received OFDM symbols.
    
    Example:
        >>> gi = GuardIntervalRemover('1/4', 2048)
        >>> symbol = gi.remove(received_symbol)
        >>> assert len(symbol) == 2048
    """
    
    def __init__(self, ratio: str = '1/4', fft_size: int = 2048):
        """
        Initialize guard interval remover.
        
        Args:
            ratio: Guard interval ratio ('1/4', '1/8', '1/16', '1/32', 'acoustic')
            fft_size: FFT size (useful symbol length)
        """
        if ratio not in GUARD_FRACTIONS:
            raise ValueError(f"Invalid guard interval: {ratio}")
        
        self.ratio = ratio
        self.fft_size = fft_size
        
        # Calculate guard length - handle fractional divisors for acoustic mode
        divisor = GUARD_FRACTIONS[ratio]
        if divisor < 1:
            self.guard_length = int(fft_size / divisor)
        else:
            self.guard_length = fft_size // int(divisor)
        
        self.symbol_length = fft_size + self.guard_length
    
    def remove(self, symbol: np.ndarray) -> np.ndarray:
        """
        Remove guard interval from OFDM symbol.
        
        Removes the first guard_length samples (the cyclic prefix).
        
        Args:
            symbol: Received symbol with guard interval
            
        Returns:
            Symbol without guard interval (length = fft_size)
        """
        if len(symbol) != self.symbol_length:
            raise ValueError(f"Expected {self.symbol_length} samples, "
                           f"got {len(symbol)}")
        
        # Remove cyclic prefix
        return symbol[self.guard_length:]
    
    def remove_multiple(self, symbols: np.ndarray) -> np.ndarray:
        """
        Remove guard intervals from multiple symbols.
        
        Args:
            symbols: Array of symbols with guard intervals
            
        Returns:
            Symbols without guard intervals
        """
        if symbols.ndim == 1:
            return self.remove(symbols)
        
        num_symbols = symbols.shape[0]
        result = np.zeros((num_symbols, self.fft_size), 
                         dtype=symbols.dtype)
        
        for i in range(num_symbols):
            result[i] = self.remove(symbols[i])
        
        return result


def get_guard_samples(ratio: str, mode: str) -> int:
    """
    Get number of guard interval samples.
    
    Args:
        ratio: Guard interval ratio ('1/4', '1/8', '1/16', '1/32')
        mode: DVB-T mode ('2K' or '8K')
        
    Returns:
        Number of guard samples
    """
    fft_sizes = {'2K': 2048, '8K': 8192}
    fft_size = fft_sizes.get(mode, 2048)
    
    return fft_size // GUARD_FRACTIONS.get(ratio, 4)


def get_symbol_samples(ratio: str, mode: str) -> Tuple[int, int]:
    """
    Get useful and total symbol sample counts.
    
    Args:
        ratio: Guard interval ratio
        mode: DVB-T mode
        
    Returns:
        Tuple of (useful_samples, total_samples)
    """
    fft_sizes = {'2K': 2048, '8K': 8192}
    fft_size = fft_sizes.get(mode, 2048)
    
    guard = fft_size // GUARD_FRACTIONS.get(ratio, 4)
    
    return fft_size, fft_size + guard
