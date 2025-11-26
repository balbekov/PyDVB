"""
DVB-T I/Q Sample Writer

Writes baseband I/Q samples in formats suitable for SDR transmission.

Supported formats:
- .cf32: Complex float32 (GNU Radio, most SDR software)
- .cs8: Complex signed 8-bit (HackRF)
- .cs16: Complex signed 16-bit (RTL-SDR, USRP)
- .cu8: Complex unsigned 8-bit (RTL-SDR raw)

Reference: Various SDR documentation
"""

import numpy as np
from pathlib import Path
from typing import Union, BinaryIO, Optional


class IQWriter:
    """
    I/Q sample writer for SDR transmission.
    
    Converts complex baseband samples to various SDR-compatible formats.
    
    Attributes:
        format: Output format ('cf32', 'cs8', 'cs16', 'cu8')
        scale: Amplitude scaling factor
        
    Example:
        >>> writer = IQWriter('cf32')
        >>> writer.write('output.cf32', samples)
    """
    
    # Format specifications
    FORMATS = {
        'cf32': {'dtype': np.complex64, 'extension': '.cf32'},
        'cs8': {'dtype': np.int8, 'extension': '.cs8'},
        'cs16': {'dtype': np.int16, 'extension': '.cs16'},
        'cu8': {'dtype': np.uint8, 'extension': '.cu8'},
    }
    
    def __init__(self, format: str = 'cf32', scale: float = 1.0):
        """
        Initialize I/Q writer.
        
        Args:
            format: Output format ('cf32', 'cs8', 'cs16', 'cu8')
            scale: Amplitude scaling factor (applies before quantization)
        """
        if format not in self.FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        self.format = format
        self.scale = scale
    
    def _convert_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert complex samples to output format.
        
        Args:
            samples: Complex baseband samples
            
        Returns:
            Converted samples ready for writing
        """
        # Apply scaling
        scaled = samples * self.scale
        
        if self.format == 'cf32':
            # Complex float32 - direct output
            return scaled.astype(np.complex64)
        
        elif self.format == 'cs8':
            # Complex signed 8-bit
            # Scale to fit in [-127, 127]
            max_val = np.max(np.abs(scaled))
            if max_val > 0:
                normalized = scaled / max_val * 127
            else:
                normalized = scaled
            
            # Interleave I and Q as signed 8-bit
            output = np.zeros(len(scaled) * 2, dtype=np.int8)
            output[0::2] = np.clip(normalized.real, -127, 127).astype(np.int8)
            output[1::2] = np.clip(normalized.imag, -127, 127).astype(np.int8)
            return output
        
        elif self.format == 'cs16':
            # Complex signed 16-bit
            max_val = np.max(np.abs(scaled))
            if max_val > 0:
                normalized = scaled / max_val * 32767
            else:
                normalized = scaled
            
            output = np.zeros(len(scaled) * 2, dtype=np.int16)
            output[0::2] = np.clip(normalized.real, -32767, 32767).astype(np.int16)
            output[1::2] = np.clip(normalized.imag, -32767, 32767).astype(np.int16)
            return output
        
        elif self.format == 'cu8':
            # Complex unsigned 8-bit (offset binary)
            max_val = np.max(np.abs(scaled))
            if max_val > 0:
                normalized = scaled / max_val * 127
            else:
                normalized = scaled
            
            output = np.zeros(len(scaled) * 2, dtype=np.uint8)
            output[0::2] = np.clip(normalized.real + 128, 0, 255).astype(np.uint8)
            output[1::2] = np.clip(normalized.imag + 128, 0, 255).astype(np.uint8)
            return output
        
        return scaled.astype(np.complex64)
    
    def write(self, path: Union[str, Path], samples: np.ndarray) -> None:
        """
        Write I/Q samples to file.
        
        Args:
            path: Output file path
            samples: Complex baseband samples
        """
        path = Path(path)
        
        converted = self._convert_samples(samples)
        converted.tofile(path)
    
    def write_stream(self, stream: BinaryIO, samples: np.ndarray) -> None:
        """
        Write I/Q samples to file-like object.
        
        Args:
            stream: Binary file-like object
            samples: Complex baseband samples
        """
        converted = self._convert_samples(samples)
        stream.write(converted.tobytes())
    
    def append(self, path: Union[str, Path], samples: np.ndarray) -> None:
        """
        Append I/Q samples to existing file.
        
        Args:
            path: Output file path
            samples: Complex baseband samples
        """
        path = Path(path)
        
        converted = self._convert_samples(samples)
        
        with open(path, 'ab') as f:
            f.write(converted.tobytes())


class IQReader:
    """
    I/Q sample reader for SDR files.
    
    Reads various I/Q file formats back to complex numpy arrays.
    
    Example:
        >>> reader = IQReader('cf32')
        >>> samples = reader.read('input.cf32')
    """
    
    def __init__(self, format: str = 'cf32'):
        """
        Initialize I/Q reader.
        
        Args:
            format: Input format ('cf32', 'cs8', 'cs16', 'cu8')
        """
        if format not in IQWriter.FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        self.format = format
    
    def read(self, path: Union[str, Path], 
             count: Optional[int] = None) -> np.ndarray:
        """
        Read I/Q samples from file.
        
        Args:
            path: Input file path
            count: Number of complex samples to read (None = all)
            
        Returns:
            Complex numpy array
        """
        path = Path(path)
        
        if self.format == 'cf32':
            data = np.fromfile(path, dtype=np.complex64, count=count or -1)
            return data
        
        elif self.format == 'cs8':
            read_count = count * 2 if count else -1
            raw = np.fromfile(path, dtype=np.int8, count=read_count)
            samples = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
            return samples / 127.0
        
        elif self.format == 'cs16':
            read_count = count * 2 if count else -1
            raw = np.fromfile(path, dtype=np.int16, count=read_count)
            samples = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
            return samples / 32767.0
        
        elif self.format == 'cu8':
            read_count = count * 2 if count else -1
            raw = np.fromfile(path, dtype=np.uint8, count=read_count)
            samples = (raw[0::2].astype(np.float32) - 128) + \
                     1j * (raw[1::2].astype(np.float32) - 128)
            return samples / 127.0
        
        return np.array([], dtype=np.complex64)


def detect_format(path: Union[str, Path]) -> str:
    """
    Detect I/Q file format from extension.
    
    Args:
        path: File path
        
    Returns:
        Format string
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    format_map = {
        '.cf32': 'cf32',
        '.fc32': 'cf32',
        '.cs8': 'cs8',
        '.s8': 'cs8',
        '.cs16': 'cs16',
        '.s16': 'cs16',
        '.cu8': 'cu8',
        '.u8': 'cu8',
        '.raw': 'cf32',  # Assume cf32 for raw
        '.iq': 'cf32',
    }
    
    return format_map.get(ext, 'cf32')


def get_sample_rate(bandwidth: str = '8MHz') -> float:
    """
    Get DVB-T sample rate for bandwidth.
    
    Args:
        bandwidth: '6MHz', '7MHz', or '8MHz'
        
    Returns:
        Sample rate in Hz
    """
    rates = {
        '8MHz': 9142857.142857143,
        '7MHz': 8000000.0,
        '6MHz': 6857142.857142857,
    }
    return rates.get(bandwidth, rates['8MHz'])
