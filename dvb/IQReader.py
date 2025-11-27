"""
DVB-T I/Q Sample Reader

Reads baseband I/Q samples from various SDR file formats for receiver processing.

Supported formats:
- .cf32: Complex float32 (GNU Radio, most SDR software)
- .cs8: Complex signed 8-bit (HackRF)
- .cs16: Complex signed 16-bit (RTL-SDR, USRP)
- .cu8: Complex unsigned 8-bit (RTL-SDR raw)

Reference: Various SDR documentation
"""

import numpy as np
from pathlib import Path
from typing import Union, Iterator, Optional, Tuple
import os


class IQReader:
    """
    I/Q sample reader for SDR files.
    
    Reads various I/Q file formats back to complex numpy arrays.
    Supports both full-file reading and chunked streaming.
    
    Attributes:
        format: Input format ('cf32', 'cs8', 'cs16', 'cu8', 'auto')
        path: File path (if set)
        sample_rate: Sample rate in Hz (optional metadata)
        
    Example:
        >>> reader = IQReader('cf32')
        >>> samples = reader.read('input.cf32')
        >>> 
        >>> # Chunked reading
        >>> for chunk in reader.iter_chunks('input.cf32', chunk_size=65536):
        ...     process(chunk)
    """
    
    # Format specifications
    FORMATS = {
        'cf32': {
            'dtype': np.complex64,
            'bytes_per_sample': 8,
            'extension': ['.cf32', '.fc32', '.cfile', '.iq'],
        },
        'cs8': {
            'dtype': np.int8,
            'bytes_per_sample': 2,
            'extension': ['.cs8', '.s8'],
        },
        'cs16': {
            'dtype': np.int16,
            'bytes_per_sample': 4,
            'extension': ['.cs16', '.s16', '.sc16'],
        },
        'cu8': {
            'dtype': np.uint8,
            'bytes_per_sample': 2,
            'extension': ['.cu8', '.u8', '.raw'],
        },
    }
    
    def __init__(self, format: str = 'auto', sample_rate: Optional[float] = None):
        """
        Initialize I/Q reader.
        
        Args:
            format: Input format ('cf32', 'cs8', 'cs16', 'cu8', 'auto')
            sample_rate: Optional sample rate in Hz (for metadata)
        """
        self.format = format
        self.sample_rate = sample_rate
        self._file_handle = None
        self._file_path = None
    
    def _detect_format(self, path: Union[str, Path]) -> str:
        """
        Detect I/Q file format from extension.
        
        Args:
            path: File path
            
        Returns:
            Format string
        """
        path = Path(path)
        ext = path.suffix.lower()
        
        for fmt, info in self.FORMATS.items():
            if ext in info['extension']:
                return fmt
        
        # Default to cf32 for unknown extensions
        return 'cf32'
    
    def _get_format(self, path: Union[str, Path]) -> str:
        """Get format, auto-detecting if needed."""
        if self.format == 'auto':
            return self._detect_format(path)
        return self.format
    
    def _convert_to_complex(self, raw_data: np.ndarray, 
                            format: str) -> np.ndarray:
        """
        Convert raw data to complex samples.
        
        Args:
            raw_data: Raw data from file
            format: File format
            
        Returns:
            Complex numpy array
        """
        if format == 'cf32':
            return raw_data.astype(np.complex64)
        
        elif format == 'cs8':
            # Interleaved signed 8-bit I/Q
            samples = (raw_data[0::2].astype(np.float32) + 
                      1j * raw_data[1::2].astype(np.float32))
            return (samples / 127.0).astype(np.complex64)
        
        elif format == 'cs16':
            # Interleaved signed 16-bit I/Q
            samples = (raw_data[0::2].astype(np.float32) + 
                      1j * raw_data[1::2].astype(np.float32))
            return (samples / 32767.0).astype(np.complex64)
        
        elif format == 'cu8':
            # Interleaved unsigned 8-bit I/Q (offset binary)
            samples = ((raw_data[0::2].astype(np.float32) - 128) + 
                      1j * (raw_data[1::2].astype(np.float32) - 128))
            return (samples / 127.0).astype(np.complex64)
        
        return raw_data.astype(np.complex64)
    
    def read(self, path: Union[str, Path], 
             count: Optional[int] = None,
             offset: int = 0) -> np.ndarray:
        """
        Read I/Q samples from file.
        
        Args:
            path: Input file path
            count: Number of complex samples to read (None = all)
            offset: Sample offset from start of file
            
        Returns:
            Complex numpy array
        """
        path = Path(path)
        format = self._get_format(path)
        
        if format == 'cf32':
            # Direct complex read
            with open(path, 'rb') as f:
                if offset > 0:
                    f.seek(offset * 8)  # 8 bytes per complex sample
                if count:
                    data = np.fromfile(f, dtype=np.complex64, count=count)
                else:
                    data = np.fromfile(f, dtype=np.complex64)
            return data
        
        else:
            # Read interleaved I/Q
            bytes_per_sample = self.FORMATS[format]['bytes_per_sample']
            dtype = self.FORMATS[format]['dtype']
            
            with open(path, 'rb') as f:
                if offset > 0:
                    f.seek(offset * bytes_per_sample)
                if count:
                    raw = np.fromfile(f, dtype=dtype, count=count * 2)
                else:
                    raw = np.fromfile(f, dtype=dtype)
            
            return self._convert_to_complex(raw, format)
    
    def iter_chunks(self, path: Union[str, Path], 
                    chunk_size: int = 65536,
                    overlap: int = 0) -> Iterator[np.ndarray]:
        """
        Iterate over file in chunks.
        
        Useful for processing large files without loading entirely into memory.
        
        Args:
            path: Input file path
            chunk_size: Number of complex samples per chunk
            overlap: Number of samples to overlap between chunks
            
        Yields:
            Complex numpy arrays
        """
        path = Path(path)
        format = self._get_format(path)
        bytes_per_sample = self.FORMATS[format]['bytes_per_sample']
        
        file_size = os.path.getsize(path)
        total_samples = file_size // bytes_per_sample
        
        if format == 'cf32':
            dtype = np.complex64
            samples_per_read = chunk_size
        else:
            dtype = self.FORMATS[format]['dtype']
            samples_per_read = chunk_size * 2  # I/Q interleaved
        
        position = 0
        stride = chunk_size - overlap
        
        with open(path, 'rb') as f:
            while position < total_samples:
                # Seek to position (accounting for overlap)
                seek_samples = max(0, position - overlap) if position > 0 else 0
                f.seek(seek_samples * bytes_per_sample)
                
                # Read chunk
                if format == 'cf32':
                    raw = np.fromfile(f, dtype=dtype, count=chunk_size + overlap)
                else:
                    raw = np.fromfile(f, dtype=dtype, 
                                     count=(chunk_size + overlap) * 2)
                
                if len(raw) == 0:
                    break
                
                # Convert to complex
                samples = self._convert_to_complex(raw, format)
                
                yield samples
                
                position += stride
                
                # Check if we've read everything
                if len(samples) < chunk_size:
                    break
    
    def get_file_info(self, path: Union[str, Path]) -> dict:
        """
        Get information about an I/Q file.
        
        Args:
            path: File path
            
        Returns:
            Dictionary with file information
        """
        path = Path(path)
        format = self._get_format(path)
        file_size = os.path.getsize(path)
        bytes_per_sample = self.FORMATS[format]['bytes_per_sample']
        
        num_samples = file_size // bytes_per_sample
        
        info = {
            'path': str(path),
            'format': format,
            'file_size_bytes': file_size,
            'num_samples': num_samples,
            'bytes_per_sample': bytes_per_sample,
        }
        
        if self.sample_rate:
            info['sample_rate'] = self.sample_rate
            info['duration_seconds'] = num_samples / self.sample_rate
        
        return info


def read_iq_file(path: Union[str, Path], 
                 format: str = 'auto',
                 count: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to read I/Q file.
    
    Args:
        path: File path
        format: File format (or 'auto' to detect)
        count: Number of samples to read
        
    Returns:
        Complex numpy array
    """
    reader = IQReader(format)
    return reader.read(path, count=count)


def detect_format(path: Union[str, Path]) -> str:
    """
    Detect I/Q file format from extension.
    
    Args:
        path: File path
        
    Returns:
        Format string ('cf32', 'cs8', 'cs16', 'cu8')
    """
    reader = IQReader('auto')
    return reader._detect_format(path)


def get_dvbt_sample_rate(bandwidth: str = '8MHz') -> float:
    """
    Get DVB-T sample rate for bandwidth.
    
    Args:
        bandwidth: '6MHz', '7MHz', or '8MHz'
        
    Returns:
        Sample rate in Hz
    """
    rates = {
        '8MHz': 9142857.142857143,  # 64/7 MHz
        '7MHz': 8000000.0,
        '6MHz': 6857142.857142857,  # 48/7 MHz
    }
    return rates.get(bandwidth, rates['8MHz'])
