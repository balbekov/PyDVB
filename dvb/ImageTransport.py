"""
DVB-T Image Transport

Encapsulates images (JPG, PNG) into transport stream packets for transmission.
This is a simplified approach that doesn't require ffmpeg - it embeds the raw
image data directly into TS packets with a simple header for recovery.

For educational purposes, this demonstrates how arbitrary data can be 
transmitted over DVB-T.

Protocol:
    - First packet: Header with magic, image size, format, dimensions
    - Following packets: Raw image data split across TS packets
    - CRC32 at end for verification
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import struct
import io

from .TransportStream import TransportStream
from .Packet import Packet, PACKET_SIZE
from .CRC import CRC32


# Magic bytes to identify image transport packets
IMAGE_MAGIC = b'DVBI'  # DVB Image

# Stream PIDs for image data
IMAGE_PID = 0x200
HEADER_PID = 0x201


class ImagePacketizer:
    """
    Converts images to transport stream packets.
    
    Creates a transport stream containing the image data with
    a simple header for recovery at the receiver.
    
    Example:
        >>> packetizer = ImagePacketizer()
        >>> ts_data = packetizer.packetize('photo.jpg')
        >>> 
        >>> # Transmit ts_data through DVB-T
    """
    
    def __init__(self, pid: int = IMAGE_PID):
        """
        Initialize image packetizer.
        
        Args:
            pid: PID for image data packets
        """
        self.pid = pid
        self.crc = CRC32()
    
    def packetize(self, image_path: Union[str, Path]) -> bytes:
        """
        Convert image file to transport stream.
        
        Args:
            image_path: Path to image file (JPG, PNG, etc.)
            
        Returns:
            Transport stream bytes
        """
        image_path = Path(image_path)
        
        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Detect format from file signature
        image_format = self._detect_format(image_data)
        
        # Try to get dimensions
        width, height = self._get_dimensions(image_data, image_format)
        
        return self.packetize_bytes(image_data, image_format, width, height)
    
    def packetize_bytes(self, image_data: bytes, 
                        image_format: str = 'jpg',
                        width: int = 0, 
                        height: int = 0) -> bytes:
        """
        Convert image bytes to transport stream.
        
        Args:
            image_data: Raw image file bytes
            image_format: Image format ('jpg', 'png', etc.)
            width: Image width (0 if unknown)
            height: Image height (0 if unknown)
            
        Returns:
            Transport stream bytes
        """
        ts = TransportStream()
        
        # Calculate CRC of image data
        crc_value = self.crc.calculate(image_data)
        
        # Create header (fits in one TS packet payload)
        # Header format:
        #   Magic (4 bytes): 'DVBI'
        #   Version (1 byte): 0x01
        #   Format (4 bytes): 'jpg\0', 'png\0', etc.
        #   Width (2 bytes): big-endian
        #   Height (2 bytes): big-endian
        #   Data length (4 bytes): big-endian
        #   CRC32 (4 bytes): big-endian
        #   Reserved (padding to 32 bytes)
        header = bytearray(32)
        header[0:4] = IMAGE_MAGIC
        header[4] = 0x01  # Version
        fmt_bytes = image_format.encode('ascii')[:4].ljust(4, b'\x00')
        header[5:9] = fmt_bytes
        struct.pack_into('>H', header, 9, width)
        struct.pack_into('>H', header, 11, height)
        struct.pack_into('>I', header, 13, len(image_data))
        struct.pack_into('>I', header, 17, crc_value)
        
        # Add header packet
        ts.append(Packet(
            pid=HEADER_PID,
            payload_unit_start=True,
            payload=bytes(header).ljust(184, b'\xff'),
            continuity_counter=0,
        ))
        
        # Split image data across packets
        # TS payload is 184 bytes
        payload_size = 184
        cc = 0
        
        for i in range(0, len(image_data), payload_size):
            chunk = image_data[i:i + payload_size]
            
            # Pad last chunk if needed
            if len(chunk) < payload_size:
                chunk = chunk.ljust(payload_size, b'\xff')
            
            ts.append(Packet(
                pid=self.pid,
                payload_unit_start=(i == 0),
                payload=chunk,
                continuity_counter=cc,
            ))
            cc = (cc + 1) % 16
        
        return ts.to_bytes()
    
    def _detect_format(self, data: bytes) -> str:
        """Detect image format from file signature."""
        if data[:2] == b'\xff\xd8':
            return 'jpg'
        elif data[:8] == b'\x89PNG\r\n\x1a\n':
            return 'png'
        elif data[:6] in (b'GIF87a', b'GIF89a'):
            return 'gif'
        elif data[:2] == b'BM':
            return 'bmp'
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return 'webp'
        else:
            return 'bin'
    
    def _get_dimensions(self, data: bytes, fmt: str) -> Tuple[int, int]:
        """Get image dimensions from header."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            return img.size
        except Exception:
            pass
        
        # Fallback: parse common formats manually
        try:
            if fmt == 'png' and len(data) >= 24:
                width = struct.unpack('>I', data[16:20])[0]
                height = struct.unpack('>I', data[20:24])[0]
                return width, height
            elif fmt == 'jpg':
                return self._parse_jpeg_dimensions(data)
        except Exception:
            pass
        
        return 0, 0
    
    def _parse_jpeg_dimensions(self, data: bytes) -> Tuple[int, int]:
        """Parse JPEG dimensions from SOF marker."""
        i = 2  # Skip SOI
        while i < len(data) - 9:
            if data[i] != 0xff:
                i += 1
                continue
            
            marker = data[i + 1]
            if marker == 0xd9:  # EOI
                break
            
            if marker in (0xc0, 0xc1, 0xc2):  # SOF markers
                height = struct.unpack('>H', data[i + 5:i + 7])[0]
                width = struct.unpack('>H', data[i + 7:i + 9])[0]
                return width, height
            
            # Skip to next marker
            length = struct.unpack('>H', data[i + 2:i + 4])[0]
            i += 2 + length
        
        return 0, 0


class ImageDepacketizer:
    """
    Extracts images from transport stream packets.
    
    Recovers image data from transport streams created by ImagePacketizer.
    
    Example:
        >>> depacketizer = ImageDepacketizer()
        >>> image_data, info = depacketizer.depacketize(ts_data)
        >>> with open('recovered.jpg', 'wb') as f:
        ...     f.write(image_data)
    """
    
    def __init__(self, pid: int = IMAGE_PID):
        """
        Initialize image depacketizer.
        
        Args:
            pid: Expected PID for image data packets
        """
        self.pid = pid
        self.crc = CRC32()
    
    def depacketize(self, ts_data: bytes) -> Tuple[Optional[bytes], dict]:
        """
        Extract image from transport stream.
        
        Args:
            ts_data: Transport stream bytes
            
        Returns:
            Tuple of (image_data or None, info dict)
        """
        ts = TransportStream.from_bytes(ts_data)
        
        info = {
            'found_header': False,
            'format': None,
            'width': 0,
            'height': 0,
            'expected_size': 0,
            'received_size': 0,
            'crc_valid': False,
            'packets': 0,
        }
        
        # Find header packet
        header_data = None
        for pkt in ts:
            if pkt.pid == HEADER_PID and pkt.payload_unit_start:
                header_data = pkt.payload
                break
        
        if header_data is None or header_data[:4] != IMAGE_MAGIC:
            return None, info
        
        info['found_header'] = True
        
        # Parse header
        version = header_data[4]
        img_format = header_data[5:9].rstrip(b'\x00').decode('ascii', errors='ignore')
        width = struct.unpack('>H', header_data[9:11])[0]
        height = struct.unpack('>H', header_data[11:13])[0]
        data_length = struct.unpack('>I', header_data[13:17])[0]
        expected_crc = struct.unpack('>I', header_data[17:21])[0]
        
        info['format'] = img_format
        info['width'] = width
        info['height'] = height
        info['expected_size'] = data_length
        
        # Collect image data packets
        image_chunks = []
        for pkt in ts:
            if pkt.pid == self.pid:
                image_chunks.append(pkt.payload)
                info['packets'] += 1
        
        if not image_chunks:
            return None, info
        
        # Concatenate and trim to expected size
        image_data = b''.join(image_chunks)
        image_data = image_data[:data_length]
        info['received_size'] = len(image_data)
        
        # Verify CRC
        if len(image_data) == data_length:
            actual_crc = self.crc.calculate(image_data)
            info['crc_valid'] = (actual_crc == expected_crc)
        
        return image_data, info


def image_to_ts(image_data: Union[str, Path, bytes]) -> bytes:
    """
    Convenience function to convert image to transport stream.
    
    Args:
        image_data: Path to image file or raw image bytes
        
    Returns:
        Transport stream bytes
    """
    packetizer = ImagePacketizer()
    if isinstance(image_data, bytes):
        return packetizer.packetize_bytes(image_data)
    return packetizer.packetize(image_data)


def ts_to_image(ts_data: bytes) -> Tuple[Optional[bytes], dict]:
    """
    Convenience function to extract image from transport stream.
    
    Args:
        ts_data: Transport stream bytes
        
    Returns:
        Tuple of (image_data, info_dict)
    """
    depacketizer = ImageDepacketizer()
    return depacketizer.depacketize(ts_data)


def send_image_audio(image_path: Union[str, Path],
                     output_wav: Optional[Union[str, Path]] = None,
                     play: bool = False,
                     sample_rate: int = 48000,
                     carrier_freq: float = 5000,
                     mono: bool = True) -> dict:
    """
    Send image via acoustic DVB-T.
    
    Args:
        image_path: Path to image file
        output_wav: Optional WAV output path
        play: Play through speaker
        sample_rate: Audio sample rate
        carrier_freq: Audio carrier frequency for mono mode
        mono: If True, modulate onto carrier (for speaker->mic).
              If False, use stereo I/Q (for file loopback).
        
    Returns:
        Transmission info dict
    """
    from .AudioOutput import AcousticDVBT
    
    # Convert image to transport stream
    packetizer = ImagePacketizer()
    ts_data = packetizer.packetize(image_path)
    
    # Create transmitter
    tx = AcousticDVBT(
        audio_sample_rate=sample_rate,
        carrier_freq=carrier_freq
    )
    
    info = {
        'image_path': str(image_path),
        'ts_bytes': len(ts_data),
        'ts_packets': len(ts_data) // 188,
        'mono': mono,
    }
    
    if output_wav:
        tx.transmit_to_file(ts_data, output_wav, stereo=not mono)
        info['wav_path'] = str(output_wav)
    
    if play:
        tx.transmit(ts_data, blocking=True)
    
    tx.close()
    return info


def receive_image_audio(input_wav: Union[str, Path],
                        output_path: Optional[Union[str, Path]] = None,
                        sample_rate: int = 48000,
                        carrier_freq: float = 13000) -> Tuple[Optional[bytes], dict]:
    """
    Receive image via acoustic DVB-T from WAV file.
    
    Args:
        input_wav: Input WAV file
        output_path: Path to save recovered image
        sample_rate: Audio sample rate
        carrier_freq: Audio carrier frequency
        
    Returns:
        Tuple of (image_data, reception_stats)
    """
    from .AudioInput import AcousticDVBTReceiver
    
    # Create receiver
    rx = AcousticDVBTReceiver(
        audio_sample_rate=sample_rate,
        carrier_freq=carrier_freq
    )
    
    # Receive transport stream from file
    ts_data, rx_stats = rx.receive_file(input_wav)
    rx.close()
    
    # Extract image
    if len(ts_data) > 0:
        image_data, img_info = ts_to_image(ts_data)
    else:
        image_data = None
        img_info = {'found_header': False}
    
    # Combine stats
    rx_stats.update(img_info)
    
    # Save if requested
    if output_path and image_data:
        with open(output_path, 'wb') as f:
            f.write(image_data)
        rx_stats['saved_to'] = str(output_path)
    
    return image_data, rx_stats

