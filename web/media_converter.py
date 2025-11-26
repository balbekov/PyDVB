"""
Media to Transport Stream Converter

Converts images (JPG, PNG) and videos (MP4, AVI) to MPEG-2 Transport Stream
format suitable for DVB-T encoding.

Requires ffmpeg to be installed on the system.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
import os

from PIL import Image
import numpy as np


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_media_info(file_path: Path) -> dict:
    """
    Get media file information using ffprobe.
    
    Args:
        file_path: Path to media file
        
    Returns:
        Dict with media info (duration, dimensions, etc.)
    """
    try:
        result = subprocess.run([
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(file_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
    except Exception:
        pass
    
    return {}


def image_to_ts(image_path: Path, output_path: Optional[Path] = None,
                duration: float = 1.0, fps: int = 25) -> Tuple[bytes, dict]:
    """
    Convert an image to MPEG-2 Transport Stream.
    
    Creates a short video from a still image, then encodes to MPEG-2 TS.
    
    Args:
        image_path: Path to input image (JPG, PNG, etc.)
        output_path: Optional path for output TS file
        duration: Duration of the video in seconds
        fps: Frames per second
        
    Returns:
        Tuple of (TS data bytes, info dict)
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")
    
    # Load image to get dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        # Ensure dimensions are even (required for video encoding)
        width = width - (width % 2)
        height = height - (height % 2)
    
    # Create temp file for output if not specified
    if output_path is None:
        fd, temp_path = tempfile.mkstemp(suffix='.ts')
        os.close(fd)
        output_path = Path(temp_path)
        cleanup = True
    else:
        cleanup = False
    
    try:
        # Convert image to MPEG-2 TS using ffmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-loop', '1',  # Loop the image
            '-i', str(image_path),
            '-t', str(duration),  # Duration
            '-c:v', 'mpeg2video',  # MPEG-2 video codec
            '-b:v', '4000k',  # Video bitrate
            '-maxrate', '4000k',
            '-bufsize', '2000k',
            '-pix_fmt', 'yuv420p',
            '-s', f'{width}x{height}',  # Resolution
            '-r', str(fps),  # Frame rate
            '-f', 'mpegts',  # MPEG-TS container
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        # Read the output
        with open(output_path, 'rb') as f:
            ts_data = f.read()
        
        info = {
            'type': 'image',
            'source': str(image_path),
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps,
            'ts_size': len(ts_data),
            'packets': len(ts_data) // 188,
        }
        
        return ts_data, info
        
    finally:
        if cleanup and output_path.exists():
            output_path.unlink()


def video_to_ts(video_path: Path, output_path: Optional[Path] = None,
                max_duration: Optional[float] = None) -> Tuple[bytes, dict]:
    """
    Convert a video file to MPEG-2 Transport Stream.
    
    Args:
        video_path: Path to input video (MP4, AVI, etc.)
        output_path: Optional path for output TS file
        max_duration: Maximum duration to process (seconds), None for full video
        
    Returns:
        Tuple of (TS data bytes, info dict)
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")
    
    # Get source video info
    media_info = get_media_info(video_path)
    
    # Create temp file for output if not specified
    if output_path is None:
        fd, temp_path = tempfile.mkstemp(suffix='.ts')
        os.close(fd)
        output_path = Path(temp_path)
        cleanup = True
    else:
        cleanup = False
    
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', str(video_path),
        ]
        
        # Limit duration if specified
        if max_duration is not None:
            cmd.extend(['-t', str(max_duration)])
        
        cmd.extend([
            '-c:v', 'mpeg2video',  # MPEG-2 video codec
            '-b:v', '4000k',  # Video bitrate
            '-maxrate', '6000k',
            '-bufsize', '2000k',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'mp2',  # MPEG audio
            '-b:a', '192k',
            '-ar', '48000',  # Audio sample rate
            '-f', 'mpegts',  # MPEG-TS container
            str(output_path)
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        # Read the output
        with open(output_path, 'rb') as f:
            ts_data = f.read()
        
        # Extract info from ffprobe data
        width = height = 0
        duration = 0.0
        fps = 25
        
        if 'streams' in media_info:
            for stream in media_info['streams']:
                if stream.get('codec_type') == 'video':
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    # Parse frame rate
                    fps_str = stream.get('r_frame_rate', '25/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = int(num) / int(den) if int(den) > 0 else 25
                    break
        
        if 'format' in media_info:
            duration = float(media_info['format'].get('duration', 0))
        
        if max_duration is not None:
            duration = min(duration, max_duration)
        
        info = {
            'type': 'video',
            'source': str(video_path),
            'width': width,
            'height': height,
            'duration': duration,
            'fps': fps,
            'ts_size': len(ts_data),
            'packets': len(ts_data) // 188,
        }
        
        return ts_data, info
        
    finally:
        if cleanup and output_path.exists():
            output_path.unlink()


def convert_to_ts(file_path: Path, max_duration: Optional[float] = 5.0) -> Tuple[bytes, dict]:
    """
    Convert any supported media file to Transport Stream.
    
    Automatically detects file type and uses appropriate converter.
    
    Args:
        file_path: Path to input media file
        max_duration: Maximum duration for videos (seconds)
        
    Returns:
        Tuple of (TS data bytes, info dict)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    # Image formats
    image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    # Video formats
    video_formats = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v'}
    
    if suffix in image_formats:
        return image_to_ts(file_path, duration=1.0)
    elif suffix in video_formats:
        return video_to_ts(file_path, max_duration=max_duration)
    else:
        # Try to detect based on content
        try:
            Image.open(file_path)
            return image_to_ts(file_path, duration=1.0)
        except Exception:
            # Assume video
            return video_to_ts(file_path, max_duration=max_duration)


def generate_test_ts(num_packets: int = 100) -> Tuple[bytes, dict]:
    """
    Generate a test Transport Stream without requiring ffmpeg.
    
    Creates synthetic TS packets for testing purposes.
    
    Args:
        num_packets: Number of packets to generate
        
    Returns:
        Tuple of (TS data bytes, info dict)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from dvb import TransportStream, Packet, PAT, PMT
    from dvb.PMT import StreamType
    
    ts = TransportStream()
    
    # Create PAT
    pat = PAT(transport_stream_id=1)
    pat.add_program(1, 0x100)
    
    # Create PMT
    pmt = PMT(program_number=1, pcr_pid=0x101)
    pmt.add_stream(StreamType.MPEG2_VIDEO, 0x101)
    
    # Add PAT packet
    pat_data = pat.to_bytes()
    ts.append(Packet(
        pid=0x0000,
        payload_unit_start=True,
        payload=bytes([0]) + pat_data,
        continuity_counter=0,
    ))
    
    # Add PMT packet
    pmt_data = pmt.to_bytes()
    ts.append(Packet(
        pid=0x100,
        payload_unit_start=True,
        payload=bytes([0]) + pmt_data,
        continuity_counter=0,
    ))
    
    # Add video packets with pseudo-random data
    np.random.seed(42)
    video_cc = 0
    
    for i in range(num_packets - 2):
        payload = np.random.randint(0, 256, size=184, dtype=np.uint8).tobytes()
        ts.append(Packet(
            pid=0x101,
            payload=payload,
            continuity_counter=video_cc,
        ))
        video_cc = (video_cc + 1) % 16
    
    ts_data = ts.to_bytes()
    
    info = {
        'type': 'synthetic',
        'source': 'generated',
        'width': 720,
        'height': 576,
        'duration': num_packets / 25.0,
        'fps': 25,
        'ts_size': len(ts_data),
        'packets': len(ts_data) // 188,
    }
    
    return ts_data, info

