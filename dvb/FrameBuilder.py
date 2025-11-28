"""
DVB-T OFDM Frame Builder

Assembles OFDM symbols into DVB-T frames and superframes.

DVB-T frame structure:
- 1 frame = 68 OFDM symbols
- 1 superframe = 4 frames = 272 symbols

Each symbol contains:
- Data carriers (QAM modulated)
- Continual pilots (fixed positions)
- Scattered pilots (rotating pattern)
- TPS carriers (transmission parameters)

Reference: ETSI EN 300 744 Section 4.5
"""

import numpy as np
from typing import Iterator, List, Optional

from .OFDM import OFDMModulator
from .GuardInterval import GuardIntervalInserter
from .Pilots import PilotInserter
from .TPS import TPSEncoder


class FrameBuilder:
    """
    DVB-T OFDM frame builder.
    
    Combines data, pilots, and TPS into complete OFDM frames.
    
    Attributes:
        mode: '2K', '8K', or 'audio'
        guard_interval: Guard interval ratio
        constellation: Modulation type
        symbols_per_frame: Number of symbols per frame (68 for 2K/8K, 16 for audio)
        
    Example:
        >>> builder = FrameBuilder('2K', '1/4', '64QAM')
        >>> frame_samples = builder.build_frame(data_symbols)
        
        >>> # Audio mode
        >>> builder = FrameBuilder('audio', '1/4', 'QPSK')
        >>> frame_samples = builder.build_frame(data_symbols)
    """
    
    SYMBOLS_PER_FRAME = 68
    SYMBOLS_PER_FRAME_AUDIO = 16  # Shorter frames for audio
    FRAMES_PER_SUPERFRAME = 4
    
    def __init__(self, mode: str = '2K', guard_interval: str = '1/4',
                 constellation: str = 'QPSK', code_rate: str = '1/2'):
        """
        Initialize frame builder.
        
        Args:
            mode: '2K', '8K', or 'audio'
            guard_interval: '1/4', '1/8', '1/16', '1/32'
            constellation: 'QPSK', '16QAM', '64QAM'
            code_rate: '1/2', '2/3', '3/4', '5/6', '7/8'
        """
        self.mode = mode
        self.guard_interval = guard_interval
        self.constellation = constellation
        self.code_rate = code_rate
        
        # Set symbols per frame based on mode
        self.symbols_per_frame = (
            self.SYMBOLS_PER_FRAME_AUDIO if mode == 'audio' 
            else self.SYMBOLS_PER_FRAME
        )
        
        # Initialize components
        self.ofdm_mod = OFDMModulator(mode)
        self.guard_inserter = GuardIntervalInserter(guard_interval, 
                                                     self.ofdm_mod.fft_size)
        self.pilot_inserter = PilotInserter(mode)
        self.tps_encoder = TPSEncoder(mode)
        
        # Set TPS parameters
        self.tps_encoder.set_parameters(
            constellation=constellation,
            hp_code_rate=code_rate,
            guard_interval=guard_interval,
            mode=mode,
        )
        
        # Calculate data capacity
        self._calc_capacity()
    
    def _calc_capacity(self) -> None:
        """Calculate data carrier capacity per symbol and frame."""
        # Get data carrier counts for each symbol pattern
        # Pattern repeats: 4 symbols for 2K/8K, 1 for audio (fixed pattern)
        pattern_len = 1 if self.mode == 'audio' else 4
        self.data_carriers_per_symbol = [
            self.pilot_inserter.get_data_carrier_count(i) 
            for i in range(pattern_len)
        ]
        
        # Total data carriers per frame
        self.data_carriers_per_frame = 0
        for sym in range(self.symbols_per_frame):
            self.data_carriers_per_frame += self.data_carriers_per_symbol[sym % pattern_len]
    
    def get_data_carriers_per_frame(self) -> int:
        """Get total number of data carriers per frame."""
        return self.data_carriers_per_frame
    
    def build_symbol(self, data_carriers: np.ndarray, 
                     symbol_index: int,
                     frame_tps: np.ndarray) -> np.ndarray:
        """
        Build one OFDM symbol with pilots and guard interval.
        
        Args:
            data_carriers: QAM-modulated data for this symbol
            symbol_index: Symbol number within frame (0-67)
            frame_tps: TPS bits for the frame
            
        Returns:
            Time-domain samples with guard interval
        """
        # Get TPS bits for this symbol
        tps_bits = self.tps_encoder.get_symbol_tps(symbol_index, frame_tps)
        
        # Insert pilots and data into carriers
        carriers = self.pilot_inserter.insert(data_carriers, tps_bits, symbol_index)
        
        # OFDM modulation (IFFT)
        time_samples = self.ofdm_mod.modulate(carriers)
        
        # Add guard interval
        symbol_with_guard = self.guard_inserter.add(time_samples)
        
        return symbol_with_guard
    
    def build_frame(self, data: np.ndarray, 
                    frame_number: int = 0) -> np.ndarray:
        """
        Build complete OFDM frame.
        
        Args:
            data: QAM-modulated data carriers for entire frame
            frame_number: Frame number within superframe (0-3)
            
        Returns:
            Time-domain samples for entire frame
        """
        # Generate TPS for this frame
        frame_tps = self.tps_encoder.encode_frame(frame_number)
        
        # Build each symbol
        symbols = []
        data_idx = 0
        pattern_len = 1 if self.mode == 'audio' else 4
        
        for sym_idx in range(self.symbols_per_frame):
            # Get data carriers needed for this symbol
            num_data = self.data_carriers_per_symbol[sym_idx % pattern_len]
            
            if data_idx + num_data <= len(data):
                sym_data = data[data_idx:data_idx + num_data]
            else:
                # Pad with zeros if not enough data
                sym_data = np.zeros(num_data, dtype=np.complex64)
                remaining = len(data) - data_idx
                if remaining > 0:
                    sym_data[:remaining] = data[data_idx:data_idx + remaining]
            
            data_idx += num_data
            
            # Build symbol
            symbol = self.build_symbol(sym_data, sym_idx, frame_tps)
            symbols.append(symbol)
        
        # Concatenate all symbols
        return np.concatenate(symbols)
    
    def build_superframe(self, data: np.ndarray) -> np.ndarray:
        """
        Build complete transmission (as many frames as needed).
        
        Args:
            data: QAM-modulated data for transmission
            
        Returns:
            Time-domain samples for all frames
        """
        frames = []
        data_per_frame = self.data_carriers_per_frame
        
        # Calculate number of frames needed
        num_frames = (len(data) + data_per_frame - 1) // data_per_frame
        num_frames = max(num_frames, self.FRAMES_PER_SUPERFRAME)  # At least one superframe
        
        for frame_num in range(num_frames):
            start = frame_num * data_per_frame
            end = start + data_per_frame
            
            frame_data = data[start:end] if start < len(data) else np.array([])
            # Frame number cycles 0-3 within each superframe
            frames.append(self.build_frame(frame_data, frame_num % self.FRAMES_PER_SUPERFRAME))
        
        return np.concatenate(frames)
    
    def iter_symbols(self, data: np.ndarray, 
                     start_frame: int = 0) -> Iterator[np.ndarray]:
        """
        Iterate over OFDM symbols.
        
        Yields symbols one at a time for streaming output.
        
        Args:
            data: QAM-modulated data
            start_frame: Starting frame number
            
        Yields:
            Time-domain samples for each symbol
        """
        data_idx = 0
        frame_num = start_frame
        pattern_len = 1 if self.mode == 'audio' else 4
        
        while data_idx < len(data):
            # Generate TPS for current frame
            frame_tps = self.tps_encoder.encode_frame(frame_num % 4)
            
            for sym_idx in range(self.symbols_per_frame):
                num_data = self.data_carriers_per_symbol[sym_idx % pattern_len]
                
                if data_idx >= len(data):
                    break
                
                sym_data = data[data_idx:data_idx + num_data]
                if len(sym_data) < num_data:
                    # Pad final symbol
                    padded = np.zeros(num_data, dtype=np.complex64)
                    padded[:len(sym_data)] = sym_data
                    sym_data = padded
                
                data_idx += num_data
                
                yield self.build_symbol(sym_data, sym_idx, frame_tps)
            
            frame_num += 1


class FrameInfo:
    """
    DVB-T frame timing and capacity information.
    """
    
    @staticmethod
    def get_frame_duration(mode: str, guard_interval: str,
                           bandwidth: str = '8MHz') -> float:
        """
        Get frame duration in seconds.
        
        Args:
            mode: '2K', '8K', or 'audio'
            guard_interval: Guard interval ratio
            bandwidth: Channel bandwidth
            
        Returns:
            Frame duration in seconds
        """
        sample_rates = {
            '8MHz': 9142857.142857143,
            '7MHz': 8000000.0,
            '6MHz': 6857142.857142857,
            'audio': 48000.0,
        }
        
        if mode == 'audio':
            sample_rate = 48000.0
        else:
            sample_rate = sample_rates.get(bandwidth, sample_rates['8MHz'])
        
        fft_sizes = {'2K': 2048, '8K': 8192, 'audio': 64}
        guard_fractions = {'1/4': 4, '1/8': 8, '1/16': 16, '1/32': 32, 'acoustic': 0.4}
        symbols_per_frame = {'2K': 68, '8K': 68, 'audio': 16}
        
        fft_size = fft_sizes.get(mode, 2048)
        guard_frac = guard_fractions.get(guard_interval, 4)
        num_symbols = symbols_per_frame.get(mode, 68)
        
        # Handle acoustic mode with fractional guard ratio
        if guard_frac < 1:
            guard_length = int(fft_size / guard_frac)
        else:
            guard_length = fft_size // int(guard_frac)
        symbol_samples = fft_size + guard_length
        symbol_duration = symbol_samples / sample_rate
        
        return symbol_duration * num_symbols
    
    @staticmethod
    def get_data_rate(mode: str, guard_interval: str, 
                      constellation: str, code_rate: str,
                      bandwidth: str = '8MHz') -> float:
        """
        Get net data rate in bits per second.
        
        Args:
            mode: '2K', '8K', or 'audio'
            guard_interval: Guard interval ratio
            constellation: Modulation type
            code_rate: FEC code rate
            bandwidth: Channel bandwidth
            
        Returns:
            Data rate in bps
        """
        # Create frame builder to get carrier count
        builder = FrameBuilder(mode, guard_interval, constellation, code_rate)
        carriers_per_frame = builder.get_data_carriers_per_frame()
        
        # Bits per carrier
        bits_per_symbol = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[constellation]
        
        # Code rate factor
        rate_values = {'1/2': 0.5, '2/3': 2/3, '3/4': 0.75, 
                      '5/6': 5/6, '7/8': 7/8}
        rate = rate_values.get(code_rate, 0.5)
        
        # Bits per frame
        bits_per_frame = carriers_per_frame * bits_per_symbol * rate
        
        # Frame duration (use 'audio' bandwidth for audio mode)
        bw = 'audio' if mode == 'audio' else bandwidth
        frame_duration = FrameInfo.get_frame_duration(mode, guard_interval, bw)
        
        return bits_per_frame / frame_duration
