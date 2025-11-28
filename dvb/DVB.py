"""
DVB-T Main Orchestrator

High-level interface that coordinates the complete DVB-T transmit pipeline:
    TS Packets → Scramble → RS Encode → Outer Interleave → 
    Conv Encode → Puncture → Inner Interleave → QAM Map → 
    Pilot Insert → OFDM Modulate → Guard Insert → I/Q Output

This module provides a simple interface for encoding transport streams
to SDR-ready baseband samples.

Reference: ETSI EN 300 744
"""

import numpy as np
from typing import Union, Optional, Iterator, Tuple
from pathlib import Path

from .TransportStream import TransportStream
from .Packet import Packet, PACKET_SIZE
from .Scrambler import Scrambler
from .ReedSolomon import ReedSolomon
from .OuterInterleaver import OuterInterleaver
from .Convolutional import ConvolutionalEncoder, ConvolutionalDecoder
from .Puncturing import Puncturer, Depuncturer
from .InnerInterleaver import BitInterleaver, SymbolInterleaver
from .QAM import QAMMapper, QAMDemapper
from .FrameBuilder import FrameBuilder
from .IQWriter import IQWriter


class DVBTModulator:
    """
    Complete DVB-T modulator.
    
    Implements the full DVB-T transmit chain from transport stream
    packets to baseband I/Q samples.
    
    Attributes:
        mode: '2K' or '8K' OFDM mode
        constellation: 'QPSK', '16QAM', or '64QAM'
        code_rate: '1/2', '2/3', '3/4', '5/6', or '7/8'
        guard_interval: '1/4', '1/8', '1/16', or '1/32'
        
    Example:
        >>> mod = DVBTModulator(mode='2K', constellation='64QAM', code_rate='2/3')
        >>> iq_samples = mod.modulate(ts_data)
        >>> mod.write_iq('output.cf32', iq_samples)
    """
    
    def __init__(self, mode: str = '2K', constellation: str = 'QPSK',
                 code_rate: str = '1/2', guard_interval: str = '1/4',
                 bandwidth: str = '8MHz'):
        """
        Initialize DVB-T modulator.
        
        Args:
            mode: OFDM mode ('2K', '8K', or 'audio')
            constellation: Modulation ('QPSK', '16QAM', '64QAM')
            code_rate: FEC rate ('1/2', '2/3', '3/4', '5/6', '7/8')
            guard_interval: Guard interval ('1/4', '1/8', '1/16', '1/32')
            bandwidth: Channel bandwidth ('6MHz', '7MHz', '8MHz', or 'audio')
        """
        # Handle audio mode: if bandwidth is 'audio', force mode to 'audio'
        if bandwidth == 'audio':
            mode = 'audio'
        
        self.mode = mode
        self.constellation = constellation
        self.code_rate = code_rate
        self.guard_interval = guard_interval
        self.bandwidth = bandwidth
        
        # Initialize processing stages
        self._init_stages()
    
    def _init_stages(self) -> None:
        """Initialize all processing stages."""
        # Energy dispersal
        self.scrambler = Scrambler()
        
        # Outer FEC
        self.rs_encoder = ReedSolomon()
        self.outer_interleaver = OuterInterleaver()
        
        # Inner FEC
        self.conv_encoder = ConvolutionalEncoder()
        self.puncturer = Puncturer(self.code_rate)
        
        # Inner interleaving
        self.bit_interleaver = BitInterleaver(self.constellation, self.mode)
        self.symbol_interleaver = SymbolInterleaver(self.mode)
        
        # Modulation
        self.qam_mapper = QAMMapper(self.constellation)
        
        # OFDM frame building
        self.frame_builder = FrameBuilder(
            self.mode, self.guard_interval,
            self.constellation, self.code_rate
        )
    
    # Outer interleaver fill delay: I * (I-1) * M / 2 * 2 = I * (I-1) * M
    # With I=12, M=17: 12 * 11 * 17 = 2244 bytes
    # We need ~12 packets of padding to flush the pipeline
    INTERLEAVER_FLUSH_PACKETS = 12
    
    def modulate(self, ts_data: Union[bytes, bytearray, 
                                       TransportStream]) -> np.ndarray:
        """
        Modulate transport stream to baseband I/Q samples.
        
        Args:
            ts_data: Transport stream data (bytes or TransportStream)
            
        Returns:
            Complex baseband samples
        """
        # Convert to bytes if TransportStream
        if isinstance(ts_data, TransportStream):
            ts_data = ts_data.to_bytes()
        
        # Ensure we have complete packets
        num_packets = len(ts_data) // PACKET_SIZE
        if num_packets == 0:
            return np.array([], dtype=np.complex64)
        
        ts_data = ts_data[:num_packets * PACKET_SIZE]
        
        # For audio mode, add padding packets to flush the outer interleaver
        # The interleaver has a fill delay of ~2244 bytes, so we add 12 null packets
        if self.mode == 'audio':
            # Create null TS packets (sync byte 0x47, then zeros, PID 0x1FFF = null)
            null_packet = bytes([0x47, 0x1F, 0xFF, 0x10]) + bytes(184)
            padding = null_packet * self.INTERLEAVER_FLUSH_PACKETS
            ts_data = ts_data + padding
            num_packets += self.INTERLEAVER_FLUSH_PACKETS
        
        # Stage 1: Energy dispersal (scrambling)
        scrambled = self.scrambler.scramble(ts_data)
        
        # Stage 2: Reed-Solomon encoding
        rs_encoded = self._rs_encode_stream(scrambled)
        
        # Stage 3: Outer interleaving
        self.outer_interleaver.reset()
        outer_interleaved = self.outer_interleaver.interleave(rs_encoded)
        
        # Stage 4: Convolutional encoding
        self.conv_encoder.reset()
        conv_encoded = self.conv_encoder.encode(outer_interleaved, terminate=False)
        
        # Stage 5: Puncturing
        punctured = self.puncturer.puncture(conv_encoded)
        
        # Stage 6: Bit interleaving
        bit_interleaved = self.bit_interleaver.interleave(punctured)
        
        # Stage 7: QAM mapping
        qam_symbols = self.qam_mapper.map(bit_interleaved)
        
        # Stage 8: Symbol interleaving
        # Pad to multiple of data carriers per symbol
        carriers_per_frame = self.frame_builder.get_data_carriers_per_frame()
        
        if len(qam_symbols) % carriers_per_frame != 0:
            pad_len = carriers_per_frame - (len(qam_symbols) % carriers_per_frame)
            qam_symbols = np.concatenate([
                qam_symbols, 
                np.zeros(pad_len, dtype=np.complex64)
            ])
        
        symbol_interleaved = self.symbol_interleaver.interleave(qam_symbols)
        
        # Stage 9: OFDM frame building (includes pilot insertion)
        iq_samples = self.frame_builder.build_superframe(symbol_interleaved)
        
        return iq_samples
    
    def _rs_encode_stream(self, data: bytes) -> bytes:
        """
        Apply RS encoding to stream of packets.
        
        Args:
            data: Scrambled TS packets
            
        Returns:
            RS encoded data (204 bytes per 188 input)
        """
        num_packets = len(data) // PACKET_SIZE
        encoded = bytearray()
        
        for i in range(num_packets):
            packet = data[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
            encoded.extend(self.rs_encoder.encode(packet))
        
        return bytes(encoded)
    
    def modulate_file(self, input_path: Union[str, Path],
                      output_path: Union[str, Path],
                      format: str = 'cf32') -> None:
        """
        Modulate transport stream file to I/Q file.
        
        Args:
            input_path: Path to input .ts file
            output_path: Path to output I/Q file
            format: Output format ('cf32', 'cs8', 'cs16', 'cu8')
        """
        # Read transport stream
        with open(input_path, 'rb') as f:
            ts_data = f.read()
        
        # Modulate
        iq_samples = self.modulate(ts_data)
        
        # Write output
        writer = IQWriter(format)
        writer.write(output_path, iq_samples)
    
    def iter_modulate(self, ts_data: Union[bytes, bytearray],
                      symbols_per_chunk: int = 68) -> Iterator[np.ndarray]:
        """
        Modulate in chunks for streaming output.
        
        Args:
            ts_data: Transport stream data
            symbols_per_chunk: OFDM symbols per output chunk
            
        Yields:
            I/Q sample chunks
        """
        # For streaming, we process frame by frame
        # This is a simplified implementation
        iq_samples = self.modulate(ts_data)
        
        samples_per_symbol = (self.frame_builder.ofdm_mod.fft_size + 
                             self.frame_builder.guard_inserter.guard_length)
        chunk_size = symbols_per_chunk * samples_per_symbol
        
        for i in range(0, len(iq_samples), chunk_size):
            yield iq_samples[i:i + chunk_size]
    
    def get_data_rate(self) -> float:
        """
        Get net data rate in bits per second.
        
        Returns:
            Data rate in bps
        """
        from .FrameBuilder import FrameInfo
        return FrameInfo.get_data_rate(
            self.mode, self.guard_interval,
            self.constellation, self.code_rate,
            self.bandwidth
        )
    
    def get_sample_rate(self) -> float:
        """
        Get output sample rate in Hz.
        
        Returns:
            Sample rate
        """
        # For audio mode, OFDM runs at 16 kHz (upsampled to 48kHz for audio output)
        if self.mode == 'audio' or self.bandwidth == 'audio':
            return 16000.0
        
        sample_rates = {
            '8MHz': 9142857.142857143,
            '7MHz': 8000000.0,
            '6MHz': 6857142.857142857,
        }
        return sample_rates.get(self.bandwidth, sample_rates['8MHz'])
    
    def write_iq(self, path: Union[str, Path], samples: np.ndarray,
                 format: str = 'cf32') -> None:
        """
        Write I/Q samples to file.
        
        Args:
            path: Output file path
            samples: Complex samples
            format: Output format
        """
        writer = IQWriter(format)
        writer.write(path, samples)


class DVBTDemodulator:
    """
    DVB-T demodulator (receiver).
    
    Implements the complete receive chain to recover transport stream
    from baseband I/Q samples, including:
    - Synchronization (time, frequency, frame)
    - Channel estimation and equalization
    - OFDM demodulation
    - FEC decoding
    
    Attributes:
        mode: '2K' or '8K' (or 'auto' for detection)
        constellation: 'QPSK', '16QAM', '64QAM'
        code_rate: '1/2', '2/3', '3/4', '5/6', '7/8'
        guard_interval: '1/4', '1/8', '1/16', '1/32'
        
    Example:
        >>> demod = DVBTDemodulator(mode='2K', constellation='64QAM')
        >>> ts_data, stats = demod.demodulate(iq_samples)
        >>> 
        >>> # Auto-detect parameters
        >>> demod = DVBTDemodulator(mode='auto')
        >>> ts_data, stats = demod.demodulate(iq_samples)
    """
    
    def __init__(self, mode: str = '2K', constellation: str = 'QPSK',
                 code_rate: str = '1/2', guard_interval: str = '1/4',
                 bandwidth: str = '8MHz',
                 equalization: str = 'zf',
                 soft_decision: bool = False,
                 progress_callback: callable = None):
        """
        Initialize DVB-T demodulator.
        
        Args:
            mode: OFDM mode ('2K', '8K', 'audio', or 'auto')
            constellation: Expected modulation
            code_rate: Expected FEC rate
            guard_interval: Expected guard interval
            bandwidth: Channel bandwidth for sample rate ('6MHz', '7MHz', '8MHz', 'audio')
            equalization: Equalization method ('zf' or 'mmse')
            soft_decision: Use soft decision decoding
            progress_callback: Optional callback(symbols_done, total_symbols, phase) for progress updates
        """
        self.progress_callback = progress_callback
        # Handle audio mode: if bandwidth is 'audio', force mode to 'audio'
        if bandwidth == 'audio':
            mode = 'audio'
        
        self.mode = mode
        self.constellation = constellation
        self.code_rate = code_rate
        self.guard_interval = guard_interval
        self.bandwidth = bandwidth
        self.equalization = equalization
        self.soft_decision = soft_decision
        
        # Sample rate
        sample_rates = {
            '8MHz': 9142857.142857143,
            '7MHz': 8000000.0,
            '6MHz': 6857142.857142857,
            'audio': 16000.0,  # OFDM sample rate (upsampled to 48kHz for audio)
        }
        self.sample_rate = sample_rates.get(bandwidth, sample_rates['8MHz'])
        
        # Auto-detect flag
        self._auto_detect = (mode == 'auto')
        
        if not self._auto_detect:
            self._init_stages()
    
    def _init_stages(self) -> None:
        """Initialize demodulation stages."""
        from .OFDM import OFDMDemodulator
        from .GuardInterval import GuardIntervalRemover
        from .Pilots import PilotExtractor
        from .Synchronizer import DVBTSynchronizer
        from .ChannelEstimator import ChannelEstimatorWithEqualization
        
        # FFT parameters
        fft_sizes = {'2K': 2048, '8K': 8192, 'audio': 64}
        self.fft_size = fft_sizes[self.mode]
        
        # Synchronization
        self.synchronizer = DVBTSynchronizer(
            self.mode, self.guard_interval, self.sample_rate
        )
        
        # Guard interval removal
        self.guard_remover = GuardIntervalRemover(self.guard_interval, self.fft_size)
        
        # OFDM demodulation
        self.ofdm_demod = OFDMDemodulator(self.mode)
        
        # Channel estimation and equalization
        self.channel_eq = ChannelEstimatorWithEqualization(
            self.mode, 
            interpolation='linear',
            method=self.equalization
        )
        
        # Pilot extraction
        self.pilot_extractor = PilotExtractor(self.mode)
        
        # Symbol deinterleaving
        self.symbol_deinterleaver = SymbolInterleaver(self.mode)
        
        # QAM demapping
        self.qam_demapper = QAMDemapper(self.constellation, self.soft_decision)
        
        # Bit deinterleaving
        self.bit_deinterleaver = BitInterleaver(self.constellation, self.mode)
        
        # Inner FEC
        self.depuncturer = Depuncturer(self.code_rate)
        self.conv_decoder = ConvolutionalDecoder(self.soft_decision)
        
        # Outer deinterleaving
        from .OuterInterleaver import OuterDeinterleaver
        self.outer_deinterleaver = OuterDeinterleaver()
        
        # RS decoding
        self.rs_decoder = ReedSolomon()
        
        # Descrambling
        self.descrambler = Scrambler()
    
    def _auto_detect_params(self, iq_samples: np.ndarray) -> None:
        """Auto-detect DVB-T parameters from signal."""
        from .Detector import DVBTDetector
        
        detector = DVBTDetector(self.sample_rate)
        params = detector.detect(iq_samples)
        
        self.mode = params.mode
        self.guard_interval = params.guard_interval
        self.constellation = params.constellation
        self.code_rate = params.code_rate
        
        self._init_stages()
        self._auto_detect = False
    
    def demodulate(self, iq_samples: np.ndarray) -> Tuple[bytes, dict]:
        """
        Demodulate I/Q samples to transport stream.
        
        Args:
            iq_samples: Complex baseband samples
            
        Returns:
            Tuple of (recovered TS data, statistics dict)
        """
        # Auto-detect if needed
        if self._auto_detect:
            self._auto_detect_params(iq_samples)
        
        stats = {
            'symbols': 0, 
            'rs_errors': 0, 
            'rs_uncorrectable': 0,
            'snr_db': 0.0,
            'cfo_hz': 0.0,
            'packets_recovered': 0,
        }
        
        # Step 1: Synchronization
        sync_result = self.synchronizer.synchronize(iq_samples)
        stats['cfo_hz'] = sync_result.coarse_cfo
        stats['snr_db'] = sync_result.snr_estimate
        
        # Apply CFO correction
        corrected_samples = self.synchronizer.coarse.correct_cfo(
            iq_samples, sync_result.coarse_cfo
        )
        
        # Align to symbol boundary
        aligned_samples = corrected_samples[sync_result.symbol_start:]
        
        # Calculate symbol length
        symbol_len = self.fft_size + self.guard_remover.guard_length
        num_symbols = len(aligned_samples) // symbol_len
        
        if num_symbols == 0:
            return b'', stats
        
        stats['symbols'] = num_symbols
        
        # Step 2: Process symbols
        all_data = []
        self.channel_eq.reset()
        
        # Symbols per frame depends on mode
        symbols_per_frame = 16 if self.mode == 'audio' else 68
        
        for sym_idx in range(num_symbols):
            start = sym_idx * symbol_len
            symbol_samples = aligned_samples[start:start + symbol_len]
            
            # Remove guard interval
            useful = self.guard_remover.remove(symbol_samples)
            
            # OFDM demodulation (FFT)
            carriers = self.ofdm_demod.demodulate(useful)
            
            # Channel estimation and equalization
            equalized, csi = self.channel_eq.process(carriers, sym_idx % symbols_per_frame)
            
            # Fine frequency/phase correction using pilots
            fine_cfo, phase = self.synchronizer.refine_sync(equalized, sym_idx % symbols_per_frame)
            equalized = self.synchronizer.fine.correct_phase(equalized, phase)
            
            # Extract data carriers (remove pilots)
            data_carriers, _, _ = self.pilot_extractor.extract(
                equalized, sym_idx % symbols_per_frame
            )
            
            all_data.append(data_carriers)
            
            # Progress callback for real-time updates
            if self.progress_callback and sym_idx % 50 == 0:
                self.progress_callback(sym_idx + 1, num_symbols, 'symbols')
        
        # Concatenate all data carriers
        data_carriers = np.concatenate(all_data)
        
        # Step 3: Symbol deinterleaving
        deinterleaved = self.symbol_deinterleaver.deinterleave(data_carriers)
        
        # Step 4: QAM demapping
        if self.soft_decision:
            llrs = self.qam_demapper.demap(deinterleaved)
            bits = llrs
        else:
            bits = self.qam_demapper.demap(deinterleaved)
        
        # Step 5: Bit deinterleaving
        bits = self.bit_deinterleaver.deinterleave(bits)
        
        # Step 6: Depuncturing
        depunctured = self.depuncturer.depuncture(bits.astype(np.float32))
        
        # Step 7: Viterbi decoding
        if self.soft_decision:
            # Soft decision: use LLRs directly
            decoded = self.conv_decoder.decode(depunctured, terminated=False)
        else:
            # Hard decision
            decoded = self.conv_decoder.decode(
                (depunctured > 0.5).astype(np.uint8),
                terminated=False
            )
        
        # Convert bits to bytes
        byte_data = np.packbits(decoded).tobytes()
        
        # Step 8: Outer deinterleaving
        deinterleaved = self.outer_deinterleaver.process(byte_data)
        
        # For audio mode, skip the interleaver fill delay
        # The fill produces ~2244 bytes of zeros at the start
        # Skip first 11 RS codewords (11 * 204 = 2244 bytes)
        if self.mode == 'audio':
            interleaver_fill = 11 * 204  # Skip first 11 RS blocks worth
            deinterleaved = deinterleaved[interleaver_fill:]
        
        # Step 9: RS decoding
        recovered = bytearray()
        num_codewords = len(deinterleaved) // 204
        
        for i in range(num_codewords):
            codeword = deinterleaved[i * 204:(i + 1) * 204]
            decoded_pkt, errors = self.rs_decoder.decode(codeword)
            
            if errors < 0:
                stats['rs_uncorrectable'] += 1
            elif errors > 0:
                stats['rs_errors'] += errors
            
            recovered.extend(decoded_pkt)
        
        # Step 10: Descrambling
        ts_data = self.descrambler.descramble(bytes(recovered))
        
        stats['packets_recovered'] = len(ts_data) // 188
        
        return ts_data, stats
    
    def demodulate_file(self, input_path: Union[str, Path],
                        output_path: Optional[Union[str, Path]] = None,
                        format: str = 'auto') -> Tuple[bytes, dict]:
        """
        Demodulate I/Q file to transport stream.
        
        Args:
            input_path: Path to input I/Q file
            output_path: Optional path to write TS output
            format: I/Q format ('cf32', 'cs8', etc. or 'auto')
            
        Returns:
            Tuple of (TS data, statistics)
        """
        from .IQReader import IQReader
        
        reader = IQReader(format, self.sample_rate)
        iq_samples = reader.read(input_path)
        
        ts_data, stats = self.demodulate(iq_samples)
        
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(ts_data)
        
        return ts_data, stats
    
    def iter_demodulate(self, iq_samples: np.ndarray,
                        frames_per_chunk: int = 1) -> Iterator[Tuple[bytes, dict]]:
        """
        Demodulate in chunks for streaming processing.
        
        Args:
            iq_samples: Input I/Q samples
            frames_per_chunk: Number of frames per output chunk
            
        Yields:
            Tuple of (TS chunk, statistics)
        """
        # Auto-detect if needed (uses first chunk)
        if self._auto_detect:
            self._auto_detect_params(iq_samples)
        
        symbol_len = self.fft_size + self.guard_remover.guard_length
        symbols_per_frame = 16 if self.mode == 'audio' else 68
        samples_per_frame = symbol_len * symbols_per_frame
        samples_per_chunk = samples_per_frame * frames_per_chunk
        
        # Initial sync
        sync_result = self.synchronizer.synchronize(iq_samples)
        corrected = self.synchronizer.coarse.correct_cfo(
            iq_samples, sync_result.coarse_cfo
        )
        aligned = corrected[sync_result.symbol_start:]
        
        # Process in chunks
        for chunk_start in range(0, len(aligned) - samples_per_chunk, samples_per_chunk):
            chunk = aligned[chunk_start:chunk_start + samples_per_chunk]
            ts_data, stats = self._demodulate_chunk(chunk, chunk_start // symbol_len)
            yield ts_data, stats
    
    def _demodulate_chunk(self, samples: np.ndarray, 
                          start_symbol: int) -> Tuple[bytes, dict]:
        """Demodulate a chunk of samples (internal)."""
        stats = {'symbols': 0, 'rs_errors': 0, 'rs_uncorrectable': 0}
        
        symbol_len = self.fft_size + self.guard_remover.guard_length
        num_symbols = len(samples) // symbol_len
        symbols_per_frame = 16 if self.mode == 'audio' else 68
        
        if num_symbols == 0:
            return b'', stats
        
        stats['symbols'] = num_symbols
        all_data = []
        
        for sym_idx in range(num_symbols):
            frame_sym_idx = (start_symbol + sym_idx) % symbols_per_frame
            start = sym_idx * symbol_len
            symbol_samples = samples[start:start + symbol_len]
            
            useful = self.guard_remover.remove(symbol_samples)
            carriers = self.ofdm_demod.demodulate(useful)
            equalized, _ = self.channel_eq.process(carriers, frame_sym_idx)
            
            _, phase = self.synchronizer.refine_sync(equalized, frame_sym_idx)
            equalized = self.synchronizer.fine.correct_phase(equalized, phase)
            
            data_carriers, _, _ = self.pilot_extractor.extract(equalized, frame_sym_idx)
            all_data.append(data_carriers)
        
        data_carriers = np.concatenate(all_data)
        deinterleaved = self.symbol_deinterleaver.deinterleave(data_carriers)
        bits = self.qam_demapper.demap(deinterleaved)
        bits = self.bit_deinterleaver.deinterleave(bits)
        depunctured = self.depuncturer.depuncture(bits.astype(np.float32))
        decoded = self.conv_decoder.decode(
            (depunctured > 0.5).astype(np.uint8), terminated=False
        )
        
        byte_data = np.packbits(decoded).tobytes()
        deinterleaved = self.outer_deinterleaver.process(byte_data)
        
        recovered = bytearray()
        num_codewords = len(deinterleaved) // 204
        
        for i in range(num_codewords):
            codeword = deinterleaved[i * 204:(i + 1) * 204]
            decoded_pkt, errors = self.rs_decoder.decode(codeword)
            
            if errors < 0:
                stats['rs_uncorrectable'] += 1
            elif errors > 0:
                stats['rs_errors'] += errors
            
            recovered.extend(decoded_pkt)
        
        ts_data = self.descrambler.descramble(bytes(recovered))
        return ts_data, stats
    
    def get_sample_rate(self) -> float:
        """Get expected sample rate in Hz."""
        return self.sample_rate
    
    def reset(self) -> None:
        """Reset demodulator state."""
        if hasattr(self, 'channel_eq'):
            self.channel_eq.reset()
        if hasattr(self, 'synchronizer'):
            self.synchronizer.fine._last_phase = 0.0


def modulate(ts_data: Union[bytes, TransportStream],
             mode: str = '2K',
             constellation: str = 'QPSK',
             code_rate: str = '1/2',
             guard_interval: str = '1/4') -> np.ndarray:
    """
    Convenience function to modulate transport stream.
    
    Args:
        ts_data: Transport stream data
        mode: OFDM mode
        constellation: Modulation type
        code_rate: FEC rate
        guard_interval: Guard interval
        
    Returns:
        Complex baseband samples
    """
    mod = DVBTModulator(mode, constellation, code_rate, guard_interval)
    return mod.modulate(ts_data)
