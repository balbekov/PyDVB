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
            mode: OFDM mode ('2K' or '8K')
            constellation: Modulation ('QPSK', '16QAM', '64QAM')
            code_rate: FEC rate ('1/2', '2/3', '3/4', '5/6', '7/8')
            guard_interval: Guard interval ('1/4', '1/8', '1/16', '1/32')
            bandwidth: Channel bandwidth ('6MHz', '7MHz', '8MHz')
        """
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
    
    Implements the receive chain to recover transport stream
    from baseband I/Q samples.
    
    Note: This is a simplified demodulator without channel estimation
    or synchronization. It's primarily for educational/testing purposes.
    """
    
    def __init__(self, mode: str = '2K', constellation: str = 'QPSK',
                 code_rate: str = '1/2', guard_interval: str = '1/4'):
        """
        Initialize DVB-T demodulator.
        
        Args:
            mode: OFDM mode
            constellation: Expected modulation
            code_rate: Expected FEC rate
            guard_interval: Expected guard interval
        """
        self.mode = mode
        self.constellation = constellation
        self.code_rate = code_rate
        self.guard_interval = guard_interval
        
        self._init_stages()
    
    def _init_stages(self) -> None:
        """Initialize demodulation stages."""
        from .OFDM import OFDMDemodulator
        from .GuardInterval import GuardIntervalRemover
        from .Pilots import PilotExtractor
        
        # OFDM demodulation
        fft_sizes = {'2K': 2048, '8K': 8192}
        self.fft_size = fft_sizes[self.mode]
        
        self.guard_remover = GuardIntervalRemover(self.guard_interval, self.fft_size)
        self.ofdm_demod = OFDMDemodulator(self.mode)
        self.pilot_extractor = PilotExtractor(self.mode)
        
        # Symbol deinterleaving
        self.symbol_deinterleaver = SymbolInterleaver(self.mode)
        
        # QAM demapping
        self.qam_demapper = QAMDemapper(self.constellation)
        
        # Bit deinterleaving
        self.bit_deinterleaver = BitInterleaver(self.constellation, self.mode)
        
        # Inner FEC
        self.depuncturer = Depuncturer(self.code_rate)
        self.conv_decoder = ConvolutionalDecoder()
        
        # Outer deinterleaving
        from .OuterInterleaver import OuterDeinterleaver
        self.outer_deinterleaver = OuterDeinterleaver()
        
        # RS decoding
        self.rs_decoder = ReedSolomon()
        
        # Descrambling
        self.descrambler = Scrambler()
    
    def demodulate(self, iq_samples: np.ndarray) -> Tuple[bytes, dict]:
        """
        Demodulate I/Q samples to transport stream.
        
        Args:
            iq_samples: Complex baseband samples
            
        Returns:
            Tuple of (recovered TS data, statistics dict)
        """
        stats = {'symbols': 0, 'rs_errors': 0, 'rs_uncorrectable': 0}
        
        # Calculate symbol length
        symbol_len = self.fft_size + self.guard_remover.guard_length
        num_symbols = len(iq_samples) // symbol_len
        
        if num_symbols == 0:
            return b'', stats
        
        stats['symbols'] = num_symbols
        
        # Process symbols
        all_data = []
        
        for sym_idx in range(num_symbols):
            start = sym_idx * symbol_len
            symbol_samples = iq_samples[start:start + symbol_len]
            
            # Remove guard interval
            useful = self.guard_remover.remove(symbol_samples)
            
            # OFDM demodulation
            carriers = self.ofdm_demod.demodulate(useful)
            
            # Extract data carriers (remove pilots)
            data_carriers, _, _ = self.pilot_extractor.extract(
                carriers, sym_idx % 68
            )
            
            all_data.append(data_carriers)
        
        # Concatenate all data
        data_carriers = np.concatenate(all_data)
        
        # Symbol deinterleave
        from .InnerInterleaver import SymbolInterleaver
        sym_deint = SymbolInterleaver(self.mode)
        deinterleaved = sym_deint.deinterleave(data_carriers)
        
        # QAM demap
        bits = self.qam_demapper.demap(deinterleaved)
        
        # Bit deinterleave
        bits = self.bit_deinterleaver.deinterleave(bits)
        
        # Depuncture
        depunctured = self.depuncturer.depuncture(bits.astype(np.float32))
        
        # Viterbi decode
        decoded = self.conv_decoder.decode(
            (depunctured > 0.5).astype(np.uint8),
            terminated=False
        )
        
        # Convert bits to bytes
        byte_data = np.packbits(decoded).tobytes()
        
        # Outer deinterleave
        deinterleaved = self.outer_deinterleaver.process(byte_data)
        
        # RS decode
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
        
        # Descramble
        ts_data = self.descrambler.descramble(bytes(recovered))
        
        return ts_data, stats


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
