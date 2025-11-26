"""
DVB-T Pipeline Wrapper with Intermediate Data Capture

Wraps the DVBTModulator to capture data at each stage of the pipeline
for visualization purposes.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dvb import (
    DVBTModulator,
    TransportStream,
    Scrambler,
    ReedSolomon,
    OuterInterleaver,
    ConvolutionalEncoder,
    Puncturer,
    BitInterleaver,
    SymbolInterleaver,
    QAMMapper,
    FrameBuilder,
)
from dvb.Packet import PACKET_SIZE


@dataclass
class DVBTParams:
    """DVB-T modulation parameters."""
    mode: str = '2K'
    constellation: str = 'QPSK'
    code_rate: str = '1/2'
    guard_interval: str = '1/4'
    bandwidth: str = '8MHz'
    
    def to_dict(self) -> dict:
        return {
            'mode': self.mode,
            'constellation': self.constellation,
            'code_rate': self.code_rate,
            'guard_interval': self.guard_interval,
            'bandwidth': self.bandwidth,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DVBTParams':
        return cls(
            mode=d.get('mode', '2K'),
            constellation=d.get('constellation', 'QPSK'),
            code_rate=d.get('code_rate', '1/2'),
            guard_interval=d.get('guard_interval', '1/4'),
            bandwidth=d.get('bandwidth', '8MHz'),
        )


@dataclass
class PipelineResults:
    """Results from pipeline processing with intermediate data."""
    
    # Input stage
    input_data: bytes = b''
    input_packets: int = 0
    pid_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Stage 1: Scrambling
    pre_scramble: np.ndarray = field(default_factory=lambda: np.array([]))
    post_scramble: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 2: Reed-Solomon
    rs_input: np.ndarray = field(default_factory=lambda: np.array([]))
    rs_output: np.ndarray = field(default_factory=lambda: np.array([]))
    rs_parity_example: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 3: Outer interleaving
    outer_interleaved: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 4: Convolutional encoding
    conv_input: np.ndarray = field(default_factory=lambda: np.array([]))
    conv_output: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 5: Puncturing
    punctured: np.ndarray = field(default_factory=lambda: np.array([]))
    puncture_pattern: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 6: Inner interleaving
    bit_interleaved: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 7: QAM mapping
    qam_symbols: np.ndarray = field(default_factory=lambda: np.array([]))
    constellation_points: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 8: Symbol interleaving
    symbol_interleaved: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Stage 9: OFDM
    ofdm_carriers: np.ndarray = field(default_factory=lambda: np.array([]))
    ofdm_symbol_example: np.ndarray = field(default_factory=lambda: np.array([]))
    pilot_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Output
    iq_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Parameters used
    params: DVBTParams = field(default_factory=DVBTParams)
    
    # Statistics
    sample_rate: float = 0.0
    data_rate: float = 0.0
    duration: float = 0.0


class DVBPipelineViz:
    """
    DVB-T pipeline with intermediate data capture for visualization.
    
    Processes Transport Stream data through the DVB-T chain while
    capturing data at each stage for visualization.
    """
    
    def __init__(self, params: Optional[DVBTParams] = None):
        """
        Initialize pipeline with parameters.
        
        Args:
            params: DVB-T modulation parameters
        """
        self.params = params or DVBTParams()
        self._init_stages()
    
    def _init_stages(self) -> None:
        """Initialize all processing stages."""
        p = self.params
        
        # Energy dispersal
        self.scrambler = Scrambler()
        
        # Outer FEC
        self.rs_encoder = ReedSolomon()
        self.outer_interleaver = OuterInterleaver()
        
        # Inner FEC
        self.conv_encoder = ConvolutionalEncoder()
        self.puncturer = Puncturer(p.code_rate)
        
        # Inner interleaving
        self.bit_interleaver = BitInterleaver(p.constellation, p.mode)
        self.symbol_interleaver = SymbolInterleaver(p.mode)
        
        # Modulation
        self.qam_mapper = QAMMapper(p.constellation)
        
        # OFDM frame building
        self.frame_builder = FrameBuilder(
            p.mode, p.guard_interval,
            p.constellation, p.code_rate
        )
    
    def process(self, ts_data: bytes) -> PipelineResults:
        """
        Process transport stream through DVB-T pipeline with data capture.
        
        Args:
            ts_data: Transport stream data (bytes)
            
        Returns:
            PipelineResults with intermediate data from each stage
        """
        results = PipelineResults(params=self.params)
        
        # Input analysis
        results.input_data = ts_data
        num_packets = len(ts_data) // PACKET_SIZE
        results.input_packets = num_packets
        
        if num_packets == 0:
            return results
        
        # Truncate to complete packets
        ts_data = ts_data[:num_packets * PACKET_SIZE]
        
        # Analyze PID distribution
        results.pid_distribution = self._analyze_pids(ts_data)
        
        # Store pre-scramble bit distribution (sample)
        sample_size = min(len(ts_data), 10000)
        results.pre_scramble = np.unpackbits(
            np.frombuffer(ts_data[:sample_size], dtype=np.uint8)
        )
        
        # Stage 1: Scrambling
        scrambled = self.scrambler.scramble(ts_data)
        results.post_scramble = np.unpackbits(
            np.frombuffer(scrambled[:sample_size], dtype=np.uint8)
        )
        
        # Stage 2: Reed-Solomon encoding
        rs_encoded = self._rs_encode_stream(scrambled)
        results.rs_input = np.frombuffer(scrambled[:PACKET_SIZE], dtype=np.uint8)
        results.rs_output = np.frombuffer(rs_encoded[:204], dtype=np.uint8)
        # Capture parity bytes from first packet
        results.rs_parity_example = np.frombuffer(rs_encoded[188:204], dtype=np.uint8)
        
        # Stage 3: Outer interleaving
        self.outer_interleaver.reset()
        outer_interleaved = self.outer_interleaver.interleave(rs_encoded)
        results.outer_interleaved = np.frombuffer(
            outer_interleaved[:1000], dtype=np.uint8
        )
        
        # Stage 4: Convolutional encoding
        self.conv_encoder.reset()
        
        # Convert bytes to bits for convolutional encoder
        outer_bits = np.unpackbits(
            np.frombuffer(outer_interleaved, dtype=np.uint8)
        )
        results.conv_input = outer_bits[:2000].copy()
        
        conv_encoded = self.conv_encoder.encode(outer_interleaved, terminate=False)
        results.conv_output = conv_encoded[:4000].copy()
        
        # Stage 5: Puncturing
        punctured = self.puncturer.puncture(conv_encoded)
        results.punctured = punctured[:2000].copy()
        
        # Get puncture pattern
        results.puncture_pattern = self._get_puncture_pattern()
        
        # Stage 6: Bit interleaving
        bit_interleaved = self.bit_interleaver.interleave(punctured)
        results.bit_interleaved = bit_interleaved[:2000].copy()
        
        # Stage 7: QAM mapping
        qam_symbols = self.qam_mapper.map(bit_interleaved)
        results.qam_symbols = qam_symbols[:5000].copy()
        results.constellation_points = self.qam_mapper._table.copy()
        
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
        results.symbol_interleaved = symbol_interleaved[:5000].copy()
        
        # Stage 9: OFDM frame building
        # Capture one complete OFDM symbol for visualization
        results.ofdm_carriers, results.pilot_positions = self._build_ofdm_example(
            symbol_interleaved
        )
        
        # Build complete superframe
        iq_samples = self.frame_builder.build_superframe(symbol_interleaved)
        results.iq_samples = iq_samples
        
        # Capture one OFDM symbol time-domain samples
        symbol_len = (self.frame_builder.ofdm_mod.fft_size + 
                      self.frame_builder.guard_inserter.guard_length)
        if len(iq_samples) >= symbol_len:
            results.ofdm_symbol_example = iq_samples[:symbol_len].copy()
        
        # Calculate statistics
        sample_rates = {
            '8MHz': 9142857.142857143,
            '7MHz': 8000000.0,
            '6MHz': 6857142.857142857,
        }
        results.sample_rate = sample_rates.get(self.params.bandwidth, sample_rates['8MHz'])
        results.duration = len(iq_samples) / results.sample_rate if results.sample_rate > 0 else 0
        
        # Calculate data rate
        bits_per_symbol = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[self.params.constellation]
        rate_values = {'1/2': 0.5, '2/3': 2/3, '3/4': 0.75, '5/6': 5/6, '7/8': 7/8}
        code_rate = rate_values.get(self.params.code_rate, 0.5)
        
        carriers_per_frame = self.frame_builder.get_data_carriers_per_frame()
        frame_duration = 68 * (self.frame_builder.ofdm_mod.fft_size + 
                               self.frame_builder.guard_inserter.guard_length) / results.sample_rate
        results.data_rate = carriers_per_frame * bits_per_symbol * code_rate / frame_duration
        
        return results
    
    def _analyze_pids(self, ts_data: bytes) -> Dict[int, int]:
        """Analyze PID distribution in TS data."""
        pid_counts = {}
        
        for i in range(0, len(ts_data), PACKET_SIZE):
            if i + 4 > len(ts_data):
                break
            
            # PID is in bytes 1-2 (13 bits)
            pid = ((ts_data[i + 1] & 0x1F) << 8) | ts_data[i + 2]
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
        
        return pid_counts
    
    def _rs_encode_stream(self, data: bytes) -> bytes:
        """Apply RS encoding to stream of packets."""
        num_packets = len(data) // PACKET_SIZE
        encoded = bytearray()
        
        for i in range(num_packets):
            packet = data[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
            encoded.extend(self.rs_encoder.encode(packet))
        
        return bytes(encoded)
    
    def _get_puncture_pattern(self) -> np.ndarray:
        """Get the puncturing pattern for current code rate."""
        patterns = {
            '1/2': np.array([1, 1]),
            '2/3': np.array([1, 1, 0, 1]),
            '3/4': np.array([1, 1, 0, 1, 1, 0]),
            '5/6': np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
            '7/8': np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
        }
        return patterns.get(self.params.code_rate, patterns['1/2'])
    
    def _build_ofdm_example(self, data: np.ndarray) -> tuple:
        """Build one OFDM symbol for visualization."""
        # Get carriers needed for one symbol
        num_data = self.frame_builder.data_carriers_per_symbol[0]
        
        if len(data) < num_data:
            sym_data = np.zeros(num_data, dtype=np.complex64)
            sym_data[:len(data)] = data
        else:
            sym_data = data[:num_data]
        
        # Get TPS for first symbol
        frame_tps = self.frame_builder.tps_encoder.encode_frame(0)
        tps_bits = self.frame_builder.tps_encoder.get_symbol_tps(0, frame_tps)
        
        # Insert pilots and data
        carriers = self.frame_builder.pilot_inserter.insert(sym_data, tps_bits, 0)
        
        # Get pilot positions
        pilot_positions = self._get_pilot_positions(0)
        
        return carriers, pilot_positions
    
    def _get_pilot_positions(self, symbol_index: int) -> np.ndarray:
        """Get positions of different pilot types."""
        mode_info = {
            '2K': {'carriers': 1705, 'continual': 45},
            '8K': {'carriers': 6817, 'continual': 177},
        }
        info = mode_info.get(self.params.mode, mode_info['2K'])
        
        # Scattered pilots: every 12th carrier, offset by (symbol_index % 4) * 3
        scattered = np.arange(
            (symbol_index % 4) * 3, 
            info['carriers'], 
            12
        )
        
        return scattered


def get_sample_rates() -> Dict[str, float]:
    """Get sample rates for each bandwidth."""
    return {
        '8MHz': 9142857.142857143,
        '7MHz': 8000000.0,
        '6MHz': 6857142.857142857,
    }


def calculate_data_rate(params: DVBTParams) -> float:
    """Calculate net data rate for given parameters."""
    from dvb.FrameBuilder import FrameInfo
    return FrameInfo.get_data_rate(
        params.mode, params.guard_interval,
        params.constellation, params.code_rate,
        params.bandwidth
    )

