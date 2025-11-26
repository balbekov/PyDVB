"""
PyDVB - Educational DVB-T Implementation

A complete DVB-T (Digital Video Broadcasting - Terrestrial) transmitter pipeline
for educational purposes. Implements the full signal processing chain from 
MPEG Transport Stream to I/Q baseband samples suitable for SDR transmission.

Modules:
    Transport Layer:
        - TransportStream: Container for TS packets
        - Packet: Single 188-byte TS packet
        - AdaptationField: Timing and PCR information
        - PES: Packetized Elementary Stream
        - PSI/PAT/PMT: Program tables
        - CRC: CRC-32 calculation

    Channel Coding (FEC):
        - Scrambler: Energy dispersal PRBS
        - ReedSolomon: RS(204,188) outer code
        - GaloisField: GF(2^8) arithmetic
        - OuterInterleaver: Forney convolutional interleaver
        - Convolutional: K=7 inner code
        - Puncturing: Rate matching
        - InnerInterleaver: Bit and symbol interleaving

    OFDM Modulation:
        - QAM: QPSK/16-QAM/64-QAM mapping
        - Pilots: Scattered, continual, and TPS pilots
        - OFDM: IFFT modulation
        - GuardInterval: Cyclic prefix insertion
        - TPS: Transmission Parameter Signaling
        - FrameBuilder: OFDM frame assembly
        - IQWriter: SDR output formats
"""

__version__ = "0.1.0"
__author__ = "PyDVB Contributors"

# DVB-T mode parameters
MODES = {
    '2K': {
        'fft_size': 2048,
        'active_carriers': 1705,
        'symbols_per_frame': 68,
        'frames_per_superframe': 4,
    },
    '8K': {
        'fft_size': 8192,
        'active_carriers': 6817,
        'symbols_per_frame': 68,
        'frames_per_superframe': 4,
    },
}

# Guard interval options (fraction of useful symbol duration)
GUARD_INTERVALS = {
    '1/4': 4,
    '1/8': 8,
    '1/16': 16,
    '1/32': 32,
}

# Constellation types and bits per symbol
CONSTELLATIONS = {
    'QPSK': 2,
    '16QAM': 4,
    '64QAM': 6,
}

# Code rates (after puncturing)
CODE_RATES = {
    '1/2': (1, 2),
    '2/3': (2, 3),
    '3/4': (3, 4),
    '5/6': (5, 6),
    '7/8': (7, 8),
}

# Channel bandwidths and corresponding sample rates
BANDWIDTHS = {
    '8MHz': {'sample_rate': 9142857.142857143},  # 64/7 MHz
    '7MHz': {'sample_rate': 8000000.0},
    '6MHz': {'sample_rate': 6857142.857142857},  # 48/7 MHz
}

# Import main classes for convenience
from .CRC import CRC32
from .Packet import Packet
from .AdaptationField import AdaptationField
from .TransportStream import TransportStream
from .PES import PES
from .PSI import PSI
from .PAT import PAT
from .PMT import PMT

from .Scrambler import Scrambler
from .GaloisField import GaloisField
from .ReedSolomon import ReedSolomon
from .OuterInterleaver import OuterInterleaver
from .Convolutional import ConvolutionalEncoder, ConvolutionalDecoder
from .Puncturing import Puncturer, Depuncturer
from .InnerInterleaver import BitInterleaver, SymbolInterleaver

from .QAM import QAMMapper, QAMDemapper
from .Pilots import PilotInserter, PilotExtractor
from .TPS import TPSEncoder, TPSDecoder
from .OFDM import OFDMModulator, OFDMDemodulator
from .GuardInterval import GuardIntervalInserter, GuardIntervalRemover
from .FrameBuilder import FrameBuilder
from .IQWriter import IQWriter

from .DVB import DVBTModulator, DVBTDemodulator

__all__ = [
    # Constants
    'MODES', 'GUARD_INTERVALS', 'CONSTELLATIONS', 'CODE_RATES', 'BANDWIDTHS',
    
    # Transport Layer
    'CRC32', 'Packet', 'AdaptationField', 'TransportStream', 'PES', 'PSI', 'PAT', 'PMT',
    
    # Channel Coding
    'Scrambler', 'GaloisField', 'ReedSolomon', 'OuterInterleaver',
    'ConvolutionalEncoder', 'ConvolutionalDecoder', 'Puncturer', 'Depuncturer',
    'BitInterleaver', 'SymbolInterleaver',
    
    # OFDM Modulation
    'QAMMapper', 'QAMDemapper', 'PilotInserter', 'PilotExtractor',
    'TPSEncoder', 'TPSDecoder', 'OFDMModulator', 'OFDMDemodulator',
    'GuardIntervalInserter', 'GuardIntervalRemover', 'FrameBuilder', 'IQWriter',
    
    # Main
    'DVBTModulator', 'DVBTDemodulator',
]
