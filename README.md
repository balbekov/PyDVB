# PyDVB - Educational DVB-T Implementation

PyDVB is a complete, educational implementation of the DVB-T (Digital Video Broadcasting - Terrestrial) transmitter pipeline in Python. It transforms MPEG-2 Transport Stream packets into I/Q baseband samples suitable for SDR (Software Defined Radio) transmission.

## Features

- **Complete DVB-T Transmit Pipeline**: From TS packets to I/Q samples
- **Educational Design**: Both fast (numpy/scipy) and slow (step-by-step) implementations
- **Modular Architecture**: Each processing stage is a separate, well-documented module
- **Full FEC Chain**: Reed-Solomon, convolutional coding, and interleaving
- **OFDM Modulation**: 2K and 8K modes with proper pilot insertion
- **Multiple Output Formats**: cf32, cs8, cs16, cu8 for various SDR hardware

## Architecture

```
          COMPLETE DVB-T TRANSMIT PIPELINE
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  TS Packets    Scramble     Outer Code    Outer         Inner Code          │
│  (188 bytes)   (energy      RS(204,188)   Interleave    Convolutional       │
│      │         dispersal)       │         (12x17)       K=7, rate 1/2       │
│      ▼            ▼             ▼            ▼              ▼               │
│   [data] ───► [PRBS XOR] ───► [+16 RS] ───► [Forney] ───► [encode] ──┐     │
│                                                                       │     │
│  ┌────────────────────────────────────────────────────────────────────┘     │
│  │                                                                          │
│  │  Puncture      Inner         Symbol Map     OFDM Frame      Guard       │
│  │  (1/2-7/8)     Interleave    (QPSK/16/64)   Assembly        Interval    │
│  │      │            │              │              │              │        │
│  │      ▼            ▼              ▼              ▼              ▼        │
│  └──► [rate] ───► [bit+sym] ───► [QAM] ───► [pilots+   ───► [+cyclic ───► I/Q
│       match]      interleave      map        IFFT]          prefix]       out
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd PyDVB

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

## Quick Start

### Command Line Interface

```bash
# Show transport stream information
python -m dvb info sample.ts

# Generate a test transport stream
python -m dvb generate -o test.ts -n 1000

# Modulate to I/Q samples
python -m dvb modulate test.ts -o output.cf32 \
    --mode 2K \
    --constellation 64QAM \
    --code-rate 2/3 \
    --guard 1/4 \
    --bandwidth 8MHz

# Demultiplex transport stream
python -m dvb demux sample.ts -o output_dir/
```

### Python API

```python
from dvb import DVBTModulator, TransportStream

# Create modulator
mod = DVBTModulator(
    mode='2K',           # 2K or 8K OFDM
    constellation='64QAM',  # QPSK, 16QAM, 64QAM
    code_rate='2/3',     # 1/2, 2/3, 3/4, 5/6, 7/8
    guard_interval='1/4', # 1/4, 1/8, 1/16, 1/32
    bandwidth='8MHz',    # 6MHz, 7MHz, 8MHz
)

# Read transport stream
with open('input.ts', 'rb') as f:
    ts_data = f.read()

# Modulate to I/Q
iq_samples = mod.modulate(ts_data)

# Write output
mod.write_iq('output.cf32', iq_samples)

# Get transmission parameters
print(f"Sample rate: {mod.get_sample_rate()} Hz")
print(f"Data rate: {mod.get_data_rate()} bps")
```

### Individual Modules

```python
# Reed-Solomon encoding
from dvb import ReedSolomon
rs = ReedSolomon()
encoded = rs.encode(packet_188_bytes)  # Returns 204 bytes

# QAM mapping
from dvb import QAMMapper
mapper = QAMMapper('64QAM')
symbols = mapper.map(bits)

# OFDM modulation
from dvb import OFDMModulator
ofdm = OFDMModulator('2K')
time_samples = ofdm.modulate(carrier_values)
```

## Module Structure

```
dvb/
├── __init__.py           # Package exports and constants
├── __main__.py           # CLI interface
├── DVB.py                # Main orchestrator class
│
│ # === Transport Layer ===
├── TransportStream.py    # TS container (collection of packets)
├── Packet.py             # Single 188-byte TS packet
├── AdaptationField.py    # Timing, PCR, stuffing
├── PES.py                # Packetized Elementary Stream
├── PSI.py                # Program Specific Information base
├── PAT.py                # Program Association Table
├── PMT.py                # Program Map Table
├── CRC.py                # CRC-32/MPEG-2 calculation
│
│ # === Channel Coding (FEC) ===
├── Scrambler.py          # Energy dispersal PRBS (x^15+x^14+1)
├── ReedSolomon.py        # Outer code: RS(204,188) encoder/decoder
├── GaloisField.py        # GF(2^8) arithmetic for RS
├── OuterInterleaver.py   # Forney convolutional interleaver (I=12)
├── Convolutional.py      # Inner code: K=7, G1=171o, G2=133o
├── Puncturing.py         # Rate matching: 1/2, 2/3, 3/4, 5/6, 7/8
├── InnerInterleaver.py   # Bit-wise + symbol interleaving
│
│ # === OFDM Modulation ===
├── QAM.py                # QPSK, 16-QAM, 64-QAM mapping
├── Pilots.py             # Scattered, continual, TPS pilots
├── OFDM.py               # IFFT modulation, FFT demodulation
├── GuardInterval.py      # Cyclic prefix: 1/4, 1/8, 1/16, 1/32
├── TPS.py                # Transmission Parameter Signaling
├── FrameBuilder.py       # Assemble OFDM symbols into frames
└── IQWriter.py           # Output to .cf32, .cs8 for SDR
```

## DVB-T Parameters

| Parameter | Options |
|-----------|---------|
| Mode | 2K (2048 FFT), 8K (8192 FFT) |
| Constellation | QPSK (2 bits), 16-QAM (4 bits), 64-QAM (6 bits) |
| Code Rate | 1/2, 2/3, 3/4, 5/6, 7/8 |
| Guard Interval | 1/4, 1/8, 1/16, 1/32 |
| Bandwidth | 6 MHz, 7 MHz, 8 MHz |

## SDR Transmission

The output I/Q files can be transmitted using various SDR hardware:

```bash
# HackRF (UHF Channel 21 = 474 MHz)
hackrf_transfer -t output.cs8 -f 474e6 -s 9142857 -x 40

# Using GNU Radio
# Load output.cf32 as complex float32 file source
```

## Educational Notes

Each module includes both optimized (numpy/scipy) and educational implementations:

- **Fast path**: Uses vectorized numpy operations for performance
- **Slow path**: Implements algorithms step-by-step for learning

Example:
```python
# Fast Reed-Solomon encoding
rs = ReedSolomon(use_fast=True)

# Educational bit-by-bit encoding
rs = ReedSolomon(use_fast=False)
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_qam.py -v

# Run with coverage
python -m pytest tests/ --cov=dvb
```

## References

- ETSI EN 300 744: DVB-T specification
- ISO/IEC 13818-1: MPEG-2 Systems
- DVB-T Implementation Guidelines (ETSI TR 101 290)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure tests pass before submitting PRs.
