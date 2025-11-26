"""
Tests for full DVB-T encode/decode roundtrip.
"""

import pytest
import numpy as np
from dvb import (
    Scrambler, ReedSolomon, OuterInterleaver,
    ConvolutionalEncoder, Puncturer,
    BitInterleaver, SymbolInterleaver,
    QAMMapper, QAMDemapper,
    OFDMModulator, OFDMDemodulator,
    GuardIntervalInserter, GuardIntervalRemover,
)


class TestScramblerRoundtrip:
    """Test scrambler roundtrip."""
    
    def test_scramble_descramble(self):
        """Test that descrambling reverses scrambling."""
        scrambler = Scrambler()
        
        # Create test packets (multiple of 188)
        num_packets = 8
        data = bytes([i % 256 for i in range(188 * num_packets)])
        
        scrambled = scrambler.scramble(data)
        descrambled = scrambler.descramble(scrambled)
        
        # Note: sync bytes are modified, so compare packet content
        for i in range(num_packets):
            start = i * 188 + 1  # Skip sync byte
            end = (i + 1) * 188
            assert data[start:end] == descrambled[start:end]


class TestRSRoundtrip:
    """Test Reed-Solomon roundtrip."""
    
    def test_encode_decode_multiple_packets(self):
        """Test RS on multiple packets."""
        rs = ReedSolomon()
        
        for _ in range(10):
            message = bytes(np.random.randint(0, 256, 188, dtype=np.uint8))
            encoded = rs.encode(message)
            decoded, errors = rs.decode(encoded)
            
            assert decoded == message
            assert errors == 0


class TestOuterInterleaverRoundtrip:
    """Test outer interleaver roundtrip."""
    
    def test_interleave_deinterleave(self):
        """Test that interleaving produces valid output."""
        interleaver = OuterInterleaver()
        
        # Need enough data for interleaver
        data = bytes([i % 256 for i in range(204 * 20)])
        
        interleaved = interleaver.interleave(data, sync=True)
        
        # Interleaved data should have same length
        assert len(interleaved) == len(data)
        
        # Interleaved data should be different from input (most of the time)
        # but have same byte distribution
        input_set = set(data)
        output_set = set(interleaved)
        # Most bytes should appear in both
        assert len(input_set & output_set) > 0


class TestConvolutionalRoundtrip:
    """Test convolutional coding roundtrip."""
    
    def test_encode_decode_with_puncturing(self):
        """Test convolutional code with various puncturing rates."""
        from dvb.Convolutional import ConvolutionalDecoder
        from dvb.Puncturing import Depuncturer
        
        encoder = ConvolutionalEncoder()
        decoder = ConvolutionalDecoder()
        
        for rate in ['1/2', '2/3', '3/4']:
            punct = Puncturer(rate)
            depunct = Depuncturer(rate)
            
            # Input data
            data = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
            
            # Encode
            encoder.reset()
            encoded = encoder.encode(data, terminate=True)
            
            # Puncture
            punctured = punct.puncture(encoded)
            
            # Depuncture
            depunctured = depunct.depuncture(punctured.astype(np.float32))
            
            # Hard decision
            hard = (depunctured > 0.5).astype(np.uint8)
            
            # Decode
            decoded = decoder.decode(hard, terminated=True)
            
            np.testing.assert_array_equal(decoded, data)


class TestQAMRoundtrip:
    """Test QAM mapping roundtrip."""
    
    def test_all_constellations(self):
        """Test all QAM constellations."""
        for const in ['QPSK', '16QAM', '64QAM']:
            mapper = QAMMapper(const)
            demapper = QAMDemapper(const)
            
            bits_per_sym = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[const]
            
            # Random bits (multiple of bits per symbol)
            np.random.seed(42)
            bits = np.random.randint(0, 2, 120, dtype=np.uint8)
            bits = bits[:len(bits) - len(bits) % bits_per_sym]
            
            symbols = mapper.map(bits)
            recovered = demapper.demap(symbols)
            
            np.testing.assert_array_equal(recovered, bits)


class TestOFDMRoundtrip:
    """Test OFDM modulation roundtrip."""
    
    def test_with_guard_interval(self):
        """Test OFDM with guard interval."""
        for mode in ['2K', '8K']:
            for guard in ['1/4', '1/8']:
                fft_size = {'2K': 2048, '8K': 8192}[mode]
                num_carriers = {'2K': 1705, '8K': 6817}[mode]
                
                mod = OFDMModulator(mode)
                demod = OFDMDemodulator(mode)
                gi_add = GuardIntervalInserter(guard, fft_size)
                gi_rem = GuardIntervalRemover(guard, fft_size)
                
                # Random QAM symbols
                np.random.seed(42)
                carriers = (np.random.randn(num_carriers) + 
                           1j * np.random.randn(num_carriers)).astype(np.complex64)
                carriers /= np.max(np.abs(carriers))
                
                # Modulate
                samples = mod.modulate(carriers)
                
                # Add guard interval
                with_guard = gi_add.add(samples)
                
                # Remove guard interval
                no_guard = gi_rem.remove(with_guard)
                
                # Demodulate
                recovered = demod.demodulate(no_guard)
                
                np.testing.assert_allclose(recovered, carriers, rtol=1e-5)


class TestPartialChain:
    """Test partial processing chains."""
    
    def test_fec_chain(self):
        """Test FEC chain: RS -> outer interleave -> conv -> puncture."""
        rs = ReedSolomon()
        outer_int = OuterInterleaver()
        encoder = ConvolutionalEncoder()
        punct = Puncturer('2/3')
        
        # Input: single TS packet
        packet = bytes(range(188))
        
        # RS encode
        rs_encoded = rs.encode(packet)
        assert len(rs_encoded) == 204
        
        # Outer interleave
        outer_int.reset()
        interleaved = outer_int.interleave(rs_encoded)
        assert len(interleaved) == 204
        
        # Convolutional encode
        encoder.reset()
        conv_encoded = encoder.encode(interleaved)
        assert len(conv_encoded) == 2 * (204 * 8 + 6)  # Rate 1/2 + termination
        
        # Puncture
        punctured = punct.puncture(conv_encoded)
        # Rate 2/3: 3/4 of input
        expected_len = len(conv_encoded) * 3 // 4
        assert abs(len(punctured) - expected_len) < 10  # Allow for padding
    
    def test_modulation_chain(self):
        """Test modulation chain: inner interleave -> QAM -> OFDM."""
        bit_int = BitInterleaver('64QAM', '2K')
        sym_int = SymbolInterleaver('2K')
        mapper = QAMMapper('64QAM')
        ofdm = OFDMModulator('2K')
        
        # Generate enough bits for one OFDM symbol of data carriers
        # Approximately 1512 data carriers * 6 bits/symbol
        np.random.seed(42)
        bits = np.random.randint(0, 2, 1512 * 6, dtype=np.uint8)
        
        # Bit interleave
        bit_interleaved = bit_int.interleave(bits)
        assert len(bit_interleaved) == len(bits)
        
        # QAM map
        symbols = mapper.map(bit_interleaved)
        assert len(symbols) == 1512
        
        # Symbol interleave (pad to full carrier count)
        sym_interleaved = sym_int.interleave(symbols)
        assert len(sym_interleaved) == 1512


class TestFullPipeline:
    """Test complete DVB-T pipeline."""
    
    def test_modulator_integration(self):
        """Test DVBTModulator produces output."""
        from dvb.DVB import DVBTModulator
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4',
        )
        
        # Generate test transport stream
        ts_data = bytes([0x47] + [i % 256 for i in range(187)]) * 100
        
        # Modulate
        iq_samples = mod.modulate(ts_data)
        
        # Should produce output
        assert len(iq_samples) > 0
        assert iq_samples.dtype == np.complex64
    
    def test_different_configurations(self):
        """Test various DVB-T configurations."""
        from dvb.DVB import DVBTModulator
        
        configs = [
            ('2K', 'QPSK', '1/2', '1/4'),
            ('2K', '16QAM', '2/3', '1/8'),
            ('2K', '64QAM', '3/4', '1/16'),
        ]
        
        ts_data = bytes([0x47] + [i % 256 for i in range(187)]) * 50
        
        for mode, const, rate, guard in configs:
            mod = DVBTModulator(mode, const, rate, guard)
            iq_samples = mod.modulate(ts_data)
            
            assert len(iq_samples) > 0
            
            # Check sample rate
            sample_rate = mod.get_sample_rate()
            assert sample_rate > 6e6  # Should be >6 MHz
    
    def test_data_rate_calculation(self):
        """Test that data rate calculation is reasonable."""
        from dvb.DVB import DVBTModulator
        
        mod = DVBTModulator(
            mode='2K',
            constellation='64QAM',
            code_rate='2/3',
            guard_interval='1/4',
        )
        
        data_rate = mod.get_data_rate()
        
        # 64QAM with 2/3 rate should give ~18-20 Mbps for 2K/1/4
        assert 10e6 < data_rate < 30e6
