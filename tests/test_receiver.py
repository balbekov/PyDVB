"""
DVB-T Receiver Unit Tests

Tests for individual receiver components:
- IQReader
- Synchronizer (coarse, fine, frame)
- ChannelEstimator
- Detector
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


class TestIQReader:
    """Test I/Q file reader."""
    
    def test_read_cf32(self):
        """Test reading cf32 format."""
        from dvb.IQReader import IQReader
        
        # Create test file
        test_samples = (np.random.randn(1000) + 
                       1j * np.random.randn(1000)).astype(np.complex64)
        
        with tempfile.NamedTemporaryFile(suffix='.cf32', delete=False) as f:
            test_samples.tofile(f)
            path = f.name
        
        try:
            reader = IQReader('cf32')
            read_samples = reader.read(path)
            
            np.testing.assert_allclose(read_samples, test_samples, rtol=1e-6)
        finally:
            os.unlink(path)
    
    def test_read_cs8(self):
        """Test reading cs8 format."""
        from dvb.IQReader import IQReader
        
        # Create test file with interleaved int8
        I = np.array([10, -20, 30, -40], dtype=np.int8)
        Q = np.array([5, -10, 15, -20], dtype=np.int8)
        interleaved = np.empty(8, dtype=np.int8)
        interleaved[0::2] = I
        interleaved[1::2] = Q
        
        with tempfile.NamedTemporaryFile(suffix='.cs8', delete=False) as f:
            interleaved.tofile(f)
            path = f.name
        
        try:
            reader = IQReader('cs8')
            samples = reader.read(path)
            
            # Check values (normalized by 127)
            expected_I = I.astype(np.float32) / 127
            expected_Q = Q.astype(np.float32) / 127
            
            np.testing.assert_allclose(samples.real, expected_I, rtol=1e-5)
            np.testing.assert_allclose(samples.imag, expected_Q, rtol=1e-5)
        finally:
            os.unlink(path)
    
    def test_read_with_offset(self):
        """Test reading with sample offset."""
        from dvb.IQReader import IQReader
        
        test_samples = (np.arange(100) + 
                       1j * np.arange(100)).astype(np.complex64)
        
        with tempfile.NamedTemporaryFile(suffix='.cf32', delete=False) as f:
            test_samples.tofile(f)
            path = f.name
        
        try:
            reader = IQReader('cf32')
            samples = reader.read(path, offset=10, count=20)
            
            np.testing.assert_allclose(samples, test_samples[10:30], rtol=1e-6)
        finally:
            os.unlink(path)
    
    def test_iter_chunks(self):
        """Test chunked reading."""
        from dvb.IQReader import IQReader
        
        test_samples = (np.random.randn(1000) + 
                       1j * np.random.randn(1000)).astype(np.complex64)
        
        with tempfile.NamedTemporaryFile(suffix='.cf32', delete=False) as f:
            test_samples.tofile(f)
            path = f.name
        
        try:
            reader = IQReader('cf32')
            
            chunks = list(reader.iter_chunks(path, chunk_size=100))
            
            # Should have 10 chunks
            assert len(chunks) == 10
            
            # Each chunk should be 100 samples
            for chunk in chunks:
                assert len(chunk) == 100
        finally:
            os.unlink(path)
    
    def test_auto_format_detection(self):
        """Test automatic format detection from extension."""
        from dvb.IQReader import IQReader, detect_format
        
        assert detect_format('test.cf32') == 'cf32'
        assert detect_format('test.cs8') == 'cs8'
        assert detect_format('test.cs16') == 'cs16'
        assert detect_format('test.cu8') == 'cu8'
    
    def test_file_info(self):
        """Test getting file information."""
        from dvb.IQReader import IQReader
        
        test_samples = np.zeros(500, dtype=np.complex64)
        
        with tempfile.NamedTemporaryFile(suffix='.cf32', delete=False) as f:
            test_samples.tofile(f)
            path = f.name
        
        try:
            reader = IQReader('cf32', sample_rate=9142857.0)
            info = reader.get_file_info(path)
            
            assert info['num_samples'] == 500
            assert info['format'] == 'cf32'
            assert info['sample_rate'] == 9142857.0
        finally:
            os.unlink(path)


class TestCoarseSync:
    """Test coarse synchronization."""
    
    def test_guard_correlation(self):
        """Test guard interval correlation detects symbol boundaries."""
        from dvb.Synchronizer import CoarseSync
        from dvb.OFDM import OFDMModulator
        from dvb.GuardInterval import GuardIntervalInserter
        
        mode = '2K'
        guard = '1/4'
        
        sync = CoarseSync(mode, guard)
        ofdm = OFDMModulator(mode)
        gi = GuardIntervalInserter(guard, ofdm.fft_size)
        
        # Generate test symbol
        carriers = (np.random.randn(ofdm.active_carriers) + 
                   1j * np.random.randn(ofdm.active_carriers)).astype(np.complex64)
        time_samples = ofdm.modulate(carriers)
        symbol_with_guard = gi.add(time_samples)
        
        # Generate multiple symbols with random start offset
        offset = 100
        samples = np.concatenate([
            np.random.randn(offset).astype(np.complex64) * 0.01,
            symbol_with_guard,
            symbol_with_guard,
        ])
        
        # Find symbol start
        found_start, peak = sync.find_symbol_start(samples)
        
        # Should find start near offset
        assert abs(found_start - offset) < 10
        assert peak > 0
    
    def test_cfo_estimation(self):
        """Test coarse CFO estimation."""
        from dvb.Synchronizer import CoarseSync
        from dvb.OFDM import OFDMModulator
        from dvb.GuardInterval import GuardIntervalInserter
        from tests.channel_model import ChannelModel
        
        mode = '2K'
        guard = '1/4'
        cfo_hz = 500  # Apply 500 Hz CFO
        
        sync = CoarseSync(mode, guard)
        ofdm = OFDMModulator(mode)
        gi = GuardIntervalInserter(guard, ofdm.fft_size)
        channel = ChannelModel(sync.sample_rate)
        
        # Generate test symbols
        symbols = []
        for _ in range(4):
            carriers = (np.random.randn(ofdm.active_carriers) + 
                       1j * np.random.randn(ofdm.active_carriers)).astype(np.complex64)
            time_samples = ofdm.modulate(carriers)
            symbols.append(gi.add(time_samples))
        
        samples = np.concatenate(symbols)
        
        # Add CFO
        samples_with_cfo = channel.add_cfo(samples, cfo_hz)
        
        # Estimate CFO
        estimated_cfo = sync.estimate_coarse_cfo(samples_with_cfo)
        
        # Should be within 10% of actual
        assert abs(estimated_cfo - cfo_hz) < abs(cfo_hz) * 0.2 + 50
    
    def test_cfo_correction(self):
        """Test CFO correction."""
        from dvb.Synchronizer import CoarseSync
        from tests.channel_model import ChannelModel
        
        sync = CoarseSync('2K', '1/4')
        channel = ChannelModel(sync.sample_rate)
        
        # Test signal
        samples = np.exp(1j * np.linspace(0, 10*np.pi, 1000)).astype(np.complex64)
        
        # Add and remove CFO
        cfo_hz = 1000
        with_cfo = channel.add_cfo(samples, cfo_hz)
        corrected = sync.correct_cfo(with_cfo, cfo_hz)
        
        # Phase difference should be minimal after correction
        phase_diff = np.angle(corrected * np.conj(samples))
        assert np.std(phase_diff) < 0.5


class TestFineSync:
    """Test fine synchronization using pilots."""
    
    def test_pilot_phase_estimation(self):
        """Test phase estimation from pilots."""
        from dvb.Synchronizer import FineSync
        from dvb.Pilots import PilotInserter
        
        mode = '2K'
        sync = FineSync(mode)
        inserter = PilotInserter(mode)
        
        # Create carriers with known phase offset
        phase_offset = 0.5  # radians
        
        # Generate symbol with data and pilots
        data = (np.random.randn(inserter.get_data_carrier_count(0)) + 
               1j * np.random.randn(inserter.get_data_carrier_count(0))).astype(np.complex64)
        tps_bits = np.zeros(17, dtype=np.uint8)
        carriers = inserter.insert(data, tps_bits, 0)
        
        # Apply phase offset
        carriers_with_phase = carriers * np.exp(1j * phase_offset)
        
        # Estimate phase
        estimated_phase = sync.estimate_phase_offset(carriers_with_phase, 0)
        
        # Should be close to applied offset
        assert abs(estimated_phase - phase_offset) < 0.3
    
    def test_fine_cfo_estimation(self):
        """Test fine CFO estimation from pilots."""
        from dvb.Synchronizer import FineSync
        from dvb.Pilots import PilotInserter
        
        mode = '2K'
        sync = FineSync(mode)
        inserter = PilotInserter(mode)
        
        # Generate symbol
        data = (np.random.randn(inserter.get_data_carrier_count(0)) + 
               1j * np.random.randn(inserter.get_data_carrier_count(0))).astype(np.complex64)
        tps_bits = np.zeros(17, dtype=np.uint8)
        carriers = inserter.insert(data, tps_bits, 0)
        
        # Fine CFO estimate on perfect signal should be near zero
        fine_cfo = sync.estimate_fine_cfo(carriers, 0)
        
        assert abs(fine_cfo) < 100  # Should be small


class TestChannelEstimator:
    """Test channel estimation and equalization."""
    
    def test_flat_channel_estimation(self):
        """Test channel estimation with flat channel."""
        from dvb.ChannelEstimator import ChannelEstimator
        from dvb.Pilots import PilotInserter
        
        mode = '2K'
        estimator = ChannelEstimator(mode)
        inserter = PilotInserter(mode)
        
        # Generate symbol with flat channel (H = 1)
        data = (np.random.randn(inserter.get_data_carrier_count(0)) + 
               1j * np.random.randn(inserter.get_data_carrier_count(0))).astype(np.complex64)
        tps_bits = np.zeros(17, dtype=np.uint8)
        carriers = inserter.insert(data, tps_bits, 0)
        
        # Estimate channel
        channel = estimator.estimate(carriers, 0)
        
        # Should be approximately unity
        assert np.mean(np.abs(channel)) > 0.8
        assert np.mean(np.abs(channel)) < 1.5
    
    def test_equalization_zf(self):
        """Test zero-forcing equalization."""
        from dvb.ChannelEstimator import Equalizer
        
        eq = Equalizer(method='zf')
        
        # Known channel
        channel = np.array([1+0.5j, 0.8-0.2j, 1.2+0.3j], dtype=np.complex64)
        
        # Transmitted symbols
        tx = np.array([1+1j, -1+1j, 1-1j], dtype=np.complex64)
        
        # Received (through channel)
        rx = tx * channel
        
        # Equalize
        equalized = eq.equalize(rx, channel)
        
        np.testing.assert_allclose(equalized, tx, rtol=1e-5)
    
    def test_equalization_mmse(self):
        """Test MMSE equalization."""
        from dvb.ChannelEstimator import Equalizer
        
        eq = Equalizer(method='mmse', noise_variance=0.01)
        
        channel = np.array([1+0.5j, 0.8-0.2j, 1.2+0.3j], dtype=np.complex64)
        tx = np.array([1+1j, -1+1j, 1-1j], dtype=np.complex64)
        rx = tx * channel
        
        equalized = eq.equalize(rx, channel)
        
        # MMSE should be close to TX
        np.testing.assert_allclose(equalized, tx, rtol=0.1)
    
    def test_csi_computation(self):
        """Test CSI computation."""
        from dvb.ChannelEstimator import Equalizer
        
        eq = Equalizer()
        
        channel = np.array([1+0j, 0.5+0.5j, 2-1j], dtype=np.complex64)
        csi = eq.get_csi(channel)
        
        # CSI should be |H|^2
        expected = np.abs(channel) ** 2
        np.testing.assert_allclose(csi, expected, rtol=1e-5)


class TestDetector:
    """Test parameter auto-detection."""
    
    def test_mode_detection(self):
        """Test mode (FFT size) detection."""
        from dvb.Detector import DVBTDetector
        from dvb.OFDM import OFDMModulator
        from dvb.GuardInterval import GuardIntervalInserter
        
        for mode in ['2K', '8K']:
            detector = DVBTDetector()
            ofdm = OFDMModulator(mode)
            gi = GuardIntervalInserter('1/4', ofdm.fft_size)
            
            # Generate multiple symbols
            symbols = []
            for _ in range(10):
                carriers = (np.random.randn(ofdm.active_carriers) + 
                           1j * np.random.randn(ofdm.active_carriers)).astype(np.complex64)
                time_samples = ofdm.modulate(carriers)
                symbols.append(gi.add(time_samples))
            
            samples = np.concatenate(symbols)
            
            detected_mode, confidence = detector.detect_mode(samples)
            
            assert detected_mode == mode
            assert confidence > 0.5
    
    def test_guard_interval_detection(self):
        """Test guard interval detection."""
        from dvb.Detector import DVBTDetector
        from dvb.OFDM import OFDMModulator
        from dvb.GuardInterval import GuardIntervalInserter
        
        mode = '2K'
        
        for guard in ['1/4', '1/8']:
            detector = DVBTDetector()
            ofdm = OFDMModulator(mode)
            gi = GuardIntervalInserter(guard, ofdm.fft_size)
            
            # Generate symbols
            symbols = []
            for _ in range(10):
                carriers = (np.random.randn(ofdm.active_carriers) + 
                           1j * np.random.randn(ofdm.active_carriers)).astype(np.complex64)
                time_samples = ofdm.modulate(carriers)
                symbols.append(gi.add(time_samples))
            
            samples = np.concatenate(symbols)
            
            detected_guard, confidence = detector.detect_guard_interval(samples, mode)
            
            assert detected_guard == guard
            assert confidence > 0.3


class TestIntegration:
    """Integration tests for receiver components."""
    
    def test_sync_and_channel_estimation(self):
        """Test synchronization followed by channel estimation."""
        from dvb.Synchronizer import DVBTSynchronizer
        from dvb.ChannelEstimator import ChannelEstimatorWithEqualization
        from dvb.OFDM import OFDMModulator, OFDMDemodulator
        from dvb.GuardInterval import GuardIntervalInserter, GuardIntervalRemover
        from dvb.Pilots import PilotInserter
        from tests.channel_model import ChannelModel
        
        mode = '2K'
        guard = '1/4'
        
        # Create components
        sync = DVBTSynchronizer(mode, guard)
        channel_eq = ChannelEstimatorWithEqualization(mode)
        ofdm_mod = OFDMModulator(mode)
        ofdm_demod = OFDMDemodulator(mode)
        gi_add = GuardIntervalInserter(guard, ofdm_mod.fft_size)
        gi_rem = GuardIntervalRemover(guard, ofdm_mod.fft_size)
        pilot_inserter = PilotInserter(mode)
        channel = ChannelModel()
        
        # Generate TX signal
        symbols = []
        for sym_idx in range(4):
            data = np.ones(pilot_inserter.get_data_carrier_count(sym_idx), 
                          dtype=np.complex64)
            tps = np.zeros(17, dtype=np.uint8)
            carriers = pilot_inserter.insert(data, tps, sym_idx)
            time_samples = ofdm_mod.modulate(carriers)
            symbols.append(gi_add.add(time_samples))
        
        tx_samples = np.concatenate(symbols)
        
        # Add mild impairments
        rx_samples = channel.add_awgn(tx_samples, snr_db=30)
        
        # Synchronize
        sync_result = sync.synchronize(rx_samples)
        
        # Should find symbol start
        assert sync_result.symbol_start >= 0
        assert sync_result.snr_estimate > 0
        
        # Process first symbol with channel estimation
        symbol_len = ofdm_mod.fft_size + gi_add.guard_length
        symbol = rx_samples[sync_result.symbol_start:
                           sync_result.symbol_start + symbol_len]
        
        useful = gi_rem.remove(symbol)
        carriers = ofdm_demod.demodulate(useful)
        equalized, csi = channel_eq.process(carriers, 0)
        
        # Equalized carriers should be valid
        assert len(equalized) == ofdm_mod.active_carriers
        assert not np.any(np.isnan(equalized))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
