"""
Tests for OFDM modulation.
"""

import pytest
import numpy as np
from dvb.OFDM import OFDMModulator, OFDMDemodulator
from dvb.GuardInterval import GuardIntervalInserter, GuardIntervalRemover


class TestOFDMModulator:
    """Test OFDM modulator."""
    
    def test_2k_output_length(self):
        """Test 2K mode output length."""
        mod = OFDMModulator('2K')
        
        carriers = np.ones(1705, dtype=np.complex64)
        samples = mod.modulate(carriers)
        
        assert len(samples) == 2048
    
    def test_8k_output_length(self):
        """Test 8K mode output length."""
        mod = OFDMModulator('8K')
        
        carriers = np.ones(6817, dtype=np.complex64)
        samples = mod.modulate(carriers)
        
        assert len(samples) == 8192
    
    def test_wrong_carrier_count(self):
        """Test that wrong carrier count raises error."""
        mod = OFDMModulator('2K')
        
        with pytest.raises(ValueError):
            mod.modulate(np.ones(1000, dtype=np.complex64))
    
    def test_slow_vs_fast(self):
        """Compare slow and fast IFFT implementations."""
        mod_fast = OFDMModulator('2K', fast=True)
        mod_slow = OFDMModulator('2K', fast=False)
        
        carriers = np.random.randn(1705) + 1j * np.random.randn(1705)
        carriers = carriers.astype(np.complex64)
        
        samples_fast = mod_fast.modulate(carriers)
        samples_slow = mod_slow.modulate(carriers)
        
        # Should be approximately equal (floating point differences)
        np.testing.assert_allclose(samples_fast, samples_slow, rtol=1e-4)


class TestOFDMDemodulator:
    """Test OFDM demodulator."""
    
    def test_modulate_demodulate_roundtrip(self):
        """Test that demodulation recovers carriers."""
        mod = OFDMModulator('2K')
        demod = OFDMDemodulator('2K')
        
        # Generate random carriers
        np.random.seed(42)
        carriers = np.random.randn(1705) + 1j * np.random.randn(1705)
        carriers = carriers.astype(np.complex64)
        
        samples = mod.modulate(carriers)
        recovered = demod.demodulate(samples)
        
        np.testing.assert_allclose(recovered, carriers, rtol=1e-5)
    
    def test_8k_roundtrip(self):
        """Test 8K mode roundtrip."""
        mod = OFDMModulator('8K')
        demod = OFDMDemodulator('8K')
        
        np.random.seed(42)
        carriers = np.random.randn(6817) + 1j * np.random.randn(6817)
        carriers = carriers.astype(np.complex64)
        
        samples = mod.modulate(carriers)
        recovered = demod.demodulate(samples)
        
        np.testing.assert_allclose(recovered, carriers, rtol=1e-5)


class TestGuardInterval:
    """Test guard interval insertion and removal."""
    
    def test_guard_length_1_4(self):
        """Test 1/4 guard interval length."""
        gi = GuardIntervalInserter('1/4', 2048)
        
        assert gi.guard_length == 512
        assert gi.symbol_length == 2560
    
    def test_guard_length_1_8(self):
        """Test 1/8 guard interval length."""
        gi = GuardIntervalInserter('1/8', 2048)
        
        assert gi.guard_length == 256
        assert gi.symbol_length == 2304
    
    def test_guard_length_1_16(self):
        """Test 1/16 guard interval length."""
        gi = GuardIntervalInserter('1/16', 2048)
        
        assert gi.guard_length == 128
        assert gi.symbol_length == 2176
    
    def test_guard_length_1_32(self):
        """Test 1/32 guard interval length."""
        gi = GuardIntervalInserter('1/32', 2048)
        
        assert gi.guard_length == 64
        assert gi.symbol_length == 2112
    
    def test_cyclic_prefix(self):
        """Test that guard is cyclic prefix."""
        gi = GuardIntervalInserter('1/4', 2048)
        
        symbol = np.arange(2048, dtype=np.complex64)
        with_guard = gi.add(symbol)
        
        # First 512 samples should equal last 512 of original
        np.testing.assert_array_equal(with_guard[:512], symbol[-512:])
    
    def test_insert_remove_roundtrip(self):
        """Test guard interval insertion and removal."""
        for ratio in ['1/4', '1/8', '1/16', '1/32']:
            inserter = GuardIntervalInserter(ratio, 2048)
            remover = GuardIntervalRemover(ratio, 2048)
            
            symbol = np.random.randn(2048).astype(np.complex64)
            
            with_guard = inserter.add(symbol)
            recovered = remover.remove(with_guard)
            
            np.testing.assert_array_equal(recovered, symbol)


class TestFullOFDMChain:
    """Test complete OFDM chain."""
    
    def test_full_chain(self):
        """Test carriers through OFDM and guard interval."""
        for mode in ['2K', '8K']:
            for guard in ['1/4', '1/8']:
                fft_size = {'2K': 2048, '8K': 8192}[mode]
                num_carriers = {'2K': 1705, '8K': 6817}[mode]
                
                mod = OFDMModulator(mode)
                demod = OFDMDemodulator(mode)
                gi_add = GuardIntervalInserter(guard, fft_size)
                gi_rem = GuardIntervalRemover(guard, fft_size)
                
                # Random carriers
                np.random.seed(42)
                carriers = (np.random.randn(num_carriers) + 
                           1j * np.random.randn(num_carriers)).astype(np.complex64)
                
                # Full chain
                samples = mod.modulate(carriers)
                with_guard = gi_add.add(samples)
                no_guard = gi_rem.remove(with_guard)
                recovered = demod.demodulate(no_guard)
                
                np.testing.assert_allclose(recovered, carriers, rtol=1e-5)
