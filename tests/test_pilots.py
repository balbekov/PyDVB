"""
Tests for DVB-T pilot carriers.
"""

import pytest
import numpy as np
from dvb.Pilots import (
    PilotGenerator, ContinualPilots, ScatteredPilots, 
    TPSPilots, PilotInserter, PILOT_BOOST
)


class TestPilotGenerator:
    """Test pilot value generation."""
    
    def test_prbs_deterministic(self):
        """Test that PRBS is deterministic."""
        gen1 = PilotGenerator('2K')
        gen2 = PilotGenerator('2K')
        
        for k in range(100):
            assert gen1.get_pilot_value(k) == gen2.get_pilot_value(k)
    
    def test_pilot_boost(self):
        """Test pilot amplitude boost."""
        gen = PilotGenerator('2K')
        
        value = gen.get_pilot_value(0, boost=True)
        assert abs(abs(value) - PILOT_BOOST) < 0.01
    
    def test_bpsk_values(self):
        """Test that pilots are BPSK (real only)."""
        gen = PilotGenerator('2K')
        
        for k in range(100):
            value = gen.get_pilot_value(k)
            assert value.imag == 0
            assert abs(abs(value.real) - PILOT_BOOST) < 0.01


class TestContinualPilots:
    """Test continual pilot positions."""
    
    def test_2k_count(self):
        """Test 2K mode has 45 continual pilots."""
        cp = ContinualPilots('2K')
        assert len(cp.positions) == 45
    
    def test_8k_count(self):
        """Test 8K mode has 177 continual pilots."""
        cp = ContinualPilots('8K')
        assert len(cp.positions) == 177
    
    def test_positions_in_range(self):
        """Test all positions are valid carrier indices."""
        for mode in ['2K', '8K']:
            cp = ContinualPilots(mode)
            max_carrier = {'2K': 1705, '8K': 6817}[mode]
            
            for pos in cp.positions:
                assert 0 <= pos < max_carrier
    
    def test_is_continual_pilot(self):
        """Test position checking."""
        cp = ContinualPilots('2K')
        
        assert cp.is_continual_pilot(0) is True
        assert cp.is_continual_pilot(48) is True
        assert cp.is_continual_pilot(1) is False


class TestScatteredPilots:
    """Test scattered pilot positions."""
    
    def test_pattern_rotation(self):
        """Test that scattered pilots rotate every symbol."""
        sp = ScatteredPilots('2K')
        
        # Get positions for first 4 symbols
        pos0 = sp.get_positions(0)
        pos1 = sp.get_positions(1)
        pos2 = sp.get_positions(2)
        pos3 = sp.get_positions(3)
        
        # Pattern should shift by 3 each symbol
        assert pos0[0] == 0
        assert pos1[0] == 3
        assert pos2[0] == 6
        assert pos3[0] == 9
    
    def test_pattern_repeats(self):
        """Test pattern repeats every 4 symbols."""
        sp = ScatteredPilots('2K')
        
        pos0 = sp.get_positions(0)
        pos4 = sp.get_positions(4)
        
        np.testing.assert_array_equal(pos0, pos4)
    
    def test_spacing(self):
        """Test scattered pilots are every 12 carriers."""
        sp = ScatteredPilots('2K')
        
        positions = sp.get_positions(0)
        
        # Check spacing
        for i in range(len(positions) - 1):
            assert positions[i + 1] - positions[i] == 12


class TestTPSPilots:
    """Test TPS pilot positions."""
    
    def test_2k_count(self):
        """Test 2K mode has 17 TPS pilots."""
        tps = TPSPilots('2K')
        assert len(tps.positions) == 17
    
    def test_8k_count(self):
        """Test 8K mode has 68 TPS pilots."""
        tps = TPSPilots('8K')
        assert len(tps.positions) == 68


class TestPilotInserter:
    """Test complete pilot insertion."""
    
    def test_data_carrier_count(self):
        """Test data carrier counting."""
        inserter = PilotInserter('2K')
        
        # Each symbol has different data carrier count due to scattered pilots
        total = sum(inserter.data_carriers_per_symbol)
        
        # Should be approximately 1512 * 4 (minus overlapping pilots)
        assert 5900 < total < 6200
    
    def test_insert_preserves_length(self):
        """Test that insertion produces correct number of carriers."""
        inserter = PilotInserter('2K')
        
        num_data = inserter.get_data_carrier_count(0)
        data = np.random.randn(num_data).astype(np.complex64)
        tps = np.zeros(17, dtype=np.uint8)
        
        carriers = inserter.insert(data, tps, 0)
        
        assert len(carriers) == 1705
    
    def test_pilot_positions_correct(self):
        """Test pilots are at expected positions."""
        inserter = PilotInserter('2K')
        
        types = inserter.get_carrier_types(0)
        
        # Check continual pilots
        cp = ContinualPilots('2K')
        for pos in cp.positions:
            assert types[pos] in [1, 2]  # Continual or overlapping scattered
    
    def test_carrier_types_sum(self):
        """Test that carrier types sum correctly."""
        inserter = PilotInserter('2K')
        
        for sym in range(4):
            types = inserter.get_carrier_types(sym)
            
            data_count = np.sum(types == 0)
            continual_count = np.sum(types == 1)
            scattered_count = np.sum(types == 2)
            tps_count = np.sum(types == 3)
            
            # Total should be 1705
            assert data_count + continual_count + scattered_count + tps_count == 1705
            
            # Data count should match our calculation
            assert data_count == inserter.get_data_carrier_count(sym)
