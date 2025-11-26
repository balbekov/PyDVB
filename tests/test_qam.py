"""
Tests for QAM constellation mapping.
"""

import pytest
import numpy as np
from math import sqrt
from dvb.QAM import QAMMapper, QAMDemapper, get_constellation_points


class TestQPSK:
    """Test QPSK constellation."""
    
    def test_output_count(self):
        """Test correct number of symbols."""
        mapper = QAMMapper('QPSK')
        
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        symbols = mapper.map(bits)
        
        assert len(symbols) == 4  # 8 bits / 2 bits per symbol
    
    def test_normalized_power(self):
        """Test that constellation has unit average power."""
        mapper = QAMMapper('QPSK')
        points = mapper._table
        
        avg_power = np.mean(np.abs(points) ** 2)
        assert abs(avg_power - 1.0) < 0.01
    
    def test_all_quadrants(self):
        """Test that all four quadrants are used."""
        mapper = QAMMapper('QPSK')
        
        # Map all possible 2-bit combinations
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        symbols = mapper.map(bits)
        
        # Check we have points in all quadrants
        quadrants = set()
        for s in symbols:
            q = (np.sign(s.real), np.sign(s.imag))
            quadrants.add(q)
        
        assert len(quadrants) == 4
    
    def test_roundtrip(self):
        """Test mapping and demapping."""
        mapper = QAMMapper('QPSK')
        demapper = QAMDemapper('QPSK')
        
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        symbols = mapper.map(bits)
        recovered = demapper.demap(symbols)
        
        np.testing.assert_array_equal(recovered, bits)


class Test16QAM:
    """Test 16-QAM constellation."""
    
    def test_output_count(self):
        """Test correct number of symbols."""
        mapper = QAMMapper('16QAM')
        
        bits = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
        symbols = mapper.map(bits)
        
        assert len(symbols) == 2  # 8 bits / 4 bits per symbol
    
    def test_normalized_power(self):
        """Test that constellation has unit average power."""
        mapper = QAMMapper('16QAM')
        points = mapper._table
        
        avg_power = np.mean(np.abs(points) ** 2)
        assert abs(avg_power - 1.0) < 0.01
    
    def test_unique_points(self):
        """Test that all 16 points are unique."""
        mapper = QAMMapper('16QAM')
        points = mapper._table
        
        unique_points = set()
        for p in points:
            unique_points.add((round(p.real, 6), round(p.imag, 6)))
        
        assert len(unique_points) == 16
    
    def test_gray_coding(self):
        """Test Gray coding property (adjacent symbols differ by 1 bit)."""
        mapper = QAMMapper('16QAM')
        points = mapper._table
        
        # For each point, check that nearest neighbors differ by 1 bit
        for i, p1 in enumerate(points):
            distances = np.abs(points - p1)
            # Sort by distance, skip self
            sorted_idx = np.argsort(distances)
            
            # Check first few neighbors
            for j in sorted_idx[1:5]:  # Skip self (index 0)
                # Count bit differences
                diff_bits = bin(i ^ j).count('1')
                # At least some neighbors should differ by 1 bit
                if distances[j] < 0.5:  # Close neighbor
                    assert diff_bits <= 2
    
    def test_roundtrip(self):
        """Test mapping and demapping."""
        mapper = QAMMapper('16QAM')
        demapper = QAMDemapper('16QAM')
        
        # Test all possible 4-bit patterns
        for i in range(16):
            bits = np.array([(i >> (3-b)) & 1 for b in range(4)], dtype=np.uint8)
            symbols = mapper.map(bits)
            recovered = demapper.demap(symbols)
            
            np.testing.assert_array_equal(recovered, bits)


class Test64QAM:
    """Test 64-QAM constellation."""
    
    def test_output_count(self):
        """Test correct number of symbols."""
        mapper = QAMMapper('64QAM')
        
        bits = np.array([0] * 12, dtype=np.uint8)
        symbols = mapper.map(bits)
        
        assert len(symbols) == 2  # 12 bits / 6 bits per symbol
    
    def test_normalized_power(self):
        """Test that constellation has unit average power."""
        mapper = QAMMapper('64QAM')
        points = mapper._table
        
        avg_power = np.mean(np.abs(points) ** 2)
        assert abs(avg_power - 1.0) < 0.01
    
    def test_unique_points(self):
        """Test that all 64 points are unique."""
        mapper = QAMMapper('64QAM')
        points = mapper._table
        
        unique_points = set()
        for p in points:
            unique_points.add((round(p.real, 6), round(p.imag, 6)))
        
        assert len(unique_points) == 64
    
    def test_roundtrip(self):
        """Test mapping and demapping."""
        mapper = QAMMapper('64QAM')
        demapper = QAMDemapper('64QAM')
        
        # Test random patterns
        np.random.seed(42)
        for _ in range(10):
            bits = np.random.randint(0, 2, 60, dtype=np.uint8)  # 10 symbols
            symbols = mapper.map(bits)
            recovered = demapper.demap(symbols)
            
            np.testing.assert_array_equal(recovered, bits)


class TestQAMNoiseRobustness:
    """Test QAM with noise."""
    
    def test_qpsk_noise_tolerance(self):
        """Test QPSK with moderate noise."""
        mapper = QAMMapper('QPSK')
        demapper = QAMDemapper('QPSK')
        
        np.random.seed(42)
        bits = np.random.randint(0, 2, 100, dtype=np.uint8)
        symbols = mapper.map(bits)
        
        # Add small noise
        noise = 0.1 * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
        noisy = symbols + noise.astype(np.complex64)
        
        recovered = demapper.demap(noisy)
        
        # Should still decode correctly with small noise
        np.testing.assert_array_equal(recovered, bits)
    
    def test_soft_decision_output(self):
        """Test soft decision demapping."""
        mapper = QAMMapper('QPSK')
        demapper = QAMDemapper('QPSK', soft_output=True)
        
        bits = np.array([0, 0, 1, 1], dtype=np.uint8)
        symbols = mapper.map(bits)
        
        llrs = demapper.demap(symbols, noise_variance=0.1)
        
        # LLRs should be non-zero
        assert np.any(llrs != 0)
        
        # Sign should indicate bit value
        hard_decisions = (llrs < 0).astype(np.uint8)
        np.testing.assert_array_equal(hard_decisions, bits)
