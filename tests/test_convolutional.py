"""
Tests for convolutional encoder and Viterbi decoder.
"""

import pytest
import numpy as np
from dvb.Convolutional import ConvolutionalEncoder, ConvolutionalDecoder
from dvb.Puncturing import Puncturer, Depuncturer


class TestConvolutionalEncoder:
    """Test convolutional encoder."""
    
    def test_output_length(self):
        """Test that output is 2x input (rate 1/2) plus termination."""
        encoder = ConvolutionalEncoder()
        
        input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        output = encoder.encode(input_bits, terminate=True)
        
        # Output should be 2 * (input + 6 termination bits)
        expected_len = 2 * (len(input_bits) + 6)
        assert len(output) == expected_len
    
    def test_without_termination(self):
        """Test encoding without trellis termination."""
        encoder = ConvolutionalEncoder()
        
        input_bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        output = encoder.encode(input_bits, terminate=False)
        
        assert len(output) == 2 * len(input_bits)
    
    def test_deterministic_output(self):
        """Test that same input produces same output."""
        encoder1 = ConvolutionalEncoder()
        encoder2 = ConvolutionalEncoder()
        
        input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        
        output1 = encoder1.encode(input_bits)
        output2 = encoder2.encode(input_bits)
        
        np.testing.assert_array_equal(output1, output2)
    
    def test_reset_state(self):
        """Test that reset clears encoder state."""
        encoder = ConvolutionalEncoder()
        
        # Encode some data
        encoder.encode(np.array([1, 1, 1, 1], dtype=np.uint8), terminate=False)
        
        # Reset
        encoder.reset()
        
        # Encode again and compare with fresh encoder
        encoder2 = ConvolutionalEncoder()
        
        input_bits = np.array([1, 0, 1, 0], dtype=np.uint8)
        output1 = encoder.encode(input_bits, terminate=False)
        output2 = encoder2.encode(input_bits, terminate=False)
        
        np.testing.assert_array_equal(output1, output2)
    
    def test_slow_vs_fast(self):
        """Compare slow and fast implementations."""
        encoder_fast = ConvolutionalEncoder(use_fast=True)
        encoder_slow = ConvolutionalEncoder(use_fast=False)
        
        input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
        
        output_fast = encoder_fast.encode(input_bits)
        output_slow = encoder_slow.encode(input_bits)
        
        np.testing.assert_array_equal(output_fast, output_slow)


class TestViterbiDecoder:
    """Test Viterbi decoder."""
    
    def test_perfect_decode(self):
        """Test decoding without errors."""
        encoder = ConvolutionalEncoder()
        decoder = ConvolutionalDecoder()
        
        input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = encoder.encode(input_bits, terminate=True)
        decoded = decoder.decode(encoded, terminated=True)
        
        np.testing.assert_array_equal(decoded, input_bits)
    
    def test_single_bit_error(self):
        """Test that single bit error is corrected."""
        encoder = ConvolutionalEncoder()
        decoder = ConvolutionalDecoder()
        
        input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = encoder.encode(input_bits, terminate=True)
        
        # Flip one bit
        encoded[5] ^= 1
        
        decoded = decoder.decode(encoded, terminated=True)
        np.testing.assert_array_equal(decoded, input_bits)
    
    def test_multiple_bit_errors(self):
        """Test correction of scattered bit errors."""
        encoder = ConvolutionalEncoder()
        decoder = ConvolutionalDecoder()
        
        input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
        encoded = encoder.encode(input_bits, terminate=True)
        
        # Flip several bits (not too many)
        error_positions = [3, 10, 20]
        for pos in error_positions:
            encoded[pos] ^= 1
        
        decoded = decoder.decode(encoded, terminated=True)
        np.testing.assert_array_equal(decoded, input_bits)
    
    def test_decode_to_bytes(self):
        """Test decoding to byte output."""
        encoder = ConvolutionalEncoder()
        decoder = ConvolutionalDecoder()
        
        input_data = b'\x55\xAA\x00\xFF'
        input_bits = np.unpackbits(np.frombuffer(input_data, dtype=np.uint8))
        
        encoded = encoder.encode(input_bits, terminate=True)
        decoded_bytes = decoder.decode_to_bytes(encoded, terminated=True)
        
        assert decoded_bytes[:len(input_data)] == input_data


class TestPuncturing:
    """Test puncturing and depuncturing."""
    
    def test_rate_half_no_puncture(self):
        """Test that rate 1/2 doesn't puncture."""
        punct = Puncturer('1/2')
        
        input_bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        output = punct.puncture(input_bits)
        
        np.testing.assert_array_equal(output, input_bits)
    
    def test_rate_two_thirds(self):
        """Test rate 2/3 puncturing."""
        punct = Puncturer('2/3')
        
        # Rate 2/3: 4 input -> 3 output
        input_bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        output = punct.puncture(input_bits)
        
        # Pattern [1,1; 1,0] means keep bits 0,1,2 drop bit 3
        assert len(output) == 3
    
    def test_rate_three_quarters(self):
        """Test rate 3/4 puncturing."""
        punct = Puncturer('3/4')
        
        # Rate 3/4: 6 input -> 4 output
        input_bits = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
        output = punct.puncture(input_bits)
        
        assert len(output) == 4
    
    def test_depuncture_restores_length(self):
        """Test that depuncturing restores original length."""
        for rate in ['1/2', '2/3', '3/4', '5/6', '7/8']:
            punct = Puncturer(rate)
            depunct = Depuncturer(rate)
            
            # Generate enough bits for complete patterns
            input_len = punct.input_per_period * 5
            input_bits = np.random.randint(0, 2, input_len, dtype=np.uint8)
            
            punctured = punct.puncture(input_bits)
            depunctured = depunct.depuncture(punctured)
            
            assert len(depunctured) == len(input_bits)
    
    def test_puncture_depuncture_roundtrip(self):
        """Test that non-erased positions are preserved."""
        for rate in ['2/3', '3/4', '5/6', '7/8']:
            punct = Puncturer(rate)
            depunct = Depuncturer(rate)
            
            input_len = punct.input_per_period * 3
            input_bits = np.random.randint(0, 2, input_len, dtype=np.uint8)
            
            punctured = punct.puncture(input_bits)
            depunctured = depunct.depuncture(punctured.astype(np.float32))
            
            # Non-erased positions should match
            for i, (orig, depunc) in enumerate(zip(input_bits, depunctured)):
                if depunc != 0.5:  # Not an erasure
                    assert depunc == orig or depunc == 0.0


class TestFullConvChain:
    """Test complete convolutional coding chain."""
    
    def test_encode_puncture_depuncture_decode(self):
        """Test full chain with puncturing."""
        encoder = ConvolutionalEncoder()
        decoder = ConvolutionalDecoder()
        
        for rate in ['1/2', '2/3', '3/4']:
            punct = Puncturer(rate)
            depunct = Depuncturer(rate)
            
            # Input data
            input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
            
            # Encode
            encoded = encoder.encode(input_bits, terminate=True)
            encoder.reset()
            
            # Puncture
            punctured = punct.puncture(encoded)
            
            # Depuncture
            depunctured = depunct.depuncture(punctured.astype(np.float32))
            
            # Convert back to hard decisions
            hard = (depunctured > 0.5).astype(np.uint8)
            
            # Decode
            decoded = decoder.decode(hard, terminated=True)
            
            # Compare only the original input length
            np.testing.assert_array_equal(decoded[:len(input_bits)], input_bits)
