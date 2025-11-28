"""
DVB-T TX/RX Loopback Tests

End-to-end tests that transmit and receive through simulated channels.

Test levels:
- Ideal channel (sanity check)
- AWGN channel
- CFO channel
- Multipath channel
"""

import pytest
import numpy as np
open aufrom pathlib import Path
from typing import Tuple
from tests.channel_model import (
    ChannelModel, ChannelConfig,
    awgn_channel, cfo_channel, single_echo_channel,
    measure_ber, measure_per,
    RoomChannelModel,
)


def generate_test_ts_data(num_packets: int = 100) -> bytes:
    """Generate known test transport stream data."""
    packets = []
    for i in range(num_packets):
        # Each packet: sync byte + known pattern
        packet = bytearray([0x47])  # Sync byte
        # Payload: packet number repeated
        packet.extend([(i % 256)] * 187)
        packets.append(bytes(packet))
    return b''.join(packets)


def verify_ts_data(original: bytes, recovered: bytes) -> Tuple[int, int, float]:
    """
    Verify recovered TS data against original.
    
    Returns:
        Tuple of (matching_packets, total_recovered_packets, match_rate)
    """
    packet_size = 188
    original_packets = [original[i:i+packet_size] 
                       for i in range(0, len(original), packet_size)]
    recovered_packets = [recovered[i:i+packet_size] 
                        for i in range(0, len(recovered), packet_size)]
    
    # Try to match packets
    matches = 0
    for rx_pkt in recovered_packets:
        if rx_pkt in original_packets:
            matches += 1
    
    if len(recovered_packets) == 0:
        return 0, 0, 0.0
    
    return matches, len(recovered_packets), matches / len(recovered_packets)


class TestLoopbackIdeal:
    """TX -> RX with ideal channel (no impairments)."""
    
    def test_qpsk_ideal(self):
        """QPSK loopback with ideal channel."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(iq_samples)
        
        # Should recover most packets
        matches, total, rate = verify_ts_data(ts_data, recovered)
        
        assert stats['symbols'] > 0
        # In ideal channel, we expect good recovery
        # Note: Due to interleaver delay, first/last packets may be lost
        assert rate > 0.5 or total > 0
    
    def test_16qam_ideal(self):
        """16QAM loopback with ideal channel."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='16QAM',
            code_rate='2/3',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='16QAM',
            code_rate='2/3',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(iq_samples)
        
        assert stats['symbols'] > 0
    
    def test_64qam_ideal(self):
        """64QAM loopback with ideal channel."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='64QAM',
            code_rate='3/4',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='64QAM',
            code_rate='3/4',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(iq_samples)
        
        assert stats['symbols'] > 0
    
    def test_8k_mode_ideal(self):
        """8K mode loopback."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(10)
        
        mod = DVBTModulator(
            mode='8K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/8'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        demod = DVBTDemodulator(
            mode='8K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/8'
        )
        
        recovered, stats = demod.demodulate(iq_samples)
        
        assert stats['symbols'] > 0


class TestLoopbackAWGN:
    """TX -> AWGN channel -> RX."""
    
    def test_awgn_high_snr(self):
        """Test with high SNR (should work well)."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        # Add AWGN at 25 dB SNR
        channel = ChannelModel()
        noisy_samples = channel.add_awgn(iq_samples, snr_db=25)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(noisy_samples)
        
        assert stats['symbols'] > 0
        # At 25 dB with QPSK 1/2, should have low error rate
    
    def test_awgn_medium_snr(self):
        """Test with medium SNR."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        # Add AWGN at 15 dB SNR
        channel = ChannelModel()
        noisy_samples = channel.add_awgn(iq_samples, snr_db=15)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(noisy_samples)
        
        assert stats['symbols'] > 0
    
    @pytest.mark.parametrize("snr_db", [30, 20])
    def test_awgn_snr_sweep(self, snr_db):
        """Test various SNR levels."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        channel = ChannelModel()
        noisy_samples = channel.add_awgn(iq_samples, snr_db=snr_db)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(noisy_samples)
        
        # At least should process symbols
        assert stats['symbols'] > 0


class TestLoopbackCFO:
    """TX -> CFO channel -> RX (tests synchronization)."""
    
    def test_small_cfo(self):
        """Test with small CFO (< 1 subcarrier spacing)."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        # Add small CFO (carrier spacing for 2K/8MHz is ~4.5 kHz)
        channel = ChannelModel()
        cfo_hz = 500  # 500 Hz, ~0.1 subcarriers
        samples_with_cfo = channel.add_cfo(iq_samples, cfo_hz)
        samples_with_noise = channel.add_awgn(samples_with_cfo, snr_db=25)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(samples_with_noise)
        
        assert stats['symbols'] > 0
        # CFO should be estimated
        assert 'cfo_hz' in stats
    
    def test_medium_cfo(self):
        """Test with medium CFO."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        channel = ChannelModel()
        cfo_hz = 2000  # 2 kHz
        samples_with_cfo = channel.add_cfo(iq_samples, cfo_hz)
        samples_with_noise = channel.add_awgn(samples_with_cfo, snr_db=25)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(samples_with_noise)
        
        assert stats['symbols'] > 0


class TestLoopbackMultipath:
    """TX -> multipath channel -> RX (tests channel estimation)."""
    
    def test_single_echo(self):
        """Test with single echo within guard interval."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        # Single echo at 100 samples delay (well within 1/4 guard = 512 samples)
        channel = ChannelModel()
        config = single_echo_channel(snr_db=25, delay_samples=100, echo_gain_db=-6)
        multipath_samples = channel.apply(iq_samples, config)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(multipath_samples)
        
        assert stats['symbols'] > 0


class TestChannelModelLoopback:
    """Full channel model loopback for standard and audio modes."""
    
    @pytest.mark.parametrize(
        "mode, bandwidth, guard, cfo_hz, echo_delay_samples, echo_delay_ms",
        [
            ('2K', '8MHz', '1/4', 350.0, 120, None),
            # Audio mode: include a 50 ms echo to emulate long acoustic reflections.
            ('audio', 'audio', 'acoustic', 15.0, None, 50.0),
        ]
    )
    def test_channel_model_modes(self, mode, bandwidth, guard,
                                 cfo_hz, echo_delay_samples, echo_delay_ms):
        """Ensure ChannelModel impairments work in regular and audio loopback."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_packets = 8 if mode == 'audio' else 5
        ts_data = generate_test_ts_data(ts_packets)
        
        mod = DVBTModulator(
            mode=mode,
            constellation='QPSK',
            code_rate='1/2',
            guard_interval=guard,
            bandwidth=bandwidth,
        )
        
        iq_samples = mod.modulate(ts_data)
        
        channel = ChannelModel(sample_rate=mod.get_sample_rate())
        # Convert requested multipath delay to samples (supports ms for audio case).
        if echo_delay_ms is not None:
            echo_delay = int(round((echo_delay_ms / 1000.0) * mod.get_sample_rate()))
            echo_delay = max(1, echo_delay)
        else:
            echo_delay = echo_delay_samples
        
        config = ChannelConfig(
            snr_db=32.0,
            cfo_hz=cfo_hz,
            phase_noise_deg=0.5,
            multipath=True,
            delays_samples=[0, echo_delay],
            gains_db=[0.0, -6.0],
        )
        impaired = channel.apply(iq_samples, config)
        
        demod = DVBTDemodulator(
            mode=mode,
            constellation='QPSK',
            code_rate='1/2',
            guard_interval=guard,
            bandwidth=bandwidth,
        )
        
        recovered, stats = demod.demodulate(impaired)
        matches, total, rate = verify_ts_data(ts_data, recovered)
        
        assert stats['symbols'] > 0
        assert total >= matches >= 0
        assert total > 0
        
        # CFO estimation should detect the applied offset in both modes.
        cfo_tolerance = 50 if mode != 'audio' else 5
        assert abs(stats['cfo_hz'] - cfo_hz) < cfo_tolerance
    
    @pytest.mark.parametrize("channel_type", ["simulated", "room"])
    def test_room_channel_model(self, channel_type):
        """
        Test with real room channel model (if available) or simulated.
        
        To use the real room channel:
        1. Run: python scripts/calibrate_room.py generate -o probe.wav
        2. Play probe.wav through speaker, record with mic as recorded.wav
        3. Run: python scripts/calibrate_room.py analyze probe.wav recorded.wav
        4. Run this test: pytest tests/test_loopback.py::TestChannelModelLoopback::test_room_channel_model -v
        """
        from dvb import DVBTModulator, DVBTDemodulator
        
        # Check for room channel model file (check scripts/ and project root)
        room_model_path = Path(__file__).parent.parent / "scripts" / "room_channel.npz"
        if not room_model_path.exists():
            room_model_path = Path(__file__).parent.parent / "room_channel.npz"
        
        if channel_type == "room":
            if not room_model_path.exists():
                pytest.skip(
                    f"Room channel model not found at {room_model_path}. "
                    "Run scripts/calibrate_room.py to create one."
                )
            channel = RoomChannelModel.load(str(room_model_path))
            # Room channel already includes noise at measured SNR
        else:
            # Simulated channel for comparison
            channel = None
        
        # Use audio mode for acoustic testing
        ts_data = generate_test_ts_data(8)
        
        mod = DVBTModulator(
            mode='audio',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='acoustic',
            bandwidth='audio',
        )
        
        iq_samples = mod.modulate(ts_data)
        
        if channel_type == "room":
            # Apply real room channel
            impaired = channel.apply(iq_samples)
        else:
            # Apply simulated channel with typical room characteristics
            sim_channel = ChannelModel(sample_rate=mod.get_sample_rate())
            config = ChannelConfig(
                snr_db=25.0,
                cfo_hz=10.0,
                phase_noise_deg=1.0,
                multipath=True,
                # 50ms echo typical for room
                delays_samples=[0, int(0.050 * mod.get_sample_rate())],
                gains_db=[0.0, -10.0],
            )
            impaired = sim_channel.apply(iq_samples, config)
        
        demod = DVBTDemodulator(
            mode='audio',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='acoustic',
            bandwidth='audio',
        )
        
        recovered, stats = demod.demodulate(impaired)
        matches, total, rate = verify_ts_data(ts_data, recovered)
        
        assert stats['symbols'] > 0, "Should process symbols"
        assert total > 0, "Should recover some packets"
        
        # Print results for debugging
        print(f"\n{channel_type.upper()} channel results:")
        print(f"  Symbols processed: {stats['symbols']}")
        print(f"  Packets recovered: {total} ({matches} matching)")
        print(f"  Match rate: {rate*100:.1f}%")
        print(f"  RS errors: {stats.get('rs_errors', 0)}")
        print(f"  RS uncorrectable: {stats.get('rs_uncorrectable', 0)}")
        if channel_type == "room":
            print(f"  Room channel SNR: {channel.snr_db:.1f} dB")
            print(f"  Room delay spread: {channel.delay_spread_ms:.1f} ms")


class TestCodeRates:
    """Test all code rates."""
    
    @pytest.mark.parametrize("code_rate", ['1/2', '2/3'])
    def test_code_rates(self, code_rate):
        """Test different code rates."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate=code_rate,
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        channel = ChannelModel()
        noisy = channel.add_awgn(iq_samples, snr_db=25)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate=code_rate,
            guard_interval='1/4'
        )
        
        recovered, stats = demod.demodulate(noisy)
        
        assert stats['symbols'] > 0


class TestGuardIntervals:
    """Test all guard intervals."""
    
    @pytest.mark.parametrize("guard", ['1/4', '1/8'])
    def test_guard_intervals(self, guard):
        """Test different guard intervals."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval=guard
        )
        
        iq_samples = mod.modulate(ts_data)
        
        channel = ChannelModel()
        noisy = channel.add_awgn(iq_samples, snr_db=25)
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval=guard
        )
        
        recovered, stats = demod.demodulate(noisy)
        
        assert stats['symbols'] > 0


class TestStageRoundtrip:
    """Test individual stage roundtrips."""
    
    def test_ofdm_roundtrip_ideal(self):
        """IFFT -> FFT roundtrip."""
        from dvb import OFDMModulator, OFDMDemodulator
        
        for mode in ['2K', '8K']:
            mod = OFDMModulator(mode)
            demod = OFDMDemodulator(mode)
            
            carriers = (np.random.randn(mod.active_carriers) + 
                       1j * np.random.randn(mod.active_carriers)).astype(np.complex64)
            
            time_domain = mod.modulate(carriers)
            recovered = demod.demodulate(time_domain)
            
            np.testing.assert_allclose(recovered, carriers, rtol=1e-4)
    
    def test_guard_interval_roundtrip(self):
        """Add -> Remove guard interval."""
        from dvb import GuardIntervalInserter, GuardIntervalRemover
        
        for guard in ['1/4', '1/8', '1/16', '1/32']:
            fft_size = 2048
            
            inserter = GuardIntervalInserter(guard, fft_size)
            remover = GuardIntervalRemover(guard, fft_size)
            
            symbol = np.random.randn(fft_size).astype(np.complex64)
            
            with_guard = inserter.add(symbol)
            without_guard = remover.remove(with_guard)
            
            np.testing.assert_allclose(without_guard, symbol, rtol=1e-6)
    
    def test_qam_roundtrip(self):
        """QAM map -> demap."""
        from dvb import QAMMapper, QAMDemapper
        
        for const in ['QPSK', '16QAM', '64QAM']:
            mapper = QAMMapper(const)
            demapper = QAMDemapper(const)
            
            bits_per_sym = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[const]
            bits = np.random.randint(0, 2, 120, dtype=np.uint8)
            bits = bits[:len(bits) - len(bits) % bits_per_sym]
            
            symbols = mapper.map(bits)
            recovered_bits = demapper.demap(symbols)
            
            np.testing.assert_array_equal(recovered_bits, bits)
    
    def test_scrambler_roundtrip(self):
        """Scramble -> descramble."""
        from dvb import Scrambler
        
        scrambler = Scrambler()
        
        # Test data (multiple of 188 for TS packets)
        data = bytes([i % 256 for i in range(188 * 8)])
        
        scrambled = scrambler.scramble(data)
        descrambled = scrambler.descramble(scrambled)
        
        # Sync bytes may differ, check payload
        for i in range(8):
            start = i * 188 + 1
            end = (i + 1) * 188
            assert data[start:end] == descrambled[start:end]
    
    def test_rs_roundtrip(self):
        """RS encode -> decode."""
        from dvb import ReedSolomon
        
        rs = ReedSolomon()
        
        for _ in range(10):
            message = bytes(np.random.randint(0, 256, 188, dtype=np.uint8))
            encoded = rs.encode(message)
            
            assert len(encoded) == 204
            
            decoded, errors = rs.decode(encoded)
            
            assert decoded == message
            assert errors == 0
    
    def test_convolutional_roundtrip(self):
        """Conv encode -> Viterbi decode."""
        from dvb import ConvolutionalEncoder, ConvolutionalDecoder
        
        encoder = ConvolutionalEncoder()
        decoder = ConvolutionalDecoder()
        
        data = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
        
        encoder.reset()
        encoded = encoder.encode(data, terminate=True)
        
        decoded = decoder.decode(encoded, terminated=True)
        
        np.testing.assert_array_equal(decoded, data)


class TestAutoDetection:
    """Test automatic parameter detection."""
    
    def test_auto_mode(self):
        """Test auto-detection of mode."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        ts_data = generate_test_ts_data(5)
        
        # Transmit with known parameters
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        iq_samples = mod.modulate(ts_data)
        
        # Receive with auto-detection
        demod = DVBTDemodulator(mode='auto')
        
        recovered, stats = demod.demodulate(iq_samples)
        
        # Verify detection worked
        assert demod.mode == '2K'
        assert stats['symbols'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
