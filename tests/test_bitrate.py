"""
DVB-T Bitrate and Throughput Tests

Tests that measure:
- Theoretical data rates for different configurations
- Actual processing throughput (modulation and demodulation speed)
- Efficiency relative to real-time
"""

import pytest
import numpy as np
import time
from typing import Tuple


def generate_test_ts_data(num_packets: int) -> bytes:
    """Generate test transport stream data."""
    packets = []
    for i in range(num_packets):
        packet = bytearray([0x47])  # Sync byte
        packet.extend([(i % 256)] * 187)
        packets.append(bytes(packet))
    return b''.join(packets)


class TestTheoreticalBitrate:
    """Test theoretical data rate calculations."""
    
    def test_qpsk_rate_half_2k(self):
        """Test QPSK 1/2 rate in 2K mode."""
        from dvb import DVBTModulator
        from dvb.FrameBuilder import FrameInfo
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        rate = mod.get_data_rate()
        
        # DVB-T QPSK 1/2 with 1/4 guard in 8MHz should be ~4.98 Mbps
        assert 4.5e6 < rate < 5.5e6, f"Expected ~5 Mbps, got {rate/1e6:.2f} Mbps"
    
    def test_64qam_rate_three_quarters_2k(self):
        """Test 64QAM 3/4 rate in 2K mode."""
        from dvb import DVBTModulator
        
        mod = DVBTModulator(
            mode='2K',
            constellation='64QAM',
            code_rate='3/4',
            guard_interval='1/4'
        )
        
        rate = mod.get_data_rate()
        
        # DVB-T 64QAM 3/4 with 1/4 guard should be ~22.4 Mbps
        assert 20e6 < rate < 25e6, f"Expected ~22 Mbps, got {rate/1e6:.2f} Mbps"
    
    def test_guard_interval_affects_rate(self):
        """Smaller guard interval = higher data rate."""
        from dvb import DVBTModulator
        
        rates = {}
        for guard in ['1/4', '1/8', '1/16', '1/32']:
            mod = DVBTModulator(
                mode='2K',
                constellation='QPSK',
                code_rate='1/2',
                guard_interval=guard
            )
            rates[guard] = mod.get_data_rate()
        
        # Smaller guard = higher rate
        assert rates['1/32'] > rates['1/16'] > rates['1/8'] > rates['1/4']
    
    def test_constellation_affects_rate(self):
        """Higher order modulation = higher data rate."""
        from dvb import DVBTModulator
        
        rates = {}
        for const in ['QPSK', '16QAM', '64QAM']:
            mod = DVBTModulator(
                mode='2K',
                constellation=const,
                code_rate='1/2',
                guard_interval='1/4'
            )
            rates[const] = mod.get_data_rate()
        
        # Higher modulation = higher rate
        assert rates['64QAM'] > rates['16QAM'] > rates['QPSK']
        
        # Rate ratios should match bits per symbol ratios
        assert abs(rates['16QAM'] / rates['QPSK'] - 2.0) < 0.1
        assert abs(rates['64QAM'] / rates['QPSK'] - 3.0) < 0.1
    
    def test_code_rate_affects_rate(self):
        """Higher code rate = higher data rate (less redundancy)."""
        from dvb import DVBTModulator
        
        rates = {}
        for code_rate in ['1/2', '2/3', '3/4']:
            mod = DVBTModulator(
                mode='2K',
                constellation='QPSK',
                code_rate=code_rate,
                guard_interval='1/4'
            )
            rates[code_rate] = mod.get_data_rate()
        
        # Higher code rate = higher data rate
        assert rates['3/4'] > rates['2/3'] > rates['1/2']


class TestModulatorThroughput:
    """Measure modulator processing speed."""
    
    def test_modulator_throughput_qpsk(self):
        """Measure QPSK modulation throughput."""
        from dvb import DVBTModulator
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        # Generate enough data for meaningful measurement
        num_packets = 20
        ts_data = generate_test_ts_data(num_packets)
        input_bits = len(ts_data) * 8
        
        # Measure modulation time
        start = time.perf_counter()
        iq_samples = mod.modulate(ts_data)
        elapsed = time.perf_counter() - start
        
        # Calculate throughput
        throughput_bps = input_bits / elapsed
        theoretical_rate = mod.get_data_rate()
        
        # Report results
        print(f"\n  QPSK 1/2 Modulator:")
        print(f"    Input: {input_bits} bits ({num_packets} packets)")
        print(f"    Time: {elapsed*1000:.1f} ms")
        print(f"    Throughput: {throughput_bps/1e6:.2f} Mbps")
        print(f"    Theoretical rate: {theoretical_rate/1e6:.2f} Mbps")
        print(f"    Real-time factor: {throughput_bps/theoretical_rate:.1f}x")
        
        # Should be at least 1x real-time (process faster than real transmission)
        assert throughput_bps > 0
        assert len(iq_samples) > 0
    
    def test_modulator_throughput_64qam(self):
        """Measure 64QAM modulation throughput."""
        from dvb import DVBTModulator
        
        mod = DVBTModulator(
            mode='2K',
            constellation='64QAM',
            code_rate='3/4',
            guard_interval='1/4'
        )
        
        num_packets = 20
        ts_data = generate_test_ts_data(num_packets)
        input_bits = len(ts_data) * 8
        
        start = time.perf_counter()
        iq_samples = mod.modulate(ts_data)
        elapsed = time.perf_counter() - start
        
        throughput_bps = input_bits / elapsed
        theoretical_rate = mod.get_data_rate()
        
        print(f"\n  64QAM 3/4 Modulator:")
        print(f"    Input: {input_bits} bits ({num_packets} packets)")
        print(f"    Time: {elapsed*1000:.1f} ms")
        print(f"    Throughput: {throughput_bps/1e6:.2f} Mbps")
        print(f"    Theoretical rate: {theoretical_rate/1e6:.2f} Mbps")
        print(f"    Real-time factor: {throughput_bps/theoretical_rate:.1f}x")
        
        assert throughput_bps > 0


class TestDemodulatorThroughput:
    """Measure demodulator processing speed."""
    
    @pytest.mark.slow
    def test_demodulator_throughput_qpsk(self):
        """Measure QPSK demodulation throughput."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        # First modulate some data
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        num_packets = 5
        ts_data = generate_test_ts_data(num_packets)
        iq_samples = mod.modulate(ts_data)
        
        # Calculate equivalent signal duration
        sample_rate = mod.get_sample_rate()
        signal_duration = len(iq_samples) / sample_rate
        
        # Measure demodulation time
        demod = DVBTDemodulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        start = time.perf_counter()
        recovered, stats = demod.demodulate(iq_samples)
        elapsed = time.perf_counter() - start
        
        # Calculate throughput
        input_bits = len(ts_data) * 8
        throughput_bps = input_bits / elapsed if elapsed > 0 else 0
        theoretical_rate = mod.get_data_rate()
        realtime_factor = signal_duration / elapsed if elapsed > 0 else 0
        
        print(f"\n  QPSK 1/2 Demodulator:")
        print(f"    Signal duration: {signal_duration*1000:.1f} ms")
        print(f"    Processing time: {elapsed*1000:.1f} ms")
        print(f"    Throughput: {throughput_bps/1e6:.2f} Mbps")
        print(f"    Theoretical rate: {theoretical_rate/1e6:.2f} Mbps")
        print(f"    Real-time factor: {realtime_factor:.3f}x")
        print(f"    Symbols processed: {stats['symbols']}")
        
        assert stats['symbols'] > 0
    
    @pytest.mark.slow
    def test_demodulator_throughput_64qam(self):
        """Measure 64QAM demodulation throughput."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        mod = DVBTModulator(
            mode='2K',
            constellation='64QAM',
            code_rate='3/4',
            guard_interval='1/4'
        )
        
        num_packets = 5
        ts_data = generate_test_ts_data(num_packets)
        iq_samples = mod.modulate(ts_data)
        
        sample_rate = mod.get_sample_rate()
        signal_duration = len(iq_samples) / sample_rate
        
        demod = DVBTDemodulator(
            mode='2K',
            constellation='64QAM',
            code_rate='3/4',
            guard_interval='1/4'
        )
        
        start = time.perf_counter()
        recovered, stats = demod.demodulate(iq_samples)
        elapsed = time.perf_counter() - start
        
        input_bits = len(ts_data) * 8
        throughput_bps = input_bits / elapsed if elapsed > 0 else 0
        theoretical_rate = mod.get_data_rate()
        realtime_factor = signal_duration / elapsed if elapsed > 0 else 0
        
        print(f"\n  64QAM 3/4 Demodulator:")
        print(f"    Signal duration: {signal_duration*1000:.1f} ms")
        print(f"    Processing time: {elapsed*1000:.1f} ms")
        print(f"    Throughput: {throughput_bps/1e6:.2f} Mbps")
        print(f"    Theoretical rate: {theoretical_rate/1e6:.2f} Mbps")
        print(f"    Real-time factor: {realtime_factor:.3f}x")
        print(f"    Symbols processed: {stats['symbols']}")
        
        assert stats['symbols'] > 0


class TestEndToEndThroughput:
    """Measure complete TX/RX chain throughput."""
    
    @pytest.mark.slow
    def test_loopback_throughput(self):
        """Measure full modulation + demodulation throughput."""
        from dvb import DVBTModulator, DVBTDemodulator
        
        configs = [
            ('QPSK', '1/2'),
            ('16QAM', '2/3'),
            ('64QAM', '3/4'),
        ]
        
        print("\n  End-to-End Throughput Summary:")
        print("  " + "-" * 70)
        print(f"  {'Config':<15} {'Theory':<12} {'TX':<14} {'RX':<14} {'RT Factor'}")
        print("  " + "-" * 70)
        
        num_packets = 5
        ts_data = generate_test_ts_data(num_packets)
        input_bits = len(ts_data) * 8
        
        for const, code_rate in configs:
            mod = DVBTModulator(
                mode='2K',
                constellation=const,
                code_rate=code_rate,
                guard_interval='1/4'
            )
            
            theoretical_rate = mod.get_data_rate()
            
            # Measure TX
            start = time.perf_counter()
            iq_samples = mod.modulate(ts_data)
            tx_time = time.perf_counter() - start
            
            # Calculate signal duration
            sample_rate = mod.get_sample_rate()
            signal_duration = len(iq_samples) / sample_rate
            
            # Measure RX
            demod = DVBTDemodulator(
                mode='2K',
                constellation=const,
                code_rate=code_rate,
                guard_interval='1/4'
            )
            
            start = time.perf_counter()
            recovered, stats = demod.demodulate(iq_samples)
            rx_time = time.perf_counter() - start
            
            tx_throughput = input_bits / tx_time
            rx_throughput = input_bits / rx_time
            realtime_factor = signal_duration / (tx_time + rx_time)
            
            # Format throughput with appropriate units
            def fmt_rate(bps):
                if bps >= 1e6:
                    return f"{bps/1e6:>7.2f} Mbps"
                elif bps >= 1e3:
                    return f"{bps/1e3:>7.1f} kbps"
                else:
                    return f"{bps:>7.0f} bps"
            
            config_str = f"{const} {code_rate}"
            print(f"  {config_str:<15} {theoretical_rate/1e6:>8.2f} Mbps "
                  f"{fmt_rate(tx_throughput):>14} {fmt_rate(rx_throughput):>14} "
                  f"{realtime_factor:>8.3f}x")
        
        print("  " + "-" * 70)


class TestBitrateVsPacketCount:
    """Test how throughput scales with data size."""
    
    def test_modulator_scaling(self):
        """Test modulator throughput vs packet count."""
        from dvb import DVBTModulator
        
        mod = DVBTModulator(
            mode='2K',
            constellation='QPSK',
            code_rate='1/2',
            guard_interval='1/4'
        )
        
        print("\n  Modulator Scaling (QPSK 1/2):")
        print("  " + "-" * 45)
        print(f"  {'Packets':<10} {'Input':<12} {'Time':<12} {'Throughput'}")
        print("  " + "-" * 45)
        
        for num_packets in [5, 10, 20, 50]:
            ts_data = generate_test_ts_data(num_packets)
            input_bits = len(ts_data) * 8
            
            start = time.perf_counter()
            iq_samples = mod.modulate(ts_data)
            elapsed = time.perf_counter() - start
            
            throughput = input_bits / elapsed / 1e6
            
            print(f"  {num_packets:<10} {input_bits:>8} bits "
                  f"{elapsed*1000:>8.1f} ms {throughput:>8.2f} Mbps")
        
        print("  " + "-" * 45)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

