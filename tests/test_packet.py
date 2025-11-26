"""
Tests for MPEG-2 Transport Stream Packet handling.
"""

import pytest
import numpy as np
from dvb.Packet import Packet, PACKET_SIZE, PID_PAT, PID_NULL, SYNC_BYTE
from dvb.AdaptationField import AdaptationField
from dvb.CRC import CRC32, crc32


class TestCRC32:
    """Test CRC-32/MPEG-2 implementation."""
    
    def test_known_value(self):
        """Test CRC against known value."""
        crc = CRC32()
        # Test vector: CRC of "123456789" should be 0x0376E6E7 for MPEG-2
        data = b'123456789'
        result = crc.calculate(data)
        assert result == 0x0376E6E7
    
    def test_empty_data(self):
        """Test CRC of empty data."""
        crc = CRC32()
        result = crc.calculate(b'')
        assert result == 0xFFFFFFFF  # Initial value, no data processed
    
    def test_slow_vs_fast(self):
        """Compare slow and fast implementations."""
        crc_fast = CRC32(use_table=True)
        crc_slow = CRC32(use_table=False)
        
        test_data = bytes(range(256))
        assert crc_fast.calculate(test_data) == crc_slow.calculate(test_data)
    
    def test_append_and_verify(self):
        """Test appending CRC to data."""
        crc = CRC32()
        data = b'Test data for CRC'
        with_crc = crc.append(data)
        
        assert len(with_crc) == len(data) + 4
        # Verify by recalculating
        assert crc.calculate(with_crc) == 0  # Should be 0 if correct


class TestPacket:
    """Test TS packet parsing and generation."""
    
    def test_create_simple_packet(self):
        """Test creating a basic packet."""
        pkt = Packet(pid=0x100, payload=b'\x00' * 184)
        data = pkt.to_bytes()
        
        assert len(data) == PACKET_SIZE
        assert data[0] == SYNC_BYTE
    
    def test_roundtrip(self):
        """Test packet serialization and parsing."""
        original = Packet(
            pid=0x1234,
            payload=bytes(range(184)),
            transport_error=False,
            payload_unit_start=True,
            transport_priority=True,
            scrambling_control=0,
            continuity_counter=7,
        )
        
        data = original.to_bytes()
        parsed = Packet.from_bytes(data)
        
        assert parsed.pid == original.pid
        assert parsed.payload_unit_start == original.payload_unit_start
        assert parsed.transport_priority == original.transport_priority
        assert parsed.continuity_counter == original.continuity_counter
    
    def test_null_packet(self):
        """Test null packet creation."""
        pkt = Packet.null_packet()
        
        assert pkt.pid == PID_NULL
        assert len(pkt.to_bytes()) == PACKET_SIZE
    
    def test_invalid_pid(self):
        """Test that invalid PID raises error."""
        with pytest.raises(ValueError):
            Packet(pid=0x2000)  # Max is 0x1FFF
    
    def test_invalid_continuity_counter(self):
        """Test that invalid CC raises error."""
        with pytest.raises(ValueError):
            Packet(pid=0x100, continuity_counter=16)  # Max is 15
    
    def test_packet_with_adaptation_field(self):
        """Test packet with adaptation field."""
        af = AdaptationField(length=7, random_access=True)
        pkt = Packet(
            pid=0x100,
            payload=b'\x00' * 176,  # Less payload due to AF
            adaptation_field=af,
        )
        
        data = pkt.to_bytes()
        assert len(data) == PACKET_SIZE
        
        # Parse back
        parsed = Packet.from_bytes(data)
        assert parsed.has_adaptation_field
        assert parsed.adaptation_field.random_access
    
    def test_payload_too_large(self):
        """Test that oversized payload raises error."""
        with pytest.raises(ValueError):
            Packet(pid=0x100, payload=b'\x00' * 185)


class TestAdaptationField:
    """Test adaptation field handling."""
    
    def test_stuffing(self):
        """Test stuffing-only adaptation field."""
        af = AdaptationField.stuffing(10)
        data = af.to_bytes()
        
        assert len(data) == 11  # 1 length byte + 10 stuffing
        assert data[0] == 10
    
    def test_pcr(self):
        """Test PCR encoding/decoding."""
        # Test timestamp conversion
        timestamp = 1.5  # 1.5 seconds
        pcr = AdaptationField.pcr_from_timestamp(timestamp)
        recovered = AdaptationField.pcr_to_timestamp(pcr)
        
        assert abs(recovered - timestamp) < 1e-6
    
    def test_pcr_roundtrip(self):
        """Test PCR in adaptation field roundtrip."""
        pcr = AdaptationField.pcr_from_timestamp(2.5)
        af = AdaptationField(length=7, pcr=pcr, random_access=True)
        
        data = af.to_bytes()
        parsed = AdaptationField.from_bytes(data)
        
        assert parsed.pcr is not None
        assert parsed.random_access
        
        # Check PCR value
        orig_time = AdaptationField.pcr_to_timestamp(pcr)
        parsed_time = AdaptationField.pcr_to_timestamp(parsed.pcr)
        assert abs(orig_time - parsed_time) < 1e-6


class TestTransportStream:
    """Test transport stream container."""
    
    def test_create_empty(self):
        """Test creating empty transport stream."""
        from dvb.TransportStream import TransportStream
        
        ts = TransportStream()
        assert len(ts) == 0
    
    def test_append_packet(self):
        """Test appending packets."""
        from dvb.TransportStream import TransportStream
        
        ts = TransportStream()
        ts.append(Packet(pid=0x100, payload=b'\x00' * 184))
        ts.append(Packet(pid=0x100, payload=b'\x00' * 184, continuity_counter=1))
        
        assert len(ts) == 2
    
    def test_iter_pid(self):
        """Test iterating over specific PID."""
        from dvb.TransportStream import TransportStream
        
        ts = TransportStream()
        ts.append(Packet(pid=0x100, payload=b'\x00' * 184))
        ts.append(Packet(pid=0x101, payload=b'\x00' * 184))
        ts.append(Packet(pid=0x100, payload=b'\x00' * 184, continuity_counter=1))
        
        packets_100 = list(ts.iter_pid(0x100))
        assert len(packets_100) == 2
    
    def test_bytes_roundtrip(self):
        """Test serialization roundtrip."""
        from dvb.TransportStream import TransportStream
        
        ts = TransportStream()
        ts.append(Packet(pid=0x100, payload=bytes(range(184))))
        ts.append(Packet(pid=0x101, payload=bytes(range(184)), continuity_counter=1))
        
        data = ts.to_bytes()
        parsed = TransportStream.from_bytes(data)
        
        assert len(parsed) == len(ts)
