"""
DVB-T Transmission Parameter Signaling (TPS)

TPS carriers convey transmission parameters to the receiver including:
- Constellation type (QPSK, 16QAM, 64QAM)
- Hierarchy information
- Code rate
- Guard interval
- Transmission mode
- Cell identifier

TPS data is spread over 68 symbols (one frame), with BCH error protection.

Reference: ETSI EN 300 744 Section 4.6
"""

import numpy as np
from typing import Dict, Optional


class TPSEncoder:
    """
    DVB-T TPS (Transmission Parameter Signaling) encoder.
    
    Encodes transmission parameters into TPS bits that are
    modulated onto TPS pilot carriers.
    
    Attributes:
        frame_size: Number of OFDM symbols per frame (68 for 2K/8K, 16 for audio)
        tps_bits_per_frame: Total TPS bits per frame
        
    Example:
        >>> tps = TPSEncoder()
        >>> tps.set_parameters(constellation='64QAM', code_rate='2/3')
        >>> frame_tps = tps.encode_frame()
    """
    
    # Frame structure
    SYMBOLS_PER_FRAME = 68
    SYMBOLS_PER_FRAME_AUDIO = 16
    TPS_BITS_PER_SYMBOL = 1  # One bit per TPS carrier per symbol
    
    # TPS bit positions within frame
    # Sync word (16 bits): symbols 0-16
    # Length indicator (6 bits): symbols 17-22
    # Frame number (2 bits): symbols 23-24
    # Constellation (2 bits): symbols 25-26
    # Hierarchy (3 bits): symbols 27-29
    # HP code rate (3 bits): symbols 30-32
    # LP code rate (3 bits): symbols 33-35
    # Guard interval (2 bits): symbols 36-37
    # Transmission mode (2 bits): symbols 38-39
    # Cell identifier (16 bits): symbols 40-55
    # Reserved (6 bits): symbols 56-61
    # BCH parity (14 bits): symbols 62-67
    # Symbol 0 is initialization
    
    # Constellation codes
    CONSTELLATION_CODES = {
        'QPSK': 0b00,
        '16QAM': 0b01,
        '64QAM': 0b10,
    }
    
    # Hierarchy codes
    HIERARCHY_CODES = {
        'non-hierarchical': 0b000,
        'α=1': 0b001,
        'α=2': 0b010,
        'α=4': 0b011,
    }
    
    # Code rate codes
    CODE_RATE_CODES = {
        '1/2': 0b000,
        '2/3': 0b001,
        '3/4': 0b010,
        '5/6': 0b011,
        '7/8': 0b100,
    }
    
    # Guard interval codes
    GUARD_INTERVAL_CODES = {
        '1/32': 0b00,
        '1/16': 0b01,
        '1/8': 0b10,
        '1/4': 0b11,
        'acoustic': 0b11,  # Uses same code as 1/4 for TPS (custom mode)
    }
    
    # Transmission mode codes
    MODE_CODES = {
        '2K': 0b00,
        '8K': 0b01,
        'audio': 0b10,  # Audio mode code
    }
    
    # Sync words for even and odd frames
    SYNC_EVEN = 0b0011010111101110
    SYNC_ODD = 0b1100101000010001
    
    # Simplified sync for audio mode (4 bits)
    SYNC_AUDIO_EVEN = 0b0101
    SYNC_AUDIO_ODD = 0b1010
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize TPS encoder.
        
        Args:
            mode: '2K', '8K', or 'audio'
        """
        self.mode = mode
        if mode == 'audio':
            self.num_tps_carriers = 4
            self.symbols_per_frame = self.SYMBOLS_PER_FRAME_AUDIO
        elif mode == '2K':
            self.num_tps_carriers = 17
            self.symbols_per_frame = self.SYMBOLS_PER_FRAME
        else:  # 8K
            self.num_tps_carriers = 68
            self.symbols_per_frame = self.SYMBOLS_PER_FRAME
        
        # Default parameters
        self._params = {
            'constellation': 'QPSK',
            'hierarchy': 'non-hierarchical',
            'hp_code_rate': '1/2',
            'lp_code_rate': '1/2',
            'guard_interval': '1/4',
            'mode': mode,
            'frame_number': 0,
            'cell_id': 0,
        }
    
    def set_parameters(self, **kwargs) -> None:
        """
        Set TPS parameters.
        
        Args:
            constellation: 'QPSK', '16QAM', or '64QAM'
            hierarchy: 'non-hierarchical', 'α=1', 'α=2', 'α=4'
            hp_code_rate: '1/2', '2/3', '3/4', '5/6', '7/8'
            lp_code_rate: '1/2', '2/3', '3/4', '5/6', '7/8'
            guard_interval: '1/4', '1/8', '1/16', '1/32'
            frame_number: 0-3
            cell_id: 16-bit cell identifier
        """
        self._params.update(kwargs)
    
    def _pack_tps_data(self, frame_number: int) -> np.ndarray:
        """
        Pack TPS parameters into bit array.
        
        Args:
            frame_number: Frame number within superframe (0-3)
            
        Returns:
            67-bit TPS data (excluding initialization bit)
        """
        bits = np.zeros(67, dtype=np.uint8)
        
        # Sync word (symbols 1-16, bits 0-15)
        sync = self.SYNC_EVEN if frame_number % 2 == 0 else self.SYNC_ODD
        for i in range(16):
            bits[i] = (sync >> (15 - i)) & 1
        
        # Length indicator (symbols 17-22, bits 16-21)
        # Fixed value for DVB-T: 0x17 (23 in decimal)
        length = 0x17
        for i in range(6):
            bits[16 + i] = (length >> (5 - i)) & 1
        
        # Frame number (symbols 23-24, bits 22-23)
        for i in range(2):
            bits[22 + i] = (frame_number >> (1 - i)) & 1
        
        # Constellation (symbols 25-26, bits 24-25)
        const_code = self.CONSTELLATION_CODES[self._params['constellation']]
        for i in range(2):
            bits[24 + i] = (const_code >> (1 - i)) & 1
        
        # Hierarchy (symbols 27-29, bits 26-28)
        hier_code = self.HIERARCHY_CODES[self._params['hierarchy']]
        for i in range(3):
            bits[26 + i] = (hier_code >> (2 - i)) & 1
        
        # HP code rate (symbols 30-32, bits 29-31)
        hp_code = self.CODE_RATE_CODES[self._params['hp_code_rate']]
        for i in range(3):
            bits[29 + i] = (hp_code >> (2 - i)) & 1
        
        # LP code rate (symbols 33-35, bits 32-34)
        lp_code = self.CODE_RATE_CODES[self._params['lp_code_rate']]
        for i in range(3):
            bits[32 + i] = (lp_code >> (2 - i)) & 1
        
        # Guard interval (symbols 36-37, bits 35-36)
        guard_code = self.GUARD_INTERVAL_CODES[self._params['guard_interval']]
        for i in range(2):
            bits[35 + i] = (guard_code >> (1 - i)) & 1
        
        # Transmission mode (symbols 38-39, bits 37-38)
        mode_code = self.MODE_CODES[self._params['mode']]
        for i in range(2):
            bits[37 + i] = (mode_code >> (1 - i)) & 1
        
        # Cell ID (symbols 40-55, bits 39-54)
        cell_id = self._params['cell_id']
        for i in range(16):
            bits[39 + i] = (cell_id >> (15 - i)) & 1
        
        # Reserved (symbols 56-61, bits 55-60) - set to 0
        # bits[55:61] already 0
        
        # BCH parity will be computed separately
        
        return bits
    
    def _compute_bch_parity(self, data: np.ndarray) -> np.ndarray:
        """
        Compute BCH(67,53,2) parity bits.
        
        Uses generator polynomial for BCH code.
        
        Args:
            data: 53 information bits
            
        Returns:
            14 parity bits
        """
        # BCH(67,53,2) generator polynomial
        # g(x) = x^14 + x^9 + x^8 + x^6 + x^5 + x^4 + x^2 + x + 1
        # = 0x4F61 (binary: 100111101100001)
        gen_poly = 0b100111101100001
        
        # Shift register implementation
        reg = 0
        
        for bit in data[:53]:  # Only first 53 bits are information
            feedback = bit ^ ((reg >> 13) & 1)
            
            if feedback:
                reg = ((reg << 1) ^ gen_poly) & 0x3FFF
            else:
                reg = (reg << 1) & 0x3FFF
        
        # Extract parity bits
        parity = np.zeros(14, dtype=np.uint8)
        for i in range(14):
            parity[i] = (reg >> (13 - i)) & 1
        
        return parity
    
    def encode_frame(self, frame_number: int = 0) -> np.ndarray:
        """
        Encode TPS for one frame.
        
        Args:
            frame_number: Frame number within superframe (0-3)
            
        Returns:
            Array of TPS bits per symbol (68 bits for 2K/8K, 16 for audio)
        """
        if self.mode == 'audio':
            return self._encode_frame_audio(frame_number)
        
        # Pack TPS data
        tps_data = self._pack_tps_data(frame_number)
        
        # Compute BCH parity
        parity = self._compute_bch_parity(tps_data[:53])
        tps_data[53:67] = parity
        
        # Symbol 0 is initialization (reference)
        # Symbols 1-67 carry the TPS data differentially
        frame_bits = np.zeros(68, dtype=np.uint8)
        frame_bits[0] = 0  # Reference bit
        
        # DBPSK: each bit is XOR of TPS data with previous transmitted bit
        for i in range(67):
            frame_bits[i + 1] = frame_bits[i] ^ tps_data[i]
        
        return frame_bits
    
    def _encode_frame_audio(self, frame_number: int) -> np.ndarray:
        """
        Encode simplified TPS for audio mode (16 symbols).
        
        Audio mode uses a simpler TPS structure:
        - Symbol 0: Reference
        - Symbols 1-4: Sync pattern
        - Symbols 5-6: Frame number
        - Symbols 7-8: Constellation
        - Symbols 9-11: Code rate
        - Symbols 12-13: Guard interval  
        - Symbols 14-15: Reserved/parity
        """
        frame_bits = np.zeros(16, dtype=np.uint8)
        frame_bits[0] = 0  # Reference
        
        # Simplified sync (4 bits)
        sync = self.SYNC_AUDIO_EVEN if frame_number % 2 == 0 else self.SYNC_AUDIO_ODD
        for i in range(4):
            frame_bits[1 + i] = (sync >> (3 - i)) & 1
        
        # Frame number (2 bits)
        for i in range(2):
            frame_bits[5 + i] = (frame_number >> (1 - i)) & 1
        
        # Constellation (2 bits)
        const_code = self.CONSTELLATION_CODES[self._params['constellation']]
        for i in range(2):
            frame_bits[7 + i] = (const_code >> (1 - i)) & 1
        
        # Code rate (3 bits)
        hp_code = self.CODE_RATE_CODES[self._params['hp_code_rate']]
        for i in range(3):
            frame_bits[9 + i] = (hp_code >> (2 - i)) & 1
        
        # Guard interval (2 bits)
        guard_code = self.GUARD_INTERVAL_CODES[self._params['guard_interval']]
        for i in range(2):
            frame_bits[12 + i] = (guard_code >> (1 - i)) & 1
        
        # Simple parity (XOR of all previous bits)
        frame_bits[14] = np.sum(frame_bits[:14]) % 2
        frame_bits[15] = frame_bits[14] ^ 1  # Complement for robustness
        
        return frame_bits
    
    def get_symbol_tps(self, symbol_index: int, 
                       frame_tps: np.ndarray) -> np.ndarray:
        """
        Get TPS bits for all TPS carriers in a symbol.
        
        In DVB-T, all TPS carriers in a symbol carry the same bit
        (with different PRBS modulation applied by pilot inserter).
        
        Args:
            symbol_index: Symbol number (0-67 for 2K/8K, 0-15 for audio)
            frame_tps: Frame TPS bits from encode_frame()
            
        Returns:
            Array of TPS bits for each TPS carrier
        """
        # Handle symbol index wrapping for different frame sizes
        idx = symbol_index % self.symbols_per_frame
        
        # All TPS carriers carry the same bit
        bit = frame_tps[idx] if idx < len(frame_tps) else 0
        return np.full(self.num_tps_carriers, bit, dtype=np.uint8)


class TPSDecoder:
    """
    DVB-T TPS decoder.
    
    Decodes TPS bits from received OFDM symbols to extract
    transmission parameters.
    """
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize TPS decoder.
        
        Args:
            mode: '2K', '8K', or 'audio'
        """
        self.mode = mode
        if mode == 'audio':
            self.num_tps_carriers = 4
        elif mode == '2K':
            self.num_tps_carriers = 17
        else:
            self.num_tps_carriers = 68
    
    def decode_frame(self, frame_tps: np.ndarray) -> Optional[Dict]:
        """
        Decode TPS from one frame.
        
        Args:
            frame_tps: Received TPS bits (68 bits for 2K/8K, 16 for audio)
            
        Returns:
            Dict of decoded parameters, or None if decode failed
        """
        if self.mode == 'audio':
            return self._decode_frame_audio(frame_tps)
        
        if len(frame_tps) != 68:
            return None
        
        # DBPSK demodulation: XOR consecutive bits
        tps_data = np.zeros(67, dtype=np.uint8)
        for i in range(67):
            tps_data[i] = frame_tps[i] ^ frame_tps[i + 1]
        
        # Check sync word
        sync = 0
        for i in range(16):
            sync = (sync << 1) | tps_data[i]
        
        if sync == TPSEncoder.SYNC_EVEN:
            frame_parity = 'even'
        elif sync == TPSEncoder.SYNC_ODD:
            frame_parity = 'odd'
        else:
            return None  # Invalid sync word
        
        # TODO: BCH error correction
        
        # Decode parameters
        def decode_bits(start, length):
            val = 0
            for i in range(length):
                val = (val << 1) | tps_data[start + i]
            return val
        
        const_code = decode_bits(24, 2)
        hier_code = decode_bits(26, 3)
        hp_code = decode_bits(29, 3)
        lp_code = decode_bits(32, 3)
        guard_code = decode_bits(35, 2)
        mode_code = decode_bits(37, 2)
        cell_id = decode_bits(39, 16)
        frame_num = decode_bits(22, 2)
        
        # Reverse lookup codes
        const_map = {v: k for k, v in TPSEncoder.CONSTELLATION_CODES.items()}
        hier_map = {v: k for k, v in TPSEncoder.HIERARCHY_CODES.items()}
        rate_map = {v: k for k, v in TPSEncoder.CODE_RATE_CODES.items()}
        guard_map = {v: k for k, v in TPSEncoder.GUARD_INTERVAL_CODES.items()}
        mode_map = {v: k for k, v in TPSEncoder.MODE_CODES.items()}
        
        return {
            'constellation': const_map.get(const_code, 'unknown'),
            'hierarchy': hier_map.get(hier_code, 'unknown'),
            'hp_code_rate': rate_map.get(hp_code, 'unknown'),
            'lp_code_rate': rate_map.get(lp_code, 'unknown'),
            'guard_interval': guard_map.get(guard_code, 'unknown'),
            'mode': mode_map.get(mode_code, 'unknown'),
            'cell_id': cell_id,
            'frame_number': frame_num,
        }
    
    def _decode_frame_audio(self, frame_tps: np.ndarray) -> Optional[Dict]:
        """Decode simplified TPS for audio mode."""
        if len(frame_tps) < 16:
            return None
        
        # Check sync pattern
        sync = 0
        for i in range(4):
            sync = (sync << 1) | frame_tps[1 + i]
        
        if sync == TPSEncoder.SYNC_AUDIO_EVEN:
            pass  # Valid even frame
        elif sync == TPSEncoder.SYNC_AUDIO_ODD:
            pass  # Valid odd frame
        else:
            return None  # Invalid sync
        
        def decode_bits(start, length):
            val = 0
            for i in range(length):
                val = (val << 1) | frame_tps[start + i]
            return val
        
        frame_num = decode_bits(5, 2)
        const_code = decode_bits(7, 2)
        hp_code = decode_bits(9, 3)
        guard_code = decode_bits(12, 2)
        
        const_map = {v: k for k, v in TPSEncoder.CONSTELLATION_CODES.items()}
        rate_map = {v: k for k, v in TPSEncoder.CODE_RATE_CODES.items()}
        guard_map = {v: k for k, v in TPSEncoder.GUARD_INTERVAL_CODES.items()}
        
        return {
            'constellation': const_map.get(const_code, 'QPSK'),
            'hierarchy': 'non-hierarchical',
            'hp_code_rate': rate_map.get(hp_code, '1/2'),
            'lp_code_rate': rate_map.get(hp_code, '1/2'),
            'guard_interval': guard_map.get(guard_code, '1/4'),
            'mode': 'audio',
            'cell_id': 0,
            'frame_number': frame_num,
        }
