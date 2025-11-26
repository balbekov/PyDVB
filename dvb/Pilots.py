"""
DVB-T Pilot Carriers

DVB-T uses three types of pilot carriers for synchronization and
channel estimation:

1. Continual Pilots: Fixed carrier positions in every symbol
   - 45 carriers in 2K mode, 177 in 8K mode
   - Used for frequency/phase tracking

2. Scattered Pilots: Rotating pattern across symbols
   - Every 12th carrier, position shifts by 3 each symbol
   - Used for channel estimation

3. TPS (Transmission Parameter Signaling) Pilots:
   - 17 carriers in 2K mode, 68 in 8K mode
   - Carry transmission parameters

All pilots are BPSK modulated with boosted power (4/3 of data carriers).

Reference: ETSI EN 300 744 Section 4.5
"""

import numpy as np
from typing import List, Set, Tuple


# Pilot amplitude boost factor
PILOT_BOOST = 4.0 / 3.0

# PRBS generator for pilot modulation
# Polynomial: x^11 + x^2 + 1
PRBS_POLY = 0b100000000101


class PilotGenerator:
    """
    Generator for DVB-T pilot values.
    
    Uses PRBS sequence to generate BPSK pilot values.
    The same PRBS sequence is used for all pilot types.
    """
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize pilot generator.
        
        Args:
            mode: '2K' or '8K'
        """
        self.mode = mode
        self.max_carrier = {'2K': 1705, '8K': 6817}[mode]
        
        # Generate PRBS sequence for all carriers
        self._prbs = self._generate_prbs()
    
    def _generate_prbs(self) -> np.ndarray:
        """
        Generate PRBS sequence.
        
        Uses polynomial x^11 + x^2 + 1, initialized to all 1s.
        """
        prbs = np.zeros(self.max_carrier, dtype=np.int8)
        
        reg = 0x7FF  # 11 bits, all 1s
        
        for k in range(self.max_carrier):
            prbs[k] = (reg & 1)
            
            # LFSR feedback: bit10 XOR bit1
            feedback = ((reg >> 10) ^ (reg >> 1)) & 1
            reg = ((reg << 1) | feedback) & 0x7FF
        
        return prbs
    
    def get_pilot_value(self, carrier: int, boost: bool = True) -> complex:
        """
        Get BPSK pilot value for a carrier.
        
        Args:
            carrier: Carrier index (0 to max_carrier-1)
            boost: Apply amplitude boost
            
        Returns:
            Complex BPSK value (+/- amplitude)
        """
        # BPSK: +1 or -1 based on PRBS
        value = 1 - 2 * self._prbs[carrier]
        
        if boost:
            value *= PILOT_BOOST
        
        return complex(value, 0)
    
    def get_pilot_values(self, carriers: np.ndarray, 
                         boost: bool = True) -> np.ndarray:
        """
        Get pilot values for multiple carriers.
        
        Args:
            carriers: Array of carrier indices
            boost: Apply amplitude boost
            
        Returns:
            Complex array of pilot values
        """
        values = (1 - 2 * self._prbs[carriers]).astype(np.complex64)
        
        if boost:
            values *= PILOT_BOOST
        
        return values


class ContinualPilots:
    """
    DVB-T continual pilot positions.
    
    Continual pilots are at fixed carrier positions in every OFDM symbol.
    They are used for fine frequency and phase tracking.
    """
    
    # Continual pilot positions for 2K mode (45 pilots)
    POSITIONS_2K = np.array([
        0, 48, 54, 87, 141, 156, 192, 201, 255, 279, 282, 333, 432, 450,
        483, 525, 531, 618, 636, 714, 759, 765, 780, 804, 873, 888, 918,
        939, 942, 969, 984, 1050, 1101, 1107, 1110, 1137, 1140, 1146, 1206,
        1269, 1323, 1377, 1491, 1683, 1704
    ], dtype=np.int32)
    
    # Continual pilot positions for 8K mode (177 pilots)
    POSITIONS_8K = np.array([
        0, 48, 54, 87, 141, 156, 192, 201, 255, 279, 282, 333, 432, 450,
        483, 525, 531, 618, 636, 714, 759, 765, 780, 804, 873, 888, 918,
        939, 942, 969, 984, 1050, 1101, 1107, 1110, 1137, 1140, 1146, 1206,
        1269, 1323, 1377, 1491, 1683, 1704, 1752, 1758, 1791, 1845, 1860,
        1896, 1905, 1959, 1983, 1986, 2037, 2136, 2154, 2187, 2229, 2235,
        2322, 2340, 2418, 2463, 2469, 2484, 2508, 2577, 2592, 2622, 2643,
        2646, 2673, 2688, 2754, 2805, 2811, 2814, 2841, 2844, 2850, 2910,
        2973, 3027, 3081, 3195, 3387, 3408, 3456, 3462, 3495, 3549, 3564,
        3600, 3609, 3663, 3687, 3690, 3741, 3840, 3858, 3891, 3933, 3939,
        4026, 4044, 4122, 4167, 4173, 4188, 4212, 4281, 4296, 4326, 4347,
        4350, 4377, 4392, 4458, 4509, 4515, 4518, 4545, 4548, 4554, 4614,
        4677, 4731, 4785, 4899, 5091, 5112, 5160, 5166, 5199, 5253, 5268,
        5304, 5313, 5367, 5391, 5394, 5445, 5544, 5562, 5595, 5637, 5643,
        5730, 5748, 5826, 5871, 5877, 5892, 5916, 5985, 6000, 6030, 6051,
        6054, 6081, 6096, 6162, 6213, 6219, 6222, 6249, 6252, 6258, 6318,
        6381, 6435, 6489, 6603, 6795, 6816
    ], dtype=np.int32)
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize continual pilots.
        
        Args:
            mode: '2K' or '8K'
        """
        self.mode = mode
        self.positions = self.POSITIONS_2K if mode == '2K' else self.POSITIONS_8K
        self._position_set = set(self.positions)
    
    def is_continual_pilot(self, carrier: int) -> bool:
        """Check if carrier is a continual pilot."""
        return carrier in self._position_set
    
    def get_positions(self) -> np.ndarray:
        """Get all continual pilot positions."""
        return self.positions.copy()


class ScatteredPilots:
    """
    DVB-T scattered pilot positions.
    
    Scattered pilots rotate their positions every symbol:
    - Symbol l: positions 3*(l mod 4) + 12*k for k = 0, 1, 2, ...
    - They cover all carriers over 4 consecutive symbols
    """
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize scattered pilots.
        
        Args:
            mode: '2K' or '8K'
        """
        self.mode = mode
        self.max_carrier = {'2K': 1705, '8K': 6817}[mode]
    
    def get_positions(self, symbol_index: int) -> np.ndarray:
        """
        Get scattered pilot positions for a symbol.
        
        Args:
            symbol_index: Symbol number within frame (0-67)
            
        Returns:
            Array of carrier indices
        """
        # Starting offset depends on symbol number mod 4
        offset = 3 * (symbol_index % 4)
        
        # Positions: offset + 12*k for valid k
        positions = np.arange(offset, self.max_carrier, 12, dtype=np.int32)
        
        return positions
    
    def is_scattered_pilot(self, carrier: int, symbol_index: int) -> bool:
        """Check if carrier is a scattered pilot in this symbol."""
        offset = 3 * (symbol_index % 4)
        return (carrier - offset) % 12 == 0 and carrier < self.max_carrier


class TPSPilots:
    """
    DVB-T TPS (Transmission Parameter Signaling) pilot positions.
    
    TPS pilots carry signaling information about transmission parameters.
    17 carriers in 2K mode, 68 in 8K mode.
    """
    
    # TPS carrier positions for 2K mode
    POSITIONS_2K = np.array([
        34, 50, 209, 346, 413, 569, 595, 688, 790, 901, 1073, 1219,
        1262, 1286, 1469, 1594, 1687
    ], dtype=np.int32)
    
    # TPS carrier positions for 8K mode
    POSITIONS_8K = np.array([
        34, 50, 209, 346, 413, 569, 595, 688, 790, 901, 1073, 1219,
        1262, 1286, 1469, 1594, 1687, 1738, 1754, 1913, 2050, 2117,
        2273, 2299, 2392, 2494, 2605, 2777, 2923, 2966, 2990, 3173,
        3298, 3391, 3442, 3458, 3617, 3754, 3821, 3977, 4003, 4096,
        4198, 4309, 4481, 4627, 4670, 4694, 4877, 5002, 5095, 5146,
        5162, 5321, 5458, 5525, 5681, 5707, 5800, 5902, 6013, 6185,
        6331, 6374, 6398, 6581, 6706, 6799
    ], dtype=np.int32)
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize TPS pilots.
        
        Args:
            mode: '2K' or '8K'
        """
        self.mode = mode
        self.positions = self.POSITIONS_2K if mode == '2K' else self.POSITIONS_8K
        self._position_set = set(self.positions)
    
    def is_tps_pilot(self, carrier: int) -> bool:
        """Check if carrier is a TPS pilot."""
        return carrier in self._position_set
    
    def get_positions(self) -> np.ndarray:
        """Get all TPS pilot positions."""
        return self.positions.copy()


class PilotInserter:
    """
    Complete pilot insertion for DVB-T OFDM symbols.
    
    Combines continual, scattered, and TPS pilots with data carriers
    to build complete OFDM symbols.
    
    Example:
        >>> inserter = PilotInserter('2K')
        >>> symbol = inserter.insert(data_carriers, tps_bits, symbol_idx)
    """
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize pilot inserter.
        
        Args:
            mode: '2K' or '8K'
        """
        self.mode = mode
        self.fft_size = {'2K': 2048, '8K': 8192}[mode]
        self.active_carriers = {'2K': 1705, '8K': 6817}[mode]
        
        # Initialize pilot components
        self.pilot_gen = PilotGenerator(mode)
        self.continual = ContinualPilots(mode)
        self.scattered = ScatteredPilots(mode)
        self.tps = TPSPilots(mode)
        
        # Calculate data carrier count per symbol
        self._calc_data_carriers()
    
    def _calc_data_carriers(self) -> None:
        """Calculate number of data carriers per symbol."""
        # Varies by symbol due to scattered pilot rotation
        self.data_carriers_per_symbol = []
        
        for sym in range(4):  # Pattern repeats every 4 symbols
            pilot_count = 0
            
            # Count all pilots
            for k in range(self.active_carriers):
                if self.continual.is_continual_pilot(k):
                    pilot_count += 1
                elif self.scattered.is_scattered_pilot(k, sym):
                    pilot_count += 1
                elif self.tps.is_tps_pilot(k):
                    pilot_count += 1
            
            self.data_carriers_per_symbol.append(self.active_carriers - pilot_count)
    
    def get_data_carrier_count(self, symbol_index: int) -> int:
        """Get number of data carriers for a symbol."""
        return self.data_carriers_per_symbol[symbol_index % 4]
    
    def insert(self, data: np.ndarray, tps_bits: np.ndarray,
               symbol_index: int) -> np.ndarray:
        """
        Insert pilots and data into OFDM symbol.
        
        Args:
            data: Data carrier values (complex)
            tps_bits: TPS bits for this symbol (one bit per TPS carrier)
            symbol_index: Symbol number within frame
            
        Returns:
            Array of active carrier values (length = active_carriers)
        """
        carriers = np.zeros(self.active_carriers, dtype=np.complex64)
        data_idx = 0
        tps_idx = 0
        
        scattered_positions = set(self.scattered.get_positions(symbol_index))
        
        for k in range(self.active_carriers):
            if self.continual.is_continual_pilot(k):
                carriers[k] = self.pilot_gen.get_pilot_value(k)
            elif k in scattered_positions:
                carriers[k] = self.pilot_gen.get_pilot_value(k)
            elif self.tps.is_tps_pilot(k):
                # TPS pilots are DBPSK with reference to first TPS carrier
                if tps_idx < len(tps_bits):
                    # BPSK based on TPS bit, boosted
                    tps_val = 1 - 2 * int(tps_bits[tps_idx])
                    carriers[k] = complex(tps_val * PILOT_BOOST, 0)
                    tps_idx += 1
                else:
                    carriers[k] = complex(PILOT_BOOST, 0)
            else:
                # Data carrier
                if data_idx < len(data):
                    carriers[k] = data[data_idx]
                    data_idx += 1
        
        return carriers
    
    def get_carrier_types(self, symbol_index: int) -> np.ndarray:
        """
        Get carrier type for each position.
        
        Returns:
            Array of type codes:
            0 = data, 1 = continual pilot, 2 = scattered pilot, 3 = TPS
        """
        types = np.zeros(self.active_carriers, dtype=np.uint8)
        
        scattered_positions = set(self.scattered.get_positions(symbol_index))
        
        for k in range(self.active_carriers):
            if self.continual.is_continual_pilot(k):
                types[k] = 1
            elif k in scattered_positions:
                types[k] = 2
            elif self.tps.is_tps_pilot(k):
                types[k] = 3
            # else: types[k] = 0 (data)
        
        return types


class PilotExtractor:
    """
    Extract pilots from received OFDM symbols.
    
    Used in receiver for channel estimation.
    """
    
    def __init__(self, mode: str = '2K'):
        """
        Initialize pilot extractor.
        
        Args:
            mode: '2K' or '8K'
        """
        self.mode = mode
        self.pilot_inserter = PilotInserter(mode)
    
    def extract(self, carriers: np.ndarray, 
                symbol_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data and pilot carriers from OFDM symbol.
        
        Args:
            carriers: Received carrier values
            symbol_index: Symbol number within frame
            
        Returns:
            Tuple of (data_carriers, pilot_carriers, pilot_positions)
        """
        types = self.pilot_inserter.get_carrier_types(symbol_index)
        
        data_mask = types == 0
        pilot_mask = (types == 1) | (types == 2)  # Continual and scattered
        
        data = carriers[data_mask]
        pilots = carriers[pilot_mask]
        pilot_positions = np.where(pilot_mask)[0]
        
        return data, pilots, pilot_positions
