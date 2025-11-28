"""
DVB-T Parameter Detection

Auto-detects DVB-T signal parameters from received I/Q samples:
- Mode (2K or 8K)
- Guard interval (1/4, 1/8, 1/16, 1/32)
- Constellation (QPSK, 16QAM, 64QAM)
- Code rate (from TPS)

Detection methods:
1. Mode: FFT size detection via correlation analysis
2. Guard interval: Cyclic prefix correlation peak width
3. Constellation/code rate: TPS decoding

Reference: ETSI EN 300 744
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DVBTParameters:
    """Detected DVB-T parameters."""
    mode: str  # '2K' or '8K'
    guard_interval: str  # '1/4', '1/8', '1/16', '1/32'
    constellation: str  # 'QPSK', '16QAM', '64QAM'
    code_rate: str  # '1/2', '2/3', '3/4', '5/6', '7/8'
    hierarchy: str  # 'none', 'alpha1', 'alpha2', 'alpha4'
    sample_rate: float
    confidence: float  # 0-1


class DVBTDetector:
    """
    Auto-detect DVB-T signal parameters.
    
    Uses correlation analysis and TPS decoding to determine
    transmission parameters.
    
    Example:
        >>> detector = DVBTDetector()
        >>> params = detector.detect(iq_samples)
        >>> print(f"Mode: {params.mode}, Guard: {params.guard_interval}")
    """
    
    # DVB-T parameter options
    MODES = ['2K', '8K']
    GUARD_INTERVALS = ['1/4', '1/8', '1/16', '1/32']
    CONSTELLATIONS = ['QPSK', '16QAM', '64QAM']
    CODE_RATES = ['1/2', '2/3', '3/4', '5/6', '7/8']
    
    # Mode parameters
    MODE_PARAMS = {
        '2K': {'fft_size': 2048, 'active_carriers': 1705},
        '8K': {'fft_size': 8192, 'active_carriers': 6817},
    }
    
    # Guard interval fractions
    GI_FRACTIONS = {'1/4': 4, '1/8': 8, '1/16': 16, '1/32': 32, 'acoustic': 0.4}
    
    def __init__(self, sample_rate: float = 9142857.142857143):
        """
        Initialize detector.
        
        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    def detect(self, samples: np.ndarray) -> DVBTParameters:
        """
        Detect all DVB-T parameters.
        
        Args:
            samples: Input I/Q samples
            
        Returns:
            DVBTParameters with detected settings
        """
        # Step 1: Detect mode (FFT size)
        mode, mode_confidence = self.detect_mode(samples)
        
        # Step 2: Detect guard interval
        guard_interval, gi_confidence = self.detect_guard_interval(samples, mode)
        
        # Step 3: Try to detect constellation and code rate from TPS
        try:
            tps_params = self.detect_from_tps(samples, mode, guard_interval)
            constellation = tps_params.get('constellation', 'QPSK')
            code_rate = tps_params.get('code_rate', '1/2')
            hierarchy = tps_params.get('hierarchy', 'none')
            tps_confidence = tps_params.get('confidence', 0.5)
        except:
            constellation = 'QPSK'
            code_rate = '1/2'
            hierarchy = 'none'
            tps_confidence = 0.0
        
        # Overall confidence
        confidence = (mode_confidence + gi_confidence + tps_confidence) / 3
        
        return DVBTParameters(
            mode=mode,
            guard_interval=guard_interval,
            constellation=constellation,
            code_rate=code_rate,
            hierarchy=hierarchy,
            sample_rate=self.sample_rate,
            confidence=float(confidence)
        )
    
    def detect_mode(self, samples: np.ndarray) -> Tuple[str, float]:
        """
        Detect DVB-T mode (2K or 8K) from FFT size.
        
        Uses autocorrelation analysis to find the FFT size.
        
        Args:
            samples: Input I/Q samples
            
        Returns:
            Tuple of (mode, confidence)
        """
        correlations = {}
        
        for mode in self.MODES:
            fft_size = self.MODE_PARAMS[mode]['fft_size']
            
            # Test with various guard intervals
            best_corr = 0
            for gi in self.GUARD_INTERVALS:
                gi_frac = self.GI_FRACTIONS[gi]
                guard_len = fft_size // gi_frac
                
                corr = self._compute_gi_correlation(samples, fft_size, guard_len)
                best_corr = max(best_corr, corr)
            
            correlations[mode] = best_corr
        
        # Select mode with highest correlation
        best_mode = max(correlations, key=correlations.get)
        
        # Confidence based on correlation ratio
        total = sum(correlations.values())
        if total > 0:
            confidence = correlations[best_mode] / total
        else:
            confidence = 0.5
        
        return best_mode, float(confidence)
    
    def detect_guard_interval(self, samples: np.ndarray,
                              mode: str) -> Tuple[str, float]:
        """
        Detect guard interval from cyclic prefix correlation.
        
        Args:
            samples: Input I/Q samples
            mode: Detected mode ('2K' or '8K')
            
        Returns:
            Tuple of (guard_interval, confidence)
        """
        fft_size = self.MODE_PARAMS[mode]['fft_size']
        correlations = {}
        
        for gi in self.GUARD_INTERVALS:
            gi_frac = self.GI_FRACTIONS[gi]
            guard_len = fft_size // gi_frac
            
            corr = self._compute_gi_correlation(samples, fft_size, guard_len)
            correlations[gi] = corr
        
        # Select guard interval with highest correlation
        best_gi = max(correlations, key=correlations.get)
        
        # Confidence
        total = sum(correlations.values())
        if total > 0:
            confidence = correlations[best_gi] / total
        else:
            confidence = 0.25
        
        return best_gi, float(confidence)
    
    def _compute_gi_correlation(self, samples: np.ndarray,
                                fft_size: int,
                                guard_len: int) -> float:
        """
        Compute guard interval correlation strength.
        
        Args:
            samples: Input samples
            fft_size: FFT size to test
            guard_len: Guard interval length
            
        Returns:
            Correlation strength (0-1)
        """
        symbol_len = fft_size + guard_len
        
        if len(samples) < 2 * symbol_len:
            return 0.0
        
        # Use multiple symbols for better estimate
        num_symbols = min(10, len(samples) // symbol_len - 1)
        
        total_corr = 0.0
        total_power = 0.0
        
        for sym in range(num_symbols):
            start = sym * symbol_len
            
            if start + symbol_len + fft_size > len(samples):
                break
            
            # Guard interval (start of symbol)
            guard = samples[start:start + guard_len]
            
            # End of symbol (should match guard)
            end = samples[start + fft_size:start + fft_size + guard_len]
            
            # Correlation
            corr = np.abs(np.sum(guard * np.conj(end)))
            power = np.sqrt(np.sum(np.abs(guard) ** 2) * np.sum(np.abs(end) ** 2))
            
            total_corr += corr
            total_power += power
        
        if total_power > 0:
            return total_corr / total_power
        return 0.0
    
    def detect_from_tps(self, samples: np.ndarray,
                        mode: str,
                        guard_interval: str) -> Dict:
        """
        Detect parameters from TPS carriers.
        
        Args:
            samples: Input I/Q samples
            mode: DVB-T mode
            guard_interval: Guard interval
            
        Returns:
            Dictionary with constellation, code_rate, etc.
        """
        from .Synchronizer import CoarseSync
        from .OFDM import OFDMDemodulator
        from .GuardInterval import GuardIntervalRemover
        from .TPS import TPSDecoder
        from .Pilots import TPSPilots, PilotGenerator
        
        fft_size = self.MODE_PARAMS[mode]['fft_size']
        gi_frac = self.GI_FRACTIONS[guard_interval]
        guard_len = fft_size // gi_frac
        symbol_len = fft_size + guard_len
        
        # Synchronize
        sync = CoarseSync(mode, guard_interval, self.sample_rate)
        symbol_start, _ = sync.find_symbol_start(samples)
        
        # CFO correction
        cfo = sync.estimate_coarse_cfo(samples, symbol_start)
        samples = sync.correct_cfo(samples, cfo)
        
        # Demodulate several symbols
        ofdm_demod = OFDMDemodulator(mode)
        gi_remover = GuardIntervalRemover(guard_interval, fft_size)
        tps_pilots = TPSPilots(mode)
        pilot_gen = PilotGenerator(mode)
        
        # Collect TPS bits
        tps_bits_all = []
        
        for sym_idx in range(min(68, (len(samples) - symbol_start) // symbol_len)):
            start = symbol_start + sym_idx * symbol_len
            
            if start + symbol_len > len(samples):
                break
            
            symbol = samples[start:start + symbol_len]
            useful = gi_remover.remove(symbol)
            carriers = ofdm_demod.demodulate(useful)
            
            # Extract TPS carriers
            tps_pos = tps_pilots.get_positions()
            tps_pos = tps_pos[tps_pos < len(carriers)]
            tps_values = carriers[tps_pos]
            
            # Demodulate TPS (DBPSK)
            if sym_idx == 0:
                prev_tps = pilot_gen.get_pilot_values(tps_pos, boost=True)
            
            tps_bits = (np.real(tps_values * np.conj(prev_tps)) < 0).astype(np.uint8)
            tps_bits_all.append(tps_bits)
            prev_tps = tps_values
        
        # Decode TPS
        if len(tps_bits_all) < 68:
            return {'confidence': 0.0}
        
        try:
            tps_decoder = TPSDecoder(mode)
            params = tps_decoder.decode_frame(np.array(tps_bits_all[:68]))
            params['confidence'] = 0.8
            return params
        except:
            return {'confidence': 0.0}
    
    def detect_constellation(self, symbols: np.ndarray) -> Tuple[str, float]:
        """
        Detect constellation from demodulated symbols.
        
        Analyzes the symbol distribution to determine QPSK/16QAM/64QAM.
        
        Args:
            symbols: Complex symbols (after equalization)
            
        Returns:
            Tuple of (constellation, confidence)
        """
        # Normalize symbols
        symbols = symbols / (np.std(symbols) + 1e-10)
        
        # Compute amplitude statistics
        amplitudes = np.abs(symbols)
        unique_amplitudes = len(np.unique(np.round(amplitudes, 1)))
        
        # QPSK: 1 amplitude level
        # 16QAM: ~3 amplitude levels
        # 64QAM: ~9 amplitude levels
        
        if unique_amplitudes <= 2:
            return 'QPSK', 0.7
        elif unique_amplitudes <= 5:
            return '16QAM', 0.7
        else:
            return '64QAM', 0.7


def detect_dvbt_parameters(samples: np.ndarray,
                           sample_rate: float = 9142857.142857143) -> DVBTParameters:
    """
    Convenience function to detect DVB-T parameters.
    
    Args:
        samples: Input I/Q samples
        sample_rate: Sample rate in Hz
        
    Returns:
        DVBTParameters with detected settings
    """
    detector = DVBTDetector(sample_rate)
    return detector.detect(samples)


def verify_dvbt_signal(samples: np.ndarray,
                       sample_rate: float = 9142857.142857143) -> bool:
    """
    Verify if samples contain a DVB-T signal.
    
    Args:
        samples: Input I/Q samples
        sample_rate: Sample rate in Hz
        
    Returns:
        True if DVB-T signal detected
    """
    detector = DVBTDetector(sample_rate)
    
    # Try to detect mode
    mode, confidence = detector.detect_mode(samples)
    
    # If correlation is reasonable, likely DVB-T
    return confidence > 0.6
