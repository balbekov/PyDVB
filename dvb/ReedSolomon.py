"""
Reed-Solomon RS(204,188) Outer Code

DVB uses a shortened RS(204,188,t=8) code derived from the mother
RS(255,239) code over GF(2^8).

Properties:
- Input: 188 bytes (transport stream packet)
- Output: 204 bytes (188 data + 16 parity)
- Correction capability: t=8 bytes (can correct up to 8 byte errors)
- Generator polynomial: g(x) = (x-α^0)(x-α^1)...(x-α^15)

The encoding is systematic: the 188 input bytes appear unchanged,
followed by 16 parity bytes.

Reference: ETSI EN 300 744 Section 4.3.2
"""

import numpy as np
from typing import Union, Tuple, Optional, List

from .GaloisField import GaloisField, get_gf


class ReedSolomon:
    """
    Reed-Solomon RS(204,188) encoder/decoder for DVB.
    
    This implements systematic RS encoding with 16 parity bytes,
    capable of correcting up to 8 byte errors.
    
    Attributes:
        n: Codeword length (204)
        k: Message length (188)
        t: Error correction capability (8)
        
    Example:
        >>> rs = ReedSolomon()
        >>> encoded = rs.encode(packet_188_bytes)
        >>> assert len(encoded) == 204
        >>> decoded, errors = rs.decode(encoded)
    """
    
    # RS(204,188) parameters
    N = 204  # Codeword length
    K = 188  # Message length
    T = 8    # Error correction capability
    PARITY = 16  # Number of parity bytes (2*T)
    
    # First consecutive root
    FCR = 0
    
    def __init__(self, use_fast: bool = True):
        """
        Initialize Reed-Solomon codec.
        
        Args:
            use_fast: Use optimized implementation (True) or 
                      educational step-by-step (False)
        """
        self.use_fast = use_fast
        self.gf = get_gf()
        
        # Generate generator polynomial
        self.generator = self._build_generator()
    
    def _build_generator(self) -> np.ndarray:
        """
        Build generator polynomial.
        
        g(x) = (x - α^0)(x - α^1)...(x - α^15)
             = x^16 + g15*x^15 + ... + g1*x + g0
        
        Returns:
            Generator polynomial coefficients (high degree first)
        """
        # Start with (x - α^0) = [1, α^0]
        g = np.array([1, self.gf.exp(self.FCR)], dtype=np.uint8)
        
        # Multiply by (x - α^i) for i = 1 to 15
        for i in range(1, 2 * self.T):
            root = self.gf.exp(self.FCR + i)
            factor = np.array([1, root], dtype=np.uint8)
            g = self.gf.poly_multiply(g, factor)
        
        return g
    
    def encode(self, message: Union[bytes, bytearray, np.ndarray]) -> bytes:
        """
        Encode message using systematic RS(204,188).
        
        The parity bytes are appended to the message.
        
        Args:
            message: 188-byte message
            
        Returns:
            204-byte codeword
        """
        if len(message) != self.K:
            raise ValueError(f"Message must be {self.K} bytes, got {len(message)}")
        
        if isinstance(message, np.ndarray):
            message = message.tobytes()
        
        message = np.array(list(message), dtype=np.uint8)
        
        if self.use_fast:
            parity = self._encode_fast(message)
        else:
            parity = self._encode_slow(message)
        
        return bytes(message) + bytes(parity)
    
    def _encode_fast(self, message: np.ndarray) -> np.ndarray:
        """
        Fast systematic encoding using shift register.
        
        Args:
            message: Message bytes as numpy array
            
        Returns:
            Parity bytes
        """
        # Shift register for division
        reg = np.zeros(self.PARITY, dtype=np.uint8)
        
        for byte in message:
            feedback = self.gf.add(byte, int(reg[0]))
            
            # Shift register
            for i in range(self.PARITY - 1):
                reg[i] = self.gf.add(
                    int(reg[i + 1]),
                    self.gf.multiply(feedback, int(self.generator[i + 1]))
                )
            
            reg[-1] = self.gf.multiply(feedback, int(self.generator[-1]))
        
        return reg
    
    def _encode_slow(self, message: np.ndarray) -> np.ndarray:
        """
        Educational encoding using polynomial division.
        
        Computes: parity = (message * x^16) mod generator
        
        Args:
            message: Message bytes
            
        Returns:
            Parity bytes
        """
        # Multiply message by x^16 (shift up)
        dividend = np.concatenate([message, np.zeros(self.PARITY, dtype=np.uint8)])
        
        # Polynomial division
        for i in range(self.K):
            if dividend[i] != 0:
                coef = dividend[i]
                for j, g in enumerate(self.generator):
                    dividend[i + j] = self.gf.add(
                        int(dividend[i + j]),
                        self.gf.multiply(coef, int(g))
                    )
        
        # Remainder is parity
        return dividend[self.K:]
    
    def decode(self, codeword: Union[bytes, bytearray, np.ndarray]) -> Tuple[bytes, int]:
        """
        Decode RS(204,188) codeword.
        
        Args:
            codeword: 204-byte codeword
            
        Returns:
            Tuple of (decoded_message, num_errors_corrected)
            Returns (-1) for num_errors if uncorrectable
        """
        if len(codeword) != self.N:
            raise ValueError(f"Codeword must be {self.N} bytes, got {len(codeword)}")
        
        if isinstance(codeword, np.ndarray):
            codeword = codeword.tobytes()
        
        codeword = np.array(list(codeword), dtype=np.uint8)
        
        # Calculate syndromes
        syndromes = self._calculate_syndromes(codeword)
        
        # Check if all syndromes are zero (no errors)
        if np.all(syndromes == 0):
            return bytes(codeword[:self.K]), 0
        
        # Find error locator polynomial using Berlekamp-Massey
        error_locator = self._berlekamp_massey(syndromes)
        
        if error_locator is None:
            return bytes(codeword[:self.K]), -1  # Uncorrectable
        
        # Find error locations using Chien search
        error_positions = self._chien_search(error_locator)
        
        if error_positions is None or len(error_positions) != len(error_locator) - 1:
            return bytes(codeword[:self.K]), -1  # Uncorrectable
        
        # Calculate error magnitudes using Forney algorithm
        error_magnitudes = self._forney_algorithm(syndromes, error_locator, error_positions)
        
        if error_magnitudes is None:
            return bytes(codeword[:self.K]), -1
        
        # Correct errors
        corrected = codeword.copy()
        for pos, mag in zip(error_positions, error_magnitudes):
            corrected[pos] = self.gf.add(int(corrected[pos]), mag)
        
        return bytes(corrected[:self.K]), len(error_positions)
    
    def _calculate_syndromes(self, codeword: np.ndarray) -> np.ndarray:
        """
        Calculate syndrome values.
        
        S_i = codeword(α^i) for i = 0 to 15
        
        Args:
            codeword: Received codeword
            
        Returns:
            Array of 16 syndrome values
        """
        syndromes = np.zeros(2 * self.T, dtype=np.uint8)
        
        for i in range(2 * self.T):
            root = self.gf.exp(self.FCR + i)
            syndromes[i] = self.gf.poly_eval(codeword, root)
        
        return syndromes
    
    def _berlekamp_massey(self, syndromes: np.ndarray) -> Optional[np.ndarray]:
        """
        Berlekamp-Massey algorithm to find error locator polynomial.
        
        Args:
            syndromes: Syndrome values
            
        Returns:
            Error locator polynomial coefficients, or None if uncorrectable
        """
        n = len(syndromes)
        
        # Initialize error locator polynomial C(x) = 1
        C = np.zeros(n + 1, dtype=np.uint8)
        C[0] = 1
        
        # Auxiliary polynomial B(x) = 1
        B = np.zeros(n + 1, dtype=np.uint8)
        B[0] = 1
        
        L = 0  # Current number of errors
        m = 1  # Number of iterations since L increased
        b = 1  # Previous discrepancy
        
        for r in range(n):
            # Calculate discrepancy
            d = int(syndromes[r])
            for i in range(1, L + 1):
                d = self.gf.add(d, self.gf.multiply(int(C[i]), int(syndromes[r - i])))
            
            if d == 0:
                m += 1
            elif 2 * L <= r:
                # L needs to increase
                T = C.copy()
                coef = self.gf.divide(d, b)
                
                for i in range(n - m):
                    C[m + i] = self.gf.add(int(C[m + i]), 
                                           self.gf.multiply(coef, int(B[i])))
                
                L = r + 1 - L
                B = T.copy()
                b = d
                m = 1
            else:
                # L stays the same
                coef = self.gf.divide(d, b)
                for i in range(n - m):
                    C[m + i] = self.gf.add(int(C[m + i]),
                                           self.gf.multiply(coef, int(B[i])))
                m += 1
        
        # Check if too many errors
        if L > self.T:
            return None
        
        # Trim polynomial to actual length
        return C[:L + 1]
    
    def _chien_search(self, error_locator: np.ndarray) -> Optional[List[int]]:
        """
        Chien search to find error positions.
        
        Evaluates error locator polynomial at all field elements
        to find roots (error positions).
        
        Args:
            error_locator: Error locator polynomial
            
        Returns:
            List of error positions, or None if search fails
        """
        positions = []
        
        for i in range(self.N):
            # Test position N-1-i (reverse order for codeword)
            # Evaluate Λ(α^(-i)) = Λ(α^(255-i))
            x = self.gf.exp(255 - i)
            
            result = 0
            for j, coef in enumerate(error_locator):
                result = self.gf.add(result, 
                                     self.gf.multiply(int(coef), self.gf.power(x, j)))
            
            if result == 0:
                positions.append(i)
        
        return positions if positions else None
    
    def _forney_algorithm(self, syndromes: np.ndarray, 
                          error_locator: np.ndarray,
                          error_positions: List[int]) -> Optional[List[int]]:
        """
        Forney algorithm to find error magnitudes.
        
        Args:
            syndromes: Syndrome values
            error_locator: Error locator polynomial
            error_positions: Error positions from Chien search
            
        Returns:
            List of error magnitudes
        """
        # Calculate error evaluator polynomial
        # Ω(x) = S(x) * Λ(x) mod x^(2t)
        omega = np.zeros(2 * self.T, dtype=np.uint8)
        
        for i in range(2 * self.T):
            for j in range(min(i + 1, len(error_locator))):
                omega[i] = self.gf.add(
                    int(omega[i]),
                    self.gf.multiply(int(error_locator[j]), int(syndromes[i - j]))
                )
        
        # Calculate formal derivative of error locator
        # Λ'(x) = Λ_1 + Λ_3*x^2 + Λ_5*x^4 + ... (odd coefficients)
        
        magnitudes = []
        
        for pos in error_positions:
            # X_i = α^(255-pos)
            X_inv = self.gf.exp(pos)  # α^pos = (α^(-pos))^(-1)
            
            # Evaluate Ω(X_i^-1)
            omega_val = 0
            x_pow = 1
            for coef in omega:
                omega_val = self.gf.add(omega_val, self.gf.multiply(int(coef), x_pow))
                x_pow = self.gf.multiply(x_pow, X_inv)
            
            # Evaluate Λ'(X_i^-1)
            lambda_deriv = 0
            x_pow = 1
            for j in range(1, len(error_locator), 2):  # Odd powers
                lambda_deriv = self.gf.add(
                    lambda_deriv,
                    self.gf.multiply(int(error_locator[j]), x_pow)
                )
                x_pow = self.gf.multiply(x_pow, self.gf.multiply(X_inv, X_inv))
            
            if lambda_deriv == 0:
                return None
            
            # Error magnitude: e_i = Ω(X_i^-1) / Λ'(X_i^-1)
            magnitude = self.gf.divide(omega_val, lambda_deriv)
            magnitudes.append(magnitude)
        
        return magnitudes
    
    def check(self, codeword: Union[bytes, bytearray, np.ndarray]) -> bool:
        """
        Check if codeword is valid (all syndromes are zero).
        
        Args:
            codeword: 204-byte codeword
            
        Returns:
            True if valid (no errors)
        """
        if len(codeword) != self.N:
            return False
        
        if isinstance(codeword, np.ndarray):
            codeword = codeword.tobytes()
        
        syndromes = self._calculate_syndromes(
            np.array(list(codeword), dtype=np.uint8)
        )
        
        return np.all(syndromes == 0)
