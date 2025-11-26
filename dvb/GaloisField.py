"""
Galois Field GF(2^8) Arithmetic

Reed-Solomon codes operate over finite fields (Galois fields).
DVB uses GF(2^8) with primitive polynomial p(x) = x^8 + x^4 + x^3 + x^2 + 1.

In GF(2^8):
- Addition is XOR
- Multiplication uses the primitive polynomial for reduction
- Each non-zero element can be represented as α^i for generator α

The field has 256 elements: {0, 1, α, α^2, ..., α^254}
where α^255 = 1 (cyclic).

Reference: ETSI EN 300 744 Section 4.3.2
"""

import numpy as np
from typing import Tuple


class GaloisField:
    """
    Galois Field GF(2^8) for Reed-Solomon coding.
    
    DVB RS codes use the primitive polynomial:
        p(x) = x^8 + x^4 + x^3 + x^2 + 1 = 0x11D
    
    The primitive element α = 0x02 generates all non-zero elements.
    
    Attributes:
        exp_table: Maps power -> element (α^i -> field element)
        log_table: Maps element -> power (field element -> i where α^i = element)
        
    Example:
        >>> gf = GaloisField()
        >>> product = gf.multiply(0x53, 0xCA)
        >>> assert gf.multiply(product, gf.inverse(0xCA)) == 0x53
    """
    
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
    PRIMITIVE_POLY = 0x11D
    
    # Primitive element (generator)
    PRIMITIVE = 0x02
    
    def __init__(self):
        """Initialize lookup tables for fast arithmetic."""
        self.exp_table, self.log_table = self._build_tables()
    
    def _build_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build exponent and logarithm lookup tables.
        
        exp_table[i] = α^i mod p(x)
        log_table[n] = i where α^i = n
        
        Returns:
            Tuple of (exp_table, log_table)
        """
        exp_table = np.zeros(512, dtype=np.uint8)  # Double size for easy modular access
        log_table = np.zeros(256, dtype=np.uint8)
        
        x = 1
        for i in range(255):
            exp_table[i] = x
            exp_table[i + 255] = x  # Wrap for easy modular arithmetic
            log_table[x] = i
            
            # Multiply by primitive element (α = 0x02)
            x <<= 1
            if x & 0x100:  # If overflow
                x ^= self.PRIMITIVE_POLY
        
        # log(0) is undefined, but we set it to 0 for convenience
        log_table[0] = 0
        
        return exp_table, log_table
    
    def add(self, a: int, b: int) -> int:
        """
        Add two field elements.
        
        In GF(2^n), addition is XOR.
        
        Args:
            a: First operand (0-255)
            b: Second operand (0-255)
            
        Returns:
            Sum a + b in GF(2^8)
        """
        return a ^ b
    
    # Subtraction is same as addition in GF(2^n)
    subtract = add
    
    def multiply(self, a: int, b: int) -> int:
        """
        Multiply two field elements.
        
        Uses log/exp tables: a * b = exp(log(a) + log(b))
        
        Args:
            a: First operand (0-255)
            b: Second operand (0-255)
            
        Returns:
            Product a * b in GF(2^8)
        """
        if a == 0 or b == 0:
            return 0
        
        log_sum = int(self.log_table[a]) + int(self.log_table[b])
        return int(self.exp_table[log_sum % 255])
    
    def multiply_slow(self, a: int, b: int) -> int:
        """
        Multiply using bit-by-bit method (educational).
        
        Implements polynomial multiplication with reduction.
        
        Args:
            a: First operand
            b: Second operand
            
        Returns:
            Product in GF(2^8)
        """
        result = 0
        
        while b:
            if b & 1:
                result ^= a
            
            # Multiply a by x (shift left)
            a <<= 1
            if a & 0x100:  # Reduce if overflow
                a ^= self.PRIMITIVE_POLY
            
            b >>= 1
        
        return result & 0xFF
    
    def divide(self, a: int, b: int) -> int:
        """
        Divide two field elements.
        
        a / b = a * inverse(b) = exp(log(a) - log(b))
        
        Args:
            a: Dividend (0-255)
            b: Divisor (1-255, non-zero)
            
        Returns:
            Quotient a / b in GF(2^8)
            
        Raises:
            ZeroDivisionError: If b is zero
        """
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(2^8)")
        if a == 0:
            return 0
        
        log_diff = int(self.log_table[a]) - int(self.log_table[b])
        return int(self.exp_table[log_diff % 255])
    
    def inverse(self, a: int) -> int:
        """
        Find multiplicative inverse of field element.
        
        inverse(a) = a^(-1) = a^253 = exp(255 - log(a))
        
        Args:
            a: Field element (1-255, non-zero)
            
        Returns:
            Inverse such that a * inverse(a) = 1
            
        Raises:
            ZeroDivisionError: If a is zero
        """
        if a == 0:
            raise ZeroDivisionError("Zero has no inverse")
        
        return int(self.exp_table[255 - self.log_table[a]])
    
    def power(self, a: int, n: int) -> int:
        """
        Raise field element to a power.
        
        a^n = exp(n * log(a))
        
        Args:
            a: Base (0-255)
            n: Exponent
            
        Returns:
            a^n in GF(2^8)
        """
        if a == 0:
            return 0 if n > 0 else 1
        
        log_a = int(self.log_table[a])
        return int(self.exp_table[(log_a * n) % 255])
    
    def exp(self, i: int) -> int:
        """
        Get α^i (exponent lookup).
        
        Args:
            i: Power
            
        Returns:
            α^i in GF(2^8)
        """
        return int(self.exp_table[i % 255])
    
    def log(self, a: int) -> int:
        """
        Get i where α^i = a (logarithm lookup).
        
        Args:
            a: Field element (1-255, non-zero)
            
        Returns:
            Power i
            
        Raises:
            ValueError: If a is zero
        """
        if a == 0:
            raise ValueError("Logarithm of zero is undefined")
        return int(self.log_table[a])
    
    def poly_multiply(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Multiply two polynomials over GF(2^8).
        
        Args:
            p1: Coefficients of first polynomial (low to high degree)
            p2: Coefficients of second polynomial
            
        Returns:
            Product polynomial coefficients
        """
        result = np.zeros(len(p1) + len(p2) - 1, dtype=np.uint8)
        
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                result[i + j] ^= self.multiply(int(c1), int(c2))
        
        return result
    
    def poly_eval(self, poly: np.ndarray, x: int) -> int:
        """
        Evaluate polynomial at point x using Horner's method.
        
        Args:
            poly: Coefficients (high degree first)
            x: Point to evaluate at
            
        Returns:
            Polynomial value at x
        """
        result = 0
        for coef in poly:
            result = self.add(self.multiply(result, x), int(coef))
        return result


# Global instance for convenience
_gf = None


def get_gf() -> GaloisField:
    """Get shared GaloisField instance."""
    global _gf
    if _gf is None:
        _gf = GaloisField()
    return _gf
