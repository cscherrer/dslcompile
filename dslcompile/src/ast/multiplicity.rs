//! Multiplicity type for exact coefficient and exponent arithmetic
//!
//! This module provides exact arithmetic for both coefficients (in Add operations)
//! and exponents (in Mul operations) using a hierarchy of Integer → Rational → Float
//! with smart normalization to preserve performance opportunities.

use std::{cmp::Ordering, fmt};

/// Represents multiplicities, coefficients, and exponents with exact arithmetic
///
/// Uses promotion hierarchy: Integer → Rational → Float to preserve exactness
/// as long as possible while enabling mathematical operations.
///
/// # Examples
///
/// ```rust
/// use dslcompile::ast::multiplicity::Multiplicity;
///
/// // Integer arithmetic stays exact
/// let a = Multiplicity::Integer(2);
/// let b = Multiplicity::Integer(3);
/// assert_eq!(a.add(b), Multiplicity::Integer(5));
///
/// // Mixed integer/rational promotes to rational
/// let c = Multiplicity::Rational(1, 3);  // 1/3
/// assert_eq!(a.add(c), Multiplicity::Rational(7, 3));  // 2 + 1/3 = 7/3
///
/// // Float normalization: 3.0 → 3 for performance
/// let d = Multiplicity::Float(3.0);
/// assert_eq!(d.normalize(), Multiplicity::Integer(3));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Multiplicity {
    /// Exact integer representation - enables optimizations like `powi()`
    Integer(i64),

    /// Exact rational representation - preserves fractions like 1/3
    Rational(i64, i64), // numerator, denominator (always in reduced form)

    /// Floating point representation - used when exactness not possible
    Float(f64),
}

impl Multiplicity {
    /// Create a zero multiplicity (additive identity)
    #[must_use]
    pub fn zero() -> Self {
        Multiplicity::Integer(0)
    }

    /// Create a one multiplicity (multiplicative identity)
    #[must_use]
    pub fn one() -> Self {
        Multiplicity::Integer(1)
    }

    /// Check if this multiplicity represents zero
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match self {
            Multiplicity::Integer(0) => true,
            Multiplicity::Rational(0, _) => true,
            Multiplicity::Float(f) => *f == 0.0,
            _ => false,
        }
    }

    /// Check if this multiplicity represents one
    #[must_use]
    pub fn is_one(&self) -> bool {
        match self {
            Multiplicity::Integer(1) => true,
            Multiplicity::Rational(n, d) => n == d,
            Multiplicity::Float(f) => *f == 1.0,
            _ => false,
        }
    }

    /// Convert to f64 for mixed arithmetic (when exactness not possible)
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        match self {
            Multiplicity::Integer(i) => *i as f64,
            Multiplicity::Rational(n, d) => (*n as f64) / (*d as f64),
            Multiplicity::Float(f) => *f,
        }
    }

    /// Create multiplicity from integer
    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        Multiplicity::Integer(value)
    }

    /// Create multiplicity from float with automatic normalization
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Multiplicity::Float(value).normalize()
    }

    /// Create rational multiplicity with automatic reduction
    #[must_use]
    pub fn from_rational(numerator: i64, denominator: i64) -> Self {
        assert!(
            (denominator != 0),
            "Cannot create rational with zero denominator"
        );
        Multiplicity::Rational(numerator, denominator).reduce()
    }

    /// Add two multiplicities with smart type promotion
    #[must_use]
    pub fn add(self, other: Multiplicity) -> Multiplicity {
        let result = match (self, other) {
            // Integer + Integer = Integer (stay exact)
            (Multiplicity::Integer(a), Multiplicity::Integer(b)) => {
                match a.checked_add(b) {
                    Some(sum) => Multiplicity::Integer(sum),
                    None => Multiplicity::Float(a as f64 + b as f64), // Overflow protection
                }
            }

            // Integer + Rational = Rational (stay exact)
            (Multiplicity::Integer(a), Multiplicity::Rational(n, d)) => {
                match a.checked_mul(d).and_then(|ad| ad.checked_add(n)) {
                    Some(numerator) => Multiplicity::Rational(numerator, d),
                    None => Multiplicity::Float(a as f64 + (n as f64) / (d as f64)),
                }
            }
            (Multiplicity::Rational(n, d), Multiplicity::Integer(a)) => {
                match a.checked_mul(d).and_then(|ad| ad.checked_add(n)) {
                    Some(numerator) => Multiplicity::Rational(numerator, d),
                    None => Multiplicity::Float((n as f64) / (d as f64) + a as f64),
                }
            }

            // Rational + Rational = Rational (stay exact)
            (Multiplicity::Rational(n1, d1), Multiplicity::Rational(n2, d2)) => {
                // (n1/d1) + (n2/d2) = (n1*d2 + n2*d1) / (d1*d2)
                match n1
                    .checked_mul(d2)
                    .zip(n2.checked_mul(d1))
                    .and_then(|(n1d2, n2d1)| n1d2.checked_add(n2d1))
                    .zip(d1.checked_mul(d2))
                {
                    Some((numerator, denominator)) => {
                        Multiplicity::Rational(numerator, denominator)
                    }
                    None => {
                        Multiplicity::Float((n1 as f64) / (d1 as f64) + (n2 as f64) / (d2 as f64))
                    }
                }
            }

            // Any + Float = Float (precision needed)
            (a, Multiplicity::Float(f)) | (Multiplicity::Float(f), a) => {
                Multiplicity::Float(a.to_f64() + f)
            }
        };

        result.reduce().normalize()
    }

    /// Multiply two multiplicities with smart type promotion
    #[must_use]
    pub fn multiply(self, other: Multiplicity) -> Multiplicity {
        let result = match (self, other) {
            // Integer * Integer = Integer (stay exact)
            (Multiplicity::Integer(a), Multiplicity::Integer(b)) => {
                match a.checked_mul(b) {
                    Some(product) => Multiplicity::Integer(product),
                    None => Multiplicity::Float(a as f64 * b as f64), // Overflow protection
                }
            }

            // Integer * Rational = Rational (stay exact)
            (Multiplicity::Integer(a), Multiplicity::Rational(n, d)) => match a.checked_mul(n) {
                Some(numerator) => Multiplicity::Rational(numerator, d),
                None => Multiplicity::Float(a as f64 * (n as f64) / (d as f64)),
            },
            (Multiplicity::Rational(n, d), Multiplicity::Integer(a)) => match a.checked_mul(n) {
                Some(numerator) => Multiplicity::Rational(numerator, d),
                None => Multiplicity::Float((n as f64) / (d as f64) * a as f64),
            },

            // Rational * Rational = Rational (stay exact)
            (Multiplicity::Rational(n1, d1), Multiplicity::Rational(n2, d2)) => {
                match n1.checked_mul(n2).zip(d1.checked_mul(d2)) {
                    Some((numerator, denominator)) => {
                        Multiplicity::Rational(numerator, denominator)
                    }
                    None => {
                        Multiplicity::Float((n1 as f64) / (d1 as f64) * (n2 as f64) / (d2 as f64))
                    }
                }
            }

            // Any * Float = Float (precision needed)
            (a, Multiplicity::Float(f)) | (Multiplicity::Float(f), a) => {
                Multiplicity::Float(a.to_f64() * f)
            }
        };

        result.reduce().normalize()
    }

    /// Normalize float values back to integers when possible (performance optimization)
    ///
    /// This is a post-processing optimization that doesn't affect egglog cost analysis
    /// but enables better code generation (e.g., powi vs powf).
    #[must_use]
    pub fn normalize(self) -> Self {
        match self {
            Multiplicity::Float(f) if can_normalize_to_integer(f) => {
                Multiplicity::Integer(f as i64)
            }
            other => other,
        }
    }

    /// Reduce rational to lowest terms
    #[must_use]
    pub fn reduce(self) -> Self {
        match self {
            Multiplicity::Rational(n, d) => {
                assert!((d != 0), "Rational with zero denominator");

                // Handle sign: ensure denominator is positive
                let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };

                // Handle exact division (rational → integer)
                if n % d == 0 {
                    return Multiplicity::Integer(n / d);
                }

                // Reduce to lowest terms
                let gcd = gcd(n.abs(), d.abs());
                Multiplicity::Rational(n / gcd, d / gcd)
            }
            other => other,
        }
    }
}

/// Check if a float can be safely normalized to an integer
fn can_normalize_to_integer(f: f64) -> bool {
    f.is_finite() &&                    // Not NaN or infinity
    f.fract() == 0.0 &&                 // No fractional part
    f >= i64::MIN as f64 &&             // Within i64 range
    f <= i64::MAX as f64
}

/// Compute greatest common divisor using Euclidean algorithm
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl fmt::Display for Multiplicity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Multiplicity::Integer(i) => write!(f, "{i}"),
            Multiplicity::Rational(n, d) => {
                if *d == 1 {
                    write!(f, "{n}")
                } else {
                    write!(f, "{n}/{d}")
                }
            }
            Multiplicity::Float(fl) => write!(f, "{fl}"),
        }
    }
}

impl PartialOrd for Multiplicity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Multiplicity {
    fn cmp(&self, other: &Self) -> Ordering {
        // Convert both to f64 for comparison
        // This is safe for ordering purposes even if not exact
        let self_f64 = self.to_f64();
        let other_f64 = other.to_f64();

        self_f64.partial_cmp(&other_f64).unwrap_or_else(|| {
            // Handle NaN cases consistently
            match (self_f64.is_nan(), other_f64.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater, // NaN sorts to end
                (false, true) => Ordering::Less,
                (false, false) => Ordering::Equal, // Should not happen
            }
        })
    }
}

impl Eq for Multiplicity {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_arithmetic() {
        let a = Multiplicity::Integer(2);
        let b = Multiplicity::Integer(3);

        assert_eq!(a.clone().add(b.clone()), Multiplicity::Integer(5));
        assert_eq!(a.multiply(b), Multiplicity::Integer(6));
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = Multiplicity::Integer(1);
        let b = Multiplicity::from_rational(1, 3); // 1/3

        // 1 + 1/3 = 4/3
        assert_eq!(a.add(b), Multiplicity::Rational(4, 3));

        // Test rational + rational
        let c = Multiplicity::from_rational(1, 2); // 1/2
        let d = Multiplicity::from_rational(1, 3); // 1/3
        // 1/2 + 1/3 = 3/6 + 2/6 = 5/6
        assert_eq!(c.add(d), Multiplicity::Rational(5, 6));
    }

    #[test]
    fn test_rational_reduction() {
        // 4/2 should reduce to 2
        let a = Multiplicity::from_rational(4, 2);
        assert_eq!(a, Multiplicity::Integer(2));

        // 6/9 should reduce to 2/3
        let b = Multiplicity::from_rational(6, 9);
        assert_eq!(b, Multiplicity::Rational(2, 3));
    }

    #[test]
    fn test_float_normalization() {
        // 3.0 should normalize to Integer(3)
        let a = Multiplicity::Float(3.0);
        assert_eq!(a.normalize(), Multiplicity::Integer(3));

        // 3.5 should stay as Float
        let b = Multiplicity::Float(3.5);
        assert_eq!(b.normalize(), Multiplicity::Float(3.5));
    }

    #[test]
    fn test_mixed_arithmetic() {
        let int = Multiplicity::Integer(2);
        let float = Multiplicity::Float(1.5);

        // Integer + Float should become Float
        assert_eq!(int.add(float), Multiplicity::Float(3.5));
    }

    #[test]
    fn test_zero_and_one() {
        assert!(Multiplicity::zero().is_zero());
        assert!(Multiplicity::one().is_one());
        assert!(Multiplicity::from_rational(0, 5).is_zero());
        assert!(Multiplicity::from_rational(3, 3).is_one());
    }

    #[test]
    fn test_normalization_after_arithmetic() {
        let a = Multiplicity::Float(1.0);
        let b = Multiplicity::Float(2.0);

        // 1.0 + 2.0 = 3.0 should normalize to Integer(3)
        let result = a.add(b);
        assert_eq!(result, Multiplicity::Integer(3));
    }

    #[test]
    fn test_ordering() {
        let a = Multiplicity::Integer(1);
        let b = Multiplicity::Float(1.5);
        let c = Multiplicity::Integer(2);

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Multiplicity::Integer(5)), "5");
        assert_eq!(format!("{}", Multiplicity::Rational(2, 3)), "2/3");
        assert_eq!(format!("{}", Multiplicity::Float(1.5)), "1.5");
        assert_eq!(format!("{}", Multiplicity::Rational(4, 1)), "4"); // Should simplify display
    }
}
