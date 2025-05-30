//! Polynomial Evaluation Utilities
//!
//! This module provides efficient polynomial evaluation using the final tagless approach.
//! Horner's method reduces the number of multiplications and provides better numerical
//! stability compared to naive polynomial evaluation.

use crate::final_tagless::traits::{MathExpr, NumericType};
use std::ops::{Add, Mul, Sub};

/// Evaluate a polynomial using Horner's method
///
/// Given coefficients [a₀, a₁, a₂, ..., aₙ] representing the polynomial:
/// a₀ + a₁x + a₂x² + ... + aₙxⁿ
///
/// Horner's method evaluates this as:
/// a₀ + x(a₁ + x(a₂ + x(...)))
///
/// This reduces the number of multiplications from O(n²) to O(n) and
/// provides better numerical stability.
///
/// # Examples
///
/// ```rust
/// use mathcompile::final_tagless::{DirectEval, polynomial::horner};
///
/// // Evaluate 1 + 3x + 2x² at x = 2
/// let coeffs = [1.0, 3.0, 2.0]; // [constant, x, x²]
/// let x = DirectEval::var("x", 2.0);
/// let result = horner::<DirectEval, f64>(&coeffs, x);
/// assert_eq!(result, 15.0); // 1 + 3(2) + 2(4) = 15
/// ```
///
/// # Type Parameters
///
/// - `E`: The expression interpreter (`DirectEval`, `PrettyPrint`, etc.)
/// - `T`: The numeric type (f64, f32, etc.)
pub fn horner<E: MathExpr, T>(coeffs: &[T], x: E::Repr<T>) -> E::Repr<T>
where
    T: NumericType + Clone + Add<Output = T> + Mul<Output = T>,
    E::Repr<T>: Clone,
{
    if coeffs.is_empty() {
        return E::constant(T::default());
    }

    if coeffs.len() == 1 {
        return E::constant(coeffs[0].clone());
    }

    // Start with the highest degree coefficient (last in ascending order)
    let mut result = E::constant(coeffs[coeffs.len() - 1].clone());

    // Work backwards through the coefficients (from highest to lowest degree)
    for coeff in coeffs.iter().rev().skip(1) {
        result = E::add(E::mul(result, x.clone()), E::constant(coeff.clone()));
    }

    result
}

/// Evaluate a polynomial with explicit coefficients using Horner's method
///
/// This is a convenience function for when you want to specify coefficients
/// as expression representations rather than raw values.
///
/// # Examples
///
/// ```rust
/// use mathcompile::final_tagless::{DirectEval, MathExpr, polynomial::horner_expr};
///
/// // Evaluate 1 + 3x + 2x² at x = 2
/// let coeffs = [
///     DirectEval::constant(1.0), // constant term
///     DirectEval::constant(3.0), // x coefficient  
///     DirectEval::constant(2.0), // x² coefficient
/// ];
/// let x = DirectEval::var("x", 2.0);
/// let result = horner_expr::<DirectEval, f64>(&coeffs, x);
/// assert_eq!(result, 15.0);
/// ```
pub fn horner_expr<E: MathExpr, T>(coeffs: &[E::Repr<T>], x: E::Repr<T>) -> E::Repr<T>
where
    T: NumericType + Add<Output = T> + Mul<Output = T>,
    E::Repr<T>: Clone,
{
    if coeffs.is_empty() {
        return E::constant(T::default());
    }

    if coeffs.len() == 1 {
        return coeffs[0].clone();
    }

    // Start with the highest degree coefficient
    let mut result = coeffs[coeffs.len() - 1].clone();

    // Work backwards through the coefficients
    for coeff in coeffs.iter().rev().skip(1) {
        result = E::add(E::mul(result, x.clone()), coeff.clone());
    }

    result
}

/// Create a polynomial from its roots using the final tagless approach
///
/// Given roots [r₁, r₂, ..., rₙ], constructs the polynomial:
/// (x - r₁)(x - r₂)...(x - rₙ)
///
/// # Examples
///
/// ```rust
/// use mathcompile::final_tagless::{DirectEval, polynomial::from_roots};
///
/// // Create polynomial with roots at 1 and 2: (x-1)(x-2) = x² - 3x + 2
/// let roots = [1.0, 2.0];
/// let x = DirectEval::var("x", 0.0);
/// let poly = from_roots::<DirectEval, f64>(&roots, x);
/// // At x=0: (0-1)(0-2) = 2
/// assert_eq!(poly, 2.0);
/// ```
pub fn from_roots<E: MathExpr, T>(roots: &[T], x: E::Repr<T>) -> E::Repr<T>
where
    T: NumericType + Clone + Sub<Output = T> + num_traits::One,
    E::Repr<T>: Clone,
{
    if roots.is_empty() {
        return E::constant(num_traits::One::one());
    }

    let mut result = E::sub(x.clone(), E::constant(roots[0].clone()));

    for root in roots.iter().skip(1) {
        let factor = E::sub(x.clone(), E::constant(root.clone()));
        result = E::mul(result, factor);
    }

    result
}

/// Evaluate the derivative of a polynomial using Horner's method
///
/// Given coefficients [a₀, a₁, a₂, ..., aₙ] representing:
/// a₀ + a₁x + a₂x² + ... + aₙxⁿ
///
/// The derivative is: a₁ + 2a₂x + 3a₃x² + ... + naₙx^(n-1)
///
/// # Examples
///
/// ```rust
/// use mathcompile::final_tagless::{DirectEval, polynomial::horner_derivative};
///
/// // Derivative of 1 + 3x + 2x² is 3 + 4x
/// let coeffs = [1.0, 3.0, 2.0]; // [constant, x, x²]
/// let x = DirectEval::var("x", 2.0);
/// let result = horner_derivative::<DirectEval, f64>(&coeffs, x);
/// assert_eq!(result, 11.0); // 3 + 4(2) = 11
/// ```
pub fn horner_derivative<E: MathExpr, T>(coeffs: &[T], x: E::Repr<T>) -> E::Repr<T>
where
    T: NumericType + Clone + Add<Output = T> + Mul<Output = T> + num_traits::FromPrimitive,
    E::Repr<T>: Clone,
{
    if coeffs.len() <= 1 {
        return E::constant(T::default());
    }

    // Create derivative coefficients: [a₁, 2a₂, 3a₃, ...]
    let mut deriv_coeffs = Vec::with_capacity(coeffs.len() - 1);
    for (i, coeff) in coeffs.iter().enumerate().skip(1) {
        // Multiply coefficient by its power
        let power = num_traits::FromPrimitive::from_usize(i).unwrap_or_else(|| T::default());
        deriv_coeffs.push(coeff.clone() * power);
    }

    horner::<E, T>(&deriv_coeffs, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{DirectEval, PrettyPrint};

    #[test]
    fn test_horner_polynomial() {
        // Test polynomial: 1 + 2x + 3x^2 at x = 2
        // Expected: 1 + 2(2) + 3(4) = 17
        let coeffs = [1.0, 2.0, 3.0];
        let x = DirectEval::var("x", 2.0);
        let result = horner::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 17.0);
    }

    #[test]
    fn test_horner_pretty_print() {
        let coeffs = [1.0, 2.0, 3.0];
        let x = PrettyPrint::var("x");
        let result = horner::<PrettyPrint, f64>(&coeffs, x);
        assert!(result.contains('x'));
    }

    #[test]
    fn test_polynomial_from_roots() {
        // Polynomial with roots at 1 and 2: (x-1)(x-2) = x^2 - 3x + 2
        // At x=0: (0-1)(0-2) = 2
        let roots = [1.0, 2.0];
        let x = DirectEval::var("x", 0.0);
        let result = from_roots::<DirectEval, f64>(&roots, x);
        assert_eq!(result, 2.0);

        // At x=3: (3-1)(3-2) = 2*1 = 2
        let x = DirectEval::var("x", 3.0);
        let result = from_roots::<DirectEval, f64>(&roots, x);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_horner_expr() {
        // Test with expression coefficients
        let coeffs = [
            DirectEval::constant(1.0),
            DirectEval::constant(2.0),
            DirectEval::constant(3.0),
        ];
        let x = DirectEval::var("x", 2.0);
        let result = horner_expr::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 17.0); // Same as test_horner_polynomial
    }

    #[test]
    fn test_horner_derivative() {
        // Derivative of 1 + 3x + 2x² is 3 + 4x
        let coeffs = [1.0, 3.0, 2.0]; // [constant, x, x²]
        let x = DirectEval::var("x", 2.0);
        let result = horner_derivative::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 11.0); // 3 + 4(2) = 11
    }

    #[test]
    fn test_empty_polynomial() {
        let coeffs: &[f64] = &[];
        let x = DirectEval::var("x", 5.0);
        let result = horner::<DirectEval, f64>(coeffs, x);
        assert_eq!(result, 0.0); // Default value
    }

    #[test]
    fn test_single_coefficient() {
        let coeffs = [42.0];
        let x = DirectEval::var("x", 5.0);
        let result = horner::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 42.0); // Just the constant
    }

    #[test]
    fn test_polynomial_from_empty_roots() {
        let roots: &[f64] = &[];
        let x = DirectEval::var("x", 5.0);
        let result = from_roots::<DirectEval, f64>(roots, x);
        assert_eq!(result, 1.0); // Identity polynomial
    }

    #[test]
    fn test_derivative_of_constant() {
        let coeffs = [42.0]; // Just a constant
        let x = DirectEval::var("x", 5.0);
        let result = horner_derivative::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 0.0); // Derivative of constant is 0
    }

    #[test]
    fn test_derivative_of_linear() {
        let coeffs = [1.0, 2.0]; // 1 + 2x, derivative is 2
        let x = DirectEval::var("x", 5.0);
        let result = horner_derivative::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 2.0); // Derivative is constant 2
    }
}
