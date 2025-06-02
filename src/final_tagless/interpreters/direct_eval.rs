//! Direct Evaluation Interpreter
//!
//! This interpreter provides immediate evaluation of mathematical expressions using native Rust
//! operations. It represents expressions directly as their computed values (`type Repr<T> = T`),
//! making it the simplest and most straightforward interpreter implementation.

use crate::final_tagless::traits::{MathExpr, NumericType, StatisticalExpr};
use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Direct evaluation interpreter for immediate computation
///
/// This interpreter provides immediate evaluation of mathematical expressions using native Rust
/// operations. It represents expressions directly as their computed values (`type Repr<T> = T`),
/// making it the simplest and most straightforward interpreter implementation.
///
/// # Characteristics
///
/// - **Zero overhead**: Direct mapping to native Rust operations
/// - **Immediate evaluation**: No intermediate representation or compilation step
/// - **Type preservation**: Works with any numeric type that implements required traits
/// - **Reference implementation**: Serves as the canonical behavior for other interpreters
///
/// # Usage Patterns
///
/// ## Simple Expression Evaluation
///
/// ```rust
/// use mathcompile::final_tagless::{DirectEval, MathExpr};
///
/// // Define a mathematical function
/// fn polynomial<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
/// where
///     E::Repr<f64>: Clone,
/// {
///     // 3x² + 2x + 1
///     let x_squared = E::pow(x.clone(), E::constant(2.0));
///     let three_x_squared = E::mul(E::constant(3.0), x_squared);
///     let two_x = E::mul(E::constant(2.0), x);
///     E::add(E::add(three_x_squared, two_x), E::constant(1.0))
/// }
///
/// // Evaluate directly with a specific value
/// let result = polynomial::<DirectEval>(DirectEval::var_with_value(0, 2.0));
/// assert_eq!(result, 17.0); // 3(4) + 2(2) + 1 = 17
/// ```
///
/// ## Working with Different Numeric Types
///
/// ```rust
/// use mathcompile::final_tagless::{DirectEval, MathExpr};
///
/// // Generic function works with any numeric type
/// fn simple_add<E: MathExpr, T>(a: E::Repr<T>, b: E::Repr<T>) -> E::Repr<T>
/// where
///     T: mathcompile::final_tagless::NumericType + std::ops::Add<Output = T>,
/// {
///     E::add(a, b)
/// }
///
/// // Works with f64
/// let result_f64 = simple_add::<DirectEval, f64>(5.0, 3.0);
/// assert_eq!(result_f64, 8.0);
///
/// // Works with f32
/// let result_f32 = simple_add::<DirectEval, f32>(5.0_f32, 3.0_f32);
/// assert_eq!(result_f32, 8.0_f32);
/// ```
///
/// # Variable Handling
///
/// For `DirectEval`, variables don't make sense in the traditional sense since we evaluate
/// immediately. Instead, use constants or the `var_with_value` helper function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectEval;

impl DirectEval {
    /// Create a variable with a specific value for direct evaluation
    ///
    /// Since `DirectEval` evaluates immediately, "variables" are just the values
    /// you want to substitute. The index parameter is ignored.
    #[must_use]
    pub fn var_with_value<T: NumericType>(_index: usize, value: T) -> T {
        value
    }

    /// Evaluate an `ASTRepr` expression with variables provided as a vector
    /// This is needed by the summation module for evaluating generated expressions
    #[must_use]
    pub fn eval_with_vars<T: NumericType + Float + Copy>(
        expr: &crate::ast::ASTRepr<T>,
        variables: &[T],
    ) -> T {
        expr.eval_with_vars(variables)
    }

    /// Evaluate a two-variable `ASTRepr` expression with specific values
    /// This is needed by various parts of the codebase for evaluation
    #[must_use]
    pub fn eval_two_vars(expr: &crate::ast::ASTRepr<f64>, x: f64, y: f64) -> f64 {
        expr.eval_two_vars(x, y)
    }
}

impl MathExpr for DirectEval {
    type Repr<T> = T;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        value
    }

    fn var<T: NumericType>(_index: usize) -> Self::Repr<T> {
        T::default()
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left + right
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left - right
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left * right
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        left / right
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        // Domain-aware power evaluation to prevent NaN results
        let result = base.powf(exp);
        if result.is_finite() && !result.is_nan() {
            result
        } else {
            // For problematic cases, try to compute a reasonable result
            // This handles cases like negative base with non-integer exponent
            if base < T::zero() {
                // For negative bases with non-integer exponents, the result is undefined in reals
                // Return NaN to indicate this, but this should be caught by domain analysis
                T::nan()
            } else {
                result // Return the original result even if it's inf/nan
            }
        }
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        -expr
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.ln()
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.exp()
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.sqrt()
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.sin()
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        expr.cos()
    }
}

// Implement StatisticalExpr for DirectEval
impl StatisticalExpr for DirectEval {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_eval_basic() {
        // Test basic arithmetic
        let result = DirectEval::add(5.0, 3.0);
        assert_eq!(result, 8.0);

        let result = DirectEval::mul(4.0, 2.5);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_direct_eval_transcendental() {
        use std::f64::consts::PI;

        // Test sin
        let result = DirectEval::sin(PI / 2.0);
        assert!((result - 1.0).abs() < 1e-15);

        // Test exp and ln
        let x = 2.0;
        let result = DirectEval::ln(DirectEval::exp(x));
        assert!((result - x).abs() < 1e-15);
    }

    #[test]
    fn test_direct_eval_power() {
        // Test power function
        let result = DirectEval::pow(2.0, 3.0);
        assert_eq!(result, 8.0);

        // Test square root
        let result = DirectEval::sqrt(9.0);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_direct_eval_statistical() {
        // Test logistic function
        let result = DirectEval::logistic(0.0);
        assert!((result - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_var_with_value() {
        let x = DirectEval::var_with_value(0, 5.0);
        assert_eq!(x, 5.0);

        // Index is ignored
        let y = DirectEval::var_with_value(999, 2.71);
        assert_eq!(y, 2.71);
    }
}
