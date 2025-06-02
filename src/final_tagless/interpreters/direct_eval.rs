//! Direct Evaluation Interpreter
//!
//! This interpreter provides immediate evaluation of mathematical expressions using native Rust
//! operations. It represents expressions directly as their computed values (`type Repr<T> = T`),
//! making it the simplest and most straightforward interpreter implementation.

use crate::ast::ASTRepr;
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
/// let result = polynomial::<DirectEval>(DirectEval::var("x", 2.0));
/// assert_eq!(result, 17.0); // 3(4) + 2(2) + 1 = 17
/// ```
///
/// ## Working with Different Numeric Types
///
/// ```rust
/// # use mathcompile::final_tagless::{DirectEval, MathExpr, NumericType};
/// // Function that works with any numeric type
/// fn linear<E: MathExpr, T>(x: E::Repr<T>, slope: T, intercept: T) -> E::Repr<T>
/// where
///     T: Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + NumericType,
/// {
///     E::add(E::mul(E::constant(slope), x), E::constant(intercept))
/// }
///
/// // Works with f32
/// let result_f32 = linear::<DirectEval, f32>(
///     DirectEval::var("x", 3.0_f32),
///     2.0_f32,
///     1.0_f32
/// );
/// assert_eq!(result_f32, 7.0_f32);
///
/// // Works with f64
/// let result_f64 = linear::<DirectEval, f64>(
///     DirectEval::var("x", 3.0_f64),
///     2.0_f64,
///     1.0_f64
/// );
/// assert_eq!(result_f64, 7.0_f64);
/// ```
///
/// ## Testing and Validation
///
/// `DirectEval` is particularly useful for testing the correctness of expressions
/// before using them with other interpreters:
///
/// ```rust
/// # use mathcompile::final_tagless::{DirectEval, MathExpr, StatisticalExpr};
/// // Test a statistical function
/// fn test_logistic<E: StatisticalExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
///     E::logistic(x)
/// }
///
/// // Verify known values
/// let result_zero = test_logistic::<DirectEval>(DirectEval::var("x", 0.0));
/// assert!((result_zero - 0.5).abs() < 1e-10); // logistic(0) = 0.5
///
/// let result_large = test_logistic::<DirectEval>(DirectEval::var("x", 10.0));
/// assert!(result_large > 0.99); // logistic(10) ≈ 1.0
/// ```
pub struct DirectEval;

impl DirectEval {
    /// Create a variable with a specific value for direct evaluation
    /// Note: This no longer registers variables globally - use `ExpressionBuilder` for that
    #[must_use]
    pub fn var<T: NumericType>(_name: &str, value: T) -> T {
        value
    }

    /// Create a variable by index with a specific value (for performance)
    #[must_use]
    pub fn var_by_index<T: NumericType>(_index: usize, value: T) -> T {
        value
    }

    /// Evaluate an expression with variables provided as a vector (efficient)
    #[must_use]
    pub fn eval_with_vars<T: NumericType + Float + Copy>(expr: &ASTRepr<T>, variables: &[T]) -> T {
        expr.eval_with_vars(variables)
    }

    /// Evaluate a two-variable expression with specific values (optimized version)
    #[must_use]
    pub fn eval_two_vars(expr: &ASTRepr<f64>, x: f64, y: f64) -> f64 {
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
    fn test_direct_eval() {
        fn linear<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
        where
            E: MathExpr,
        {
            E::add(E::mul(E::constant(2.0), x), E::constant(1.0))
        }

        let result = linear::<DirectEval>(DirectEval::var("x", 5.0));
        assert_eq!(result, 11.0); // 2*5 + 1 = 11
    }

    #[test]
    fn test_statistical_extension() {
        fn logistic_expr<E: StatisticalExpr>(x: E::Repr<f64>) -> E::Repr<f64>
        where
            E: StatisticalExpr,
        {
            E::logistic(x)
        }

        let result = logistic_expr::<DirectEval>(DirectEval::var("x", 0.0));
        assert!((result - 0.5).abs() < 1e-10); // logistic(0) = 0.5
    }

    #[test]
    fn test_division_operations() {
        let div_1_3: f64 = DirectEval::div(DirectEval::constant(1.0), DirectEval::constant(3.0));
        assert!((div_1_3 - 1.0 / 3.0).abs() < 1e-10);

        let div_10_2: f64 = DirectEval::div(DirectEval::constant(10.0), DirectEval::constant(2.0));
        assert!((div_10_2 - 5.0).abs() < 1e-10);

        // Test division by one
        let div_by_one: f64 =
            DirectEval::div(DirectEval::constant(42.0), DirectEval::constant(1.0));
        assert!((div_by_one - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_transcendental_functions() {
        // Test natural logarithm
        let ln_e: f64 = DirectEval::ln(DirectEval::constant(std::f64::consts::E));
        assert!((ln_e - 1.0).abs() < 1e-10);

        // Test exponential
        let exp_1: f64 = DirectEval::exp(DirectEval::constant(1.0));
        assert!((exp_1 - std::f64::consts::E).abs() < 1e-10);

        // Test square root
        let sqrt_4: f64 = DirectEval::sqrt(DirectEval::constant(4.0));
        assert!((sqrt_4 - 2.0).abs() < 1e-10);

        // Test sine
        let sin_pi_2: f64 = DirectEval::sin(DirectEval::constant(std::f64::consts::PI / 2.0));
        assert!((sin_pi_2 - 1.0).abs() < 1e-10);

        // Test cosine
        let cos_0: f64 = DirectEval::cos(DirectEval::constant(0.0));
        assert!((cos_0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ast_evaluation_integration() {
        // Test that DirectEval can work with AST expressions
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(5.0)),
        );

        let result = DirectEval::eval_with_vars(&expr, &[3.0]);
        assert_eq!(result, 8.0); // 3 + 5 = 8

        // Test two-variable evaluation
        let expr2 = ASTRepr::Mul(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        let result2 = DirectEval::eval_two_vars(&expr2, 4.0, 6.0);
        assert_eq!(result2, 24.0); // 4 * 6 = 24
    }
}
