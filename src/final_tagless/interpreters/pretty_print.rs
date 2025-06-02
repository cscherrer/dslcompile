//! Pretty Print Interpreter
//!
//! This interpreter converts final tagless expressions into human-readable mathematical notation.
//! It generates parenthesized infix expressions that clearly show the structure and precedence
//! of operations.

use crate::final_tagless::traits::{MathExpr, NumericType, StatisticalExpr};
use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// String representation interpreter for mathematical expressions
///
/// This interpreter converts final tagless expressions into human-readable mathematical notation.
/// It generates parenthesized infix expressions that clearly show the structure and precedence
/// of operations. This is useful for debugging, documentation, and displaying expressions to users.
///
/// # Output Format
///
/// - **Arithmetic operations**: Infix notation with parentheses `(a + b)`, `(a * b)`
/// - **Functions**: Function call notation `ln(x)`, `exp(x)`, `sqrt(x)`
/// - **Variables**: Variable names as `var_0`, `var_1`, etc.
/// - **Constants**: Numeric literals `2`, `3.14159`, `-1.5`
///
/// # Usage Examples
///
/// ## Basic Expression Formatting
///
/// ```rust
/// use mathcompile::final_tagless::{PrettyPrint, MathExpr};
///
/// // Simple quadratic: x² + 2x + 1
/// fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
/// where
///     E::Repr<f64>: Clone,
/// {
///     let x_squared = E::pow(x.clone(), E::constant(2.0));
///     let two_x = E::mul(E::constant(2.0), x);
///     E::add(E::add(x_squared, two_x), E::constant(1.0))
/// }
///
/// let pretty = quadratic::<PrettyPrint>(PrettyPrint::var(0));
/// println!("Quadratic: {}", pretty);
/// // Output: "((var_0 ^ 2) + (2 * var_0)) + 1"
/// ```
///
/// ## Complex Mathematical Expressions
///
/// ```rust
/// # use mathcompile::final_tagless::{PrettyPrint, MathExpr, StatisticalExpr};
/// // Logistic regression: 1 / (1 + exp(-θx))
/// fn logistic_regression<E: StatisticalExpr>(x: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
///     E::logistic(E::mul(theta, x))
/// }
///
/// let pretty = logistic_regression::<PrettyPrint>(
///     PrettyPrint::var(0),
///     PrettyPrint::var(1)
/// );
/// println!("Logistic: {}", pretty);
/// // Output shows the expanded logistic function structure
/// ```
///
/// ## Transcendental Functions
///
/// ```rust
/// # use mathcompile::final_tagless::{PrettyPrint, MathExpr};
/// // Gaussian: exp(-x²/2) / sqrt(2π)
/// fn gaussian_kernel<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
/// where
///     E::Repr<f64>: Clone,
/// {
///     let x_squared = E::pow(x, E::constant(2.0));
///     let neg_half_x_squared = E::div(E::neg(x_squared), E::constant(2.0));
///     let numerator = E::exp(neg_half_x_squared);
///     let denominator = E::sqrt(E::mul(E::constant(2.0), E::constant(3.14159)));
///     E::div(numerator, denominator)
/// }
///
/// let pretty = gaussian_kernel::<PrettyPrint>(PrettyPrint::var(0));
/// println!("Gaussian: {}", pretty);
/// // Output: "(exp((-(var_0 ^ 2)) / 2) / sqrt((2 * 3.14159)))"
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrettyPrint;

impl PrettyPrint {
    /// Create a variable for pretty printing by index
    #[must_use]
    pub fn var(index: usize) -> String {
        format!("var_{index}")
    }
}

impl MathExpr for PrettyPrint {
    type Repr<T> = String;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        format!("{value}")
    }

    fn var<T: NumericType>(index: usize) -> Self::Repr<T> {
        format!("var_{index}")
    }

    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} + {right})")
    }

    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} - {right})")
    }

    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} * {right})")
    }

    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        format!("({left} / {right})")
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        format!("({base} ^ {exp})")
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("(-{expr})")
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("ln({expr})")
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("exp({expr})")
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("sqrt({expr})")
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("sin({expr})")
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        format!("cos({expr})")
    }
}

// Implement StatisticalExpr for PrettyPrint
impl StatisticalExpr for PrettyPrint {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretty_print() {
        fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
        where
            E: MathExpr,
            E::Repr<f64>: Clone,
        {
            let a = E::constant(2.0);
            let b = E::constant(3.0);
            let c = E::constant(1.0);

            E::add(
                E::add(E::mul(a, E::pow(x.clone(), E::constant(2.0))), E::mul(b, x)),
                c,
            )
        }

        let expr = quadratic::<PrettyPrint>(PrettyPrint::var(0));
        assert!(expr.contains("var_0"));
        assert!(expr.contains('2'));
        assert!(expr.contains('3'));
        assert!(expr.contains('1'));
    }

    #[test]
    fn test_pretty_print_basic() {
        // Test variable creation
        let var_x = PrettyPrint::var(0);
        assert_eq!(var_x, "var_0");

        let var_y = PrettyPrint::var(1);
        assert_eq!(var_y, "var_1");

        // Test constant creation
        let const_5 = PrettyPrint::constant::<f64>(5.0);
        assert_eq!(const_5, "5");

        // Test addition
        let add_expr =
            PrettyPrint::add::<f64, f64, f64>(PrettyPrint::var(0), PrettyPrint::constant(1.0));
        assert_eq!(add_expr, "(var_0 + 1)");
    }

    #[test]
    fn test_pretty_print_transcendental() {
        let x = PrettyPrint::var(0);
        
        // Test sin
        let sin_expr = PrettyPrint::sin::<f64>(x.clone());
        assert_eq!(sin_expr, "sin(var_0)");

        // Test ln
        let ln_expr = PrettyPrint::ln::<f64>(x.clone());
        assert_eq!(ln_expr, "ln(var_0)");

        // Test exp
        let exp_expr = PrettyPrint::exp::<f64>(x);
        assert_eq!(exp_expr, "exp(var_0)");
    }

    #[test]
    fn test_pretty_print_statistical() {
        let x = PrettyPrint::var(0);
        let logistic_expr = PrettyPrint::logistic::<f64>(x);
        
        // Should contain the logistic expansion
        assert!(logistic_expr.contains("exp"));
        assert!(logistic_expr.contains("var_0"));
    }

    #[test]
    fn test_pretty_print_complex() {
        // Test nested expression: (x + 1) * (x - 1)
        let x = PrettyPrint::var(0);
        let one = PrettyPrint::constant(1.0);
        
        let left = PrettyPrint::add::<f64, f64, f64>(x.clone(), one.clone());
        let right = PrettyPrint::sub::<f64, f64, f64>(x, one);
        let result = PrettyPrint::mul::<f64, f64, f64>(left, right);
        
        assert_eq!(result, "((var_0 + 1) * (var_0 - 1))");
    }
}
