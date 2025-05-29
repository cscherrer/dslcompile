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
/// - **Variables**: Variable names as provided `x`, `theta`, `data`
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
/// let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
/// println!("Quadratic: {}", pretty);
/// // Output: "((x ^ 2) + (2 * x)) + 1"
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
///     PrettyPrint::var("x"),
///     PrettyPrint::var("theta")
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
/// let pretty = gaussian_kernel::<PrettyPrint>(PrettyPrint::var("x"));
/// println!("Gaussian: {}", pretty);
/// // Output: "(exp((-(x ^ 2)) / 2) / sqrt((2 * 3.14159)))"
/// ```
pub struct PrettyPrint;

impl PrettyPrint {
    /// Create a variable for pretty printing
    #[must_use]
    pub fn var(name: &str) -> String {
        name.to_string()
    }
}

impl MathExpr for PrettyPrint {
    type Repr<T> = String;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        format!("{value}")
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        name.to_string()
    }

    fn var_by_index<T: NumericType>(_index: usize) -> Self::Repr<T> {
        T::default().to_string()
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

        let expr = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
        assert!(expr.contains('x'));
        assert!(expr.contains('2'));
        assert!(expr.contains('3'));
        assert!(expr.contains('1'));
    }

    #[test]
    fn test_pretty_print_basic() {
        // Test variable creation
        let var_x = PrettyPrint::var("x");
        assert_eq!(var_x, "x");

        // Test constant creation
        let const_5 = PrettyPrint::constant::<f64>(5.0);
        assert_eq!(const_5, "5");

        // Test addition
        let add_expr =
            PrettyPrint::add::<f64, f64, f64>(PrettyPrint::var("x"), PrettyPrint::constant(1.0));
        assert_eq!(add_expr, "(x + 1)");
    }

    #[test]
    fn test_transcendental_pretty_print() {
        // Test sine
        let sin_expr = PrettyPrint::sin::<f64>(PrettyPrint::var("x"));
        assert_eq!(sin_expr, "sin(x)");

        // Test exponential
        let exp_expr = PrettyPrint::exp::<f64>(PrettyPrint::var("x"));
        assert_eq!(exp_expr, "exp(x)");

        // Test natural logarithm
        let ln_expr = PrettyPrint::ln::<f64>(PrettyPrint::var("x"));
        assert_eq!(ln_expr, "ln(x)");

        // Test square root
        let sqrt_expr = PrettyPrint::sqrt::<f64>(PrettyPrint::var("x"));
        assert_eq!(sqrt_expr, "sqrt(x)");
    }

    #[test]
    fn test_complex_expression_pretty_print() {
        // Test a complex expression: sin(x^2) + exp(y)
        let x = PrettyPrint::var("x");
        let y = PrettyPrint::var("y");
        let two = PrettyPrint::constant::<f64>(2.0);

        let x_squared = PrettyPrint::pow::<f64>(x, two);
        let sin_x_squared = PrettyPrint::sin::<f64>(x_squared);
        let exp_y = PrettyPrint::exp::<f64>(y);
        let result = PrettyPrint::add::<f64, f64, f64>(sin_x_squared, exp_y);

        assert!(result.contains("sin"));
        assert!(result.contains("exp"));
        assert!(result.contains("x"));
        assert!(result.contains("y"));
        assert!(result.contains("^"));
        assert!(result.contains("+"));
    }

    #[test]
    fn test_statistical_functions_pretty_print() {
        // Test logistic function pretty printing
        fn test_logistic<E: StatisticalExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::logistic(x)
        }

        let result = test_logistic::<PrettyPrint>(PrettyPrint::var("x"));
        
        // The logistic function should expand to its definition
        assert!(result.contains("exp"));
        assert!(result.contains("x"));
        // Should contain the structure of 1 / (1 + exp(-x))
        assert!(result.contains("/"));
        assert!(result.contains("+"));
    }

    #[test]
    fn test_negation_pretty_print() {
        let x = PrettyPrint::var("x");
        let neg_x = PrettyPrint::neg::<f64>(x);
        assert_eq!(neg_x, "(-x)");

        // Test negation of complex expression
        let complex = PrettyPrint::add::<f64, f64, f64>(
            PrettyPrint::var("x"),
            PrettyPrint::constant(1.0)
        );
        let neg_complex = PrettyPrint::neg::<f64>(complex);
        assert_eq!(neg_complex, "(-(x + 1))");
    }
} 