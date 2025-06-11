//! Macro-Based Zero-Overhead Expression System
//!
//! This module provides a macro-based approach to building mathematical expressions
//! with flexible arity and mixed types, achieving true zero-overhead abstraction
//! by generating direct function calls at compile time.
//!
//! # Key Benefits
//! - **Zero Runtime Overhead**: Generates direct function calls
//! - **Flexible Arity**: Support 1-N parameters without storage overhead
//! - **Mixed Types**: Natural support for f64, &[f64], usize, etc.
//! - **Type Safety**: Compile-time type checking
//! - **Natural Syntax**: Function-like parameter lists
//! - **Heterogeneous Support**: Native operations on multiple types
//!
//! # Examples
//!
//! ```rust
//! use dslcompile::{expr, hetero_expr};
//!
//! // Simple binary operation (f64 output)
//! let add_fn = expr!(|x: f64, y: f64| x + y);
//! assert_eq!(add_fn(3.0, 4.0), 7.0);
//!
//! // Heterogeneous operation (any output type)
//! let array_index = hetero_expr!(|arr: &[f64], idx: usize| -> f64 { arr[idx] });
//! let data = [1.0, 2.0, 3.0];
//! assert_eq!(array_index(&data, 1), 2.0);
//!
//! // Vector operations (Vec<f64> output)
//! let vector_scale = hetero_expr!(|v: &[f64], scale: f64| -> Vec<f64> {
//!     v.iter().map(|x| x * scale).collect()
//! });
//!
//! // Mixed types - neural network layer
//! let neural_fn = expr!(|weights: &[f64], input: f64, bias: f64|
//!     weights[0] * input + bias
//! );
//! let weights = [0.5, 0.3];
//! assert_eq!(neural_fn(&weights, 2.0, 0.1), 1.1);
//!
//! // Complex mathematical expressions
//! let quadratic_fn = expr!(|a: f64, b: f64, c: f64, x: f64|
//!     a * x * x + b * x + c
//! );
//! assert_eq!(quadratic_fn(1.0, 2.0, 3.0, 4.0), 27.0);
//! ```

/// Helper functions for mathematical operations
/// These are re-exported to make them available in macro-generated code
pub use std::f64::consts::PI;

/// Square root function
#[inline]
#[must_use]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Sine function
#[inline]
#[must_use]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine function
#[inline]
#[must_use]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Natural logarithm
#[inline]
#[must_use]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// Exponential function
#[inline]
#[must_use]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Power function
#[inline]
#[must_use]
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Tangent function
#[inline]
#[must_use]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// Arc sine function
#[inline]
#[must_use]
pub fn asin(x: f64) -> f64 {
    x.asin()
}

/// Arc cosine function
#[inline]
#[must_use]
pub fn acos(x: f64) -> f64 {
    x.acos()
}

/// Arc tangent function
#[inline]
#[must_use]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Absolute value function
#[inline]
#[must_use]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// Floor function
#[inline]
#[must_use]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Ceiling function
#[inline]
#[must_use]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Round function
#[inline]
#[must_use]
pub fn round(x: f64) -> f64 {
    x.round()
}

/// Main macro for creating zero-overhead mathematical expressions
///
/// This macro generates direct function calls with no runtime overhead,
/// supporting flexible arity and mixed types.
///
/// # Syntax
/// ```text
/// expr!(|param1: Type1, param2: Type2, ...| expression)
/// ```
///
/// # Working Examples
/// ```rust
/// use dslcompile::expr;
///
/// let add = expr!(|x: f64, y: f64| x + y);
/// assert_eq!(add(3.0, 4.0), 7.0);
/// ```
///
/// # Supported Operations
/// - Arithmetic: `+`, `-`, `*`, `/`
/// - Mathematical: `pow`, `sqrt`, `sin`, `cos`, `ln`, `exp`
/// - Array access: `arr[index]`
/// - Comparisons: `<`, `>`, `<=`, `>=`, `==`, `!=`
/// - Conditionals: `if condition { true_expr } else { false_expr }`
///
/// # Examples
/// ```rust
/// use dslcompile::expr;
/// use dslcompile::contexts::static_context::macro_expressions::{sqrt, sin, cos, pow};
///
/// // Basic arithmetic
/// let add = expr!(|x: f64, y: f64| x + y);
/// let multiply = expr!(|x: f64, y: f64| x * y);
///
/// // Mathematical functions
/// let distance = expr!(|x1: f64, y1: f64, x2: f64, y2: f64|
///     sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
/// );
///
/// // Array operations
/// let dot_product = expr!(|a: &[f64], b: &[f64]|
///     a[0] * b[0] + a[1] * b[1]  // Simplified 2D case
/// );
///
/// // Conditional logic
/// let relu = expr!(|x: f64| if x > 0.0 { x } else { 0.0 });
/// ```
#[macro_export]
macro_rules! expr {
    // Main pattern - capture parameters and expression body
    (|$($param:ident: $type:ty),*| $body:expr) => {
        {

            |$($param: $type),*| -> f64 {
                #[allow(unused_parens)]
                { ($body) as f64 }
            }
        }
    };
}

/// Heterogeneous macro for creating zero-overhead expressions with explicit return types
///
/// This macro supports native heterogeneous operations with any return type,
/// enabling array indexing, vector operations, and mixed-type expressions.
///
/// # Syntax
/// ```text
/// hetero_expr!(|param1: Type1, param2: Type2, ...| -> ReturnType { expression })
/// ```
///
/// # Examples
/// ```rust
/// use dslcompile::hetero_expr;
///
/// // Array indexing (Vec<f64> + usize -> f64)
/// let array_index = hetero_expr!(|arr: &[f64], idx: usize| -> f64 { arr[idx] });
/// let data = [1.0, 2.0, 3.0];
/// assert_eq!(array_index(&data, 1), 2.0);
///
/// // Vector scaling (Vec<f64> + f64 -> Vec<f64>)
/// let vector_scale = hetero_expr!(|v: &[f64], scale: f64| -> Vec<f64> {
///     v.iter().map(|x| x * scale).collect()
/// });
///
/// // Boolean operations (f64 + f64 -> bool)
/// let greater_than = hetero_expr!(|x: f64, y: f64| -> bool { x > y });
/// assert_eq!(greater_than(5.0, 3.0), true);
///
/// // String operations (usize -> String)
/// let repeat_char = hetero_expr!(|count: usize| -> String {
///     "x".repeat(count)
/// });
/// assert_eq!(repeat_char(3), "xxx");
/// ```
#[macro_export]
macro_rules! hetero_expr {
    // Pattern with explicit return type
    (|$($param:ident: $type:ty),*| -> $ret_type:ty { $body:expr }) => {
        {
            use $crate::contexts::static_context::macro_expressions::*;
            |$($param: $type),*| -> $ret_type {
                #[allow(unused_parens)]
                { $body }
            }
        }
    };

    // Pattern without explicit return type (inferred)
    (|$($param:ident: $type:ty),*| $body:expr) => {
        {
            use $crate::contexts::static_context::macro_expressions::*;
            |$($param: $type),*| {
                #[allow(unused_parens)]
                { $body }
            }
        }
    };
}

/// Build a linear combination expression
///
/// Creates a function that computes the dot product of coefficients and values.
#[must_use]
pub fn linear_combination<const N: usize>() -> impl Fn(&[f64; N], &[f64; N]) -> f64 {
    |coeffs: &[f64; N], values: &[f64; N]| -> f64 {
        coeffs.iter().zip(values.iter()).map(|(c, v)| c * v).sum()
    }
}

/// Build a polynomial expression
///
/// Creates a function that evaluates a polynomial with given coefficients at point x.
#[must_use]
pub fn polynomial<const N: usize>() -> impl Fn(&[f64; N], f64) -> f64 {
    |coeffs: &[f64; N], x: f64| -> f64 {
        coeffs
            .iter()
            .enumerate()
            .map(|(i, c)| c * x.powi(i as i32))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience macro for common mathematical patterns (test-only)
    ///
    /// This macro provides shortcuts for frequently used mathematical expressions.
    macro_rules! math_expr {
        // Linear function: ax + b
        (linear |$x:ident: f64, $a:ident: f64, $b:ident: f64|) => {
            expr!(|$x: f64, $a: f64, $b: f64| $a * $x + $b)
        };

        // Quadratic function: ax² + bx + c
        (quadratic |$x:ident: f64, $a:ident: f64, $b:ident: f64, $c:ident: f64|) => {
            expr!(|$x: f64, $a: f64, $b: f64, $c: f64| $a * $x * $x + $b * $x + $c)
        };

        // Euclidean distance in 2D
        (distance_2d |$x1:ident: f64, $y1:ident: f64, $x2:ident: f64, $y2:ident: f64|) => {
            expr!(|$x1: f64, $y1: f64, $x2: f64, $y2: f64| sqrt(
                ($x2 - $x1) * ($x2 - $x1) + ($y2 - $y1) * ($y2 - $y1)
            ))
        };
    }

    #[test]
    fn test_basic_arithmetic() {
        let add = expr!(|x: f64, y: f64| x + y);
        assert_eq!(add(3.0, 4.0), 7.0);

        let multiply = expr!(|x: f64, y: f64| x * y);
        assert_eq!(multiply(3.0, 4.0), 12.0);

        let subtract = expr!(|x: f64, y: f64| x - y);
        assert_eq!(subtract(7.0, 3.0), 4.0);

        let divide = expr!(|x: f64, y: f64| x / y);
        assert_eq!(divide(8.0, 2.0), 4.0);
    }

    #[test]
    fn test_mathematical_functions() {
        let sqrt_expr = expr!(|x: f64| sqrt(x));
        assert_eq!(sqrt_expr(9.0), 3.0);

        let sin_expr = expr!(|x: f64| sin(x));
        assert!((sin_expr(0.0) - 0.0).abs() < 1e-10);

        let exp_expr = expr!(|x: f64| exp(x));
        assert!((exp_expr(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_array_access() {
        let weights = [0.1, 0.2, 0.3, 0.4];
        let array_expr = expr!(|arr: &[f64], idx: usize| arr[idx]);
        assert_eq!(array_expr(&weights, 1), 0.2);

        let weighted_expr = expr!(|arr: &[f64], idx: usize, factor: f64| arr[idx] * factor);
        assert_eq!(weighted_expr(&weights, 1, 2.0), 0.4);
    }

    #[test]
    fn test_mixed_types() {
        let neural_expr = expr!(|weights: &[f64], input: f64, bias: f64| weights[0] * input + bias);
        let weights = [0.5, 0.3];
        assert_eq!(neural_expr(&weights, 2.0, 0.1), 1.1);
    }

    #[test]
    fn test_conditional_expressions() {
        let relu = expr!(|x: f64| if x > 0.0 { x } else { 0.0 });
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-3.0), 0.0);
    }

    #[test]
    fn test_math_expr_shortcuts() {
        let linear = math_expr!(linear |x: f64, a: f64, b: f64|);
        assert_eq!(linear(2.0, 3.0, 1.0), 7.0); // 3*2 + 1 = 7

        let quadratic = math_expr!(quadratic |x: f64, a: f64, b: f64, c: f64|);
        assert_eq!(quadratic(2.0, 1.0, 2.0, 3.0), 11.0); // 1*4 + 2*2 + 3 = 11
    }

    #[test]
    fn test_utility_functions() {
        let linear_comb = linear_combination::<3>();
        let coeffs = [1.0, 2.0, 3.0];
        let values = [4.0, 5.0, 6.0];
        assert_eq!(linear_comb(&coeffs, &values), 32.0); // 1*4 + 2*5 + 3*6 = 32

        let poly = polynomial::<3>();
        let coeffs = [1.0, 2.0, 3.0]; // 1 + 2x + 3x²
        assert_eq!(poly(&coeffs, 2.0), 17.0); // 1 + 2*2 + 3*4 = 17
    }

    #[test]
    fn test_math_expr_convenience_macros() {
        // Test linear function
        let linear = math_expr!(linear |x: f64, a: f64, b: f64|);
        assert_eq!(linear(2.0, 3.0, 1.0), 7.0); // 3*2 + 1 = 7

        // Test quadratic function
        let quadratic = math_expr!(quadratic |x: f64, a: f64, b: f64, c: f64|);
        assert_eq!(quadratic(2.0, 1.0, 2.0, 3.0), 11.0); // 1*4 + 2*2 + 3 = 11

        // Test 2D distance function
        let distance = math_expr!(distance_2d |x1: f64, y1: f64, x2: f64, y2: f64|);
        assert!((distance(0.0, 0.0, 3.0, 4.0) - 5.0).abs() < 1e-10); // 3-4-5 triangle
    }
}
