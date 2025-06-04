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
//!
//! # Examples
//!
//! ```rust
//! use dslcompile::expr;
//!
//! // Simple binary operation
//! let add_fn = expr!(|x: f64, y: f64| x + y);
//! assert_eq!(add_fn(3.0, 4.0), 7.0);
//!
//! // Ternary operation
//! let sum3_fn = expr!(|x: f64, y: f64, z: f64| x + y + z);
//! assert_eq!(sum3_fn(1.0, 2.0, 3.0), 6.0);
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
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Sine function
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine function
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Natural logarithm
#[inline]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// Exponential function
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Power function
#[inline]
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Tangent function
#[inline]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// Arc sine function
#[inline]
pub fn asin(x: f64) -> f64 {
    x.asin()
}

/// Arc cosine function
#[inline]
pub fn acos(x: f64) -> f64 {
    x.acos()
}

/// Arc tangent function
#[inline]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Absolute value function
#[inline]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// Floor function
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Ceiling function
#[inline]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Round function
#[inline]
pub fn round(x: f64) -> f64 {
    x.round()
}

/// Main macro for creating zero-overhead mathematical expressions
///
/// This macro generates direct function calls with no runtime overhead,
/// supporting flexible arity and mixed types.
///
/// # Syntax
/// ```rust
/// use dslcompile::expr;
/// expr!(|param1: Type1, param2: Type2, ...| expression)
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
/// use dslcompile::compile_time::macro_expressions::{sqrt, sin, cos, pow};
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
            use $crate::compile_time::macro_expressions::*;
            |$($param: $type),*| -> f64 { 
                #[allow(unused_parens)]
                { ($body) as f64 }
            }
        }
    };
}

/// Convenience macro for common mathematical patterns
///
/// This macro provides shortcuts for frequently used mathematical expressions.
#[macro_export]
macro_rules! math_expr {
    // Linear function: ax + b
    (linear |$x:ident: f64, $a:ident: f64, $b:ident: f64|) => {
        expr!(|$x: f64, $a: f64, $b: f64| $a * $x + $b)
    };
    
    // Quadratic function: ax² + bx + c
    (quadratic |$x:ident: f64, $a:ident: f64, $b:ident: f64, $c:ident: f64|) => {
        expr!(|$x: f64, $a: f64, $b: f64, $c: f64| $a * $x * $x + $b * $x + $c)
    };
    
    // Gaussian/Normal distribution
    (gaussian |$x:ident: f64, $mu:ident: f64, $sigma:ident: f64|) => {
        expr!(|$x: f64, $mu: f64, $sigma: f64| 
            exp(-0.5 * pow(($x - $mu) / $sigma, 2.0)) / ($sigma * sqrt(2.0 * 3.14159265359))
        )
    };
    
    // Sigmoid activation function
    (sigmoid |$x:ident: f64|) => {
        expr!(|$x: f64| 1.0 / (1.0 + exp(-$x)))
    };
    
    // ReLU activation function
    (relu |$x:ident: f64|) => {
        expr!(|$x: f64| if $x > 0.0 { $x } else { 0.0 })
    };
    
    // Euclidean distance in 2D
    (distance_2d |$x1:ident: f64, $y1:ident: f64, $x2:ident: f64, $y2:ident: f64|) => {
        expr!(|$x1: f64, $y1: f64, $x2: f64, $y2: f64|
            sqrt(($x2 - $x1) * ($x2 - $x1) + ($y2 - $y1) * ($y2 - $y1))
        )
    };
}

/// Builder pattern for complex expressions
///
/// This provides a more structured way to build complex mathematical expressions
/// when the macro syntax becomes unwieldy.
pub struct ExpressionBuilder {
    // This will be expanded as needed
}

impl ExpressionBuilder {
    /// Create a new expression builder
    pub fn new() -> Self {
        Self {}
    }
    
    /// Build a linear combination expression
    pub fn linear_combination<const N: usize>() -> impl Fn(&[f64; N], &[f64; N]) -> f64 {
        |coeffs: &[f64; N], values: &[f64; N]| -> f64 {
            coeffs.iter().zip(values.iter()).map(|(c, v)| c * v).sum()
        }
    }
    
    /// Build a polynomial expression
    pub fn polynomial<const N: usize>() -> impl Fn(&[f64; N], f64) -> f64 {
        |coeffs: &[f64; N], x: f64| -> f64 {
            coeffs.iter().enumerate().map(|(i, c)| c * x.powi(i as i32)).sum()
        }
    }
}

impl Default for ExpressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
        
        let relu = math_expr!(relu |x: f64|);
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-3.0), 0.0);
    }
    
    #[test]
    fn test_expression_builder() {
        let linear_comb = ExpressionBuilder::linear_combination::<3>();
        let coeffs = [1.0, 2.0, 3.0];
        let values = [4.0, 5.0, 6.0];
        assert_eq!(linear_comb(&coeffs, &values), 32.0); // 1*4 + 2*5 + 3*6 = 32
        
        let poly = ExpressionBuilder::polynomial::<3>();
        let coeffs = [1.0, 2.0, 3.0]; // 1 + 2x + 3x²
        assert_eq!(poly(&coeffs, 2.0), 17.0); // 1 + 2*2 + 3*4 = 17
    }
} 