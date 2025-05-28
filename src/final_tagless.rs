//! Final Tagless Approach for Symbolic Mathematical Expressions
//!
//! This module implements the final tagless approach to solve the expression problem in symbolic
//! mathematics. The final tagless approach uses traits with Generic Associated Types (GATs) to
//! represent mathematical operations, enabling both easy extension of operations and interpreters
//! without modifying existing code.
//!
//! # Technical Motivation
//!
//! Traditional approaches to symbolic mathematics face the expression problem: adding new operations
//! requires modifying existing interpreter code, while adding new interpreters requires modifying
//! existing operation definitions. The final tagless approach solves this by:
//!
//! 1. **Parameterizing representation types**: Operations are defined over abstract representation
//!    types `Repr<T>`, allowing different interpreters to use different concrete representations
//! 2. **Trait-based extensibility**: New operations can be added via trait extension without
//!    modifying existing code
//! 3. **Zero intermediate representation**: Expressions compile directly to target representations
//!    without building intermediate ASTs
//!
//! # Architecture
//!
//! ## Core Traits
//!
//! - **`MathExpr`**: Defines basic mathematical operations (arithmetic, transcendental functions)
//! - **`StatisticalExpr`**: Extends `MathExpr` with statistical functions (logistic, softplus)
//! - **`NumericType`**: Helper trait bundling common numeric type requirements
//!
//! ## Interpreters
//!
//! - **`DirectEval`**: Immediate evaluation using native Rust operations (`type Repr<T> = T`)
//! - **`PrettyPrint`**: String representation generation (`type Repr<T> = String`)
//!
//! # Usage Patterns
//!
//! ## Polymorphic Expression Definition
//!
//! Define mathematical expressions that work with any interpreter:
//!
//! ```rust
//! use mathjit::final_tagless::*;
//!
//! // Define a quadratic function: 2x² + 3x + 1
//! fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! where
//!     E::Repr<f64>: Clone,
//! {
//!     let a = E::constant(2.0);
//!     let b = E::constant(3.0);
//!     let c = E::constant(1.0);
//!     
//!     E::add(
//!         E::add(
//!             E::mul(a, E::pow(x.clone(), E::constant(2.0))),
//!             E::mul(b, x)
//!         ),
//!         c
//!     )
//! }
//! ```
//!
//! ## Direct Evaluation
//!
//! Evaluate expressions immediately using native Rust operations:
//!
//! ```rust
//! # use mathjit::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));
//! assert_eq!(result, 15.0); // 2(4) + 3(2) + 1 = 15
//! ```
//!
//! ## Pretty Printing
//!
//! Generate human-readable mathematical notation:
//!
//! ```rust
//! # use mathjit::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
//! println!("Expression: {}", pretty);
//! // Output: "((2 * (x ^ 2)) + (3 * x)) + 1"
//! ```
//!
//! # Extension Example
//!
//! Adding new operations requires only trait extension:
//!
//! ```rust
//! use mathjit::final_tagless::*;
//! use num_traits::Float;
//!
//! // Extend with hyperbolic functions
//! trait HyperbolicExpr: MathExpr {
//!     fn tanh<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T>
//!     where
//!         Self::Repr<T>: Clone,
//!     {
//!         let exp_x = Self::exp(x.clone());
//!         let exp_neg_x = Self::exp(Self::neg(x));
//!         let numerator = Self::sub(exp_x.clone(), exp_neg_x.clone());
//!         let denominator = Self::add(exp_x, exp_neg_x);
//!         Self::div(numerator, denominator)
//!     }
//! }
//!
//! // Automatically works with all existing interpreters
//! impl HyperbolicExpr for DirectEval {}
//! impl HyperbolicExpr for PrettyPrint {}
//! ```

use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Helper trait that bundles all the common trait bounds for numeric types
/// This makes the main `MathExpr` trait much cleaner and easier to read
pub trait NumericType:
    Clone + Default + Send + Sync + 'static + std::fmt::Display + std::fmt::Debug
{
}

/// Blanket implementation for all types that satisfy the bounds
impl<T> NumericType for T where
    T: Clone + Default + Send + Sync + 'static + std::fmt::Display + std::fmt::Debug
{
}

/// Core trait for mathematical expressions using Generic Associated Types (GATs)
/// This follows the final tagless approach where the representation type is parameterized
/// and works with generic numeric types including AD types
pub trait MathExpr {
    /// The representation type parameterized by the value type
    type Repr<T>;

    /// Create a constant value
    fn constant<T: NumericType>(value: T) -> Self::Repr<T>;

    /// Create a variable reference
    fn var<T: NumericType>(name: &str) -> Self::Repr<T>;

    // Arithmetic operations with flexible type parameters
    /// Addition operation
    fn add<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Subtraction operation
    fn sub<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Multiplication operation
    fn mul<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Division operation
    fn div<L, R, Output>(left: Self::Repr<L>, right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType;

    /// Power operation
    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T>;

    /// Negation operation
    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Natural logarithm
    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Exponential function
    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Square root
    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Sine function
    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;

    /// Cosine function
    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T>;
}

/// Polynomial evaluation utilities using Horner's method
///
/// This module provides efficient polynomial evaluation using the final tagless approach.
/// Horner's method reduces the number of multiplications and provides better numerical
/// stability compared to naive polynomial evaluation.
pub mod polynomial {
    use super::{MathExpr, NumericType};
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
    /// use mathjit::final_tagless::{DirectEval, polynomial::horner};
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
    /// use mathjit::final_tagless::{DirectEval, MathExpr, polynomial::horner_expr};
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
    /// use mathjit::final_tagless::{DirectEval, polynomial::from_roots};
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
    /// use mathjit::final_tagless::{DirectEval, polynomial::horner_derivative};
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
}

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
/// use mathjit::final_tagless::{DirectEval, MathExpr};
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
/// # use mathjit::final_tagless::{DirectEval, MathExpr, NumericType};
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
/// # use mathjit::final_tagless::{DirectEval, MathExpr, StatisticalExpr};
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
    #[must_use]
    pub fn var<T: NumericType>(_name: &str, value: T) -> T {
        value
    }

    /// Evaluate an expression with variables provided as a vector (efficient)
    #[must_use]
    pub fn eval_with_vars<T: NumericType + Float + Copy>(expr: &ASTRepr<T>, variables: &[T]) -> T {
        Self::eval_vars_optimized(expr, variables)
    }

    /// Optimized variable evaluation without additional allocations
    #[must_use]
    pub fn eval_vars_optimized<T: NumericType + Float + Copy>(
        expr: &ASTRepr<T>,
        variables: &[T],
    ) -> T {
        match expr {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => variables.get(*index).copied().unwrap_or_else(|| T::zero()),
            ASTRepr::VariableByName(name) => {
                // Fast path for common variable names
                match name.as_str() {
                    "x" => variables.first().copied().unwrap_or_else(|| T::zero()),
                    "y" => variables.get(1).copied().unwrap_or_else(|| T::zero()),
                    "z" => variables.get(2).copied().unwrap_or_else(|| T::zero()),
                    _ => T::zero(), // Default for unknown variables
                }
            }
            ASTRepr::Add(left, right) => {
                Self::eval_vars_optimized(left, variables)
                    + Self::eval_vars_optimized(right, variables)
            }
            ASTRepr::Sub(left, right) => {
                Self::eval_vars_optimized(left, variables)
                    - Self::eval_vars_optimized(right, variables)
            }
            ASTRepr::Mul(left, right) => {
                Self::eval_vars_optimized(left, variables)
                    * Self::eval_vars_optimized(right, variables)
            }
            ASTRepr::Div(left, right) => {
                Self::eval_vars_optimized(left, variables)
                    / Self::eval_vars_optimized(right, variables)
            }
            ASTRepr::Pow(base, exp) => Self::eval_vars_optimized(base, variables)
                .powf(Self::eval_vars_optimized(exp, variables)),
            ASTRepr::Neg(inner) => -Self::eval_vars_optimized(inner, variables),
            ASTRepr::Ln(inner) => Self::eval_vars_optimized(inner, variables).ln(),
            ASTRepr::Exp(inner) => Self::eval_vars_optimized(inner, variables).exp(),
            ASTRepr::Sin(inner) => Self::eval_vars_optimized(inner, variables).sin(),
            ASTRepr::Cos(inner) => Self::eval_vars_optimized(inner, variables).cos(),
            ASTRepr::Sqrt(inner) => Self::eval_vars_optimized(inner, variables).sqrt(),
        }
    }

    /// Evaluate a two-variable expression with specific values (optimized version)
    #[must_use]
    pub fn eval_two_vars(expr: &ASTRepr<f64>, x: f64, y: f64) -> f64 {
        Self::eval_two_vars_fast(expr, x, y)
    }

    /// Fast evaluation without heap allocation for two variables
    #[must_use]
    pub fn eval_two_vars_fast(expr: &ASTRepr<f64>, x: f64, y: f64) -> f64 {
        match expr {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => match *index {
                0 => x,
                1 => y,
                _ => 0.0, // Default for out-of-bounds
            },
            ASTRepr::VariableByName(name) => match name.as_str() {
                "x" => x,
                "y" => y,
                _ => 0.0, // Default for unknown variables
            },
            ASTRepr::Add(left, right) => {
                Self::eval_two_vars_fast(left, x, y) + Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Sub(left, right) => {
                Self::eval_two_vars_fast(left, x, y) - Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Mul(left, right) => {
                Self::eval_two_vars_fast(left, x, y) * Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Div(left, right) => {
                Self::eval_two_vars_fast(left, x, y) / Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Pow(base, exp) => {
                Self::eval_two_vars_fast(base, x, y).powf(Self::eval_two_vars_fast(exp, x, y))
            }
            ASTRepr::Neg(inner) => -Self::eval_two_vars_fast(inner, x, y),
            ASTRepr::Ln(inner) => Self::eval_two_vars_fast(inner, x, y).ln(),
            ASTRepr::Exp(inner) => Self::eval_two_vars_fast(inner, x, y).exp(),
            ASTRepr::Sin(inner) => Self::eval_two_vars_fast(inner, x, y).sin(),
            ASTRepr::Cos(inner) => Self::eval_two_vars_fast(inner, x, y).cos(),
            ASTRepr::Sqrt(inner) => Self::eval_two_vars_fast(inner, x, y).sqrt(),
        }
    }
}

impl MathExpr for DirectEval {
    type Repr<T> = T;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        value
    }

    fn var<T: NumericType>(_name: &str) -> Self::Repr<T> {
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
        base.powf(exp)
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

/// Extension trait for statistical operations
pub trait StatisticalExpr: MathExpr {
    /// Logistic function: 1 / (1 + exp(-x))
    fn logistic<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        let one = Self::constant(T::one());
        let neg_x = Self::neg(x);
        let exp_neg_x = Self::exp(neg_x);
        let denominator = Self::add(one, exp_neg_x);
        Self::div(Self::constant(T::one()), denominator)
    }

    /// Softplus function: ln(1 + exp(x))
    fn softplus<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        let one = Self::constant(T::one());
        let exp_x = Self::exp(x);
        let one_plus_exp_x = Self::add(one, exp_x);
        Self::ln(one_plus_exp_x)
    }

    /// Sigmoid function (alias for logistic)
    fn sigmoid<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T> {
        Self::logistic(x)
    }
}

// Implement StatisticalExpr for DirectEval
impl StatisticalExpr for DirectEval {}

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
/// use mathjit::final_tagless::{PrettyPrint, MathExpr};
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
/// # use mathjit::final_tagless::{PrettyPrint, MathExpr, StatisticalExpr};
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
/// # use mathjit::final_tagless::{PrettyPrint, MathExpr};
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

/// JIT compilation representation for mathematical expressions
///
/// This enum represents mathematical expressions in a form suitable for JIT compilation
/// using Cranelift. Each variant corresponds to a mathematical operation that can be
/// compiled to native machine code.
///
/// # Performance Note
///
/// For optimal performance with `DirectEval`, use `Variable(usize)` instead of
/// `VariableByName(String)`. This allows `DirectEval` to use vector indexing instead
/// of string lookups:
///
/// ```rust
/// use mathjit::final_tagless::{ASTRepr, DirectEval};
///
/// // Efficient: uses vector indexing
/// let efficient_expr = ASTRepr::Add(
///     Box::new(ASTRepr::Variable(0)), // x
///     Box::new(ASTRepr::Variable(1)), // y
/// );
/// let result = DirectEval::eval_with_vars(&efficient_expr, &[2.0, 3.0]);
/// assert_eq!(result, 5.0);
///
/// // Less efficient: uses string matching
/// let less_efficient_expr = ASTRepr::Add(
///     Box::new(ASTRepr::VariableByName("x".to_string())),
///     Box::new(ASTRepr::VariableByName("y".to_string())),
/// );
/// let result = DirectEval::eval_with_vars(&less_efficient_expr, &[2.0, 3.0]);
/// assert_eq!(result, 5.0);
/// ```
#[derive(Debug, Clone)]
pub enum ASTRepr<T> {
    /// Constant value
    Constant(T),
    /// Variable reference by index (efficient for evaluation)
    Variable(usize),
    /// Variable reference by name (for backwards compatibility, less efficient)
    VariableByName(String),
    /// Addition of two expressions
    Add(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    /// Subtraction of two expressions
    Sub(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    /// Multiplication of two expressions
    Mul(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    /// Division of two expressions
    Div(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    /// Power operation
    Pow(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    /// Negation
    Neg(Box<ASTRepr<T>>),
    /// Natural logarithm
    Ln(Box<ASTRepr<T>>),
    /// Exponential function
    Exp(Box<ASTRepr<T>>),
    /// Square root
    Sqrt(Box<ASTRepr<T>>),
    /// Sine function
    Sin(Box<ASTRepr<T>>),
    /// Cosine function
    Cos(Box<ASTRepr<T>>),
}

impl<T> ASTRepr<T> {
    /// Count the total number of operations in the expression tree
    pub fn count_operations(&self) -> usize {
        match self {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::VariableByName(_) => 0,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => 1 + left.count_operations() + right.count_operations(),
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => 1 + inner.count_operations(),
        }
    }

    /// Get the variable index if this is a variable, otherwise None
    pub fn variable_index(&self) -> Option<usize> {
        match self {
            ASTRepr::Variable(index) => Some(*index),
            _ => None,
        }
    }

    /// Get the variable name if this is a named variable, otherwise None
    pub fn variable_name(&self) -> Option<&str> {
        match self {
            ASTRepr::VariableByName(name) => Some(name),
            _ => None,
        }
    }
}

/// JIT evaluation interpreter that builds an intermediate representation
/// suitable for compilation with Cranelift or Rust codegen
///
/// This interpreter constructs a `ASTRepr` tree that can later be compiled
/// to native machine code for high-performance evaluation.
pub struct ASTEval;

impl ASTEval {
    /// Create a variable reference for JIT compilation using an index (efficient)
    #[must_use]
    pub fn var<T: NumericType>(index: usize) -> ASTRepr<T> {
        ASTRepr::Variable(index)
    }

    /// Create a variable reference for JIT compilation by name (backwards compatible)
    #[must_use]
    pub fn var_by_name<T: NumericType>(name: &str) -> ASTRepr<T> {
        ASTRepr::VariableByName(name.to_string())
    }
}

/// Simplified trait for JIT compilation that works with homogeneous f64 types
/// This is a practical compromise for JIT compilation while maintaining the final tagless approach
pub trait ASTMathExpr {
    /// The representation type for JIT compilation (always f64 for practical reasons)
    type Repr;

    /// Create a constant value
    fn constant(value: f64) -> Self::Repr;

    /// Create a variable reference
    fn var(name: &str) -> Self::Repr;

    /// Addition operation
    fn add(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Subtraction operation
    fn sub(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Multiplication operation
    fn mul(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Division operation
    fn div(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Power operation
    fn pow(base: Self::Repr, exp: Self::Repr) -> Self::Repr;

    /// Negation operation
    fn neg(expr: Self::Repr) -> Self::Repr;

    /// Natural logarithm
    fn ln(expr: Self::Repr) -> Self::Repr;

    /// Exponential function
    fn exp(expr: Self::Repr) -> Self::Repr;

    /// Square root
    fn sqrt(expr: Self::Repr) -> Self::Repr;

    /// Sine function
    fn sin(expr: Self::Repr) -> Self::Repr;

    /// Cosine function
    fn cos(expr: Self::Repr) -> Self::Repr;
}

impl ASTMathExpr for ASTEval {
    type Repr = ASTRepr<f64>;

    fn constant(value: f64) -> Self::Repr {
        ASTRepr::Constant(value)
    }

    fn var(name: &str) -> Self::Repr {
        ASTRepr::VariableByName(name.to_string())
    }

    fn add(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Add(Box::new(left), Box::new(right))
    }

    fn sub(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Sub(Box::new(left), Box::new(right))
    }

    fn mul(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Mul(Box::new(left), Box::new(right))
    }

    fn div(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Div(Box::new(left), Box::new(right))
    }

    fn pow(base: Self::Repr, exp: Self::Repr) -> Self::Repr {
        ASTRepr::Pow(Box::new(base), Box::new(exp))
    }

    fn neg(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Neg(Box::new(expr))
    }

    fn ln(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Ln(Box::new(expr))
    }

    fn exp(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Exp(Box::new(expr))
    }

    fn sqrt(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Sqrt(Box::new(expr))
    }

    fn sin(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Sin(Box::new(expr))
    }

    fn cos(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Cos(Box::new(expr))
    }
}

/// For compatibility with the main `MathExpr` trait, we provide a limited implementation
/// that works only with f64 types
impl MathExpr for ASTEval {
    type Repr<T> = ASTRepr<T>;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        ASTRepr::Constant(value)
    }

    fn var<T: NumericType>(name: &str) -> Self::Repr<T> {
        ASTRepr::VariableByName(name.to_string())
    }

    fn add<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        // This is a design limitation - JIT compilation works best with homogeneous types
        // For practical JIT usage, use the ASTMathExpr trait instead
        unimplemented!("Use ASTMathExpr trait for practical JIT compilation with f64 types")
    }

    fn sub<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr trait for practical JIT compilation with f64 types")
    }

    fn mul<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr trait for practical JIT compilation with f64 types")
    }

    fn div<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr trait for practical JIT compilation with f64 types")
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Pow(Box::new(base), Box::new(exp))
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Neg(Box::new(expr))
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Ln(Box::new(expr))
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Exp(Box::new(expr))
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Sqrt(Box::new(expr))
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Sin(Box::new(expr))
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Cos(Box::new(expr))
    }
}

impl StatisticalExpr for ASTEval {}

/// Backwards-compatible trait for f64-specific AST math expressions
/// This trait delegates to the generic `ASTMathExpr`<f64> implementation
pub trait ASTMathExprf64 {
    /// The representation type for f64 compilation
    type Repr;

    /// Create a constant value
    fn constant(value: f64) -> Self::Repr;

    /// Create a variable reference
    fn var(name: &str) -> Self::Repr;

    /// Addition operation
    fn add(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Subtraction operation
    fn sub(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Multiplication operation
    fn mul(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Division operation
    fn div(left: Self::Repr, right: Self::Repr) -> Self::Repr;

    /// Power operation
    fn pow(base: Self::Repr, exp: Self::Repr) -> Self::Repr;

    /// Negation operation
    fn neg(expr: Self::Repr) -> Self::Repr;

    /// Natural logarithm
    fn ln(expr: Self::Repr) -> Self::Repr;

    /// Exponential function
    fn exp(expr: Self::Repr) -> Self::Repr;

    /// Square root
    fn sqrt(expr: Self::Repr) -> Self::Repr;

    /// Sine function
    fn sin(expr: Self::Repr) -> Self::Repr;

    /// Cosine function
    fn cos(expr: Self::Repr) -> Self::Repr;
}

impl ASTMathExprf64 for ASTEval {
    type Repr = ASTRepr<f64>;

    fn constant(value: f64) -> Self::Repr {
        <ASTEval as ASTMathExpr>::constant(value)
    }

    fn var(name: &str) -> Self::Repr {
        <ASTEval as ASTMathExpr>::var(name)
    }

    fn add(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::add(left, right)
    }

    fn sub(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::sub(left, right)
    }

    fn mul(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::mul(left, right)
    }

    fn div(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::div(left, right)
    }

    fn pow(base: Self::Repr, exp: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::pow(base, exp)
    }

    fn neg(expr: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::neg(expr)
    }

    fn ln(expr: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::ln(expr)
    }

    fn exp(expr: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::exp(expr)
    }

    fn sqrt(expr: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::sqrt(expr)
    }

    fn sin(expr: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::sin(expr)
    }

    fn cos(expr: Self::Repr) -> Self::Repr {
        <ASTEval as ASTMathExpr>::cos(expr)
    }
}

// ============================================================================
// Summation Infrastructure Implementation
// ============================================================================

/// Trait for range-like types in summations
///
/// This trait defines the interface for different types of ranges that can be used
/// in summations, from simple integer ranges to symbolic ranges with expression bounds.
pub trait RangeType: Clone + Send + Sync + 'static + std::fmt::Debug {
    /// The type of values in this range
    type IndexType: NumericType;

    /// Start of the range (inclusive)
    fn start(&self) -> Self::IndexType;

    /// End of the range (inclusive)  
    fn end(&self) -> Self::IndexType;

    /// Check if the range contains a value
    fn contains(&self, value: &Self::IndexType) -> bool;

    /// Get the length of the range (end - start + 1)
    fn len(&self) -> Self::IndexType;

    /// Check if the range is empty
    fn is_empty(&self) -> bool;
}

/// Simple integer range for summations
///
/// Represents ranges like 1..=n, 0..=100, etc. This is the most common
/// type of range used in mathematical summations.
///
/// # Examples
///
/// ```rust
/// use mathjit::final_tagless::IntRange;
///
/// let range = IntRange::new(1, 10);  // Range from 1 to 10 inclusive
/// assert_eq!(range.len(), 10);
/// assert!(range.contains(&5));
/// assert!(!range.contains(&15));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntRange {
    pub start: i64,
    pub end: i64, // inclusive
}

impl IntRange {
    /// Create a new integer range
    #[must_use]
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    /// Create a range from 1 to n (common mathematical convention)
    #[must_use]
    pub fn one_to_n(n: i64) -> Self {
        Self::new(1, n)
    }

    /// Create a range from 0 to n-1 (common programming convention)
    #[must_use]
    pub fn zero_to_n_minus_one(n: i64) -> Self {
        Self::new(0, n - 1)
    }

    /// Iterate over the range values
    pub fn iter(&self) -> impl Iterator<Item = i64> {
        self.start..=self.end
    }
}

impl RangeType for IntRange {
    type IndexType = i64;

    fn start(&self) -> Self::IndexType {
        self.start
    }

    fn end(&self) -> Self::IndexType {
        self.end
    }

    fn contains(&self, value: &Self::IndexType) -> bool {
        *value >= self.start && *value <= self.end
    }

    fn len(&self) -> Self::IndexType {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            0
        }
    }

    fn is_empty(&self) -> bool {
        self.end < self.start
    }
}

/// Floating-point range for summations
///
/// Represents ranges with floating-point bounds. Less common than integer ranges
/// but useful for continuous approximations or when bounds are computed values.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatRange {
    pub start: f64,
    pub end: f64,
    pub step: f64,
}

impl FloatRange {
    /// Create a new floating-point range with step size
    #[must_use]
    pub fn new(start: f64, end: f64, step: f64) -> Self {
        Self { start, end, step }
    }

    /// Create a range with step size 1.0
    #[must_use]
    pub fn unit_step(start: f64, end: f64) -> Self {
        Self::new(start, end, 1.0)
    }
}

impl RangeType for FloatRange {
    type IndexType = f64;

    fn start(&self) -> Self::IndexType {
        self.start
    }

    fn end(&self) -> Self::IndexType {
        self.end
    }

    fn contains(&self, value: &Self::IndexType) -> bool {
        *value >= self.start && *value <= self.end
    }

    fn len(&self) -> Self::IndexType {
        if self.end >= self.start && self.step > 0.0 {
            ((self.end - self.start) / self.step).floor() + 1.0
        } else {
            0.0
        }
    }

    fn is_empty(&self) -> bool {
        self.end < self.start || self.step <= 0.0
    }
}

/// Symbolic range with expression bounds
///
/// Represents ranges where the start and/or end are expressions rather than
/// concrete values. This enables symbolic manipulation of summation bounds.
///
/// # Examples
///
/// ```rust
/// use mathjit::final_tagless::{SymbolicRange, ASTRepr};
///
/// // Range from 1 to n (where n is a variable)
/// let range = SymbolicRange::new(
///     ASTRepr::Constant(1.0),
///     ASTRepr::VariableByName("n".to_string())
/// );
/// ```
#[derive(Debug, Clone)]
pub struct SymbolicRange<T> {
    pub start: Box<ASTRepr<T>>,
    pub end: Box<ASTRepr<T>>,
}

impl<T: NumericType> SymbolicRange<T> {
    /// Create a new symbolic range
    pub fn new(start: ASTRepr<T>, end: ASTRepr<T>) -> Self {
        Self {
            start: Box::new(start),
            end: Box::new(end),
        }
    }

    /// Create a range from 1 to a symbolic expression
    pub fn one_to_expr(end: ASTRepr<T>) -> Self
    where
        T: num_traits::One,
    {
        Self::new(ASTRepr::Constant(T::one()), end)
    }

    /// Evaluate the range bounds with given variable values
    pub fn evaluate_bounds(&self, variables: &[T]) -> Option<(T, T)>
    where
        T: Float + Copy,
    {
        let start_val = DirectEval::eval_with_vars(&self.start, variables);
        let end_val = DirectEval::eval_with_vars(&self.end, variables);
        Some((start_val, end_val))
    }
}

// Note: SymbolicRange doesn't implement RangeType because it requires evaluation
// to determine concrete bounds. It's used in a different way in the summation system.

/// Trait for function-like expressions in summations
///
/// The function must not be opaque to enable factor extraction and algebraic
/// manipulation. This trait provides access to the function's internal structure.
pub trait SummandFunction<T>: Clone + std::fmt::Debug {
    /// The expression representing the function body
    type Body: Clone;

    /// The variable name for the summation index
    fn index_var(&self) -> &str;

    /// Get the function body expression
    fn body(&self) -> &Self::Body;

    /// Apply the function to a specific index value (for evaluation)
    fn apply(&self, index: T) -> Self::Body;

    /// Check if the function depends on the index variable
    fn depends_on_index(&self) -> bool;

    /// Extract factors that don't depend on the index variable
    /// Returns (`independent_factors`, `remaining_expression`)
    fn extract_independent_factors(&self) -> (Vec<Self::Body>, Self::Body);
}

/// Concrete implementation for AST-based functions
///
/// This represents a function as an AST expression with a designated index variable.
/// It provides the foundation for algebraic manipulation of summands.
///
/// # Examples
///
/// ```rust
/// use mathjit::final_tagless::{ASTFunction, ASTRepr};
///
/// // Function f(i) = 2*i + 3
/// let func = ASTFunction::new(
///     "i",
///     ASTRepr::Add(
///         Box::new(ASTRepr::Mul(
///             Box::new(ASTRepr::Constant(2.0)),
///             Box::new(ASTRepr::VariableByName("i".to_string()))
///         )),
///         Box::new(ASTRepr::Constant(3.0))
///     )
/// );
/// ```
#[derive(Debug, Clone)]
pub struct ASTFunction<T> {
    pub index_var: String,
    pub body: ASTRepr<T>,
}

impl<T: NumericType> ASTFunction<T> {
    /// Create a new AST-based function
    pub fn new(index_var: &str, body: ASTRepr<T>) -> Self {
        Self {
            index_var: index_var.to_string(),
            body,
        }
    }

    /// Create a simple linear function: a*i + b
    pub fn linear(index_var: &str, coefficient: T, constant: T) -> Self {
        let body = ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Constant(coefficient)),
                Box::new(ASTRepr::VariableByName(index_var.to_string())),
            )),
            Box::new(ASTRepr::Constant(constant)),
        );
        Self::new(index_var, body)
    }

    /// Create a power function: i^k
    pub fn power(index_var: &str, exponent: T) -> Self {
        let body = ASTRepr::Pow(
            Box::new(ASTRepr::VariableByName(index_var.to_string())),
            Box::new(ASTRepr::Constant(exponent)),
        );
        Self::new(index_var, body)
    }

    /// Create a constant function (doesn't depend on index)
    pub fn constant_func(index_var: &str, value: T) -> Self {
        let body = ASTRepr::Constant(value);
        Self::new(index_var, body)
    }
}

impl<T: NumericType + Float + Copy> SummandFunction<T> for ASTFunction<T> {
    type Body = ASTRepr<T>;

    fn index_var(&self) -> &str {
        &self.index_var
    }

    fn body(&self) -> &Self::Body {
        &self.body
    }

    fn apply(&self, index: T) -> Self::Body {
        // Create a simple substitution - in a full implementation,
        // this would do proper variable substitution in the AST
        self.substitute_variable(&self.index_var, index)
    }

    fn depends_on_index(&self) -> bool {
        self.contains_variable(&self.body, &self.index_var)
    }

    fn extract_independent_factors(&self) -> (Vec<Self::Body>, Self::Body) {
        // Basic implementation - in practice, this would do sophisticated
        // algebraic analysis to extract factors
        self.extract_factors_recursive(&self.body)
    }
}

impl<T: NumericType + Copy> ASTFunction<T> {
    /// Substitute a variable with a concrete value (simplified implementation)
    fn substitute_variable(&self, var_name: &str, value: T) -> ASTRepr<T> {
        self.substitute_in_expr(&self.body, var_name, value)
    }

    /// Recursive variable substitution
    fn substitute_in_expr(&self, expr: &ASTRepr<T>, var_name: &str, value: T) -> ASTRepr<T> {
        match expr {
            ASTRepr::Constant(c) => ASTRepr::Constant(*c),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::VariableByName(name) => {
                if name == var_name {
                    ASTRepr::Constant(value)
                } else {
                    ASTRepr::VariableByName(name.clone())
                }
            }
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(self.substitute_in_expr(left, var_name, value)),
                Box::new(self.substitute_in_expr(right, var_name, value)),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(self.substitute_in_expr(left, var_name, value)),
                Box::new(self.substitute_in_expr(right, var_name, value)),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(self.substitute_in_expr(left, var_name, value)),
                Box::new(self.substitute_in_expr(right, var_name, value)),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(self.substitute_in_expr(left, var_name, value)),
                Box::new(self.substitute_in_expr(right, var_name, value)),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(self.substitute_in_expr(base, var_name, value)),
                Box::new(self.substitute_in_expr(exp, var_name, value)),
            ),
            ASTRepr::Neg(inner) => {
                ASTRepr::Neg(Box::new(self.substitute_in_expr(inner, var_name, value)))
            }
            ASTRepr::Ln(inner) => {
                ASTRepr::Ln(Box::new(self.substitute_in_expr(inner, var_name, value)))
            }
            ASTRepr::Exp(inner) => {
                ASTRepr::Exp(Box::new(self.substitute_in_expr(inner, var_name, value)))
            }
            ASTRepr::Sin(inner) => {
                ASTRepr::Sin(Box::new(self.substitute_in_expr(inner, var_name, value)))
            }
            ASTRepr::Cos(inner) => {
                ASTRepr::Cos(Box::new(self.substitute_in_expr(inner, var_name, value)))
            }
            ASTRepr::Sqrt(inner) => {
                ASTRepr::Sqrt(Box::new(self.substitute_in_expr(inner, var_name, value)))
            }
        }
    }

    /// Check if an expression contains a specific variable
    fn contains_variable(&self, expr: &ASTRepr<T>, var_name: &str) -> bool {
        match expr {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => false,
            ASTRepr::VariableByName(name) => name == var_name,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.contains_variable(left, var_name) || self.contains_variable(right, var_name)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.contains_variable(inner, var_name),
        }
    }

    /// Extract factors that don't depend on the index variable (simplified implementation)
    fn extract_factors_recursive(&self, expr: &ASTRepr<T>) -> (Vec<ASTRepr<T>>, ASTRepr<T>)
    where
        T: One,
    {
        match expr {
            // For multiplication, we can extract independent factors
            ASTRepr::Mul(left, right) => {
                let left_depends = self.contains_variable(left, &self.index_var);
                let right_depends = self.contains_variable(right, &self.index_var);

                match (left_depends, right_depends) {
                    (false, false) => {
                        // Both factors are independent
                        (vec![expr.clone()], ASTRepr::Constant(T::one()))
                    }
                    (false, true) => {
                        // Left factor is independent
                        (vec![(**left).clone()], (**right).clone())
                    }
                    (true, false) => {
                        // Right factor is independent
                        (vec![(**right).clone()], (**left).clone())
                    }
                    (true, true) => {
                        // Both factors depend on index, can't extract
                        (vec![], expr.clone())
                    }
                }
            }
            // For other operations, basic handling
            _ => {
                if self.contains_variable(expr, &self.index_var) {
                    (vec![], expr.clone())
                } else {
                    (vec![expr.clone()], ASTRepr::Constant(T::one()))
                }
            }
        }
    }
}

// Helper trait to provide one() method for numeric types
use num_traits::One;

/// Extension trait for summation operations
///
/// This trait extends the final tagless approach to support summations with
/// algebraic manipulation capabilities. It provides methods for creating
/// various types of summations and will eventually support automatic simplification.
pub trait SummationExpr: MathExpr {
    /// Create a finite summation: Σ(i=start to end) f(i)
    ///
    /// This is the most general form of finite summation, where both the range
    /// and the function can be represented using any interpreter.
    fn sum_finite<T, R, F>(range: Self::Repr<R>, function: Self::Repr<F>) -> Self::Repr<T>
    where
        T: NumericType,
        R: RangeType,
        F: SummandFunction<T>,
        Self::Repr<T>: Clone;

    /// Create an infinite summation: Σ(i=start to ∞) f(i)  
    ///
    /// For infinite summations, convergence analysis and special handling
    /// would be needed in a complete implementation.
    fn sum_infinite<T, F>(start: Self::Repr<T>, function: Self::Repr<F>) -> Self::Repr<T>
    where
        T: NumericType,
        F: SummandFunction<T>,
        Self::Repr<T>: Clone;

    /// Create a telescoping sum for automatic simplification
    ///
    /// Telescoping sums have the special property that consecutive terms cancel,
    /// allowing for closed-form evaluation: Σ(f(i+1) - f(i)) = f(end+1) - f(start)
    fn sum_telescoping<T, F>(range: Self::Repr<IntRange>, function: Self::Repr<F>) -> Self::Repr<T>
    where
        T: NumericType,
        F: SummandFunction<T>;

    /// Create a simple integer range for summations
    fn range_to<T: NumericType>(start: Self::Repr<T>, end: Self::Repr<T>) -> Self::Repr<IntRange>;

    /// Create a function representation for summands
    fn function<T: NumericType>(index_var: &str, body: Self::Repr<T>)
        -> Self::Repr<ASTFunction<T>>;
}

// Extension to ASTRepr to support summation operations
impl<T> ASTRepr<T> {
    /// Add summation support to the AST representation
    ///
    /// These variants would be added to the enum in a complete implementation:
    /// - SumFinite(Box<`ASTRepr`<IntRange>>, Box<`ASTRepr`<`ASTFunction`<T>>>)
    /// - SumInfinite(Box<`ASTRepr`<T>>, Box<`ASTRepr`<`ASTFunction`<T>>>)
    /// - SumTelescoping(Box<`ASTRepr`<IntRange>>, Box<`ASTRepr`<`ASTFunction`<T>>>)
    /// - Range(i64, i64)
    /// - Function(String, Box<`ASTRepr`<T>>)

    /// Placeholder for future summation operation counting
    pub fn count_summation_operations(&self) -> usize {
        // This would count summation-specific operations in addition to
        // the basic operations already counted by count_operations()
        0
    }
}

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
    fn test_horner_polynomial() {
        // Test polynomial: 1 + 2x + 3x^2 at x = 2
        // Expected: 1 + 2(2) + 3(4) = 17
        let coeffs = [1.0, 2.0, 3.0];
        let x = DirectEval::var("x", 2.0);
        let result = polynomial::horner::<DirectEval, f64>(&coeffs, x);
        assert_eq!(result, 17.0);
    }

    #[test]
    fn test_horner_pretty_print() {
        let coeffs = [1.0, 2.0, 3.0];
        let x = PrettyPrint::var("x");
        let result = polynomial::horner::<PrettyPrint, f64>(&coeffs, x);
        assert!(result.contains('x'));
    }

    #[test]
    fn test_polynomial_from_roots() {
        // Polynomial with roots at 1 and 2: (x-1)(x-2) = x^2 - 3x + 2
        // At x=0: (0-1)(0-2) = 2
        let roots = [1.0, 2.0];
        let x = DirectEval::var("x", 0.0);
        let result = polynomial::from_roots::<DirectEval, f64>(&roots, x);
        assert_eq!(result, 2.0);

        // At x=3: (3-1)(3-2) = 2*1 = 2
        let x = DirectEval::var("x", 3.0);
        let result = polynomial::from_roots::<DirectEval, f64>(&roots, x);
        assert_eq!(result, 2.0);
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
    fn test_efficient_variable_indexing() {
        // Test efficient index-based variables
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = DirectEval::eval_with_vars(&expr, &[2.0, 3.0]);
        assert_eq!(result, 5.0);

        // Test multiplication with index-based variables
        let expr = ASTRepr::Mul(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = DirectEval::eval_with_vars(&expr, &[4.0, 5.0]);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_mixed_variable_types() {
        // Test mixing index-based and name-based variables
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),                     // x
            Box::new(ASTRepr::VariableByName("y".to_string())), // y
        );
        let result = DirectEval::eval_with_vars(&expr, &[2.0, 3.0]);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_variable_index_access() {
        let expr: ASTRepr<f64> = ASTRepr::Variable(5);
        assert_eq!(expr.variable_index(), Some(5));

        let expr: ASTRepr<f64> = ASTRepr::VariableByName("test".to_string());
        assert_eq!(expr.variable_index(), None);
        assert_eq!(expr.variable_name(), Some("test"));
    }

    #[test]
    fn test_out_of_bounds_variable_index() {
        // Test behavior when variable index is out of bounds
        let expr = ASTRepr::Variable(10); // Index 10, but only 2 variables provided
        let result = DirectEval::eval_with_vars(&expr, &[1.0, 2.0]);
        assert_eq!(result, 0.0); // Should return zero for out-of-bounds index
    }

    // ============================================================================
    // Summation Infrastructure Tests
    // ============================================================================

    #[test]
    fn test_int_range() {
        let range = IntRange::new(1, 10);
        assert_eq!(range.start(), 1);
        assert_eq!(range.end(), 10);
        assert_eq!(range.len(), 10);
        assert!(range.contains(&5));
        assert!(!range.contains(&15));
        assert!(!range.is_empty());

        let empty_range = IntRange::new(5, 3);
        assert!(empty_range.is_empty());
        assert_eq!(empty_range.len(), 0);
    }

    #[test]
    fn test_float_range() {
        let range = FloatRange::new(1.0, 10.0, 1.0);
        assert_eq!(range.start(), 1.0);
        assert_eq!(range.end(), 10.0);
        assert_eq!(range.len(), 10.0);
        assert!(range.contains(&5.5));
        assert!(!range.contains(&15.0));

        let empty_range = FloatRange::new(5.0, 3.0, 1.0);
        assert!(empty_range.is_empty());
    }

    #[test]
    fn test_symbolic_range() {
        // Test with index-based variable (more reliable for evaluation)
        let range = SymbolicRange::new(
            ASTRepr::Constant(1.0),
            ASTRepr::Variable(0), // First variable in the array
        );

        // Test evaluation with variable at index 0 = 10
        let bounds = range.evaluate_bounds(&[10.0]);
        assert_eq!(bounds, Some((1.0, 10.0)));

        // Test with both bounds as variables
        let range2 = SymbolicRange::new(
            ASTRepr::Variable(0), // Start from first variable
            ASTRepr::Variable(1), // End at second variable
        );

        let bounds2 = range2.evaluate_bounds(&[2.0, 8.0]);
        assert_eq!(bounds2, Some((2.0, 8.0)));
    }

    #[test]
    fn test_ast_function_creation() {
        // Test linear function: 2*i + 3
        let func = ASTFunction::linear("i", 2.0, 3.0);
        assert_eq!(func.index_var(), "i");
        assert!(func.depends_on_index());

        // Test constant function
        let const_func = ASTFunction::constant_func("i", 42.0);
        assert!(!const_func.depends_on_index());
    }

    #[test]
    fn test_ast_function_substitution() {
        // Test function application: f(i) = 2*i + 3, evaluate at i = 5
        let func = ASTFunction::linear("i", 2.0, 3.0);
        let result = func.apply(5.0);

        // The result should be a constant expression with value 13.0
        let evaluated = DirectEval::eval_with_vars(&result, &[]);
        assert_eq!(evaluated, 13.0); // 2*5 + 3 = 13
    }

    #[test]
    fn test_ast_function_factor_extraction() {
        // Test factor extraction for: 3 * i
        let func = ASTFunction::new(
            "i",
            ASTRepr::Mul(
                Box::new(ASTRepr::Constant(3.0)),
                Box::new(ASTRepr::VariableByName("i".to_string())),
            ),
        );

        let (factors, remaining) = func.extract_independent_factors();
        assert_eq!(factors.len(), 1); // Should extract the constant factor 3

        // Verify the extracted factor
        if let Some(ASTRepr::Constant(value)) = factors.first() {
            assert_eq!(*value, 3.0);
        } else {
            panic!("Expected constant factor");
        }
    }

    #[test]
    fn test_range_convenience_methods() {
        let range_1_to_n = IntRange::one_to_n(10);
        assert_eq!(range_1_to_n.start(), 1);
        assert_eq!(range_1_to_n.end(), 10);

        let range_0_to_n_minus_1 = IntRange::zero_to_n_minus_one(10);
        assert_eq!(range_0_to_n_minus_1.start(), 0);
        assert_eq!(range_0_to_n_minus_1.end(), 9);
    }

    #[test]
    fn test_power_function() {
        // Test power function: i^2
        let func = ASTFunction::power("i", 2.0);
        assert!(func.depends_on_index());

        // Test evaluation at i = 3 (should give 9)
        let result = func.apply(3.0);
        let evaluated = DirectEval::eval_with_vars(&result, &[]);
        assert_eq!(evaluated, 9.0);
    }
}
