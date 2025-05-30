//! Core Traits for the Final Tagless Approach
//!
//! This module defines the fundamental traits that enable the final tagless approach
//! to mathematical expression representation and evaluation.

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

    /// Create a variable reference by name (registers variable automatically)
    fn var<T: NumericType>(name: &str) -> Self::Repr<T>;

    /// Create a variable reference by index (for performance-critical code)
    fn var_by_index<T: NumericType>(index: usize) -> Self::Repr<T>;

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
    fn sum_telescoping<T, F>(
        range: Self::Repr<crate::final_tagless::IntRange>,
        function: Self::Repr<F>,
    ) -> Self::Repr<T>
    where
        T: NumericType,
        F: SummandFunction<T>;

    /// Create a simple integer range for summations
    fn range_to<T: NumericType>(
        start: Self::Repr<T>,
        end: Self::Repr<T>,
    ) -> Self::Repr<crate::final_tagless::IntRange>;

    /// Create a function representation for summands
    fn function<T: NumericType>(
        index_var: &str,
        body: Self::Repr<T>,
    ) -> Self::Repr<crate::final_tagless::ASTFunction<T>>;
}

/// Simplified trait for JIT compilation that works with homogeneous f64 types
/// This is a practical compromise for JIT compilation while maintaining the final tagless approach
pub trait ASTMathExpr {
    /// The representation type for JIT compilation (always f64 for practical reasons)
    type Repr;

    /// Create a constant value
    fn constant(value: f64) -> Self::Repr;

    /// Create a variable reference by index
    fn var(index: usize) -> Self::Repr;

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
