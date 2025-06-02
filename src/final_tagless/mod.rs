//! Final Tagless Mathematical Interpreter System
//!
//! This module provides a clean, composable interface for mathematical expressions
//! using the final tagless style. It offers multiple interpreters and ensures type safety.

use crate::ast::ast_utils::{contains_variable_by_index, transform_expression};

// Core infrastructure
pub mod interpreters;
pub mod polynomial;
pub mod traits;
pub mod variables;

// Re-export everything
pub use interpreters::*;
pub use traits::*;
pub use variables::*;

// Direct re-exports for common types
pub use crate::ast::ASTRepr;

/// Re-export of `NumericType` trait for convenience
pub use crate::final_tagless::traits::NumericType;

// Convenience aliases for the most commonly used interpreters
/// Alias for `ASTEval` for users who prefer the shorter name
pub type AST = ASTEval;

/// Alias for `DirectEval` for backward compatibility
pub type Eval = DirectEval;

/// Simple integer range for summations
///
/// Represents ranges like 1..=n, 0..=100, etc. This is the most common
/// type of range used in mathematical summations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntRange {
    /// Start of the range (inclusive)
    pub start: i64,
    /// End of the range (inclusive)
    pub end: i64,
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

/// AST-based function for summations
#[derive(Debug, Clone)]
pub struct ASTFunction<T> {
    /// The variable name for the summation index
    pub index_var: String,
    /// The expression representing the function body
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

    /// Create a constant function: f(i) = c
    pub fn constant_func(index_var: &str, value: T) -> Self {
        Self::new(index_var, ASTRepr::Constant(value))
    }

    /// Create a polynomial function: f(i) = coefficients[0] + coefficients[1]*i + coefficients[2]*i² + ...
    /// Coefficients are in ascending order of powers
    pub fn poly(index_var: &str, coefficients: &[T]) -> Self
    where
        T: Clone,
    {
        if coefficients.is_empty() {
            return Self::new(index_var, ASTRepr::Constant(T::default()));
        }

        if coefficients.len() == 1 {
            return Self::new(index_var, ASTRepr::Constant(coefficients[0].clone()));
        }

        let i = ASTRepr::Variable(0); // Assume index variable is at position 0

        // Build polynomial using Horner's method: a₀ + i*(a₁ + i*(a₂ + ...))
        let mut result = ASTRepr::Constant(coefficients[coefficients.len() - 1].clone());

        for coeff in coefficients.iter().rev().skip(1) {
            result = ASTRepr::Add(
                Box::new(ASTRepr::Constant(coeff.clone())),
                Box::new(ASTRepr::Mul(Box::new(i.clone()), Box::new(result))),
            );
        }

        Self::new(index_var, result)
    }

    /// Create a power function: f(i) = i^exponent
    pub fn power(index_var: &str, exponent: T) -> Self {
        let i = ASTRepr::Variable(0); // Assume index variable is at position 0
        let exp_expr = ASTRepr::Constant(exponent);
        let body = ASTRepr::Pow(Box::new(i), Box::new(exp_expr));
        Self::new(index_var, body)
    }
}

// Placeholder implementation for SummandFunction
impl<T: NumericType + Clone> SummandFunction<T> for ASTFunction<T> {
    type Body = ASTRepr<T>;

    fn index_var(&self) -> &str {
        &self.index_var
    }

    fn body(&self) -> &Self::Body {
        &self.body
    }

    fn apply(&self, index: T) -> Self::Body {
        // Substitute Variable(0) with Constant(index)
        transform_expression(&self.body, &|expr| match expr {
            ASTRepr::Variable(0) => Some(ASTRepr::Constant(index.clone())),
            _ => None,
        })
    }

    fn depends_on_index(&self) -> bool {
        contains_variable_by_index(&self.body, 0)
    }

    fn extract_independent_factors(&self) -> (Vec<Self::Body>, Self::Body) {
        // Placeholder implementation
        (vec![], self.body.clone())
    }
}
