//! AST Evaluation Utilities
//!
//! This module provides efficient evaluation methods for AST expressions,
//! including optimized variable handling and specialized evaluation functions.

use super::ast_repr::ASTRepr;
use crate::final_tagless::traits::NumericType;
use num_traits::Float;

/// Optimized evaluation methods for AST expressions
impl<T> ASTRepr<T>
where
    T: NumericType + Float + Copy,
{
    /// Evaluate an expression with variables provided as a vector (efficient)
    #[must_use]
    pub fn eval_with_vars(&self, variables: &[T]) -> T {
        Self::eval_vars_optimized(self, variables)
    }

    /// Optimized variable evaluation without additional allocations
    #[must_use]
    pub fn eval_vars_optimized(expr: &ASTRepr<T>, variables: &[T]) -> T {
        match expr {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => variables.get(*index).copied().unwrap_or_else(|| T::zero()),
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
}

/// Specialized evaluation methods for f64 expressions
impl ASTRepr<f64> {
    /// Evaluate a two-variable expression with specific values (optimized version)
    #[must_use]
    pub fn eval_two_vars(&self, x: f64, y: f64) -> f64 {
        Self::eval_two_vars_fast(self, x, y)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficient_variable_indexing() {
        // Test efficient index-based variables
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = expr.eval_with_vars(&[2.0, 3.0]);
        assert_eq!(result, 5.0);

        // Test multiplication with index-based variables
        let expr = ASTRepr::Mul(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = expr.eval_with_vars(&[4.0, 5.0]);
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_out_of_bounds_variable_index() {
        // Test behavior when variable index is out of bounds
        let expr = ASTRepr::Variable(10); // Index 10, but only 2 variables provided
        let result = expr.eval_with_vars(&[1.0, 2.0]);
        assert_eq!(result, 0.0); // Should return zero for out-of-bounds index
    }

    #[test]
    fn test_two_variable_evaluation() {
        // Test two-variable evaluation: x + y
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = expr.eval_two_vars(3.0, 4.0);
        assert_eq!(result, 7.0);

        // Test more complex expression: x * y + 1
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)), // x
                Box::new(ASTRepr::Variable(1)), // y
            )),
            Box::new(ASTRepr::Constant(1.0)),
        );
        let result = expr.eval_two_vars(2.0, 3.0);
        assert_eq!(result, 7.0); // 2 * 3 + 1 = 7
    }

    #[test]
    fn test_transcendental_evaluation() {
        // Test sine evaluation
        let expr = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[0.0]);
        assert!((result - 0.0).abs() < 1e-10); // sin(0) = 0

        // Test exponential evaluation
        let expr = ASTRepr::Exp(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[0.0]);
        assert!((result - 1.0).abs() < 1e-10); // exp(0) = 1

        // Test natural logarithm evaluation
        let expr = ASTRepr::Ln(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[1.0]);
        assert!((result - 0.0).abs() < 1e-10); // ln(1) = 0
    }

    #[test]
    fn test_power_evaluation() {
        // Test power evaluation: x^2
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        );
        let result = expr.eval_with_vars(&[3.0]);
        assert_eq!(result, 9.0); // 3^2 = 9

        // Test fractional power: x^0.5 (square root)
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(0.5)),
        );
        let result = expr.eval_with_vars(&[4.0]);
        assert!((result - 2.0).abs() < 1e-10); // 4^0.5 = 2
    }
}
