//! AST Evaluation Utilities
//!
//! This module provides efficient evaluation methods for AST expressions,
//! including optimized variable handling and specialized evaluation functions.

use crate::ast::{ASTRepr, NumericType};
use num_traits::Float;

/// Optimized evaluation methods for AST expressions
impl<T> ASTRepr<T>
where
    T: NumericType + Float + Copy,
{
    // TODO: Macro version that takes arbitrary number of variables?
    // Note that arguments may have different types, so slice won't work
    /// Evaluate the expression with variables provided as a vector
    #[must_use]
    pub fn eval_with_vars(&self, variables: &[T]) -> T {
        match self {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => {
                variables.get(*index).copied().unwrap_or_else(|| {
                    panic!(
                        "Variable index {} is out of bounds! Expression uses Variable({}) but only {} variable values were provided. \
                        Expected variable array of length at least {}. \
                        Hint: When using the same ExpressionBuilder instance, variable indices increment with each math.var() call.",
                        index, index, variables.len(), index + 1
                    )
                })
            }
            ASTRepr::Add(left, right) => {
                left.eval_with_vars(variables) + right.eval_with_vars(variables)
            }
            ASTRepr::Sub(left, right) => {
                left.eval_with_vars(variables) - right.eval_with_vars(variables)
            }
            ASTRepr::Mul(left, right) => {
                left.eval_with_vars(variables) * right.eval_with_vars(variables)
            }
            ASTRepr::Div(left, right) => {
                left.eval_with_vars(variables) / right.eval_with_vars(variables)
            }
            ASTRepr::Pow(base, exp) => {
                let base_val = base.eval_with_vars(variables);
                let exp_val = exp.eval_with_vars(variables);
                base_val.powf(exp_val)
            }
            ASTRepr::Neg(expr) => -expr.eval_with_vars(variables),
            ASTRepr::Ln(expr) => expr.eval_with_vars(variables).ln(),
            ASTRepr::Exp(expr) => expr.eval_with_vars(variables).exp(),
            ASTRepr::Sin(expr) => expr.eval_with_vars(variables).sin(),
            ASTRepr::Cos(expr) => expr.eval_with_vars(variables).cos(),
            ASTRepr::Sqrt(expr) => expr.eval_with_vars(variables).sqrt(),
            // NOTE: Future Sum variant evaluation will go here
        }
    }

    /// Evaluate a two-variable expression with specific values
    #[must_use]
    pub fn eval_two_vars(&self, x: T, y: T) -> T {
        self.eval_with_vars(&[x, y])
    }

    /// Evaluate with a single variable value
    #[must_use]
    pub fn eval_one_var(&self, value: T) -> T {
        self.eval_with_vars(&[value])
    }
}

/// Specialized evaluation methods for f64 expressions
impl ASTRepr<f64> {
    /// Fast evaluation without heap allocation for two variables
    #[must_use]
    pub fn eval_two_vars_fast(expr: &ASTRepr<f64>, x: f64, y: f64) -> f64 {
        match expr {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => match *index {
                0 => x,
                1 => y,
                _ => panic!(
                    "Variable index {} is out of bounds for two-variable evaluation! \
                    eval_two_vars_fast only supports Variable(0) and Variable(1). \
                    Use eval_with_vars() for expressions with more variables.",
                    index
                ),
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
            // NOTE: Future Sum variant fast evaluation will go here
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
    #[should_panic(expected = "Variable index 10 is out of bounds")]
    fn test_out_of_bounds_variable_index() {
        // Test behavior when variable index is out of bounds - should panic
        let expr = ASTRepr::Variable(10); // Index 10, but only 2 variables provided
        let _result = expr.eval_with_vars(&[1.0, 2.0]); // Should panic!
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

    #[test]
    #[should_panic(expected = "Variable index 2 is out of bounds for two-variable evaluation")]
    fn test_two_vars_fast_out_of_bounds() {
        // Test that eval_two_vars_fast panics for Variable(2) and higher
        let expr = ASTRepr::Variable(2); // Index 2, but only supports 0 and 1
        let _result = ASTRepr::eval_two_vars_fast(&expr, 1.0, 2.0); // Should panic!
    }
}
