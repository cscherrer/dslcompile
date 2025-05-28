//! AST Utility Functions
//!
//! This module provides common utility functions for working with `ASTRepr` expressions,
//! consolidating functionality that was previously duplicated across multiple modules.
//!
//! # Features
//!
//! - **Expression Equality**: Unified structural equality checking with configurable tolerance
//! - **Variable Analysis**: Variable detection, collection, and dependency analysis
//! - **Expression Traversal**: Generic traversal patterns for AST manipulation
//! - **Optimization Helpers**: Common optimization patterns and utilities

use crate::final_tagless::{ASTRepr, NumericType, VariableRegistry};
use num_traits::Float;
use std::collections::HashSet;

/// Configuration for AST utility operations
#[derive(Debug, Clone)]
pub struct ASTUtilConfig {
    /// Tolerance for floating-point comparisons
    pub tolerance: f64,
    /// Whether to use strict structural equality (no tolerance)
    pub strict_equality: bool,
}

impl Default for ASTUtilConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            strict_equality: false,
        }
    }
}

/// Unified expression equality checking with configurable tolerance
pub fn expressions_equal<T: NumericType + Float>(
    expr1: &ASTRepr<T>,
    expr2: &ASTRepr<T>,
    config: &ASTUtilConfig,
) -> bool {
    match (expr1, expr2) {
        (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
            if config.strict_equality {
                a == b
            } else {
                let diff = (*a - *b).abs();
                diff < T::from(config.tolerance).unwrap_or_else(|| T::from(1e-12).unwrap())
            }
        }
        (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,
        (ASTRepr::Add(a1, a2), ASTRepr::Add(b1, b2))
        | (ASTRepr::Sub(a1, a2), ASTRepr::Sub(b1, b2))
        | (ASTRepr::Mul(a1, a2), ASTRepr::Mul(b1, b2))
        | (ASTRepr::Div(a1, a2), ASTRepr::Div(b1, b2))
        | (ASTRepr::Pow(a1, a2), ASTRepr::Pow(b1, b2)) => {
            expressions_equal(a1, b1, config) && expressions_equal(a2, b2, config)
        }
        (ASTRepr::Neg(a), ASTRepr::Neg(b))
        | (ASTRepr::Ln(a), ASTRepr::Ln(b))
        | (ASTRepr::Exp(a), ASTRepr::Exp(b))
        | (ASTRepr::Sin(a), ASTRepr::Sin(b))
        | (ASTRepr::Cos(a), ASTRepr::Cos(b))
        | (ASTRepr::Sqrt(a), ASTRepr::Sqrt(b)) => expressions_equal(a, b, config),
        _ => false,
    }
}

/// Convenience function for default expression equality checking
pub fn expressions_equal_default<T: NumericType + Float>(
    expr1: &ASTRepr<T>,
    expr2: &ASTRepr<T>,
) -> bool {
    expressions_equal(expr1, expr2, &ASTUtilConfig::default())
}

/// Check if an expression contains a variable by index
pub fn contains_variable_by_index<T: NumericType>(
    expr: &ASTRepr<T>,
    var_index: usize,
) -> bool {
    match expr {
        ASTRepr::Constant(_) => false,
        ASTRepr::Variable(index) => *index == var_index,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => {
            contains_variable_by_index(left, var_index) || contains_variable_by_index(right, var_index)
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => contains_variable_by_index(inner, var_index),
    }
}

/// Check if an expression contains a variable by name using a registry
pub fn contains_variable_by_name<T: NumericType>(
    expr: &ASTRepr<T>,
    var_name: &str,
    registry: &VariableRegistry,
) -> bool {
    if let Some(var_index) = registry.get_index(var_name) {
        contains_variable_by_index(expr, var_index)
    } else {
        false
    }
}

/// Legacy variable name mapping for backward compatibility
pub fn contains_variable_by_name_legacy<T: NumericType>(
    expr: &ASTRepr<T>,
    var_name: &str,
) -> bool {
    let expected_index = match var_name {
        "i" | "x" => 0,
        "j" | "y" => 1,
        "k" | "z" => 2,
        _ => return false,
    };
    contains_variable_by_index(expr, expected_index)
}

/// Collect all variable indices used in an expression
pub fn collect_variable_indices<T: NumericType>(expr: &ASTRepr<T>) -> HashSet<usize> {
    let mut variables = HashSet::new();
    collect_variable_indices_recursive(expr, &mut variables);
    variables
}

/// Recursive helper for collecting variable indices
fn collect_variable_indices_recursive<T: NumericType>(
    expr: &ASTRepr<T>,
    variables: &mut HashSet<usize>,
) {
    match expr {
        ASTRepr::Constant(_) => {}
        ASTRepr::Variable(index) => {
            variables.insert(*index);
        }
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => {
            collect_variable_indices_recursive(left, variables);
            collect_variable_indices_recursive(right, variables);
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => {
            collect_variable_indices_recursive(inner, variables);
        }
    }
}

/// Collect variable names using a registry
pub fn collect_variable_names<T: NumericType>(
    expr: &ASTRepr<T>,
    registry: &VariableRegistry,
) -> Vec<String> {
    let indices = collect_variable_indices(expr);
    let mut names = Vec::new();
    
    for index in indices {
        if let Some(name) = registry.get_name(index) {
            names.push(name.to_string());
        } else {
            names.push(format!("var_{index}"));
        }
    }
    
    names.sort();
    names
}

/// Generic expression traversal with a visitor function
pub fn traverse_expression<T: NumericType, F>(expr: &ASTRepr<T>, mut visitor: F)
where
    F: FnMut(&ASTRepr<T>),
{
    visitor(expr);
    
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => {}
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => {
            traverse_expression(left, &mut visitor);
            traverse_expression(right, &mut visitor);
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => {
            traverse_expression(inner, &mut visitor);
        }
    }
}

/// Transform an expression using a visitor function
pub fn transform_expression<T: NumericType + Clone, F>(
    expr: &ASTRepr<T>,
    transformer: &F,
) -> ASTRepr<T>
where
    F: Fn(&ASTRepr<T>) -> Option<ASTRepr<T>>,
{
    // First try to transform the current expression
    if let Some(transformed) = transformer(expr) {
        return transformed;
    }
    
    // If no transformation, recursively transform children
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => expr.clone(),
        ASTRepr::Add(left, right) => {
            let left_transformed = transform_expression(left, transformer);
            let right_transformed = transform_expression(right, transformer);
            ASTRepr::Add(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Sub(left, right) => {
            let left_transformed = transform_expression(left, transformer);
            let right_transformed = transform_expression(right, transformer);
            ASTRepr::Sub(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Mul(left, right) => {
            let left_transformed = transform_expression(left, transformer);
            let right_transformed = transform_expression(right, transformer);
            ASTRepr::Mul(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Div(left, right) => {
            let left_transformed = transform_expression(left, transformer);
            let right_transformed = transform_expression(right, transformer);
            ASTRepr::Div(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Pow(left, right) => {
            let left_transformed = transform_expression(left, transformer);
            let right_transformed = transform_expression(right, transformer);
            ASTRepr::Pow(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Neg(inner) => {
            let inner_transformed = transform_expression(inner, transformer);
            ASTRepr::Neg(Box::new(inner_transformed))
        }
        ASTRepr::Ln(inner) => {
            let inner_transformed = transform_expression(inner, transformer);
            ASTRepr::Ln(Box::new(inner_transformed))
        }
        ASTRepr::Exp(inner) => {
            let inner_transformed = transform_expression(inner, transformer);
            ASTRepr::Exp(Box::new(inner_transformed))
        }
        ASTRepr::Sin(inner) => {
            let inner_transformed = transform_expression(inner, transformer);
            ASTRepr::Sin(Box::new(inner_transformed))
        }
        ASTRepr::Cos(inner) => {
            let inner_transformed = transform_expression(inner, transformer);
            ASTRepr::Cos(Box::new(inner_transformed))
        }
        ASTRepr::Sqrt(inner) => {
            let inner_transformed = transform_expression(inner, transformer);
            ASTRepr::Sqrt(Box::new(inner_transformed))
        }
    }
}

/// Check if an expression is a constant
pub fn is_constant<T: NumericType>(expr: &ASTRepr<T>) -> bool {
    matches!(expr, ASTRepr::Constant(_))
}

/// Check if an expression is a variable
pub fn is_variable<T: NumericType>(expr: &ASTRepr<T>) -> bool {
    matches!(expr, ASTRepr::Variable(_))
}

/// Check if an expression is zero (constant 0)
pub fn is_zero<T: NumericType + Float>(expr: &ASTRepr<T>, tolerance: Option<f64>) -> bool {
    if let ASTRepr::Constant(value) = expr {
        let tol = tolerance.unwrap_or(1e-12);
        value.abs() < T::from(tol).unwrap_or_else(|| T::from(1e-12).unwrap())
    } else {
        false
    }
}

/// Check if an expression is one (constant 1)
pub fn is_one<T: NumericType + Float>(expr: &ASTRepr<T>, tolerance: Option<f64>) -> bool {
    if let ASTRepr::Constant(value) = expr {
        let tol = tolerance.unwrap_or(1e-12);
        let diff = (*value - T::one()).abs();
        diff < T::from(tol).unwrap_or_else(|| T::from(1e-12).unwrap())
    } else {
        false
    }
}

/// Extract the constant value if the expression is a constant
pub fn extract_constant<T: NumericType>(expr: &ASTRepr<T>) -> Option<T> {
    match expr {
        ASTRepr::Constant(value) => Some(value.clone()),
        _ => None,
    }
}

/// Extract the variable index if the expression is a variable
pub fn extract_variable_index<T: NumericType>(expr: &ASTRepr<T>) -> Option<usize> {
    if let ASTRepr::Variable(index) = expr {
        Some(*index)
    } else {
        None
    }
}

/// Count the total number of nodes in an expression tree
pub fn count_nodes<T: NumericType>(expr: &ASTRepr<T>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => 1 + count_nodes(left) + count_nodes(right),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + count_nodes(inner),
    }
}

/// Calculate the depth of an expression tree
pub fn expression_depth<T: NumericType>(expr: &ASTRepr<T>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => {
            1 + expression_depth(left).max(expression_depth(right))
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + expression_depth(inner),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    fn test_expressions_equal() {
        let expr1 = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        let expr2 = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        let expr3 = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.1));

        assert!(expressions_equal_default(&expr1, &expr2));
        assert!(!expressions_equal_default(&expr1, &expr3));
    }

    #[test]
    fn test_contains_variable() {
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        
        assert!(contains_variable_by_index(&expr, 0));
        assert!(!contains_variable_by_index(&expr, 1));
    }

    #[test]
    fn test_collect_variables() {
        let expr = ASTEval::add(
            ASTEval::mul(ASTEval::var(0), ASTEval::var(1)),
            ASTEval::var(2),
        );
        
        let variables = collect_variable_indices(&expr);
        assert_eq!(variables.len(), 3);
        assert!(variables.contains(&0));
        assert!(variables.contains(&1));
        assert!(variables.contains(&2));
    }

    #[test]
    fn test_is_constant_and_variable() {
        let const_expr = ASTEval::constant(5.0);
        let var_expr: ASTRepr<f64> = ASTEval::var(0);
        
        assert!(is_constant(&const_expr));
        assert!(!is_constant(&var_expr));
        assert!(is_variable(&var_expr));
        assert!(!is_variable(&const_expr));
    }

    #[test]
    fn test_is_zero_and_one() {
        let zero_expr = ASTEval::constant(0.0);
        let one_expr = ASTEval::constant(1.0);
        let other_expr = ASTEval::constant(2.0);
        
        assert!(is_zero(&zero_expr, None));
        assert!(!is_zero(&one_expr, None));
        assert!(is_one(&one_expr, None));
        assert!(!is_one(&other_expr, None));
    }

    #[test]
    fn test_count_nodes_and_depth() {
        let simple_expr = ASTEval::constant(1.0);
        let complex_expr = ASTEval::add(
            ASTEval::mul(ASTEval::var(0), ASTEval::var(1)),
            ASTEval::constant(1.0),
        );
        
        assert_eq!(count_nodes(&simple_expr), 1);
        assert_eq!(count_nodes(&complex_expr), 5); // add + mul + var(0) + var(1) + const(1)
        
        assert_eq!(expression_depth(&simple_expr), 1);
        assert_eq!(expression_depth(&complex_expr), 3); // add -> mul -> var
    }

    #[test]
    fn test_transform_expression() {
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(0.0));
        
        // Transform to remove addition with zero
        let transformer = |e: &ASTRepr<f64>| -> Option<ASTRepr<f64>> {
            if let ASTRepr::Add(left, right) = e {
                if is_zero(right, None) {
                    return Some((**left).clone());
                }
                if is_zero(left, None) {
                    return Some((**right).clone());
                }
            }
            None
        };
        
        let transformed = transform_expression(&expr, &transformer);
        assert!(is_variable(&transformed));
        assert_eq!(extract_variable_index(&transformed), Some(0));
    }
} 