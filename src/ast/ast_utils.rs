//! Utility functions for AST manipulation and analysis
//!
//! This module provides helper functions for working with AST expressions,
//! including equality checking, variable analysis, and expression transformation.

use crate::ast::ASTRepr;
use crate::final_tagless::traits::NumericType;
use num_traits::Float;

/// Configuration for expression equality checking
#[derive(Debug, Clone)]
pub struct EqualityConfig {
    /// Whether to consider expressions equal if they differ only by commutativity
    pub commutative_equality: bool,
    /// Tolerance for floating-point constant comparison
    pub float_tolerance: f64,
}

impl Default for EqualityConfig {
    fn default() -> Self {
        Self {
            commutative_equality: true,
            float_tolerance: 1e-10,
        }
    }
}

/// Check if two expressions are structurally equal
pub fn expressions_equal<T>(left: &ASTRepr<T>, right: &ASTRepr<T>, config: &EqualityConfig) -> bool
where
    T: NumericType + PartialEq + std::fmt::Debug + Clone + Default + Send + Sync,
{
    match (left, right) {
        // Constants
        (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
            // For floating point, use tolerance
            if let (Ok(a_f64), Ok(b_f64)) =
                (a.to_string().parse::<f64>(), b.to_string().parse::<f64>())
            {
                (a_f64 - b_f64).abs() < config.float_tolerance
            } else {
                a == b
            }
        }

        // Variables
        (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,

        // Binary operations
        (ASTRepr::Add(a1, a2), ASTRepr::Add(b1, b2))
        | (ASTRepr::Sub(a1, a2), ASTRepr::Sub(b1, b2))
        | (ASTRepr::Mul(a1, a2), ASTRepr::Mul(b1, b2))
        | (ASTRepr::Div(a1, a2), ASTRepr::Div(b1, b2))
        | (ASTRepr::Pow(a1, a2), ASTRepr::Pow(b1, b2)) => {
            if config.commutative_equality {
                // For commutative operations, check both orders
                match left {
                    ASTRepr::Add(_, _) | ASTRepr::Mul(_, _) => {
                        (expressions_equal(a1, b1, config) && expressions_equal(a2, b2, config))
                            || (expressions_equal(a1, b2, config)
                                && expressions_equal(a2, b1, config))
                    }
                    _ => expressions_equal(a1, b1, config) && expressions_equal(a2, b2, config),
                }
            } else {
                expressions_equal(a1, b1, config) && expressions_equal(a2, b2, config)
            }
        }

        // Unary operations
        (ASTRepr::Neg(a), ASTRepr::Neg(b)) => expressions_equal(a, b, config),

        // Trigonometric functions
        (ASTRepr::Trig(a), ASTRepr::Trig(b)) => {
            // For now, just check structural equality
            // In a full implementation, we'd compare the function types and arguments
            a == b
        }

        // Hyperbolic functions
        (ASTRepr::Hyperbolic(a), ASTRepr::Hyperbolic(b)) => a == b,

        // Feature-gated functions
        #[cfg(feature = "logexp")]
        (ASTRepr::Log(a), ASTRepr::Log(b)) | (ASTRepr::Exp(a), ASTRepr::Exp(b)) => {
            expressions_equal(a, b, config)
        }

        #[cfg(feature = "logexp")]
        (ASTRepr::LogExp(a), ASTRepr::LogExp(b)) => a == b,

        #[cfg(feature = "special")]
        (ASTRepr::Special(a), ASTRepr::Special(b)) => a == b,

        #[cfg(feature = "linear_algebra")]
        (ASTRepr::LinearAlgebra(a), ASTRepr::LinearAlgebra(b)) => a == b,

        // Different types are not equal
        _ => false,
    }
}

/// Convenience function for default expression equality checking (backward compatibility)
pub fn expressions_equal_default<T>(left: &ASTRepr<T>, right: &ASTRepr<T>) -> bool
where
    T: NumericType + PartialEq + std::fmt::Debug + Clone + Default + Send + Sync,
{
    expressions_equal(left, right, &EqualityConfig::default())
}

/// Check if an expression contains a specific variable by index
pub fn contains_variable_by_index<T>(expr: &ASTRepr<T>, var_index: usize) -> bool
where
    T: NumericType + std::fmt::Debug + Clone + Default + Send + Sync,
{
    match expr {
        ASTRepr::Constant(_) => false,
        ASTRepr::Variable(index) => *index == var_index,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => {
            contains_variable_by_index(left, var_index)
                || contains_variable_by_index(right, var_index)
        }
        ASTRepr::Neg(inner) => contains_variable_by_index(inner, var_index),

        // Feature-gated functions
        #[cfg(feature = "logexp")]
        ASTRepr::Log(inner) | ASTRepr::Exp(inner) => contains_variable_by_index(inner, var_index),

        // Function categories - for now, return false (placeholder)
        // In a full implementation, we'd check their inner expressions
        ASTRepr::Trig(_) => false,
        ASTRepr::Hyperbolic(_) => false,

        #[cfg(feature = "logexp")]
        ASTRepr::LogExp(_) => false,

        #[cfg(feature = "special")]
        ASTRepr::Special(_) => false,

        #[cfg(feature = "linear_algebra")]
        ASTRepr::LinearAlgebra(_) => false,
    }
}

/// Get all variable indices used in an expression
pub fn get_variable_indices<T>(expr: &ASTRepr<T>) -> Vec<usize>
where
    T: NumericType + std::fmt::Debug + Clone + Default + Send + Sync,
{
    let mut indices = Vec::new();
    collect_variable_indices(expr, &mut indices);
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Collect variable indices into a vector (public for backward compatibility)
pub fn collect_variable_indices<T>(expr: &ASTRepr<T>, indices: &mut Vec<usize>)
where
    T: NumericType + std::fmt::Debug + Clone + Default + Send + Sync,
{
    match expr {
        ASTRepr::Constant(_) => {}
        ASTRepr::Variable(index) => indices.push(*index),
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => {
            collect_variable_indices(left, indices);
            collect_variable_indices(right, indices);
        }
        ASTRepr::Neg(inner) => collect_variable_indices(inner, indices),

        // Feature-gated functions
        #[cfg(feature = "logexp")]
        ASTRepr::Log(inner) | ASTRepr::Exp(inner) => collect_variable_indices(inner, indices),

        // Function categories - for now, do nothing (placeholder)
        // In a full implementation, we'd check their inner expressions
        ASTRepr::Trig(_) => {}
        ASTRepr::Hyperbolic(_) => {}

        #[cfg(feature = "logexp")]
        ASTRepr::LogExp(_) => {}

        #[cfg(feature = "special")]
        ASTRepr::Special(_) => {}

        #[cfg(feature = "linear_algebra")]
        ASTRepr::LinearAlgebra(_) => {}
    }
}

/// Transform an expression by applying a function to each node
pub fn transform_expression<T, F>(expr: &ASTRepr<T>, transform_fn: &F) -> ASTRepr<T>
where
    T: NumericType + std::fmt::Debug + Clone + Default + Send + Sync,
    F: Fn(&ASTRepr<T>) -> ASTRepr<T>,
{
    let transformed = match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => expr.clone(),
        ASTRepr::Add(left, right) => {
            let left_transformed = transform_expression(left, transform_fn);
            let right_transformed = transform_expression(right, transform_fn);
            ASTRepr::Add(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Sub(left, right) => {
            let left_transformed = transform_expression(left, transform_fn);
            let right_transformed = transform_expression(right, transform_fn);
            ASTRepr::Sub(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Mul(left, right) => {
            let left_transformed = transform_expression(left, transform_fn);
            let right_transformed = transform_expression(right, transform_fn);
            ASTRepr::Mul(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Div(left, right) => {
            let left_transformed = transform_expression(left, transform_fn);
            let right_transformed = transform_expression(right, transform_fn);
            ASTRepr::Div(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Pow(base, exp) => {
            let base_transformed = transform_expression(base, transform_fn);
            let exp_transformed = transform_expression(exp, transform_fn);
            ASTRepr::Pow(Box::new(base_transformed), Box::new(exp_transformed))
        }
        ASTRepr::Neg(inner) => {
            let inner_transformed = transform_expression(inner, transform_fn);
            ASTRepr::Neg(Box::new(inner_transformed))
        }

        // Feature-gated functions
        #[cfg(feature = "logexp")]
        ASTRepr::Log(inner) => {
            let inner_transformed = transform_expression(inner, transform_fn);
            ASTRepr::Log(Box::new(inner_transformed))
        }
        #[cfg(feature = "logexp")]
        ASTRepr::Exp(inner) => {
            let inner_transformed = transform_expression(inner, transform_fn);
            ASTRepr::Exp(Box::new(inner_transformed))
        }

        // Function categories - for now, just clone them
        // In a full implementation, we'd transform their inner expressions
        ASTRepr::Trig(category) => ASTRepr::Trig(category.clone()),
        ASTRepr::Hyperbolic(category) => ASTRepr::Hyperbolic(category.clone()),

        #[cfg(feature = "logexp")]
        ASTRepr::LogExp(category) => ASTRepr::LogExp(category.clone()),

        #[cfg(feature = "special")]
        ASTRepr::Special(category) => ASTRepr::Special(category.clone()),

        #[cfg(feature = "linear_algebra")]
        ASTRepr::LinearAlgebra(category) => ASTRepr::LinearAlgebra(category.clone()),
    };

    transform_fn(&transformed)
}

/// Count the total number of nodes in an expression tree
pub fn count_nodes<T>(expr: &ASTRepr<T>) -> usize
where
    T: NumericType + std::fmt::Debug + Clone + Default + Send + Sync,
{
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => 1 + count_nodes(left) + count_nodes(right),
        ASTRepr::Neg(inner) => 1 + count_nodes(inner),

        // Feature-gated functions
        #[cfg(feature = "logexp")]
        ASTRepr::Log(inner) | ASTRepr::Exp(inner) => 1 + count_nodes(inner),

        // Function categories
        ASTRepr::Trig(_) => 1,
        ASTRepr::Hyperbolic(_) => 1,

        #[cfg(feature = "logexp")]
        ASTRepr::LogExp(_) => 1,

        #[cfg(feature = "special")]
        ASTRepr::Special(_) => 1,

        #[cfg(feature = "linear_algebra")]
        ASTRepr::LinearAlgebra(_) => 1,
    }
}

/// Calculate the depth of an expression tree
pub fn expression_depth<T>(expr: &ASTRepr<T>) -> usize
where
    T: NumericType + std::fmt::Debug + Clone + Default + Send + Sync,
{
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => 1 + expression_depth(left).max(expression_depth(right)),
        ASTRepr::Neg(inner) => 1 + expression_depth(inner),

        // Feature-gated functions
        #[cfg(feature = "logexp")]
        ASTRepr::Log(inner) | ASTRepr::Exp(inner) => 1 + expression_depth(inner),

        // Function categories
        ASTRepr::Trig(_) => 1,
        ASTRepr::Hyperbolic(_) => 1,

        #[cfg(feature = "logexp")]
        ASTRepr::LogExp(_) => 1,

        #[cfg(feature = "special")]
        ASTRepr::Special(_) => 1,

        #[cfg(feature = "linear_algebra")]
        ASTRepr::LinearAlgebra(_) => 1,
    }
}
