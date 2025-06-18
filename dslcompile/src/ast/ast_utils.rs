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
//! - **Stack-based Visitors**: Non-recursive analysis to prevent stack overflow

use crate::ast::{
    ASTRepr, Scalar, VariableRegistry,
    ast_repr::{Collection, Lambda},
};
use num_traits::Float;
use std::collections::HashSet;

/// Configuration for AST utilities
#[derive(Debug, Clone)]
pub struct ASTUtilConfig {
    /// Tolerance for floating-point equality comparisons
    pub tolerance: f64,
    /// Maximum recursion depth for expression analysis
    pub max_depth: usize,
}

impl Default for ASTUtilConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_depth: 100,
        }
    }
}

/// Unified expression equality checking with configurable tolerance
pub fn expressions_equal<T: Scalar + Float>(
    expr1: &ASTRepr<T>,
    expr2: &ASTRepr<T>,
    config: &ASTUtilConfig,
) -> bool {
    match (expr1, expr2) {
        (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
            let diff = (*a - *b).abs();
            diff < T::from(config.tolerance).unwrap_or_else(|| T::from(1e-12).unwrap())
        }
        (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,
        (ASTRepr::Add(a_terms), ASTRepr::Add(b_terms)) => {
            // Multiset equality: same elements regardless of order
            if a_terms.len() == b_terms.len() {
                // For now, simple element-wise comparison (could be optimized for true multiset comparison)
                // MultiSet is already sorted, so we can compare elements directly
                let a_elements: Vec<_> = a_terms.elements().collect();
                let b_elements: Vec<_> = b_terms.elements().collect();
                a_elements
                    .iter()
                    .zip(b_elements.iter())
                    .all(|(a, b)| expressions_equal(a, b, config))
            } else {
                false
            }
        }
        (ASTRepr::Mul(a_factors), ASTRepr::Mul(b_factors)) => {
            // Multiset equality: same elements regardless of order
            if a_factors.len() == b_factors.len() {
                // For now, simple element-wise comparison (could be optimized for true multiset comparison)
                // MultiSet is already sorted, so we can compare elements directly
                let a_elements: Vec<_> = a_factors.elements().collect();
                let b_elements: Vec<_> = b_factors.elements().collect();
                a_elements
                    .iter()
                    .zip(b_elements.iter())
                    .all(|(a, b)| expressions_equal(a, b, config))
            } else {
                false
            }
        }
        (ASTRepr::Sub(a1, a2), ASTRepr::Sub(b1, b2))
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
pub fn expressions_equal_default<T: Scalar + Float>(
    expr1: &ASTRepr<T>,
    expr2: &ASTRepr<T>,
) -> bool {
    expressions_equal(expr1, expr2, &ASTUtilConfig::default())
}

/// Check if an expression contains a variable by index
pub fn contains_variable_by_index<T: Scalar>(expr: &ASTRepr<T>, var_index: usize) -> bool {
    match expr {
        ASTRepr::Constant(_) => false,
        ASTRepr::Variable(index) => *index == var_index,
        ASTRepr::Add(terms) => terms
            .elements()
            .any(|term| contains_variable_by_index(term, var_index)),
        ASTRepr::Mul(factors) => factors
            .elements()
            .any(|factor| contains_variable_by_index(factor, var_index)),
        ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) | ASTRepr::Pow(left, right) => {
            contains_variable_by_index(left, var_index)
                || contains_variable_by_index(right, var_index)
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => contains_variable_by_index(inner, var_index),
        ASTRepr::Sum(_collection) => {
            // TODO: Check Collection for variable usage in new format
            false // Placeholder until Collection variable checking is implemented
        }
        ASTRepr::BoundVar(index) => *index == var_index,
        ASTRepr::Let(_, binding, body) => {
            contains_variable_by_index(binding, var_index)
                || contains_variable_by_index(body, var_index)
        }
        ASTRepr::Lambda(lambda) => {
            // Check if the lambda body contains the variable, but ignore lambda's own variables
            if lambda.var_indices.contains(&var_index) {
                false // Variable is bound by this lambda
            } else {
                contains_variable_by_index(&lambda.body, var_index)
            }
        }
    }
}

/// Collect all variable indices used in an expression
pub fn collect_variable_indices<T: Scalar>(expr: &ASTRepr<T>) -> HashSet<usize> {
    let mut variables = HashSet::new();
    collect_variable_indices_recursive(expr, &mut variables);
    variables
}

/// Recursive helper for collecting variable indices
fn collect_variable_indices_recursive<T: Scalar>(
    expr: &ASTRepr<T>,
    variables: &mut HashSet<usize>,
) {
    match expr {
        ASTRepr::Constant(_) => {}
        ASTRepr::Variable(index) => {
            variables.insert(*index);
        }
        ASTRepr::Add(terms) => {
            for term in terms.elements() {
                collect_variable_indices_recursive(term, variables);
            }
        }
        ASTRepr::Mul(factors) => {
            for factor in factors.elements() {
                collect_variable_indices_recursive(factor, variables);
            }
        }
        ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) | ASTRepr::Pow(left, right) => {
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
        ASTRepr::Sum(collection) => {
            collect_variables_from_collection(collection, variables);
        }
        ASTRepr::BoundVar(index) => {
            variables.insert(*index);
        }
        ASTRepr::Let(_, binding, body) => {
            collect_variable_indices_recursive(binding, variables);
            collect_variable_indices_recursive(body, variables);
        }
        ASTRepr::Lambda(lambda) => {
            collect_variables_from_lambda(lambda, variables);
        }
    }
}

/// Collect variables from Collection structures
fn collect_variables_from_collection<T: Scalar>(
    collection: &Collection<T>,
    variables: &mut HashSet<usize>,
) {
    use crate::ast::ast_repr::Collection;
    match collection {
        Collection::Empty => {}
        Collection::Singleton(expr) => {
            collect_variable_indices_recursive(expr, variables);
        }
        Collection::Range { start, end } => {
            collect_variable_indices_recursive(start, variables);
            collect_variable_indices_recursive(end, variables);
        }
        Collection::Variable(index) => {
            variables.insert(*index);
        }
        Collection::Filter {
            collection,
            predicate,
        } => {
            collect_variables_from_collection(collection, variables);
            collect_variable_indices_recursive(predicate, variables);
        }
        Collection::Map { lambda, collection } => {
            collect_variables_from_lambda(lambda, variables);
            collect_variables_from_collection(collection, variables);
        }
        Collection::DataArray(_) => {
            // Embedded data arrays don't contain variables
        }
    }
}

/// Collect variables from Lambda structures
fn collect_variables_from_lambda<T: Scalar>(lambda: &Lambda<T>, variables: &mut HashSet<usize>) {
    // Note: lambda var_indices are bound variables, not free variables
    // We only collect variables from the body expression
    collect_variable_indices_recursive(&lambda.body, variables);
}

/// Generate debug names for variables using a registry
#[must_use]
pub fn generate_variable_names(
    indices: &HashSet<usize>,
    registry: &VariableRegistry,
) -> Vec<String> {
    let mut names = Vec::new();

    for &index in indices {
        if index < registry.len() {
            names.push(registry.debug_name(index));
        } else {
            names.push(format!("var_{index}"));
        }
    }

    names.sort();
    names
}

/// Generic expression traversal with a visitor function
pub fn traverse_expression<T: Scalar, F>(expr: &ASTRepr<T>, mut visitor: F)
where
    F: FnMut(&ASTRepr<T>),
{
    visitor(expr);

    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => {}
        ASTRepr::Add(terms) => {
            for term in terms.elements() {
                traverse_expression(term, &mut visitor);
            }
        }
        ASTRepr::Mul(factors) => {
            for factor in factors.elements() {
                traverse_expression(factor, &mut visitor);
            }
        }
        ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) | ASTRepr::Pow(left, right) => {
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
        ASTRepr::Sum(_collection) => {
            // TODO: Traverse Collection in new format
            // Placeholder until Collection traversal is implemented
        }
        ASTRepr::BoundVar(_) => {
            // BoundVar doesn't need traversal
        }
        ASTRepr::Let(_, binding, body) => {
            traverse_expression(binding, &mut visitor);
            traverse_expression(body, &mut visitor);
        }
        ASTRepr::Lambda(lambda) => {
            traverse_expression(&lambda.body, &mut visitor);
        }
    }
}

/// Transform an expression using a visitor function
pub fn transform_expression<T: Scalar + Clone + num_traits::Zero + num_traits::One, F>(
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
        ASTRepr::Add(terms) => {
            let transformed_terms: Vec<ASTRepr<T>> = terms
                .elements()
                .map(|term| transform_expression(term, transformer))
                .collect();
            ASTRepr::add_multiset(transformed_terms)
        }
        ASTRepr::Sub(left, right) => {
            let left_transformed = transform_expression(left, transformer);
            let right_transformed = transform_expression(right, transformer);
            ASTRepr::Sub(Box::new(left_transformed), Box::new(right_transformed))
        }
        ASTRepr::Mul(factors) => {
            let transformed_factors: Vec<ASTRepr<T>> = factors
                .elements()
                .map(|factor| transform_expression(factor, transformer))
                .collect();
            ASTRepr::mul_multiset(transformed_factors)
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
        ASTRepr::Sum(_collection) => {
            // TODO: Transform Collection format
            expr.clone() // Placeholder until Collection transformation is implemented
        }
        ASTRepr::BoundVar(_) => expr.clone(),
        ASTRepr::Let(var_index, binding, body) => {
            let binding_transformed = transform_expression(binding, transformer);
            let body_transformed = transform_expression(body, transformer);
            ASTRepr::Let(
                *var_index,
                Box::new(binding_transformed),
                Box::new(body_transformed),
            )
        }
        ASTRepr::Lambda(lambda) => {
            let body_transformed = transform_expression(&lambda.body, transformer);
            let transformed_lambda = Lambda {
                var_indices: lambda.var_indices.clone(),
                body: Box::new(body_transformed),
            };
            ASTRepr::Lambda(Box::new(transformed_lambda))
        }
    }
}

/// Check if an expression is a constant
pub fn is_constant<T: Scalar>(expr: &ASTRepr<T>) -> bool {
    matches!(expr, ASTRepr::Constant(_))
}

/// Check if an expression is a variable
pub fn is_variable<T: Scalar>(expr: &ASTRepr<T>) -> bool {
    matches!(expr, ASTRepr::Variable(_))
}

/// Check if an expression is zero (constant 0)
pub fn is_zero<T: Scalar + Float>(expr: &ASTRepr<T>, tolerance: Option<f64>) -> bool {
    if let ASTRepr::Constant(value) = expr {
        let tol = tolerance.unwrap_or(1e-12);
        value.abs() < T::from(tol).unwrap_or_else(|| T::from(1e-12).unwrap())
    } else {
        false
    }
}

/// Check if an expression is one (constant 1)
pub fn is_one<T: Scalar + Float>(expr: &ASTRepr<T>, tolerance: Option<f64>) -> bool {
    if let ASTRepr::Constant(value) = expr {
        let tol = tolerance.unwrap_or(1e-12);
        let diff = (*value - T::one()).abs();
        diff < T::from(tol).unwrap_or_else(|| T::from(1e-12).unwrap())
    } else {
        false
    }
}

/// Extract the constant value if the expression is a constant
pub fn extract_constant<T: Scalar>(expr: &ASTRepr<T>) -> Option<T> {
    match expr {
        ASTRepr::Constant(value) => Some(value.clone()),
        _ => None,
    }
}

/// Extract the variable index if the expression is a variable
pub fn extract_variable_index<T: Scalar>(expr: &ASTRepr<T>) -> Option<usize> {
    if let ASTRepr::Variable(index) = expr {
        Some(*index)
    } else {
        None
    }
}

/// Count the total number of nodes in an expression tree
pub fn count_nodes<T: Scalar>(expr: &ASTRepr<T>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(terms) => 1 + terms.elements().map(count_nodes).sum::<usize>(),
        ASTRepr::Mul(factors) => 1 + factors.elements().map(count_nodes).sum::<usize>(),
        ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) | ASTRepr::Pow(left, right) => {
            1 + count_nodes(left) + count_nodes(right)
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + count_nodes(inner),
        ASTRepr::Sum(_collection) => {
            // TODO: Count nodes in Collection format
            1 // Placeholder until Collection node counting is implemented
        }
        ASTRepr::BoundVar(_) => 1,
        ASTRepr::Let(_, expr, body) => 1 + count_nodes(expr) + count_nodes(body),
        ASTRepr::Lambda(lambda) => 1 + count_nodes(&lambda.body),
    }
}

/// Calculate the depth of an expression tree
pub fn expression_depth<T: Scalar>(expr: &ASTRepr<T>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(terms) => 1 + terms.elements().map(expression_depth).max().unwrap_or(0),
        ASTRepr::Mul(factors) => 1 + factors.elements().map(expression_depth).max().unwrap_or(0),
        ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) | ASTRepr::Pow(left, right) => {
            1 + expression_depth(left).max(expression_depth(right))
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + expression_depth(inner),
        ASTRepr::Sum(_collection) => {
            // TODO: Calculate depth for Collection format
            1 // Placeholder until Collection depth calculation is implemented
        }
        ASTRepr::BoundVar(_) => 1,
        ASTRepr::Let(_, expr, body) => 1 + expression_depth(expr).max(expression_depth(body)),
        ASTRepr::Lambda(lambda) => 1 + expression_depth(&lambda.body),
    }
}

/// Shared AST conversion utilities to eliminate duplication across modules
pub mod conversion {
    use crate::ast::{
        Scalar,
        ast_repr::{ASTRepr, Collection, Lambda},
    };

    /// Convert AST from one numeric type to f64
    pub fn convert_ast_to_f64<T: Scalar>(ast: &ASTRepr<T>) -> ASTRepr<f64>
    where
        T: Into<f64> + Clone,
    {
        match ast {
            ASTRepr::Constant(val) => ASTRepr::Constant(val.clone().into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(terms) => {
                ASTRepr::add_multiset(terms.elements().map(convert_ast_to_f64).collect())
            }
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(convert_ast_to_f64(left)),
                Box::new(convert_ast_to_f64(right)),
            ),
            ASTRepr::Mul(factors) => {
                ASTRepr::mul_multiset(factors.elements().map(convert_ast_to_f64).collect())
            }
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(convert_ast_to_f64(left)),
                Box::new(convert_ast_to_f64(right)),
            ),
            ASTRepr::Pow(left, right) => ASTRepr::Pow(
                Box::new(convert_ast_to_f64(left)),
                Box::new(convert_ast_to_f64(right)),
            ),
            ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(convert_ast_to_f64(inner))),
            ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(convert_ast_to_f64(inner))),
            ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(convert_ast_to_f64(inner))),
            ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(convert_ast_to_f64(inner))),
            ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(convert_ast_to_f64(inner))),
            ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(convert_ast_to_f64(inner))),
            ASTRepr::Sum(collection) => {
                ASTRepr::Sum(Box::new(convert_collection_to_f64(collection)))
            }
            ASTRepr::BoundVar(idx) => ASTRepr::BoundVar(*idx),
            ASTRepr::Let(id, expr, body) => ASTRepr::Let(
                *id,
                Box::new(convert_ast_to_f64(expr)),
                Box::new(convert_ast_to_f64(body)),
            ),
            ASTRepr::Lambda(lambda) => ASTRepr::Lambda(Box::new(convert_lambda_to_f64(lambda))),
        }
    }

    /// Convert AST from one numeric type to f32
    pub fn convert_ast_to_f32<T: Scalar>(ast: &ASTRepr<T>) -> ASTRepr<f32>
    where
        T: Into<f32> + Clone,
    {
        match ast {
            ASTRepr::Constant(val) => ASTRepr::Constant(val.clone().into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(terms) => {
                ASTRepr::add_multiset(terms.elements().map(convert_ast_to_f32).collect())
            }
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(convert_ast_to_f32(left)),
                Box::new(convert_ast_to_f32(right)),
            ),
            ASTRepr::Mul(factors) => {
                ASTRepr::mul_multiset(factors.elements().map(convert_ast_to_f32).collect())
            }
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(convert_ast_to_f32(left)),
                Box::new(convert_ast_to_f32(right)),
            ),
            ASTRepr::Pow(left, right) => ASTRepr::Pow(
                Box::new(convert_ast_to_f32(left)),
                Box::new(convert_ast_to_f32(right)),
            ),
            ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(convert_ast_to_f32(inner))),
            ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(convert_ast_to_f32(inner))),
            ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(convert_ast_to_f32(inner))),
            ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(convert_ast_to_f32(inner))),
            ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(convert_ast_to_f32(inner))),
            ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(convert_ast_to_f32(inner))),
            ASTRepr::Sum(collection) => {
                ASTRepr::Sum(Box::new(convert_collection_to_f32(collection)))
            }
            ASTRepr::BoundVar(idx) => ASTRepr::BoundVar(*idx),
            ASTRepr::Let(id, expr, body) => ASTRepr::Let(
                *id,
                Box::new(convert_ast_to_f32(expr)),
                Box::new(convert_ast_to_f32(body)),
            ),
            ASTRepr::Lambda(lambda) => ASTRepr::Lambda(Box::new(convert_lambda_to_f32(lambda))),
        }
    }

    /// Convert Collection from one numeric type to f64
    #[must_use]
    pub fn convert_collection_to_f64<T: Scalar>(collection: &Collection<T>) -> Collection<f64>
    where
        T: Into<f64> + Clone,
    {
        match collection {
            Collection::Empty => Collection::Empty,
            Collection::Singleton(expr) => {
                Collection::Singleton(Box::new(convert_ast_to_f64(expr)))
            }
            Collection::Range { start, end } => Collection::Range {
                start: Box::new(convert_ast_to_f64(start)),
                end: Box::new(convert_ast_to_f64(end)),
            },
            Collection::Variable(index) => Collection::Variable(*index),
            Collection::Filter {
                collection,
                predicate,
            } => Collection::Filter {
                collection: Box::new(convert_collection_to_f64(collection)),
                predicate: Box::new(convert_ast_to_f64(predicate)),
            },
            Collection::Map { lambda, collection } => Collection::Map {
                lambda: Box::new(convert_lambda_to_f64(lambda)),
                collection: Box::new(convert_collection_to_f64(collection)),
            },
            Collection::DataArray(_data) => {
                // DataArray type conversion not supported - should use proper type-safe conversion
                panic!(
                    "DataArray type conversion from {} to f64 not implemented",
                    std::any::type_name::<T>()
                )
            }
        }
    }

    /// Convert Collection from one numeric type to f32
    #[must_use]
    pub fn convert_collection_to_f32<T: Scalar>(collection: &Collection<T>) -> Collection<f32>
    where
        T: Into<f32> + Clone,
    {
        match collection {
            Collection::Empty => Collection::Empty,
            Collection::Singleton(expr) => {
                Collection::Singleton(Box::new(convert_ast_to_f32(expr)))
            }
            Collection::Range { start, end } => Collection::Range {
                start: Box::new(convert_ast_to_f32(start)),
                end: Box::new(convert_ast_to_f32(end)),
            },

            Collection::Variable(index) => Collection::Variable(*index),
            Collection::Filter {
                collection,
                predicate,
            } => Collection::Filter {
                collection: Box::new(convert_collection_to_f32(collection)),
                predicate: Box::new(convert_ast_to_f32(predicate)),
            },
            Collection::Map { lambda, collection } => Collection::Map {
                lambda: Box::new(convert_lambda_to_f32(lambda)),
                collection: Box::new(convert_collection_to_f32(collection)),
            },
            Collection::DataArray(_data) => {
                // DataArray type conversion not supported - should use proper type-safe conversion
                panic!(
                    "DataArray type conversion from {} to f32 not implemented",
                    std::any::type_name::<T>()
                )
            }
        }
    }

    /// Convert Lambda from one numeric type to f64
    #[must_use]
    pub fn convert_lambda_to_f64<T: Scalar>(lambda: &Lambda<T>) -> Lambda<f64>
    where
        T: Into<f64> + Clone,
    {
        Lambda {
            var_indices: lambda.var_indices.clone(),
            body: Box::new(convert_ast_to_f64(&lambda.body)),
        }
    }

    /// Convert Lambda from one numeric type to f32
    #[must_use]
    pub fn convert_lambda_to_f32<T: Scalar>(lambda: &Lambda<T>) -> Lambda<f32>
    where
        T: Into<f32> + Clone,
    {
        Lambda {
            var_indices: lambda.var_indices.clone(),
            body: Box::new(convert_ast_to_f32(&lambda.body)),
        }
    }
}

/// Stack-based visitor implementations to prevent stack overflow on deep expressions
pub mod visitors {
    use super::{ASTRepr, Float, Scalar};
    use crate::ast::visitor::ASTVisitor;

    /// Visitor for counting operations using stack-based traversal
    pub struct OperationCountVisitor {
        count: usize,
    }

    impl Default for OperationCountVisitor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl OperationCountVisitor {
        #[must_use]
        pub fn new() -> Self {
            Self { count: 0 }
        }

        pub fn count_operations<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
            let mut visitor = Self::new();
            visitor.visit(expr).unwrap_or(0)
        }
    }

    impl<T: Scalar + Clone> ASTVisitor<T> for OperationCountVisitor {
        type Output = usize;
        type Error = ();

        fn visit_constant(&mut self, _value: &T) -> Result<Self::Output, Self::Error> {
            Ok(0) // Constants are not operations
        }

        fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(0) // Variables are not operations
        }

        fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(0) // Bound variables are not operations
        }

        // Override the main visit method to count operations correctly
        fn visit(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
            let mut total_ops = 0;

            // Count this operation if it's an operation node
            match expr {
                ASTRepr::Add(terms) => {
                    total_ops += 1; // Count the addition operation
                    for term in terms.elements() {
                        total_ops += self.visit(term)?;
                    }
                }
                ASTRepr::Sub(left, right) => {
                    total_ops += 1; // Count the subtraction operation
                    total_ops += self.visit(left)?;
                    total_ops += self.visit(right)?;
                }
                ASTRepr::Mul(factors) => {
                    total_ops += 1; // Count the multiplication operation
                    for factor in factors.elements() {
                        total_ops += self.visit(factor)?;
                    }
                }
                ASTRepr::Div(left, right) => {
                    total_ops += 1; // Count the division operation
                    total_ops += self.visit(left)?;
                    total_ops += self.visit(right)?;
                }
                ASTRepr::Pow(base, exp) => {
                    total_ops += 1; // Count the power operation
                    total_ops += self.visit(base)?;
                    total_ops += self.visit(exp)?;
                }
                ASTRepr::Neg(inner) => {
                    total_ops += 1; // Count the negation operation
                    total_ops += self.visit(inner)?;
                }
                ASTRepr::Sin(inner) => {
                    total_ops += 1; // Count the sin operation
                    total_ops += self.visit(inner)?;
                }
                ASTRepr::Cos(inner) => {
                    total_ops += 1; // Count the cos operation
                    total_ops += self.visit(inner)?;
                }
                ASTRepr::Ln(inner) => {
                    total_ops += 1; // Count the ln operation
                    total_ops += self.visit(inner)?;
                }
                ASTRepr::Exp(inner) => {
                    total_ops += 1; // Count the exp operation
                    total_ops += self.visit(inner)?;
                }
                ASTRepr::Sqrt(inner) => {
                    total_ops += 1; // Count the sqrt operation
                    total_ops += self.visit(inner)?;
                }
                ASTRepr::Sum(collection) => {
                    total_ops += 1; // Count the sum operation
                    total_ops += self.visit_collection(collection)?;
                }
                ASTRepr::Lambda(lambda) => {
                    total_ops += 1; // Count the lambda operation
                    total_ops += self.visit(&lambda.body)?;
                }
                ASTRepr::Let(_, expr, body) => {
                    total_ops += 1; // Count the let operation
                    total_ops += self.visit(expr)?;
                    total_ops += self.visit(body)?;
                }
                // Leaf nodes don't count as operations
                ASTRepr::Constant(_) => {
                    // Constants are not operations
                }
                ASTRepr::Variable(_) => {
                    // Variables are not operations
                }
                ASTRepr::BoundVar(_) => {
                    // Bound variables are not operations
                }
            }

            Ok(total_ops)
        }

        fn visit_collection(
            &mut self,
            collection: &crate::ast::ast_repr::Collection<T>,
        ) -> Result<Self::Output, Self::Error> {
            use crate::ast::ast_repr::Collection;
            let mut total_ops = 0;

            match collection {
                Collection::Empty => {
                    // Empty collections have no operations
                }
                Collection::Singleton(expr) => {
                    total_ops += self.visit(expr)?;
                }
                Collection::Range { start, end } => {
                    total_ops += self.visit(start)?;
                    total_ops += self.visit(end)?;
                }
                Collection::Variable(_) => {
                    // Collection variables are not operations
                }

                Collection::Filter {
                    collection,
                    predicate,
                } => {
                    total_ops += 1; // Count the filter operation
                    total_ops += self.visit_collection(collection)?;
                    total_ops += self.visit(predicate)?;
                }
                Collection::Map { lambda, collection } => {
                    total_ops += 1; // Count the map operation
                    total_ops += self.visit(&lambda.body)?;
                    total_ops += self.visit_collection(collection)?;
                }
                Collection::DataArray(_) => {
                    // Embedded data has no operations
                }
            }

            Ok(total_ops)
        }

        fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(1) // Default: each node is one operation
        }

        fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(0) // Empty collections have no operations
        }

        fn visit_collection_variable(
            &mut self,
            _index: usize,
        ) -> Result<Self::Output, Self::Error> {
            Ok(0) // Collection variables are not operations
        }
    }

    /// Visitor for counting summations using stack-based traversal
    pub struct SummationCountVisitor;

    impl SummationCountVisitor {
        pub fn count_summations<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
            let mut visitor = Self;
            visitor.visit(expr).unwrap_or(0)
        }
    }

    impl<T: Scalar + Clone> ASTVisitor<T> for SummationCountVisitor {
        type Output = usize;
        type Error = ();

        fn visit_constant(&mut self, _value: &T) -> Result<Self::Output, Self::Error> {
            Ok(0)
        }

        fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(0)
        }

        fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(0)
        }

        // Override the main visit method to count summations correctly
        fn visit(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
            let mut total_sums = 0;

            match expr {
                ASTRepr::Sum(collection) => {
                    total_sums += 1; // Count this summation
                    total_sums += self.visit_collection(collection)?;
                }
                // For all other nodes, recursively count summations in children
                ASTRepr::Add(terms) => {
                    for term in terms.elements() {
                        total_sums += self.visit(term)?;
                    }
                }
                ASTRepr::Mul(factors) => {
                    for factor in factors.elements() {
                        total_sums += self.visit(factor)?;
                    }
                }
                ASTRepr::Sub(left, right)
                | ASTRepr::Div(left, right)
                | ASTRepr::Pow(left, right) => {
                    total_sums += self.visit(left)?;
                    total_sums += self.visit(right)?;
                }
                ASTRepr::Neg(inner)
                | ASTRepr::Sin(inner)
                | ASTRepr::Cos(inner)
                | ASTRepr::Ln(inner)
                | ASTRepr::Exp(inner)
                | ASTRepr::Sqrt(inner) => {
                    total_sums += self.visit(inner)?;
                }
                ASTRepr::Lambda(lambda) => {
                    total_sums += self.visit(&lambda.body)?;
                }
                ASTRepr::Let(_, expr, body) => {
                    total_sums += self.visit(expr)?;
                    total_sums += self.visit(body)?;
                }
                // Leaf nodes contain no summations
                ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => {
                    // No summations in leaf nodes
                }
            }

            Ok(total_sums)
        }

        fn visit_collection(
            &mut self,
            collection: &crate::ast::ast_repr::Collection<T>,
        ) -> Result<Self::Output, Self::Error> {
            use crate::ast::ast_repr::Collection;
            let mut total_sums = 0;

            match collection {
                Collection::Empty | Collection::Variable(_) => {
                    // No summations
                }
                Collection::Singleton(expr) => {
                    total_sums += self.visit(expr)?;
                }
                Collection::Range { start, end } => {
                    total_sums += self.visit(start)?;
                    total_sums += self.visit(end)?;
                }

                Collection::Filter {
                    collection,
                    predicate,
                } => {
                    total_sums += self.visit_collection(collection)?;
                    total_sums += self.visit(predicate)?;
                }
                Collection::Map { lambda, collection } => {
                    total_sums += self.visit(&lambda.body)?;
                    total_sums += self.visit_collection(collection)?;
                }
                Collection::DataArray(_) => {
                    // Embedded data has no summations
                }
            }

            Ok(total_sums)
        }

        fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(0) // Most nodes don't contain summations
        }

        fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(0)
        }

        fn visit_collection_variable(
            &mut self,
            _index: usize,
        ) -> Result<Self::Output, Self::Error> {
            Ok(0)
        }
    }

    /// Visitor for computing summation-aware cost that accounts for runtime domain sizes
    ///
    /// This visitor provides a more realistic cost model for expressions containing summations
    /// by estimating the runtime cost based on expected summation domain sizes.
    pub struct SummationAwareCostVisitor {
        /// Default assumed size for unknown summation domains
        default_domain_size: usize,
        /// Override domain size for all summations (if Some)
        override_domain_size: Option<usize>,
    }

    impl Default for SummationAwareCostVisitor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SummationAwareCostVisitor {
        #[must_use]
        pub fn new() -> Self {
            Self {
                default_domain_size: 1000, // Assume large domains by default
                override_domain_size: None,
            }
        }

        #[must_use]
        pub fn with_default_domain_size(default_size: usize) -> Self {
            Self {
                default_domain_size: default_size,
                override_domain_size: None,
            }
        }

        #[must_use]
        pub fn with_override_domain_size(domain_size: usize) -> Self {
            Self {
                default_domain_size: 1000,
                override_domain_size: Some(domain_size),
            }
        }

        pub fn compute_cost<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
            let mut visitor = Self::new();
            visitor.visit(expr).unwrap_or(0)
        }

        pub fn compute_cost_with_domain_size<T: Scalar + Clone>(
            expr: &ASTRepr<T>,
            domain_size: usize,
        ) -> usize {
            let mut visitor = Self::with_override_domain_size(domain_size);
            visitor.visit(expr).unwrap_or(0)
        }

        /// Estimate the domain size for a collection
        fn estimate_collection_size<T: Scalar + Clone>(
            &self,
            collection: &crate::ast::ast_repr::Collection<T>,
        ) -> usize {
            use crate::ast::ast_repr::Collection;

            // If we have an override domain size, use it for all collections
            if let Some(override_size) = self.override_domain_size {
                return override_size;
            }

            match collection {
                Collection::Empty => 0,
                Collection::Singleton(_) => 1,
                Collection::Range { start, end } => {
                    // Try to extract constant range bounds
                    match (start.as_ref(), end.as_ref()) {
                        (ASTRepr::Constant(s), ASTRepr::Constant(e)) => {
                            // Try to convert to integers and compute range size
                            // This is a best-effort estimation for constant ranges
                            // For now, assume a default size for constant ranges
                            // TODO: Implement proper constant range size estimation
                            self.default_domain_size
                        }
                        _ => self.default_domain_size, // Unknown range size
                    }
                }
                Collection::Variable(_) => self.default_domain_size, // Runtime data size unknown

                Collection::Filter { collection, .. } => {
                    // Filtering reduces size, assume 50% pass through
                    self.estimate_collection_size(collection) / 2
                }
                Collection::Map { collection, .. } => {
                    // Mapping preserves size
                    self.estimate_collection_size(collection)
                }
                Collection::DataArray(data) => data.len(),
            }
        }
    }

    impl<T: Scalar + Clone> ASTVisitor<T> for SummationAwareCostVisitor {
        type Output = usize;
        type Error = ();

        fn visit_constant(&mut self, _value: &T) -> Result<Self::Output, Self::Error> {
            Ok(0) // Constants are free
        }

        fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(0) // Variables are free
        }

        fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(0) // Bound variables are free
        }

        fn visit(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
            let cost = match expr {
                // Basic arithmetic operations (1 unit each)
                ASTRepr::Add(terms) => {
                    1 + terms
                        .elements()
                        .map(|t| self.visit(t))
                        .collect::<Result<Vec<_>, _>>()?
                        .iter()
                        .sum::<usize>()
                }
                ASTRepr::Sub(left, right) => 1 + self.visit(left)? + self.visit(right)?,
                ASTRepr::Mul(factors) => {
                    1 + factors
                        .elements()
                        .map(|f| self.visit(f))
                        .collect::<Result<Vec<_>, _>>()?
                        .iter()
                        .sum::<usize>()
                }
                ASTRepr::Div(left, right) => {
                    5 + self.visit(left)? + self.visit(right)? // Division is more expensive
                }
                ASTRepr::Pow(base, exp) => {
                    10 + self.visit(base)? + self.visit(exp)? // Power is expensive
                }
                ASTRepr::Neg(inner) => 1 + self.visit(inner)?,

                // Transcendental functions (expensive)
                ASTRepr::Sin(inner) => 75 + self.visit(inner)?,
                ASTRepr::Cos(inner) => 75 + self.visit(inner)?,
                ASTRepr::Ln(inner) => 30 + self.visit(inner)?,
                ASTRepr::Exp(inner) => 40 + self.visit(inner)?,
                ASTRepr::Sqrt(inner) => 8 + self.visit(inner)?,

                // CRITICAL: Summations multiply inner cost by domain size!
                ASTRepr::Sum(collection) => {
                    let domain_size = self.estimate_collection_size(collection);
                    let collection_cost = self.visit_collection(collection)?;

                    // The key insight: summation cost = domain_size × inner_expression_cost + overhead
                    let summation_overhead = 10; // Fixed cost for setting up the loop
                    summation_overhead + domain_size * collection_cost
                }

                ASTRepr::Lambda(lambda) => {
                    // Lambda cost is the cost of its body
                    self.visit(&lambda.body)?
                }
                ASTRepr::Let(_, expr, body) => {
                    // Let binding: cost of computing expr + cost of body
                    self.visit(expr)? + self.visit(body)?
                }

                // Leaf nodes are free
                ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => 0,
            };

            Ok(cost)
        }

        fn visit_collection(
            &mut self,
            collection: &crate::ast::ast_repr::Collection<T>,
        ) -> Result<Self::Output, Self::Error> {
            use crate::ast::ast_repr::Collection;

            let cost = match collection {
                Collection::Empty => 0,
                Collection::Variable(_) => 0, // Variable reference has no cost
                Collection::Singleton(expr) => self.visit(expr)?,
                Collection::Range { start, end } => self.visit(start)? + self.visit(end)?,

                Collection::Filter {
                    collection,
                    predicate,
                } => 1 + self.visit_collection(collection)? + self.visit(predicate)?,
                Collection::Map { lambda, collection } => {
                    // Map cost: lambda body cost (will be multiplied by domain size in Sum)
                    self.visit(&lambda.body)? + self.visit_collection(collection)?
                }
                Collection::DataArray(_) => {
                    // Embedded data has no cost
                    0
                }
            };

            Ok(cost)
        }

        fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(1) // Default cost
        }

        fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(0)
        }

        fn visit_collection_variable(
            &mut self,
            _index: usize,
        ) -> Result<Self::Output, Self::Error> {
            Ok(0)
        }
    }

    /// Visitor for computing expression depth using stack-based traversal
    pub struct DepthVisitor;

    impl DepthVisitor {
        pub fn compute_depth<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
            let mut visitor = Self;
            visitor.visit(expr).unwrap_or(1)
        }
    }

    impl<T: Scalar + Clone> ASTVisitor<T> for DepthVisitor {
        type Output = usize;
        type Error = ();

        fn visit_constant(&mut self, _value: &T) -> Result<Self::Output, Self::Error> {
            Ok(1)
        }

        fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(1)
        }

        fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            Ok(1)
        }

        fn visit(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
            match expr {
                // Leaf nodes have depth 1
                ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => Ok(1),
                // Binary operations: depth = 1 + max(left_depth, right_depth)
                ASTRepr::Add(terms) => {
                    let max_depth = terms
                        .elements()
                        .map(|t| self.visit(t))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .max()
                        .unwrap_or(0);
                    Ok(1 + max_depth)
                }
                ASTRepr::Mul(factors) => {
                    let max_depth = factors
                        .elements()
                        .map(|f| self.visit(f))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .max()
                        .unwrap_or(0);
                    Ok(1 + max_depth)
                }
                ASTRepr::Sub(left, right)
                | ASTRepr::Div(left, right)
                | ASTRepr::Pow(left, right) => {
                    let left_depth = self.visit(left)?;
                    let right_depth = self.visit(right)?;
                    Ok(1 + left_depth.max(right_depth))
                }
                // Unary operations: depth = 1 + inner_depth
                ASTRepr::Neg(inner)
                | ASTRepr::Sin(inner)
                | ASTRepr::Cos(inner)
                | ASTRepr::Ln(inner)
                | ASTRepr::Exp(inner)
                | ASTRepr::Sqrt(inner) => {
                    let inner_depth = self.visit(inner)?;
                    Ok(1 + inner_depth)
                }
                ASTRepr::Sum(collection) => {
                    let collection_depth = self.visit_collection(collection)?;
                    Ok(1 + collection_depth)
                }
                ASTRepr::Lambda(lambda) => {
                    let body_depth = self.visit(&lambda.body)?;
                    Ok(1 + body_depth)
                }
                ASTRepr::Let(_, expr, body) => {
                    let expr_depth = self.visit(expr)?;
                    let body_depth = self.visit(body)?;
                    Ok(1 + expr_depth.max(body_depth))
                }
            }
        }

        fn visit_collection(
            &mut self,
            collection: &crate::ast::ast_repr::Collection<T>,
        ) -> Result<Self::Output, Self::Error> {
            use crate::ast::ast_repr::Collection;

            match collection {
                Collection::Empty | Collection::Variable(_) | Collection::DataArray(_) => Ok(1),
                Collection::Singleton(expr) => self.visit(expr),
                Collection::Range { start, end } => {
                    let start_depth = self.visit(start)?;
                    let end_depth = self.visit(end)?;
                    Ok(start_depth.max(end_depth))
                }

                Collection::Filter {
                    collection,
                    predicate,
                } => {
                    let collection_depth = self.visit_collection(collection)?;
                    let predicate_depth = self.visit(predicate)?;
                    Ok(1 + collection_depth.max(predicate_depth))
                }
                Collection::Map { lambda, collection } => {
                    let lambda_depth = self.visit(&lambda.body)?;
                    let collection_depth = self.visit_collection(collection)?;
                    Ok(1 + lambda_depth.max(collection_depth))
                }
            }
        }

        fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(1) // Default depth contribution
        }

        fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
            Ok(1)
        }

        fn visit_collection_variable(
            &mut self,
            _index: usize,
        ) -> Result<Self::Output, Self::Error> {
            Ok(1)
        }
    }
}

/// Non-recursive versions of analysis functions using visitor pattern
pub fn count_operations_visitor<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
    visitors::OperationCountVisitor::count_operations(expr)
}

pub fn count_summations_visitor<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
    visitors::SummationCountVisitor::count_summations(expr)
}

pub fn expression_depth_visitor<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
    visitors::DepthVisitor::compute_depth(expr)
}

pub fn summation_aware_cost_visitor<T: Scalar + Clone>(expr: &ASTRepr<T>) -> usize {
    visitors::SummationAwareCostVisitor::compute_cost(expr)
}

pub fn summation_aware_cost_visitor_with_domain_size<T: Scalar + Clone>(
    expr: &ASTRepr<T>,
    domain_size: usize,
) -> usize {
    visitors::SummationAwareCostVisitor::compute_cost_with_domain_size(expr, domain_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_equality() {
        // Test with direct ASTRepr construction
        let x = ASTRepr::<f64>::Variable(0);
        let one = ASTRepr::<f64>::Constant(1.0);
        let one_point_one = ASTRepr::<f64>::Constant(1.1);

        let expr1 = x.clone() + one.clone();
        let expr2 = x.clone() + one;
        let expr3 = x + one_point_one;

        assert!(expressions_equal_default(&expr1, &expr2));
        assert!(!expressions_equal_default(&expr1, &expr3));
    }

    #[test]
    fn test_variable_collection() {
        // Test with direct ASTRepr construction
        let x = ASTRepr::<f64>::Variable(0);
        let one = ASTRepr::<f64>::Constant(1.0);
        let expr = x + one;

        let variables = collect_variable_indices(&expr);
        assert!(variables.contains(&0)); // x should be at index 0
    }

    #[test]
    fn test_complex_variable_collection() {
        // Test with direct ASTRepr construction
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let z = ASTRepr::<f64>::Variable(2);

        let xy = x * y;
        let expr = xy + z;

        let variables = collect_variable_indices(&expr);
        assert_eq!(variables.len(), 3);
        assert!(variables.contains(&0)); // x
        assert!(variables.contains(&1)); // y
        assert!(variables.contains(&2)); // z
    }

    #[test]
    fn test_expression_depth() {
        let const_expr = ASTRepr::<f64>::Constant(5.0);
        let var_expr = ASTRepr::<f64>::Variable(0);

        assert_eq!(expression_depth(&const_expr), 1);
        assert_eq!(expression_depth(&var_expr), 1);

        // Test nested expression
        let nested = const_expr + var_expr;
        assert_eq!(expression_depth(&nested), 2);
    }

    #[test]
    fn test_is_constant_zero_one() {
        let zero_expr = ASTRepr::<f64>::Constant(0.0);
        let one_expr = ASTRepr::<f64>::Constant(1.0);
        let other_expr = ASTRepr::<f64>::Constant(2.0);

        assert!(is_zero(&zero_expr, None));
        assert!(is_one(&one_expr, None));
        assert!(!is_zero(&other_expr, None));
        assert!(!is_one(&other_expr, None));
    }

    #[test]
    fn test_expression_complexity() {
        let simple_expr = ASTRepr::<f64>::Constant(1.0);
        let x = ASTRepr::<f64>::Variable(0);
        let one = ASTRepr::<f64>::Constant(1.0);

        let x_squared = ASTRepr::mul_from_array([x.clone(), x]);
        let complex_expr = x_squared + one;

        assert!(count_nodes(&simple_expr) < count_nodes(&complex_expr));
    }

    #[test]
    fn test_contains_variable() {
        let x = ASTRepr::<f64>::Variable(0);
        let zero = ASTRepr::<f64>::Constant(0.0);
        let expr = x + zero;

        assert!(contains_variable_by_index(&expr, 0)); // Should contain variable at index 0
        assert!(!contains_variable_by_index(&expr, 1)); // Should not contain variable at index 1
    }
}
