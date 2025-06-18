//! Type Conversions for DSLCompile Expression Types
//!
//! This module provides type conversion capabilities between different expression types,
//! including AST conversions, explicit From implementations, and helper functions for
//! cross-type operations.
//!
//! ## Key Components
//!
//! - **Explicit Conversions**: From implementations for safe type conversions
//! - **AST Conversions**: Helper functions for converting AST representations between types
//! - **Pure Rust Conversions**: Type-safe conversions using standard Rust From trait
//! - **Cross-Type Support**: Infrastructure for heterogeneous expression operations

use crate::{
    ast::{
        Scalar,
        ast_repr::{ASTRepr, Collection, Lambda},
    },
    contexts::dynamic::{expression_builder::DynamicExpr, typed_registry::VariableRegistry},
};
use std::{cell::RefCell, sync::Arc};

// ============================================================================
// EXPLICIT CONVERSIONS (No auto-promotion!)
// ============================================================================

/// Explicit conversion from f32 expressions to f64 expressions
impl<const SCOPE: usize> From<DynamicExpr<f32, SCOPE>> for DynamicExpr<f64, SCOPE> {
    fn from(expr: DynamicExpr<f32, SCOPE>) -> Self {
        expr.to_f64()
    }
}

/// Explicit conversion from i32 expressions to f64 expressions  
impl<const SCOPE: usize> From<DynamicExpr<i32, SCOPE>> for DynamicExpr<f64, SCOPE> {
    fn from(expr: DynamicExpr<i32, SCOPE>) -> Self {
        DynamicExpr::new(convert_i32_ast_to_f64(&expr.ast), expr.registry)
    }
}

// ============================================================================
// AST CONVERSION HELPERS
// ============================================================================

/// Helper to convert i32 AST to f64 AST
pub fn convert_i32_ast_to_f64(ast: &ASTRepr<i32>) -> ASTRepr<f64> {
    match ast {
        ASTRepr::Constant(value) => ASTRepr::Constant(f64::from(*value)),
        ASTRepr::Variable(index) => ASTRepr::Variable(*index),
        ASTRepr::Add(terms) => {
            let converted_terms: Vec<_> = terms.elements().map(convert_i32_ast_to_f64).collect();
            ASTRepr::Add(crate::ast::multiset::MultiSet::from_iter(converted_terms))
        }
        ASTRepr::Sub(left, right) => ASTRepr::Sub(
            Box::new(convert_i32_ast_to_f64(left)),
            Box::new(convert_i32_ast_to_f64(right)),
        ),
        ASTRepr::Mul(factors) => {
            let converted_factors: Vec<_> =
                factors.elements().map(convert_i32_ast_to_f64).collect();
            ASTRepr::Mul(crate::ast::multiset::MultiSet::from_iter(converted_factors))
        }
        ASTRepr::Div(left, right) => ASTRepr::Div(
            Box::new(convert_i32_ast_to_f64(left)),
            Box::new(convert_i32_ast_to_f64(right)),
        ),
        ASTRepr::Pow(base, exp) => ASTRepr::Pow(
            Box::new(convert_i32_ast_to_f64(base)),
            Box::new(convert_i32_ast_to_f64(exp)),
        ),
        ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(convert_i32_ast_to_f64(inner))),
        // Transcendental functions don't make sense for i32, but we'll convert anyway
        ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Sum(collection) => {
            // Convert collection from i32 to f64
            ASTRepr::Sum(Box::new(convert_collection_pure_rust(collection)))
        }
        ASTRepr::Lambda(lambda) => {
            // Convert lambda to f64
            ASTRepr::Lambda(Box::new(Lambda {
                var_indices: lambda.var_indices.clone(),
                body: Box::new(convert_i32_ast_to_f64(&lambda.body)),
            }))
        }
        ASTRepr::BoundVar(index) => {
            // BoundVar index stays the same across type conversions
            ASTRepr::BoundVar(*index)
        }
        ASTRepr::Let(binding_id, expr, body) => {
            // Convert both the bound expression and body
            ASTRepr::Let(
                *binding_id,
                Box::new(convert_i32_ast_to_f64(expr)),
                Box::new(convert_i32_ast_to_f64(body)),
            )
        }
    }
}

// ============================================================================
// PURE RUST FROM/INTO CONVERSIONS (The Right Way!)
// ============================================================================

/// Generic AST conversion using ONLY standard Rust From trait
pub fn convert_ast_pure_rust<T: Scalar, U: Scalar>(ast: &ASTRepr<T>) -> ASTRepr<U>
where
    U: From<T>,
{
    match ast {
        // Use Rust's built-in From trait for primitives
        ASTRepr::Constant(value) => ASTRepr::Constant(U::from(value.clone())),
        ASTRepr::Variable(index) => ASTRepr::Variable(*index),
        ASTRepr::Add(terms) => {
            let converted_terms: Vec<_> = terms
                .elements()
                .map(|term| convert_ast_pure_rust(term))
                .collect();
            ASTRepr::Add(crate::ast::multiset::MultiSet::from_iter(converted_terms))
        }
        ASTRepr::Sub(left, right) => ASTRepr::Sub(
            Box::new(convert_ast_pure_rust(left)),
            Box::new(convert_ast_pure_rust(right)),
        ),
        ASTRepr::Mul(factors) => {
            let converted_factors: Vec<_> = factors
                .elements()
                .map(|factor| convert_ast_pure_rust(factor))
                .collect();
            ASTRepr::Mul(crate::ast::multiset::MultiSet::from_iter(converted_factors))
        }
        ASTRepr::Div(left, right) => ASTRepr::Div(
            Box::new(convert_ast_pure_rust(left)),
            Box::new(convert_ast_pure_rust(right)),
        ),
        ASTRepr::Pow(base, exp) => ASTRepr::Pow(
            Box::new(convert_ast_pure_rust(base)),
            Box::new(convert_ast_pure_rust(exp)),
        ),
        ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Sum(collection) => {
            ASTRepr::Sum(Box::new(convert_collection_pure_rust(collection)))
        }
        ASTRepr::Lambda(lambda) => ASTRepr::Lambda(Box::new(convert_lambda_pure_rust(lambda))),
        ASTRepr::BoundVar(index) => {
            // BoundVar index stays the same across type conversions
            ASTRepr::BoundVar(*index)
        }
        ASTRepr::Let(binding_id, expr, body) => {
            // Convert both the bound expression and body
            ASTRepr::Let(
                *binding_id,
                Box::new(convert_ast_pure_rust(expr)),
                Box::new(convert_ast_pure_rust(body)),
            )
        }
    }
}

/// Convert Collection using standard Rust From trait
#[must_use]
pub fn convert_collection_pure_rust<T: Scalar, U: Scalar>(
    collection: &Collection<T>,
) -> Collection<U>
where
    U: From<T>,
{
    match collection {
        Collection::Empty => Collection::Empty,
        Collection::Singleton(expr) => Collection::Singleton(Box::new(convert_ast_pure_rust(expr))),
        Collection::Range { start, end } => Collection::Range {
            start: Box::new(convert_ast_pure_rust(start)),
            end: Box::new(convert_ast_pure_rust(end)),
        },

        Collection::Variable(index) => Collection::Variable(*index),
        Collection::Filter {
            collection,
            predicate,
        } => Collection::Filter {
            collection: Box::new(convert_collection_pure_rust(collection)),
            predicate: Box::new(convert_ast_pure_rust(predicate)),
        },
        Collection::Map { lambda, collection } => Collection::Map {
            lambda: Box::new(convert_lambda_pure_rust(lambda)),
            collection: Box::new(convert_collection_pure_rust(collection)),
        },
        Collection::DataArray(data) => {
            Collection::DataArray(data.iter().map(|x| U::from(x.clone())).collect())
        }
    }
}

/// Convert Lambda using standard Rust From trait  
#[must_use]
pub fn convert_lambda_pure_rust<T: Scalar, U: Scalar>(lambda: &Lambda<T>) -> Lambda<U>
where
    U: From<T>,
{
    Lambda {
        var_indices: lambda.var_indices.clone(),
        body: Box::new(convert_ast_pure_rust(&lambda.body)),
    }
}

// ============================================================================
// FROM IMPLEMENTATIONS FOR SCALAR TYPES
// ============================================================================

// Add From implementations for scalar types
impl<const SCOPE: usize> From<f64> for DynamicExpr<f64, SCOPE> {
    fn from(value: f64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl<const SCOPE: usize> From<f32> for DynamicExpr<f32, SCOPE> {
    fn from(value: f32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl<const SCOPE: usize> From<i32> for DynamicExpr<i32, SCOPE> {
    fn from(value: i32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl<const SCOPE: usize> From<i64> for DynamicExpr<i64, SCOPE> {
    fn from(value: i64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

// Add cross-type From implementations
impl<const SCOPE: usize> From<i32> for DynamicExpr<f64, SCOPE> {
    fn from(value: i32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(f64::from(value)), registry)
    }
}

impl<const SCOPE: usize> From<i64> for DynamicExpr<f64, SCOPE> {
    fn from(value: i64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

impl<const SCOPE: usize> From<f32> for DynamicExpr<f64, SCOPE> {
    fn from(value: f32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(f64::from(value)), registry)
    }
}

impl<const SCOPE: usize> From<usize> for DynamicExpr<f64, SCOPE> {
    fn from(value: usize) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

// ============================================================================
// CONVERSION METHODS FOR DYNAMICEXPR
// ============================================================================

impl<const SCOPE: usize> DynamicExpr<f32, SCOPE> {
    /// Convert f32 expression to f64 expression
    #[must_use]
    pub fn to_f64(self) -> DynamicExpr<f64, SCOPE> {
        DynamicExpr::new(convert_ast_pure_rust(&self.ast), self.registry)
    }
}

impl<const SCOPE: usize> DynamicExpr<f64, SCOPE> {
    /// Convert f64 expression to f64 expression (identity operation)
    #[must_use]
    pub fn to_f64(self) -> DynamicExpr<f64, SCOPE> {
        self
    }
}

impl<const SCOPE: usize> DynamicExpr<i32, SCOPE> {
    /// Convert i32 expression to f64 expression
    #[must_use]
    pub fn to_f64(self) -> DynamicExpr<f64, SCOPE> {
        DynamicExpr::new(convert_i32_ast_to_f64(&self.ast), self.registry)
    }
}

// Add missing From implementation for DynamicExpr to ASTRepr conversion
impl<T: Scalar, const SCOPE: usize> From<DynamicExpr<T, SCOPE>> for ASTRepr<T> {
    fn from(expr: DynamicExpr<T, SCOPE>) -> Self {
        expr.ast
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contexts::dynamic::expression_builder::DynamicContext;

    #[test]
    fn test_from_numeric_types() {
        let mut builder = DynamicContext::new();
        let x = builder.var();

        // Test From implementations for numeric types
        let expr1: DynamicExpr<f64> = 2.0.into();
        let expr2: DynamicExpr<f64> = 3i32.into();
        let expr3: DynamicExpr<f64> = 4i64.into();
        let expr4: DynamicExpr<f64> = 5usize.into();
        let expr5: DynamicExpr<f32> = 2.5f32.into();

        // Verify they create constant expressions
        match expr1.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 2.0),
            _ => panic!("Expected constant"),
        }

        match expr2.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 3.0),
            _ => panic!("Expected constant"),
        }

        match expr3.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 4.0),
            _ => panic!("Expected constant"),
        }

        match expr4.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 5.0),
            _ => panic!("Expected constant"),
        }

        match expr5.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 2.5),
            _ => panic!("Expected constant"),
        }

        // Test that these can be used in expressions naturally
        let combined = &x + expr1 + expr2; // x + 2.0 + 3.0
        match combined.as_ast() {
            ASTRepr::Add(_) => {}
            _ => panic!("Expected addition"),
        }

        // Test evaluation works with new unified API
        let result = builder.eval(&combined, frunk::hlist![1.0]); // x = 1.0
        assert_eq!(result, 6.0); // 1.0 + 2.0 + 3.0 = 6.0
    }

    #[test]
    fn test_cross_type_operations() {
        let mut builder_f64 = DynamicContext::new();
        let mut builder_f32 = DynamicContext::new();

        let x_f64 = builder_f64.var::<f64>();
        let y_f32 = builder_f32.var::<f32>();

        // Convert f32 expression to f64 for cross-type operation
        let mixed_sum = x_f64 + y_f32.to_f64();

        // Result should be f64
        match mixed_sum.as_ast() {
            ASTRepr::Add(_) => {}
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_ast_conversion_functions() {
        let i32_ast = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Constant(42i32)]);

        let f64_ast = convert_i32_ast_to_f64(&i32_ast);

        match f64_ast {
            ASTRepr::Add(terms) => {
                assert_eq!(terms.len(), 2);
                let terms_vec: Vec<_> = terms.elements().collect();

                // MultiSet ordering is deterministic but may not match array order
                // Check if the constant 42.0 appears at either index
                let found_constant = terms_vec
                    .iter()
                    .any(|term| matches!(term, ASTRepr::Constant(val) if *val == 42.0));

                if !found_constant {
                    panic!(
                        "Expected to find constant 42.0 in terms, but got: {:?}",
                        terms_vec
                    );
                }

                // Also verify there's a Variable(0) term
                let found_variable = terms_vec
                    .iter()
                    .any(|term| matches!(term, ASTRepr::Variable(0)));

                if !found_variable {
                    panic!(
                        "Expected to find Variable(0) in terms, but got: {:?}",
                        terms_vec
                    );
                }
            }
            _ => panic!("Expected addition"),
        }
    }
}
