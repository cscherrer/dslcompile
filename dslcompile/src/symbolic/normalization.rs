//! Expression normalization utilities
//!
//! This module provides utilities for normalizing mathematical expressions
//! to canonical forms for optimization.

use crate::ast::{ASTRepr, Scalar};

/// Normalize an expression to a canonical form
pub fn normalize<T: Scalar>(expr: &ASTRepr<T>) -> ASTRepr<T> {
    // For now, just return a clone
    // TODO: Implement proper normalization
    expr.clone()
}

/// Check if an expression requires normalization
pub fn needs_normalization<T: Scalar>(expr: &ASTRepr<T>) -> bool {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => false,
        ASTRepr::Add(_) | ASTRepr::Sub(_, _) | ASTRepr::Mul(_) | ASTRepr::Div(_, _) | ASTRepr::Pow(_, _) => true,
        ASTRepr::Neg(_) | ASTRepr::Ln(_) | ASTRepr::Exp(_) | ASTRepr::Sin(_) | ASTRepr::Cos(_) | ASTRepr::Sqrt(_) => true,
        ASTRepr::Sum(_) => {
            // TODO: Implement Sum variant for normalization
            true // Treat as complex expression requiring normalization
        }
        _ => true,
    }
}