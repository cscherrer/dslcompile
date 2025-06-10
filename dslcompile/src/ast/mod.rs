//! AST (Abstract Syntax Tree) module
//!
//! This module provides the core AST representation and utilities for mathematical expressions.
//! It serves as the foundation for all expression manipulation and evaluation.

use std::fmt::{Debug, Display};

// Core numeric trait for mathematical operations
pub trait Scalar:
    Clone
    + Default
    + Send
    + Sync
    + Display
    + Debug
    + PartialEq
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
}

// Implement Scalar for standard numeric types
impl Scalar for f64 {}
impl Scalar for f32 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
impl Scalar for usize {}

pub mod ast_repr;
pub mod ast_utils;
pub mod evaluation;
pub mod normalization;
pub mod operators;
pub mod pretty;
pub mod runtime; // Runtime expression building

// Re-export core types
pub use ast_repr::{ASTRepr, Collection, Lambda};
pub use runtime::typed_registry::VariableRegistry;

// Re-export runtime expression building (new names)
pub use runtime::{DynamicContext, TypeCategory, TypedBuilderExpr, TypedVar};

// Deprecated compatibility exports (will be removed in future versions)
#[allow(deprecated)]
// Legacy type aliases removed - use DynamicContext directly for runtime expression building

// Re-export evaluation functionality

// Re-export commonly used items
pub use ast_utils::*;
pub use normalization::{denormalize, is_canonical, normalize};
pub use pretty::*;

// The operator overloading is automatically available when ASTRepr is in scope
// due to the trait implementations in the operators module

/// Advanced APIs for custom backends and internal use
///
/// ⚠️ **WARNING**: These APIs are unstable and may break between versions.
/// Use `DynamicContext` for normal expression building.
///
/// This module provides controlled access to internal AST types for:
/// - Custom backend implementations
/// - Advanced optimization passes  
/// - Internal library development
/// - Testing infrastructure
pub mod advanced {
    //! Low-level AST access for advanced use cases
    //!
    //! Most users should use `DynamicContext` instead of these APIs.

    // Import AST types for internal use
    use super::ast_repr::{ASTRepr, Collection, Lambda};

    /// Type alias for AST representation (advanced use only)
    pub type AstRepr<T> = ASTRepr<T>;

    /// Type alias for Collection (advanced use only)
    pub type AstCollection<T> = Collection<T>;

    /// Type alias for Lambda (advanced use only)
    pub type AstLambda<T> = Lambda<T>;

    /// Extract the underlying AST from a typed expression
    ///
    /// Note: This function is for advanced use cases only.
    pub fn ast_from_expr<T: super::Scalar>(expr: &super::TypedBuilderExpr<T>) -> &ASTRepr<T> {
        expr.as_ast()
    }

    /// Create an AST variable node (for internal optimization passes)
    ///
    /// ⚠️ **WARNING**: Manual variable index management can cause bugs.
    /// Use `DynamicContext::var()` instead for normal expression building.
    pub fn create_variable_node<T>(index: usize) -> ASTRepr<T> {
        ASTRepr::Variable(index)
    }

    /// Create an AST constant node (for internal optimization passes)
    pub fn create_constant_node<T>(value: T) -> ASTRepr<T> {
        ASTRepr::Constant(value)
    }
}
