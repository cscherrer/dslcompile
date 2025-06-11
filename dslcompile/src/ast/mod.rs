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
pub(crate) mod ast_utils; // Internal utilities
pub(crate) mod evaluation; // Internal evaluation logic
pub(crate) mod normalization; // Internal normalization - only used by egglog optimization
pub(crate) mod operators; // Operator overloading - automatically available via traits
pub(crate) mod pretty; // Pretty printing - controlled exports below
pub mod runtime; // Runtime expression building

// Re-export core types that external users need
pub use ast_repr::ASTRepr;

// Internal AST node types - users should use DynamicContext instead of constructing these directly

// Re-export variable registry for pretty printing and backends
pub use runtime::typed_registry::VariableRegistry;

// Re-export runtime expression building (main user-facing API)
pub use runtime::{DynamicContext, TypeCategory, TypedBuilderExpr, TypedVar};

// Selective re-exports from utilities - only what's actually needed externally
pub use ast_utils::{
    collect_variable_indices,  // Used in rust codegen backend
    expressions_equal_default, // Used in symbolic optimization
};

// Selective re-exports from pretty printing
pub use pretty::pretty_ast; // Main pretty printing function
// pretty_anf is internal - only used in tests and anf module

// Normalization functions are internal - only used by egglog optimization
pub(crate) use normalization::normalize;

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

    // Re-export internal utilities for advanced use
    pub use super::{ast_utils::*, normalization::*, pretty::*};
}
