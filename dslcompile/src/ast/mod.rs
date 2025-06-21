//! AST (Abstract Syntax Tree) module
//!
//! This module provides the core AST representation and utilities for mathematical expressions.
//! It serves as the foundation for all expression manipulation and evaluation.

pub mod multiplicity;
pub mod multiset;

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
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
}

// Blanket implementation for all types that satisfy the numeric requirements
impl<T> Scalar for T where
    T: Clone
        + Default
        + Send
        + Sync
        + Display
        + Debug
        + PartialEq
        + PartialOrd
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
{
}

/// Trait for types that can be stored as variables in the system
/// This is the most general trait - all variable types implement this
pub trait Variable: Clone + Debug + Send + Sync + 'static {
    /// Type name for debugging and registry purposes
    fn type_name() -> &'static str;
}

/// Marker trait for types that currently participate in scalar mathematical expressions
/// This is a subset of Variable - not all variables are currently mathematical
pub trait CurrentlyMathematical: Variable {}

// Blanket implementation - all Variables can be stored, regardless of mathematical capability
impl<T> Variable for T where
    T: Clone + Debug + Send + Sync + 'static
{
    fn type_name() -> &'static str {
        std::any::type_name::<T>()
    }
}

// Current Scalar types are currently mathematical
impl<T> CurrentlyMathematical for T where T: Scalar + 'static {}

pub mod arena; // Arena-based allocation for memory efficiency
pub mod arena_conversion; // Conversion utilities between Box and arena ASTs
pub mod ast_repr;
pub mod ast_utils; // Internal utilities - now public for visitor pattern
pub(crate) mod evaluation; // Internal evaluation logic
pub mod normalization; // Normalization module - used by egglog optimization and tests
pub(crate) mod operators; // Operator overloading - automatically available via traits
pub(crate) mod pretty; // Pretty printing - controlled exports below
pub mod stack_visitor;
pub mod visitor; // Visitor pattern for clean AST traversal // Stack-based visitor pattern for deep AST traversal without stack overflow

// Re-export core types that external users need
pub use ast_repr::ASTRepr;

// Re-export arena types for memory-efficient AST construction
pub use arena::{ArenaAST, ArenaCollection, ArenaLambda, ArenaMultiSet, ExprArena, ExprId};

// Re-export conversion utilities
pub use arena_conversion::{arena_to_ast, ast_to_arena};

// Internal AST node types - users should use DynamicContext instead of constructing these directly

// Re-export variable registry for pretty printing and backends
pub use crate::contexts::VariableRegistry;

// Re-export runtime expression building (main user-facing API)
pub use crate::contexts::{DynamicContext, DynamicExpr, TypeCategory, TypedVar};

// Selective re-exports from utilities - only what's actually needed externally
pub use ast_utils::{
    collect_variable_indices,  // Used in rust codegen backend
    count_nodes,               // AST analysis utilities
    expression_depth,          // AST analysis utilities
    expressions_equal_default, // Used in symbolic optimization
};

// Selective re-exports from pretty printing
pub use pretty::pretty_ast; // Main pretty printing function

// Visitor pattern for clean AST traversal
pub use visitor::{ASTMutVisitor, ASTVisitor, visit_ast, visit_ast_mut};

// Stack-based visitor pattern for deep AST traversal without stack overflow
pub use stack_visitor::{StackBasedMutVisitor, StackBasedVisitor};

// Normalization functions - used by egglog optimization

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
    pub fn ast_from_expr<T: super::Scalar>(expr: &super::DynamicExpr<T>) -> &ASTRepr<T> {
        expr.as_ast()
    }

    /// Create an AST variable node (for internal optimization passes)
    ///
    /// ⚠️ **WARNING**: Manual variable index management can cause bugs.
    /// Use `DynamicContext::var()` instead for normal expression building.
    #[must_use]
    pub fn create_variable_node<T: super::Scalar>(index: usize) -> ASTRepr<T> {
        ASTRepr::Variable(index)
    }

    /// Create an AST constant node (for internal optimization passes)
    pub fn create_constant_node<T: super::Scalar>(value: T) -> ASTRepr<T> {
        ASTRepr::Constant(value)
    }

    // Re-export internal utilities for advanced use
    pub use super::{ast_utils::*, normalization::*, pretty::*};
}
