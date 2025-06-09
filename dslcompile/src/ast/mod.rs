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
pub use ast_repr::ASTRepr;
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
