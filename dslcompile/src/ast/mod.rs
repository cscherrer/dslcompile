//! AST (Abstract Syntax Tree) Module
//!
//! This module contains the AST representation and related utilities for
//! mathematical expressions in the final tagless approach.

pub mod ast_repr;
pub mod ast_utils;
pub mod evaluation;
pub mod normalization;
pub mod operators;
pub mod pretty;

// Re-export commonly used items
pub use ast_repr::ASTRepr;
pub use ast_utils::*;
pub use normalization::{denormalize, is_canonical, normalize};
pub use pretty::*;

// The operator overloading is automatically available when ASTRepr is in scope
// due to the trait implementations in the operators module
