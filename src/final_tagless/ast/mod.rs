//! AST (Abstract Syntax Tree) Module
//!
//! This module contains the AST representation and related utilities for
//! mathematical expressions in the final tagless approach.

pub mod ast_repr;
pub mod operators;
pub mod evaluation;

// Re-export the main types for convenience
pub use ast_repr::ASTRepr;

// The operator overloading is automatically available when ASTRepr is in scope
// due to the trait implementations in the operators module 