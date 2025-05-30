//! Abstract Syntax Tree for Mathematical Expressions
//!
//! This module provides the core AST representation with optional composable function categories
//! as an extension layer for enhanced mathematical operations.

pub mod ast_repr;
pub mod ast_utils;
pub mod evaluation;
pub mod function_categories;
pub mod operators;
pub mod pretty;

// Re-export the existing, working AST
pub use ast_repr::ASTRepr;
