//! Variable Management Module
//!
//! This module provides variable management for mathematical expressions,
//! including registries for mapping names to indices and expression builders
//! for convenient expression construction.
//!
//! # New Typed Variable System
//!
//! The module now includes a type-safe variable system that provides compile-time
//! type checking while maintaining full backward compatibility with the existing API.

pub mod builder;
pub mod registry;
pub mod typed_builder;
pub mod typed_registry;

// Re-export the main types for convenience

// Original untyped system (backward compatibility)
pub use builder::ExpressionBuilder;
pub use registry::{
    VariableRegistry, clear_global_registry, create_variable_map, get_variable_index,
    get_variable_name, register_variable,
};

// New typed system
pub use typed_builder::{TypedBuilderExpr, TypedExpressionBuilder};
pub use typed_registry::{TypeCategory, TypedVar, TypedVariableRegistry};

// Convenience alias for the new primary API
pub type MathBuilder = TypedExpressionBuilder;
