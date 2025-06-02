//! Variable Management Module
//!
//! This module provides variable management for mathematical expressions.
//! The primary system is now index-based for maximum performance, with optional
//! string-based convenience for legacy compatibility.

pub mod builder;
pub mod registry;
pub mod typed_builder;
pub mod typed_registry;

// Re-export the main types for convenience

// Legacy string-based system (for backward compatibility)
pub use builder::ExpressionBuilder;
pub use registry::{
    VariableRegistry as StringBasedRegistry, clear_global_registry, create_variable_map, 
    get_variable_index, get_variable_name, global_registry, register_variable,
};

// New index-based typed system (primary API)
pub use typed_registry::{TypeCategory, TypedVar, TypedVariableRegistry};
pub use typed_builder::{TypedBuilderExpr, TypedExpressionBuilder};

// Convenience alias for the new primary API
pub type MathBuilder = TypedExpressionBuilder;

// For now, also export the old VariableRegistry under its original name
pub use registry::VariableRegistry;
