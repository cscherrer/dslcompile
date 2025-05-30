//! Variable Management Module
//!
//! This module provides variable management for mathematical expressions,
//! including registries for mapping names to indices and expression builders
//! for convenient expression construction.

pub mod builder;
pub mod registry;

// Re-export the main types for convenience
pub use builder::ExpressionBuilder;
pub use registry::{
    VariableRegistry, clear_global_registry, create_variable_map, get_variable_index,
    get_variable_name, register_variable,
};
