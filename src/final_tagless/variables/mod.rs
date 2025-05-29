//! Variable Management Module
//!
//! This module provides variable management for mathematical expressions,
//! including registries for mapping names to indices and expression builders
//! for convenient expression construction.

pub mod registry;
pub mod builder;

// Re-export the main types for convenience
pub use registry::{VariableRegistry, register_variable, get_variable_index, get_variable_name, create_variable_map, clear_global_registry};
pub use builder::ExpressionBuilder; 