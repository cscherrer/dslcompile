//! Variable Management Module
//!
//! This module provides high-performance index-only variable management for mathematical expressions.
//! Variables are tracked by index for maximum performance with optional type safety.

pub mod typed_builder;
pub mod typed_registry;

// Re-export the main types for convenience
pub use typed_builder::{TypedBuilderExpr, TypedExpressionBuilder};
pub use typed_registry::{TypeCategory, TypedVar, TypedVariableRegistry};

// Convenience alias for the primary API
pub type MathBuilder = TypedExpressionBuilder;
pub type VariableRegistry = TypedVariableRegistry; // Alias for compatibility during migration
