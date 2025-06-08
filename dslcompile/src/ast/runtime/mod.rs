//! Runtime Expression Building
//!
//! This module provides the runtime expression building system that enables
//! data-aware expression construction with pattern recognition and optimization.

pub mod expression_builder;
pub mod summation_types;
pub mod typed_registry;

// Re-export the main types
pub use expression_builder::{DynamicContext, TypedBuilderExpr};
pub use typed_registry::{TypeCategory, TypedVar, VariableRegistry};

// Legacy type aliases removed - use DynamicContext directly for runtime expression building
// Use StaticContext for compile-time optimized expressions
