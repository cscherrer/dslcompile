//! Runtime Expression Building
//!
//! This module provides the runtime expression building system that enables
//! data-aware expression construction with pattern recognition and optimization.

pub mod expression_builder;
pub mod typed_registry;

// Re-export the main types
pub use expression_builder::{DynamicContext, TypedBuilderExpr};
pub use typed_registry::{TypeCategory, TypedVar, VariableRegistry};

// Legacy aliases for backward compatibility (will be removed in future versions)
pub type ExpressionBuilder = DynamicContext;
pub type MathBuilder = DynamicContext;
