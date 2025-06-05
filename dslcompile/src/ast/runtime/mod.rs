//! Runtime Expression Building
//!
//! This module provides the runtime expression building system that enables
//! data-aware expression construction with pattern recognition and optimization.

pub mod expression_builder;
pub mod typed_registry;

// Re-export the main types
pub use expression_builder::{DynamicContext, TypedBuilderExpr};
pub use typed_registry::{TypeCategory, TypedVar, VariableRegistry};

// Deprecated aliases for backward compatibility (will be removed in future versions)
// TODO: Remove these after systematic migration to DynamicContext
#[deprecated(since = "0.3.0", note = "Use `DynamicContext` directly instead")]
pub type ExpressionBuilder = DynamicContext;

#[deprecated(since = "0.3.0", note = "Use `DynamicContext` directly instead")]
pub type MathBuilder = DynamicContext;
