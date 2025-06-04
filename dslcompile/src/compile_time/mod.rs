//! Compile-Time Mathematical Expression System
//!
//! This module provides a compile-time mathematical expression system that leverages
//! Rust's type system to perform optimizations at compile time with perfect composability.
//!
//! ## Scoped Variables System
//!
//! The scoped variables system provides:
//! - **Type-safe composition**: Variable scopes prevent collisions at compile time
//! - **Zero runtime overhead**: All scope resolution happens at compile time
//! - **Automatic variable remapping**: Functions compose seamlessly without manual index management
//! - **Perfect composability**: Build mathematical libraries without variable index conflicts
//!
//! ## Example
//!
//! ```rust
//! use dslcompile::prelude::*;
//!
//! let mut builder = Context::new();
//!
//! // Define f(x) = x² in scope 0
//! let f = builder.new_scope(|scope| {
//!     let (x, _scope) = scope.auto_var();
//!     x.clone().mul(x)
//! });
//!
//! // Advance to next scope
//! let mut builder = builder.next();
//!
//! // Define g(y) = 2y in scope 1 (no collision!)
//! let g = builder.new_scope(|scope| {
//!     let (y, scope) = scope.auto_var();
//!     y.mul(scope.constant(2.0))
//! });
//!
//! // Perfect composition with automatic variable remapping
//! let composed = compose(f, g);
//! let combined = composed.add(); // h(x,y) = x² + 2y
//!
//! let result = combined.eval(&[3.0, 4.0]);
//! assert_eq!(result, 17.0); // 3² + 2*4 = 9 + 8 = 17
//! ```

pub mod optimized;
pub mod scoped;
pub mod type_level_logic;

// Re-export the scoped variables system (recommended)
pub use scoped::{
    ScopeBuilder, ScopedConst, Context, ScopedMathExpr, ScopedVar, ScopedVarArray,
    compose,
};

// Legacy alias for backward compatibility (will be removed in future versions)
pub use scoped::Context as ScopedExpressionBuilder;
