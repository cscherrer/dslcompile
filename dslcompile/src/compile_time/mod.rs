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

pub mod scoped;
pub mod heterogeneous;

// Re-export the main types for convenience
pub use scoped::{
    Context, ScopedMathExpr, ScopedVarArray, ScopedVar, ScopedConst,
    ScopedAdd, ScopedMul, ScopedSub, ScopedDiv, ScopedPow,
    ScopedExp, ScopedLn, ScopedSin, ScopedCos, ScopedSqrt, ScopedNeg,
    compose,
};

pub use heterogeneous::{
    HeteroContext, HeteroInputs, HeteroVar, HeteroConst, HeteroExpr,
    HeteroAdd, HeteroMul, HeteroArrayIndex,
    hetero_add, hetero_mul, hetero_array_index,
};

// Legacy aliases for backward compatibility
pub type ScopedExpressionBuilder<T, const SCOPE: usize> = Context<T, SCOPE>;

// Type aliases for common use cases
pub type Context32 = Context<f32, 0>;
pub type Context64 = Context<f64, 0>;

// Common heterogeneous contexts
pub type HeteroContext8 = HeteroContext<0, 8>;
pub type HeteroContext16 = HeteroContext<0, 16>;
pub type HeteroContext32 = HeteroContext<0, 32>;

/// Trait for compile-time expression evaluation
pub trait CompileTimeEval<T> {
    /// Evaluate the expression with the given variable values
    fn eval(&self, vars: &[T]) -> T;
    
    /// Convert to AST representation for analysis
    fn to_ast(&self) -> crate::ast::ASTRepr<T>;
}
