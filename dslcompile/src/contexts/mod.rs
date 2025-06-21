//! Context Systems for `DSLCompile`
//!
//! This module provides the two core context systems for building mathematical expressions:
//!
//! ## `DynamicContext` (Runtime Flexibility)
//!
//! Provides runtime expression building with JIT compilation, symbolic optimization,
//! and flexible heterogeneous type support:
//!
//! ```rust
//! use dslcompile::contexts::DynamicContext;
//! use frunk::hlist;
//!
//! let mut ctx = DynamicContext::new();
//! let x = ctx.var();
//! let expr = &x * &x + 2.0 * &x + 1.0;
//! let result = ctx.eval(&expr, hlist![3.0]); // 3² + 2*3 + 1 = 16
//! ```
//!
//! ## `StaticContext` (Compile-time Optimization)
//!
//! Provides zero-overhead compile-time expression building with automatic scope management
//! and `HList` heterogeneous support:
//!
//! ```rust
//! use dslcompile::contexts::{StaticContext, IntoHListEvaluable};
//! use frunk::hlist;
//!
//! let mut ctx = StaticContext::new();
//! let f = ctx.new_scope(|scope| {
//!     let (x, scope) = scope.auto_var::<f64>();
//!     let (y, scope) = scope.auto_var::<f64>();
//!     x.clone() * x + scope.constant(2.0) * y  // x² + 2y
//! });
//!
//! let result = f.eval(hlist![3.0, 4.0]); // 3² + 2*4 = 17
//! ```

pub mod dynamic;
pub mod scope_merging;
pub mod shared;
pub mod static_context;

use crate::ast::Scalar;

// ============================================================================
// SEPARATE INTERFACES FOR STATIC AND DYNAMIC EXPRESSION SYSTEMS
// ============================================================================

// Each expression system now has its own optimized interface:
// - StaticExpr: Compile-time interface with zero-cost abstractions
// - DynamicExpr: Runtime interface with flexible analysis capabilities
// 
// Bridge functions provide explicit conversion points for cross-validation
// and interoperability when needed.

// Re-export DynamicContext and related types
pub use dynamic::{DynamicContext, DynamicExpr, TypeCategory, TypedVar, VariableRegistry};

// Re-export StaticContext and related types
pub use static_context::{
    HListEval, HListStorage, IntoHListEvaluable, StaticAdd, StaticConst, StaticContext, StaticExpr,
    StaticMul, StaticScopeBuilder, StaticVar, static_add, static_mul,
};

// Re-export scope merging functionality
pub use scope_merging::{MergedScope, ScopeInfo, ScopeMerger};

// Each context system provides distinct, optimized interfaces for their computational model
