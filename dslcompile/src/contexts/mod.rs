//! Context Systems for DSLCompile
//!
//! This module provides the two core context systems for building mathematical expressions:
//!
//! ## DynamicContext (Runtime Flexibility)
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
//! ## StaticContext (Compile-time Optimization)
//!
//! Provides zero-overhead compile-time expression building with automatic scope management
//! and HList heterogeneous support:
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

use crate::ast::{ASTRepr, Scalar};
use std::collections::HashSet;

// ============================================================================
// UNIFIED EXPRESSION TRAIT - COMMON INTERFACE FOR BOTH CONTEXTS
// ============================================================================

/// Universal trait for all mathematical expressions in DSLCompile
///
/// This trait provides a common interface for both static (compile-time optimized)
/// and dynamic (runtime flexible) expressions, enabling generic algorithms and
/// unified APIs while preserving the architectural strengths of each approach.
pub trait Expr<T: Scalar> {
    /// Convert to AST representation for analysis and optimization
    fn to_ast(&self) -> ASTRepr<T>;

    /// Pretty print the expression in human-readable mathematical notation
    fn pretty_print(&self) -> String;

    /// Get all variable indices used in this expression
    fn get_variables(&self) -> HashSet<usize>;

    /// Get the complexity (operation count) of this expression
    fn complexity(&self) -> usize {
        use crate::ast::ast_utils::visitors::OperationCountVisitor;
        OperationCountVisitor::count_operations(&self.to_ast())
    }

    /// Get the depth (nesting level) of this expression
    fn depth(&self) -> usize {
        use crate::ast::ast_utils::visitors::DepthVisitor;
        DepthVisitor::compute_depth(&self.to_ast())
    }
}

/// Input provider trait for unified evaluation interface
///
/// This trait abstracts over different input sources (HLists, Vec, etc.)
/// to provide a common evaluation interface for the Expr trait.
pub trait InputProvider<T: Scalar> {
    /// Get variable value by index
    fn get_var(&self, index: usize) -> T;

    /// Check if variable exists at index
    fn has_var(&self, index: usize) -> bool;
}

// Re-export DynamicContext and related types
pub use dynamic::{DynamicContext, DynamicExpr, TypeCategory, TypedVar, VariableRegistry};

// Re-export StaticContext and related types
pub use static_context::{
    HListEval, HListStorage, IntoHListEvaluable, StaticAdd, StaticConst, StaticContext, StaticExpr,
    StaticMul, StaticScopeBuilder, StaticVar, static_add, static_mul,
};

// Re-export scope merging functionality
pub use scope_merging::{MergedScope, ScopeInfo, ScopeMerger};

// Unified traits are defined in this module and available for use
