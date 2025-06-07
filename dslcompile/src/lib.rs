//! `DSLCompile`: Mathematical Expression Compilation
//!
//! `DSLCompile` provides a three-layer optimization strategy for mathematical expressions:
//! 1. **Final Tagless Approach**: Type-safe expression building with multiple interpreters
//! 2. **Symbolic Optimization**: Algebraic simplification using egglog
//! 3. **Compilation Backends**: Rust hot-loading (primary) and optional Cranelift JIT
//!
//! # Clean Two-Context Architecture
//!
//! After consolidation, DSLCompile provides exactly **two clean interfaces**:
//!
//! ## StaticContext (Compile-time optimization)
//!
//! ```rust
//! use dslcompile::prelude::*;
//! use frunk::hlist;
//!
//! // Zero-overhead, compile-time scoped variables with HList heterogeneous support
//! let mut ctx = StaticContext::new();
//!
//! let f = ctx.new_scope(|scope| {
//!     let (x, scope) = scope.auto_var::<f64>();
//!     let (y, _scope) = scope.auto_var::<f32>();
//!     x.clone() * x + scope.constant(2.0) * y  // x² + 2y
//! });
//!
//! // Evaluate with heterogeneous inputs - zero overhead
//! let result = f.eval_hlist(hlist![3.0, 4.0f32]); // 3² + 2*4 = 17
//! ```
//!
//! ## DynamicContext (Runtime flexibility)
//!
//! ```rust
//! use dslcompile::prelude::*;
//!
//! // Runtime flexibility, JIT compilation, symbolic optimization
//! let ctx = DynamicContext::new();
//! let x = ctx.var();
//! let expr = &x * &x + 2.0 * &x + 1.0;
//! let result = ctx.eval(&expr, &[3.0]); // 3² + 2*3 + 1 = 16
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]
#![feature(generic_const_exprs)]

// Core modules
pub mod ast;
pub mod backends;
pub mod compile_time;
pub mod error;
pub mod interval_domain;
pub mod symbolic;

// Re-export commonly used types
pub use ast::{ASTRepr, NumericType, VariableRegistry};
pub use error::{DSLCompileError, Result};
pub use expr::Expr;

// TWO CORE CONTEXTS - CLEAN ARCHITECTURE
// ============================================================================

// 1. STATIC CONTEXT - Compile-time optimization with automatic scope management + HList heterogeneous support
pub use compile_time::{
    HListEval, HListStorage, IntoHListEvaluable, StaticAdd, StaticConst, StaticContext, StaticExpr,
    StaticMul, StaticScopeBuilder, StaticVar, static_add, static_mul,
};

// 2. DYNAMIC CONTEXT - Runtime flexibility with JIT and symbolic optimization
pub use ast::{DynamicContext, TypedBuilderExpr, TypedVar};

// Legacy compatibility exports removed - use StaticContext and DynamicContext instead

// Evaluation functionality
pub use symbolic::symbolic::{
    CompilationApproach, CompilationStrategy, OptimizationConfig, SymbolicOptimizer,
};

pub use symbolic::anf;

// Primary backend exports (Rust codegen)
pub use backends::{CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel};

// Summation exports - DEPRECATED: Use DynamicContext.sum() instead
#[deprecated(
    note = "Use DynamicContext.sum() for summations. LegacySummationProcessor will be removed in future versions."
)]
pub use symbolic::summation::{SummationConfig, SummationPattern, SummationResult};

/// Version information for the `DSLCompile` library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for imports
///
/// This module re-exports the most commonly used types and functions for easy access.
/// Import this module to get started with `DSLCompile`.
///
/// # Examples
///
/// ## StaticContext (Recommended - Zero Overhead)
///
/// ```rust
/// use dslcompile::prelude::*;
/// use frunk::hlist;
///
/// // Zero-overhead, compile-time scoped variables with HList heterogeneous support
/// let mut ctx = StaticContext::new();
///
/// let f = ctx.new_scope(|scope| {
///     let (x, scope) = scope.auto_var::<f64>();
///     let (y, _scope) = scope.auto_var::<f32>();
///     x.clone() * x + scope.constant(2.0) * y  // x² + 2y
/// });
///
/// // Evaluate with heterogeneous inputs - zero overhead
/// let result = f.eval_hlist(hlist![3.0, 4.0f32]); // 3² + 2*4 = 17
/// ```
///
/// ## DynamicContext (Runtime Flexibility)
///
/// ```rust
/// use dslcompile::prelude::*;
///
/// // Runtime flexibility, ergonomic syntax
/// let ctx = DynamicContext::new();
/// let x = ctx.var();
/// let expr = &x * &x + 2.0 * &x + 1.0;
/// let result = ctx.eval(&expr, &[3.0]);
/// ```
pub mod prelude {
    // Core expression types from ast module
    pub use crate::ast::{ASTRepr, NumericType, VariableRegistry};

    // Static context (compile-time, zero-overhead - RECOMMENDED)
    // Automatic scope management + HList heterogeneous support
    pub use crate::compile_time::{
        HListEval, HListStorage, IntoHListEvaluable, StaticAdd, StaticConst, StaticContext,
        StaticExpr, StaticMul, StaticScopeBuilder, StaticVar, static_add, static_mul,
    };

    // Legacy compatibility removed - use StaticContext and DynamicContext instead

    // Dynamic context (runtime flexibility)
    pub use crate::ast::{DynamicContext, TypedBuilderExpr, TypedVar};

    // Error handling
    pub use crate::error::{DSLCompileError, Result};

    // Symbolic optimization
    pub use crate::symbolic::symbolic::{OptimizationConfig, SymbolicOptimizer};

    // Automatic differentiation
    pub use crate::symbolic::symbolic_ad::{
        SymbolicAD, SymbolicADConfig, convenience as ad_convenience,
    };

    // Compilation backends
    pub use crate::backends::{
        CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel,
    };

    // Operator overloading wrapper
    pub use crate::expr::Expr;

    // ANF utilities
    pub use crate::symbolic::anf::{
        ANFCodeGen, ANFConverter, ANFExpr, DomainAwareANFConverter, DomainAwareOptimizationStats,
        convert_to_anf, generate_rust_code,
    };

    // Summation utilities - Use DynamicContext.sum() (main API)
    // SummationResult kept for backward compatibility but deprecated
    #[deprecated(
        note = "Use DynamicContext.sum() for summations. Returns TypedBuilderExpr<f64> directly."
    )]
    pub use crate::symbolic::summation::SummationResult;
}

/// Ergonomic wrapper for expressions with operator overloading
/// This is now just an alias to the ast-based `TypedBuilderExpr` system
pub mod expr {
    pub use crate::ast::TypedBuilderExpr as Expr;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        println!("DSLCompile version: {VERSION}");
    }

    #[test]
    fn test_ergonomic_api() {
        // Test that basic expression building works with the new natural syntax
        let math = DynamicContext::new();
        let x = math.var();

        // Build expression: 2x + 1 using natural operator overloading
        let expr = &x * 2.0 + 1.0;

        // Test evaluation with indexed variables
        let result = math.eval(&expr, &[3.0]);
        assert_eq!(result, 7.0); // 2*3 + 1 = 7

        // Test with multiple variables using natural syntax
        let y = math.var();
        let expr2 = &x * 2.0 + &y;
        let result2 = math.eval(&expr2, &[3.0, 4.0]);
        assert_eq!(result2, 10.0); // 2*3 + 4 = 10
    }

    #[test]
    fn test_optimization_pipeline() {
        // Test that optimizations properly reduce expressions using natural syntax
        let math = DynamicContext::new();
        let x = math.var();

        // Create an expression that should optimize to zero: x - x
        let expr = x.clone() - x.clone();

        // With optimization
        let optimized_result = math.eval(&expr, &[5.0]);
        assert_eq!(optimized_result, 0.0);

        // Test evaluation with two variables using natural syntax
        let y = math.var();
        let expr = &x * 2.0 + &y;
        let result = math.eval(&expr, &[3.0, 4.0]);
        assert_eq!(result, 10.0); // 2*3 + 4 = 10
    }

    #[test]
    fn test_transcendental_functions() {
        let math = DynamicContext::new();
        let x = math.var();

        // Test trigonometric functions
        let result = math.eval(&x.sin(), &[0.0]);
        assert!((result - 0.0).abs() < 1e-10); // sin(0) = 0
    }

    #[test]
    fn test_rust_code_generation() {
        // Test Rust code generation with natural syntax
        let math = DynamicContext::new();
        let x = math.var();
        let _expr = &x * 2.0 + 1.0;

        // Convert to AST for code generation (until backends are updated)
        let _traditional_expr = math.to_ast(&_expr);

        let codegen = RustCodeGenerator::new();
        let rust_code = codegen
            .generate_function(&_traditional_expr, "test_func")
            .unwrap();

        assert!(rust_code.contains("test_func"));
        assert!(rust_code.contains("var_0 * 2"));
        assert!(rust_code.contains("+ 1"));
    }
}

/// Integration tests for the complete pipeline
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_pipeline() {
        // Create a complex expression using the new API
        let math = DynamicContext::new();
        let x = math.var();
        let y = math.var();

        // Build: (x + 0) * 2 + (y - 0) should optimize to x * 2 + y
        let expr = (x.clone() + 0.0) * 2.0 + (y.clone() - 0.0);

        // Step 1: Optimize symbolically
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let ast_expr = math.to_ast(&expr);
        let optimized = optimizer.optimize(&ast_expr).unwrap();

        // Step 2: Generate Rust code
        let codegen = RustCodeGenerator::new();
        let rust_code = codegen
            .generate_function(&optimized, "optimized_func")
            .unwrap();

        // Step 3: Test that we can still evaluate directly
        let direct_result = optimized.eval_two_vars(3.0, 4.0);
        assert_eq!(direct_result, 10.0); // 2*3 + 4 = 10

        // Verify the generated code looks reasonable
        assert!(rust_code.contains("optimized_func"));
        println!("Generated optimized Rust code:\n{rust_code}");
    }

    #[test]
    fn test_adaptive_compilation_strategy() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
            call_threshold: 3,
            complexity_threshold: 10,
        });

        let math = DynamicContext::new();
        let x = math.var();
        let expr = x + 1.0;
        let ast_expr = math.to_ast(&expr);

        // First few calls should use Rust compilation
        for i in 0..5 {
            let approach = optimizer.choose_compilation_approach(&ast_expr, "adaptive_test");
            println!("Call {i}: {approach:?}");

            optimizer.record_execution("adaptive_test", 1000);
        }

        // After threshold, should upgrade to Rust
        let approach = optimizer.choose_compilation_approach(&ast_expr, "adaptive_test");
        assert!(matches!(
            approach,
            CompilationApproach::UpgradeOptimization | CompilationApproach::RustHotLoad
        ));
    }
}
