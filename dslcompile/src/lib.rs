//! `DSLCompile`: Mathematical Expression Compilation
//!
//! `DSLCompile` provides a three-layer optimization strategy for mathematical expressions:
//! 1. **Final Tagless Approach**: Type-safe expression building with multiple interpreters
//! 2. **Symbolic Optimization**: Algebraic simplification using egglog
//! 3. **Compilation Backends**: Rust hot-loading (primary) and optional Cranelift JIT
//!
//! # Typed Variable System
//!
//! The library includes a type-safe variable system that provides compile-time type checking
//! with operator overloading syntax and full backward compatibility.
//!
//! ## Quick Start with Typed Variables
//!
//! ```rust
//! use dslcompile::prelude::*;
//!
//! // Create a typed math builder
//! let math = MathBuilder::new();
//!
//! // Create typed variables
//! let x: TypedVar<f64> = math.typed_var();
//! let y: TypedVar<f32> = math.typed_var();
//!
//! // Build expressions with syntax and type safety
//! let x_expr = math.expr_from(x);
//! let y_expr = math.expr_from(y);
//! let expr = &x_expr * &x_expr + y_expr;  // f32 auto-promotes to f64
//!
//! // Backward compatible API still works
//! let old_style = math.var();  // Defaults to f64
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Final Tagless Layer                     │
//! │  (Expression Building & Type Safety)                       │
//! └─────────────────────┬───────────────────────────────────────┘
//!                       │
//! ┌─────────────────────▼───────────────────────────────────────┐
//! │                 Symbolic Optimization                       │
//! │  (Algebraic Simplification & Rewrite Rules)                │
//! └─────────────────────┬───────────────────────────────────────┘
//!                       │
//! ┌─────────────────────▼───────────────────────────────────────┐
//! │                 Compilation Backends                        │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │    Rust     │  │  Cranelift  │  │  Future Backends    │  │
//! │  │ Hot-Loading │  │     JIT     │  │   (LLVM, etc.)      │  │
//! │  │ (Primary)   │  │ (Optional)  │  │                     │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]
#![feature(generic_const_exprs)]

// Core modules
pub mod error;

// Optimization layer
pub mod symbolic;

// Compilation backends
pub mod backends;

// Re-export commonly used types
pub use error::{DSLCompileError, Result};
pub use expr::Expr;

// Core types now come from ast module (replacing final_tagless)
pub use ast::{ASTRepr, NumericType, VariableRegistry};

// Runtime expression building (the future of the system)
pub use ast::{DynamicContext, MathBuilder, TypedBuilderExpr, TypedVar};

// Compile-time expression building with scoped variables (recommended as default)
pub use compile_time::{Context, ScopeBuilder, ScopedMathExpr, ScopedVar, ScopedVarArray, compose};

// Legacy compatibility exports
pub use ast::ExpressionBuilder;
pub use compile_time::ScopedExpressionBuilder;

// Evaluation functionality

pub use symbolic::symbolic::{
    CompilationApproach, CompilationStrategy, OptimizationConfig, SymbolicOptimizer,
};

pub use symbolic::anf;

// Primary backend exports (Rust codegen)
pub use backends::{CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel};

// Optional backend exports (Cranelift)
#[cfg(feature = "cranelift")]
pub use backends::{
    CompilationMetadata, CraneliftCompiledFunction, CraneliftCompiler, CraneliftFunctionSignature,
    CraneliftOptLevel,
};

// Conditional exports based on features
#[cfg(feature = "cranelift")]
pub use backends::cranelift;

// Summation exports - Type-safe closure-based system
pub use symbolic::summation::{
    SummationConfig, SummationPattern, SummationProcessor, SummationResult,
};

/// Version information for the `DSLCompile` library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for imports
///
/// This module re-exports the most commonly used types and functions for easy access.
/// Import this module to get started with `DSLCompile`.
///
/// # Examples
///
/// ## Static Context (Recommended - Zero Overhead)
///
/// ```rust
/// use dslcompile::prelude::*;
///
/// // Zero-overhead, compile-time scoped variables
/// let mut ctx = Context::new_f64();
///
/// let f = ctx.new_scope(|scope| {
///     let (x, _scope) = scope.auto_var();
///     x.clone() * x  // x²
/// });
/// ```
///
/// ## Dynamic Context (Runtime Flexibility)
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
///
/// ## Backward Compatible API
///
/// ```rust
/// use dslcompile::prelude::*;
///
/// // Old API still works (defaults to f64)
/// let math = MathBuilder::new();
/// let x = math.var();
/// let expr = &x * &x + 2.0 * &x + 1.0;
/// ```
pub mod prelude {
    // Core expression types from ast module
    pub use crate::ast::{ASTRepr, NumericType, VariableRegistry};

    // Static context (compile-time, zero-overhead - RECOMMENDED)
    pub use crate::compile_time::{
        Context, ScopeBuilder, ScopedMathExpr, ScopedVar, ScopedVarArray, compose,
    };

    // Dynamic context (runtime flexibility)
    pub use crate::ast::{DynamicContext, TypedBuilderExpr, TypedVar};

    // Legacy compatibility aliases
    pub use crate::ast::{ExpressionBuilder, MathBuilder};
    pub use crate::compile_time::ScopedExpressionBuilder;

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

    // Optional Cranelift backend
    #[cfg(feature = "cranelift")]
    pub use crate::backends::cranelift::{
        CompilationMetadata, CompiledFunction, CraneliftCompiler, FunctionSignature,
    };

    // Operator overloading wrapper
    pub use crate::expr::Expr;

    // ANF utilities
    pub use crate::symbolic::anf::{
        ANFCodeGen, ANFConverter, ANFExpr, DomainAwareANFConverter, DomainAwareOptimizationStats,
        convert_to_anf, generate_rust_code,
    };

    // Summation utilities - Type-safe closure-based system
    pub use crate::symbolic::summation::{SummationProcessor, SummationResult};
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
        let math = MathBuilder::new();
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
        let math = MathBuilder::new();
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
        let math = MathBuilder::new();
        let x = math.var();

        // Test trigonometric functions
        let result = math.eval(&x.sin(), &[0.0]);
        assert!((result - 0.0).abs() < 1e-10); // sin(0) = 0
    }

    #[cfg(feature = "cranelift")]
    #[test]
    #[ignore] // TODO: Update for new Cranelift API
    fn test_cranelift_compilation() {
        // Test Cranelift compilation with natural syntax
        let math = MathBuilder::new();
        let x = math.var();
        let _expr = &x * 2.0 + 1.0;

        // Convert to AST for compilation (until backends are updated)
        let _traditional_expr = math.to_ast(&_expr);

        // TODO: Update to use new CraneliftCompiler API
        // let compiler = CraneliftCompiler::new(OptimizationLevel::Basic).unwrap();
        // let compiled = compiler.compile_expression(&traditional_expr, &registry).unwrap();
        // let result = compiled.call(&[3.0]).unwrap();
        // assert_eq!(result, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_rust_code_generation() {
        // Test Rust code generation with natural syntax
        let math = MathBuilder::new();
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
        let math = MathBuilder::new();
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

    #[cfg(feature = "cranelift")]
    #[test]
    fn test_adaptive_compilation_strategy() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
            call_threshold: 3,
            complexity_threshold: 10,
        });

        let math = MathBuilder::new();
        let x = math.var();
        let expr = x + 1.0;
        let ast_expr = math.to_ast(&expr);

        // First few calls should use Cranelift
        for i in 0..5 {
            let approach = optimizer.choose_compilation_approach(&ast_expr, "adaptive_test");
            println!("Call {i}: {approach:?}");

            if i < 2 {
                assert_eq!(approach, CompilationApproach::Cranelift);
            }

            optimizer.record_execution("adaptive_test", 1000);
        }

        // After threshold, should upgrade to Rust
        let approach = optimizer.choose_compilation_approach(&ast_expr, "adaptive_test");
        assert!(matches!(
            approach,
            CompilationApproach::UpgradeToRust | CompilationApproach::RustHotLoad
        ));
    }
}

/// Interval-based domain analysis with endpoint specification
pub mod interval_domain;

pub mod ast;
pub mod compile_time;
