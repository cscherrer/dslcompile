//! `MathJIT`: High-Performance Mathematical Expression Compilation
//!
//! `MathJIT` provides a three-layer optimization strategy for mathematical expressions:
//! 1. **Final Tagless Approach**: Type-safe expression building with multiple interpreters
//! 2. **Symbolic Optimization**: Algebraic simplification using egglog
//! 3. **JIT Compilation**: Multiple backends (Cranelift, Rust hot-loading)
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
//! │  │  Cranelift  │  │    Rust     │  │  Future Backends    │  │
//! │  │     JIT     │  │ Hot-Loading │  │   (LLVM, etc.)      │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

// Core modules
pub mod error;
pub mod final_tagless;

// Optimization layer
pub mod symbolic;

// Egglog integration module (optional)
#[cfg(feature = "optimization")]
pub mod egglog_integration;

// Compilation backends
pub mod backends;

// Utilities
pub mod transcendental;

// Re-export commonly used types
pub use error::{MathJITError, Result};
pub use final_tagless::{
    DirectEval, JITEval, JITMathExpr, JITRepr, MathExpr, NumericType, PrettyPrint, StatisticalExpr,
};
pub use symbolic::{
    CompilationApproach, CompilationStrategy, OptimizationConfig, RustOptLevel, SymbolicOptimizer,
};

// Backend-specific exports
#[cfg(feature = "jit")]
pub use backends::cranelift::{CompilationStats, JITCompiler, JITFunction, JITSignature};

pub use backends::{RustCodeGenerator, RustCompiler};

// Conditional exports based on features
#[cfg(feature = "jit")]
pub use backends::cranelift;

/// Version information for the `MathJIT` library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::final_tagless::{DirectEval, JITEval, JITMathExpr, MathExpr};
    pub use crate::symbolic::{CompilationStrategy, SymbolicOptimizer};

    #[cfg(feature = "jit")]
    pub use crate::backends::cranelift::{JITCompiler, JITFunction};

    pub use crate::backends::rust_codegen::RustCodeGenerator;
    pub use crate::{MathJITError, Result};
}

/// Ergonomic wrapper for final tagless expressions with operator overloading
pub mod expr {
    use crate::final_tagless::MathExpr;
    use std::marker::PhantomData;

    /// Wrapper type that enables operator overloading for final tagless expressions
    pub struct Expr<E: MathExpr, T> {
        pub(crate) repr: E::Repr<T>,
        _phantom: PhantomData<E>,
    }

    impl<E: MathExpr, T> Expr<E, T> {
        /// Create a new expression wrapper
        pub fn new(repr: E::Repr<T>) -> Self {
            Self {
                repr,
                _phantom: PhantomData,
            }
        }

        /// Extract the underlying representation
        pub fn into_repr(self) -> E::Repr<T> {
            self.repr
        }

        /// Get a reference to the underlying representation
        pub fn as_repr(&self) -> &E::Repr<T> {
            &self.repr
        }

        /// Create a variable expression
        #[must_use]
        pub fn var(name: &str) -> Self
        where
            E::Repr<T>: Clone,
            T: crate::final_tagless::NumericType,
        {
            Self::new(E::var(name))
        }

        /// Create a constant expression
        pub fn constant(value: T) -> Self
        where
            T: crate::final_tagless::NumericType,
        {
            Self::new(E::constant(value))
        }
    }

    // Operator overloading implementations will be added here
    // This provides ergonomic syntax like: x + y * constant(2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        println!("MathJIT version: {VERSION}");
    }

    #[test]
    fn test_basic_expression_building() {
        use crate::final_tagless::JITMathExpr;

        // Test that basic expression building works
        let expr = <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::mul(
                <JITEval as JITMathExpr>::var("x"),
                <JITEval as JITMathExpr>::constant(2.0),
            ),
            <JITEval as JITMathExpr>::constant(1.0),
        );

        // Should be able to evaluate directly
        let result = DirectEval::eval_two_vars(&expr, 3.0, 0.0);
        assert_eq!(result, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_optimization_pipeline() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Expression that can be optimized: x + 0
        let expr = <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::var("x"),
            <JITEval as JITMathExpr>::constant(0.0),
        );

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should optimize to just x
        match optimized {
            JITRepr::Variable(name) => assert_eq!(name, "x"),
            _ => panic!("Expected optimization to reduce x + 0 to x"),
        }
    }

    #[cfg(feature = "jit")]
    #[test]
    fn test_cranelift_compilation() {
        let expr = <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::mul(
                <JITEval as JITMathExpr>::var("x"),
                <JITEval as JITMathExpr>::constant(2.0),
            ),
            <JITEval as JITMathExpr>::constant(1.0),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_single_var(&expr, "x").unwrap();

        let result = jit_func.call_single(3.0);
        assert_eq!(result, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_rust_code_generation() {
        let expr = <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::mul(
                <JITEval as JITMathExpr>::var("x"),
                <JITEval as JITMathExpr>::constant(2.0),
            ),
            <JITEval as JITMathExpr>::constant(1.0),
        );

        let codegen = RustCodeGenerator::new();
        let rust_code = codegen.generate_function(&expr, "test_func").unwrap();

        assert!(rust_code.contains("test_func"));
        assert!(rust_code.contains("x * 2"));
        assert!(rust_code.contains("+ 1"));
    }
}

/// Integration tests for the complete pipeline
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_pipeline() {
        // Create a complex expression
        let expr = <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::mul(
                <JITEval as JITMathExpr>::add(
                    <JITEval as JITMathExpr>::var("x"),
                    <JITEval as JITMathExpr>::constant(0.0),
                ), // Should optimize to x
                <JITEval as JITMathExpr>::constant(2.0),
            ),
            <JITEval as JITMathExpr>::sub(
                <JITEval as JITMathExpr>::var("y"),
                <JITEval as JITMathExpr>::constant(0.0),
            ), // Should optimize to y
        );

        // Step 1: Optimize symbolically
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let optimized = optimizer.optimize(&expr).unwrap();

        // Step 2: Generate Rust code
        let codegen = RustCodeGenerator::new();
        let rust_code = codegen
            .generate_function(&optimized, "optimized_func")
            .unwrap();

        // Step 3: Test that we can still evaluate directly
        let direct_result = DirectEval::eval_two_vars(&optimized, 3.0, 4.0);
        assert_eq!(direct_result, 10.0); // 2*3 + 4 = 10

        // Verify the generated code looks reasonable
        assert!(rust_code.contains("optimized_func"));
        println!("Generated optimized Rust code:\n{rust_code}");
    }

    #[cfg(feature = "jit")]
    #[test]
    fn test_adaptive_compilation_strategy() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
            call_threshold: 3,
            complexity_threshold: 10,
        });

        let expr = <JITEval as JITMathExpr>::add(
            <JITEval as JITMathExpr>::var("x"),
            <JITEval as JITMathExpr>::constant(1.0),
        );

        // First few calls should use Cranelift
        for i in 0..5 {
            let approach = optimizer.choose_compilation_approach(&expr, "adaptive_test");
            println!("Call {i}: {approach:?}");

            if i < 2 {
                assert_eq!(approach, CompilationApproach::Cranelift);
            }

            optimizer.record_execution("adaptive_test", 1000);
        }

        // After threshold, should upgrade to Rust
        let approach = optimizer.choose_compilation_approach(&expr, "adaptive_test");
        assert!(matches!(
            approach,
            CompilationApproach::UpgradeToRust | CompilationApproach::RustHotLoad
        ));
    }
}
