//! `MathCompile`: Mathematical Expression Compilation
//!
//! `MathCompile` provides a three-layer optimization strategy for mathematical expressions:
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
//! use mathcompile::prelude::*;
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

// Core modules
pub mod error;
pub mod final_tagless;

// Optimization layer
pub mod symbolic;

// Compilation backends
pub mod backends;

// Ergonomics and user-friendly API
pub mod ergonomics;

// Re-export commonly used types
pub use error::{MathCompileError, Result};
pub use expr::Expr;
pub use final_tagless::{
    ASTEval,
    ASTMathExpr,
    ASTRepr,
    DirectEval,
    // New typed variable system
    MathBuilder,
    MathExpr,
    NumericType,
    PrettyPrint,
    StatisticalExpr,
    TypeCategory,
    TypedBuilderExpr,
    TypedVar,
    TypedVariableRegistry,
};
pub use symbolic::symbolic::{
    CompilationApproach, CompilationStrategy, OptimizationConfig, SymbolicOptimizer,
};

pub use symbolic::anf;

// Primary backend exports (Rust codegen)
pub use backends::{CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel};

// Ergonomics exports (keeping for backward compatibility)
pub use ergonomics::presets;

// Optional backend exports (Cranelift)
#[cfg(feature = "cranelift")]
pub use backends::cranelift::{CompilationStats, JITCompiler, JITFunction, JITSignature};

// Conditional exports based on features
#[cfg(feature = "cranelift")]
pub use backends::cranelift;

// Summation exports
pub use symbolic::summation::{SumResult, SummationConfig, SummationPattern, SummationSimplifier};

// ANF exports
pub use symbolic::anf::{
    ANFAtom, ANFCodeGen, ANFComputation, ANFConverter, ANFExpr, ANFVarGen, DomainAwareANFConverter,
    DomainAwareOptimizationStats, VarRef, convert_to_anf, generate_rust_code,
};

/// Version information for the `MathCompile` library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for imports
///
/// This module re-exports the most commonly used types and functions for easy access.
/// Import this module to get started with `MathCompile`.
///
/// # Examples
///
/// ## Typed API
///
/// ```rust
/// use mathcompile::prelude::*;
///
/// // Type-safe variable creation
/// let math = MathBuilder::new();
/// let x: TypedVar<f64> = math.typed_var();
/// let y: TypedVar<f32> = math.typed_var();
///
/// // Mathematical syntax with type safety
/// let x_expr = math.expr_from(x);
/// let y_expr = math.expr_from(y);
/// let expr = &x_expr * &x_expr + y_expr;  // Automatic f32 → f64 promotion
/// ```
///
/// ## Backward Compatible API
///
/// ```rust
/// use mathcompile::prelude::*;
///
/// // Old API still works (defaults to f64)
/// let math = MathBuilder::new();
/// let x = math.var();
/// let expr = &x * &x + 2.0 * &x + 1.0;
/// ```
pub mod prelude {
    // Core expression types
    pub use crate::final_tagless::{
        ASTEval,
        ASTMathExpr,
        ASTRepr,
        DirectEval,
        // New typed variable system
        MathBuilder,
        MathExpr,
        NumericType,
        PrettyPrint,
        StatisticalExpr,
        TypeCategory,
        TypedBuilderExpr,
        TypedExpressionBuilder,
        TypedVar,
        TypedVariableRegistry,
        VariableRegistry,
    };

    // Error handling
    pub use crate::error::{MathCompileError, Result};

    // Ergonomic API (keeping old for compatibility, but MathBuilder is now primary)
    pub use crate::ergonomics::presets;

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
        CompilationStats, JITCompiler, JITFunction, JITSignature,
    };

    // Operator overloading wrapper
    pub use crate::expr::Expr;

    // Summation utilities
    pub use crate::symbolic::summation::{SummationConfig, SummationSimplifier};

    // ANF utilities
    pub use crate::symbolic::anf::{
        ANFCodeGen, ANFConverter, ANFExpr, DomainAwareANFConverter, DomainAwareOptimizationStats,
        convert_to_anf, generate_rust_code,
    };
}

/// Ergonomic wrapper for final tagless expressions with operator overloading
pub mod expr {
    use crate::final_tagless::{DirectEval, MathExpr, NumericType, PrettyPrint};
    use num_traits::Float;
    use std::marker::PhantomData;
    use std::ops::{Add, Div, Mul, Neg, Sub};

    /// Wrapper type that enables operator overloading for final tagless expressions
    #[derive(Debug)]
    pub struct Expr<E: MathExpr, T> {
        pub(crate) repr: E::Repr<T>,
        _phantom: PhantomData<E>,
    }

    impl<E: MathExpr, T> Clone for Expr<E, T>
    where
        E::Repr<T>: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                repr: self.repr.clone(),
                _phantom: PhantomData,
            }
        }
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

        /// Create a constant expression
        pub fn constant(value: T) -> Self
        where
            T: NumericType,
        {
            Self::new(E::constant(value))
        }

        /// Create a variable reference by index
        #[must_use]
        pub fn var(index: usize) -> Self
        where
            T: NumericType,
        {
            Self::new(E::var(index))
        }

        /// Power operation
        pub fn pow(self, exp: Self) -> Self
        where
            T: NumericType + Float,
        {
            Self::new(E::pow(self.repr, exp.repr))
        }

        /// Natural logarithm
        pub fn ln(self) -> Self
        where
            T: NumericType + Float,
        {
            Self::new(E::ln(self.repr))
        }

        /// Exponential function
        pub fn exp(self) -> Self
        where
            T: NumericType + Float,
        {
            Self::new(E::exp(self.repr))
        }

        /// Square root
        pub fn sqrt(self) -> Self
        where
            T: NumericType + Float,
        {
            Self::new(E::sqrt(self.repr))
        }

        /// Sine function
        pub fn sin(self) -> Self
        where
            T: NumericType + Float,
        {
            Self::new(E::sin(self.repr))
        }

        /// Cosine function
        pub fn cos(self) -> Self
        where
            T: NumericType + Float,
        {
            Self::new(E::cos(self.repr))
        }
    }

    /// Special methods for `DirectEval` expressions
    impl<T> Expr<DirectEval, T> {
        /// Create a variable with a specific value for direct evaluation
        pub fn var_with_value(_index: usize, value: T) -> Self
        where
            T: NumericType,
        {
            // For DirectEval, we just return the value directly since it evaluates immediately
            Self::new(value)
        }

        /// Evaluate the expression directly (only available for `DirectEval`)
        pub fn eval(self) -> T {
            self.repr
        }
    }

    /// Special methods for `PrettyPrint` expressions
    impl<T> Expr<PrettyPrint, T> {
        /// Get the string representation (only available for `PrettyPrint`)
        #[must_use]
        pub fn to_string(self) -> String {
            self.repr
        }
    }

    /// Addition operator overloading
    impl<E: MathExpr, L, R, Output> Add<Expr<E, R>> for Expr<E, L>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        type Output = Expr<E, Output>;

        fn add(self, rhs: Expr<E, R>) -> Self::Output {
            Expr::new(E::add(self.repr, rhs.repr))
        }
    }

    /// Subtraction operator overloading
    impl<E: MathExpr, L, R, Output> Sub<Expr<E, R>> for Expr<E, L>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        type Output = Expr<E, Output>;

        fn sub(self, rhs: Expr<E, R>) -> Self::Output {
            Expr::new(E::sub(self.repr, rhs.repr))
        }
    }

    /// Multiplication operator overloading
    impl<E: MathExpr, L, R, Output> Mul<Expr<E, R>> for Expr<E, L>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        type Output = Expr<E, Output>;

        fn mul(self, rhs: Expr<E, R>) -> Self::Output {
            Expr::new(E::mul(self.repr, rhs.repr))
        }
    }

    /// Division operator overloading
    impl<E: MathExpr, L, R, Output> Div<Expr<E, R>> for Expr<E, L>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        type Output = Expr<E, Output>;

        fn div(self, rhs: Expr<E, R>) -> Self::Output {
            Expr::new(E::div(self.repr, rhs.repr))
        }
    }

    /// Negation operator overloading
    impl<E: MathExpr, T> Neg for Expr<E, T>
    where
        T: NumericType + Neg<Output = T>,
    {
        type Output = Expr<E, T>;

        fn neg(self) -> Self::Output {
            Expr::new(E::neg(self.repr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        println!("MathCompile version: {VERSION}");
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
        let expr = &x - &x;

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
    fn test_cranelift_compilation() {
        // Test Cranelift compilation with natural syntax
        let math = MathBuilder::new();
        let x = math.var();
        let _expr = &x * 2.0 + 1.0;

        // Convert to traditional AST for compilation (until backends are updated)
        use crate::final_tagless::ASTMathExpr;
        let traditional_expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::mul(
                <ASTEval as ASTMathExpr>::var(0),
                <ASTEval as ASTMathExpr>::constant(2.0),
            ),
            <ASTEval as ASTMathExpr>::constant(1.0),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_single_var(&traditional_expr, "x").unwrap();

        let result = jit_func.call_single(3.0);
        assert_eq!(result, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_rust_code_generation() {
        // Test Rust code generation with natural syntax
        let math = MathBuilder::new();
        let x = math.var();
        let _expr = &x * 2.0 + 1.0;

        // Convert to traditional AST for code generation (until backends are updated)
        use crate::final_tagless::ASTMathExpr;
        let traditional_expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::mul(
                <ASTEval as ASTMathExpr>::var(0),
                <ASTEval as ASTMathExpr>::constant(2.0),
            ),
            <ASTEval as ASTMathExpr>::constant(1.0),
        );

        let codegen = RustCodeGenerator::new();
        let rust_code = codegen
            .generate_function(&traditional_expr, "test_func")
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
        // Create a complex expression
        let expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::mul(
                <ASTEval as ASTMathExpr>::add(
                    <ASTEval as ASTMathExpr>::var(0),
                    <ASTEval as ASTMathExpr>::constant(0.0),
                ), // Should optimize to x
                <ASTEval as ASTMathExpr>::constant(2.0),
            ),
            <ASTEval as ASTMathExpr>::sub(
                <ASTEval as ASTMathExpr>::var(1),
                <ASTEval as ASTMathExpr>::constant(0.0),
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

    #[cfg(feature = "cranelift")]
    #[test]
    fn test_adaptive_compilation_strategy() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
            call_threshold: 3,
            complexity_threshold: 10,
        });

        let expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::var(0),
            <ASTEval as ASTMathExpr>::constant(1.0),
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

/// Interval-based domain analysis with endpoint specification
pub mod interval_domain;

// Re-export polynomial utilities at the crate level for convenience
pub mod polynomial {
    pub use crate::final_tagless::polynomial::*;
}

pub mod ast;

pub mod compile_time;
