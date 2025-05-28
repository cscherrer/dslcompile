//! `MathJIT`: High-Performance Mathematical Expression Compilation
//!
//! `MathJIT` provides a three-layer optimization strategy for mathematical expressions:
//! 1. **Final Tagless Approach**: Type-safe expression building with multiple interpreters
//! 2. **Symbolic Optimization**: Algebraic simplification using egglog
//! 3. **Compilation Backends**: Rust hot-loading (primary) and optional Cranelift JIT
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

// Egglog integration module (optional)
#[cfg(feature = "optimization")]
pub mod egglog_integration;

// Symbolic automatic differentiation module
pub mod symbolic_ad;

// Compilation backends
pub mod backends;

// Utilities
pub mod transcendental;

// Re-export commonly used types
pub use error::{MathJITError, Result};
pub use expr::Expr;
pub use final_tagless::{
    ASTEval, ASTMathExpr, ASTRepr, DirectEval, MathExpr, NumericType, PrettyPrint, StatisticalExpr,
};
pub use symbolic::{
    CompilationApproach, CompilationStrategy, OptimizationConfig, SymbolicOptimizer,
};

// Primary backend exports (Rust codegen)
pub use backends::{RustCodeGenerator, RustCompiler, RustOptLevel};

// Optional backend exports (Cranelift)
#[cfg(feature = "cranelift")]
pub use backends::cranelift::{CompilationStats, JITCompiler, JITFunction, JITSignature};

// Conditional exports based on features
#[cfg(feature = "cranelift")]
pub use backends::cranelift;

// Symbolic AD exports
pub use symbolic_ad::{FunctionWithDerivatives, SymbolicAD, SymbolicADConfig, SymbolicADStats};

/// Version information for the `MathJIT` library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::expr::Expr;
    pub use crate::final_tagless::{ASTEval, ASTMathExpr, DirectEval, MathExpr};
    pub use crate::symbolic::{CompilationStrategy, SymbolicOptimizer};

    #[cfg(feature = "cranelift")]
    pub use crate::backends::cranelift::{JITCompiler, JITFunction};

    pub use crate::backends::rust_codegen::RustCodeGenerator;
    pub use crate::{MathJITError, Result};

    // Symbolic AD
    pub use crate::symbolic_ad::{SymbolicAD, SymbolicADConfig};
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

        /// Create a variable expression
        #[must_use]
        pub fn var(name: &str) -> Self
        where
            T: NumericType,
        {
            Self::new(E::var(name))
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
        pub fn var_with_value(name: &str, value: T) -> Self
        where
            T: NumericType,
        {
            Self::new(DirectEval::var(name, value))
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
        println!("MathJIT version: {VERSION}");
    }

    #[test]
    fn test_basic_expression_building() {
        use crate::final_tagless::ASTMathExpr;

        // Test that basic expression building works
        let expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::mul(
                <ASTEval as ASTMathExpr>::var("x"),
                <ASTEval as ASTMathExpr>::constant(2.0),
            ),
            <ASTEval as ASTMathExpr>::constant(1.0),
        );

        // Should be able to evaluate directly
        let result = DirectEval::eval_two_vars(&expr, 3.0, 0.0);
        assert_eq!(result, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_optimization_pipeline() {
        // Test that optimizations properly reduce expressions
        let expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::var("x"),
            <ASTEval as ASTMathExpr>::constant(0.0),
        );

        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let optimized = optimizer.optimize(&expr).unwrap();

        // Should optimize to just x (either Variable or VariableByName)
        match optimized {
            ASTRepr::VariableByName(name) => assert_eq!(name, "x"),
            ASTRepr::Variable(_) => {
                // Also acceptable - indexed variable
            }
            _ => panic!("Expected optimization to reduce x + 0 to x, got {:?}", optimized),
        }
    }

    #[cfg(feature = "cranelift")]
    #[test]
    fn test_cranelift_compilation() {
        let expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::mul(
                <ASTEval as ASTMathExpr>::var("x"),
                <ASTEval as ASTMathExpr>::constant(2.0),
            ),
            <ASTEval as ASTMathExpr>::constant(1.0),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_single_var(&expr, "x").unwrap();

        let result = jit_func.call_single(3.0);
        assert_eq!(result, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_rust_code_generation() {
        let expr = <ASTEval as ASTMathExpr>::add(
            <ASTEval as ASTMathExpr>::mul(
                <ASTEval as ASTMathExpr>::var("x"),
                <ASTEval as ASTMathExpr>::constant(2.0),
            ),
            <ASTEval as ASTMathExpr>::constant(1.0),
        );

        let codegen = RustCodeGenerator::new();
        let rust_code = codegen.generate_function(&expr, "test_func").unwrap();

        assert!(rust_code.contains("test_func"));
        assert!(rust_code.contains("x * 2"));
        assert!(rust_code.contains("+ 1"));
    }

    #[test]
    fn test_expr_operator_overloading() {
        use crate::expr::Expr;

        // Test the new ergonomic Expr wrapper with operator overloading

        // Define a quadratic function using natural syntax: 2x² + 3x + 1
        fn quadratic(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
            let a = Expr::constant(2.0);
            let b = Expr::constant(3.0);
            let c = Expr::constant(1.0);

            // Natural mathematical syntax!
            a * x.clone() * x.clone() + b * x + c
        }

        // Test with x = 2: 2(4) + 3(2) + 1 = 15
        let x = Expr::var_with_value("x", 2.0);
        let result = quadratic(x);
        assert_eq!(result.eval(), 15.0);

        // Test with x = 0: 2(0) + 3(0) + 1 = 1
        let x = Expr::var_with_value("x", 0.0);
        let result = quadratic(x);
        assert_eq!(result.eval(), 1.0);
    }

    #[test]
    fn test_expr_transcendental_functions() {
        use crate::expr::Expr;

        // Test transcendental functions with the Expr wrapper

        // Test: exp(ln(x)) = x
        let x = Expr::var_with_value("x", 5.0);
        let result = x.ln().exp();
        assert!((result.eval() - 5.0_f64).abs() < 1e-10);

        // Test: sin²(x) + cos²(x) = 1
        let x = Expr::var_with_value("x", 1.5_f64);
        let sin_x = x.clone().sin();
        let cos_x = x.cos();
        let result = sin_x.clone() * sin_x + cos_x.clone() * cos_x;
        assert!((result.eval() - 1.0_f64).abs() < 1e-10);
    }

    #[test]
    fn test_expr_pretty_print() {
        use crate::expr::Expr;

        // Test pretty printing with the Expr wrapper

        fn simple_expr(x: Expr<PrettyPrint, f64>) -> Expr<PrettyPrint, f64> {
            let two = Expr::constant(2.0);
            let three = Expr::constant(3.0);
            two * x + three
        }

        let x = Expr::<PrettyPrint, f64>::var("x");
        let pretty = simple_expr(x);
        let result = pretty.to_string();

        // Should contain the key components
        assert!(result.contains('x'));
        assert!(result.contains('2'));
        assert!(result.contains('3'));
        assert!(result.contains('*'));
        assert!(result.contains('+'));
    }

    #[test]
    fn test_expr_negation() {
        use crate::expr::Expr;

        // Test negation operator
        let x = Expr::var_with_value("x", 5.0);
        let neg_x = -x;
        assert_eq!(neg_x.eval(), -5.0);

        // Test: -(x + y) = -x - y
        let x = Expr::var_with_value("x", 3.0);
        let y = Expr::var_with_value("y", 2.0);
        let result = -(x.clone() + y.clone());
        let expected = -x - y;
        let result_val = result.eval();
        assert_eq!(result_val, expected.eval());
        assert_eq!(result_val, -5.0);
    }

    #[test]
    fn test_expr_mixed_operations() {
        use crate::expr::Expr;

        // Test complex expressions with mixed operations

        // Test: (x + 1) * (x - 1) = x² - 1
        let x = Expr::var_with_value("x", 4.0);
        let one = Expr::constant(1.0);

        let left = x.clone() + one.clone();
        let right = x.clone() - one;
        let result = left * right;

        // At x=4: (4+1)*(4-1) = 5*3 = 15
        let result_val = result.eval();
        assert_eq!(result_val, 15.0);

        // Verify it equals x² - 1
        let x_squared_minus_one = x.clone() * x - Expr::constant(1.0);
        assert_eq!(result_val, x_squared_minus_one.eval());
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
                    <ASTEval as ASTMathExpr>::var("x"),
                    <ASTEval as ASTMathExpr>::constant(0.0),
                ), // Should optimize to x
                <ASTEval as ASTMathExpr>::constant(2.0),
            ),
            <ASTEval as ASTMathExpr>::sub(
                <ASTEval as ASTMathExpr>::var("y"),
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
            <ASTEval as ASTMathExpr>::var("x"),
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
