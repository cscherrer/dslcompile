//! Compile-Time Mathematical Expression System
//!
//! This module provides a compile-time mathematical expression system that leverages
//! Rust's type system to perform optimizations at compile time.
//!
//! ## Low-Overhead Abstraction
//!
//! The system achieves low-overhead abstraction through:
//! - **Compile-time optimization**: Mathematical simplifications happen during compilation
//! - **Type-level computation**: Expression structure encoded in types
//! - **Direct evaluation**: Runtime evaluation compiles to simple operations
//! - **LLVM optimization**: Compiler can inline and optimize across expression boundaries

pub mod optimized;
pub mod scoped;

// Re-export the main types for convenience
pub use scoped::{
    ScopedConst, ScopedMathExpr, ScopedVar, ScopedVarArray, compose, scoped_constant, scoped_var,
};

/// Core trait for compile-time mathematical expressions
pub trait MathExpr: Clone + Sized {
    /// Evaluate the expression with the given variable values
    fn eval(&self, vars: &[f64]) -> f64;

    /// Add two expressions
    fn add<T: MathExpr>(self, other: T) -> Add<Self, T> {
        Add {
            left: self,
            right: other,
        }
    }

    /// Multiply two expressions
    fn mul<T: MathExpr>(self, other: T) -> Mul<Self, T> {
        Mul {
            left: self,
            right: other,
        }
    }

    /// Subtract two expressions
    fn sub<T: MathExpr>(self, other: T) -> Sub<Self, T> {
        Sub {
            left: self,
            right: other,
        }
    }

    /// Divide two expressions
    fn div<T: MathExpr>(self, other: T) -> Div<Self, T> {
        Div {
            left: self,
            right: other,
        }
    }

    /// Raise expression to a power
    fn pow<T: MathExpr>(self, exponent: T) -> Pow<Self, T> {
        Pow {
            base: self,
            exponent,
        }
    }

    /// Natural exponential
    fn exp(self) -> Exp<Self> {
        Exp { inner: self }
    }

    /// Natural logarithm
    fn ln(self) -> Ln<Self> {
        Ln { inner: self }
    }

    /// Sine function
    fn sin(self) -> Sin<Self> {
        Sin { inner: self }
    }

    /// Cosine function
    fn cos(self) -> Cos<Self> {
        Cos { inner: self }
    }

    /// Square root
    fn sqrt(self) -> Sqrt<Self> {
        Sqrt { inner: self }
    }

    /// Negation
    fn neg(self) -> Neg<Self> {
        Neg { inner: self }
    }
}

/// Trait for expressions that can be optimized at compile time
pub trait Optimize: MathExpr {
    type Optimized: MathExpr;

    /// Apply compile-time optimizations
    fn optimize(self) -> Self::Optimized;
}

/// Variable reference
#[derive(Clone, Debug)]
pub struct Var<const ID: usize>;

impl<const ID: usize> MathExpr for Var<ID> {
    fn eval(&self, vars: &[f64]) -> f64 {
        // Use safe bounds checking - the compiler optimizes this to zero overhead in release builds
        // ID is a compile-time constant, so the compiler can often eliminate bounds checks entirely
        vars.get(ID).copied().unwrap_or(0.0)
    }
}

/// Constant value
#[derive(Clone, Debug)]
pub struct Const<const BITS: u64>;

impl<const BITS: u64> Const<BITS> {
    /// Create a new constant (using bit representation to work around f64 const generic limitation)
    #[must_use]
    pub fn new(_value: f64) -> Self {
        // In practice, we'd need a more sophisticated encoding
        Self
    }

    /// Get the f64 value
    #[must_use]
    pub fn value(&self) -> f64 {
        f64::from_bits(BITS)
    }
}

impl<const BITS: u64> MathExpr for Const<BITS> {
    fn eval(&self, _vars: &[f64]) -> f64 {
        self.value()
    }
}

/// Addition expression
#[derive(Clone, Debug)]
pub struct Add<L: MathExpr, R: MathExpr> {
    left: L,
    right: R,
}

impl<L: MathExpr, R: MathExpr> MathExpr for Add<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.left.eval(vars) + self.right.eval(vars)
    }
}

/// Multiplication expression
#[derive(Clone, Debug)]
pub struct Mul<L: MathExpr, R: MathExpr> {
    left: L,
    right: R,
}

impl<L: MathExpr, R: MathExpr> MathExpr for Mul<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.left.eval(vars) * self.right.eval(vars)
    }
}

/// Subtraction expression
#[derive(Clone, Debug)]
pub struct Sub<L: MathExpr, R: MathExpr> {
    left: L,
    right: R,
}

impl<L: MathExpr, R: MathExpr> MathExpr for Sub<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.left.eval(vars) - self.right.eval(vars)
    }
}

/// Division expression
#[derive(Clone, Debug)]
pub struct Div<L: MathExpr, R: MathExpr> {
    left: L,
    right: R,
}

impl<L: MathExpr, R: MathExpr> MathExpr for Div<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.left.eval(vars) / self.right.eval(vars)
    }
}

/// Power expression
#[derive(Clone, Debug)]
pub struct Pow<B: MathExpr, E: MathExpr> {
    base: B,
    exponent: E,
}

impl<B: MathExpr, E: MathExpr> MathExpr for Pow<B, E> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.base.eval(vars).powf(self.exponent.eval(vars))
    }
}

/// Exponential expression
#[derive(Clone, Debug)]
pub struct Exp<T: MathExpr> {
    inner: T,
}

impl<T: MathExpr> MathExpr for Exp<T> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.inner.eval(vars).exp()
    }
}

/// Natural logarithm expression
#[derive(Clone, Debug)]
pub struct Ln<T: MathExpr> {
    inner: T,
}

impl<T: MathExpr> MathExpr for Ln<T> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.inner.eval(vars).ln()
    }
}

/// Sine expression
#[derive(Clone, Debug)]
pub struct Sin<T: MathExpr> {
    inner: T,
}

impl<T: MathExpr> MathExpr for Sin<T> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.inner.eval(vars).sin()
    }
}

/// Cosine expression
#[derive(Clone, Debug)]
pub struct Cos<T: MathExpr> {
    inner: T,
}

impl<T: MathExpr> MathExpr for Cos<T> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.inner.eval(vars).cos()
    }
}

/// Square root expression
#[derive(Clone, Debug)]
pub struct Sqrt<T: MathExpr> {
    inner: T,
}

impl<T: MathExpr> MathExpr for Sqrt<T> {
    fn eval(&self, vars: &[f64]) -> f64 {
        self.inner.eval(vars).sqrt()
    }
}

/// Negation expression
#[derive(Clone, Debug)]
pub struct Neg<T: MathExpr> {
    inner: T,
}

impl<T: MathExpr> MathExpr for Neg<T> {
    fn eval(&self, vars: &[f64]) -> f64 {
        -self.inner.eval(vars)
    }
}

// ============================================================================
// COMPILE-TIME OPTIMIZATIONS
// ============================================================================

/// ln(exp(x)) → x optimization
impl<T: MathExpr> Optimize for Ln<Exp<T>> {
    type Optimized = T;

    fn optimize(self) -> T {
        self.inner.inner
    }
}

/// exp(ln(x)) → x optimization  
impl<T: MathExpr> Optimize for Exp<Ln<T>> {
    type Optimized = T;

    fn optimize(self) -> T {
        self.inner.inner
    }
}

/// x + 0 → x optimization (only for Var to avoid conflicts)
impl<const ID: usize> Optimize for Add<Var<ID>, Const<0>> {
    type Optimized = Var<ID>;

    fn optimize(self) -> Var<ID> {
        self.left
    }
}

/// 0 + x → x optimization (only for Var to avoid conflicts)
impl<const ID: usize> Optimize for Add<Const<0>, Var<ID>> {
    type Optimized = Var<ID>;

    fn optimize(self) -> Var<ID> {
        self.right
    }
}

/// x * 1 → x optimization (only for Var to avoid conflicts)
impl<const ID: usize> Optimize for Mul<Var<ID>, Const<4607182418800017408>> {
    // 1.0 in bits
    type Optimized = Var<ID>;

    fn optimize(self) -> Var<ID> {
        self.left
    }
}

/// 1 * x → x optimization (only for Var to avoid conflicts)
impl<const ID: usize> Optimize for Mul<Const<4607182418800017408>, Var<ID>> {
    // 1.0 in bits
    type Optimized = Var<ID>;

    fn optimize(self) -> Var<ID> {
        self.right
    }
}

/// x * 0 → 0 optimization (only for Var to avoid conflicts)
impl<const ID: usize> Optimize for Mul<Var<ID>, Const<0>> {
    type Optimized = Const<0>;

    fn optimize(self) -> Const<0> {
        Const
    }
}

/// 0 * x → 0 optimization (only for Var to avoid conflicts)
impl<const ID: usize> Optimize for Mul<Const<0>, Var<ID>> {
    type Optimized = Const<0>;

    fn optimize(self) -> Const<0> {
        Const
    }
}

/// ln(a * b) → ln(a) + ln(b) optimization
impl<A: MathExpr, B: MathExpr> Optimize for Ln<Mul<A, B>> {
    type Optimized = Add<Ln<A>, Ln<B>>;

    fn optimize(self) -> Add<Ln<A>, Ln<B>> {
        Add {
            left: Ln {
                inner: self.inner.left,
            },
            right: Ln {
                inner: self.inner.right,
            },
        }
    }
}

/// exp(a + b) → exp(a) * exp(b) optimization
impl<A: MathExpr, B: MathExpr> Optimize for Exp<Add<A, B>> {
    type Optimized = Mul<Exp<A>, Exp<B>>;

    fn optimize(self) -> Mul<Exp<A>, Exp<B>> {
        Mul {
            left: Exp {
                inner: self.inner.left,
            },
            right: Exp {
                inner: self.inner.right,
            },
        }
    }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Create a variable reference
#[must_use]
pub const fn var<const ID: usize>() -> Var<ID> {
    Var
}

/// Create a constant (helper function to avoid bit manipulation)
#[must_use]
pub fn constant(value: f64) -> impl MathExpr + optimized::ToAst {
    // We need to return different types for different values
    // This is a workaround for the const generic limitation
    ConstantValue { value }
}

/// Runtime constant that can hold any f64 value
#[derive(Clone, Debug)]
pub struct ConstantValue {
    value: f64,
}

impl MathExpr for ConstantValue {
    fn eval(&self, _vars: &[f64]) -> f64 {
        self.value
    }
}

impl optimized::ToAst for ConstantValue {
    fn to_ast(&self) -> crate::ast::ASTRepr<f64> {
        crate::ast::ASTRepr::Constant(self.value)
    }
}

/// Zero constant
#[must_use]
pub const fn zero() -> Const<0> {
    Const
}

/// One constant (1.0 in f64 bit representation)
#[must_use]
pub const fn one() -> Const<4607182418800017408> {
    Const
}

// Re-export the procedural macro for true compile-time optimization
pub use dslcompile_macros::optimize_compile_time;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_evaluation() {
        let x = var::<0>();
        let y = var::<1>();

        let expr = x.clone().add(y.clone());
        let result = expr.eval(&[2.0, 3.0]);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_complex_expression() {
        let x = var::<0>();
        let y = var::<1>();

        let expr = x.clone().mul(y.clone()).add(x.clone());
        let result = expr.eval(&[2.0, 3.0]);
        assert_eq!(result, 8.0); // 2*3 + 2 = 8
    }

    #[test]
    fn test_transcendental_functions() {
        let x = var::<0>();

        let expr = x.clone().exp().ln();
        let result = expr.eval(&[2.0]);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_exp_optimization() {
        let x = var::<0>();

        // ln(exp(x)) should optimize to x
        let original = x.clone().exp().ln();
        let optimized = original.clone().optimize();

        let original_result = original.eval(&[2.0]);
        let optimized_result = optimized.eval(&[2.0]);

        assert!((original_result - optimized_result).abs() < 1e-10);
        assert!((optimized_result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_addition_optimization() {
        let x = var::<0>();
        let zero_const = zero();

        // x + 0 should optimize to x
        let original = x.clone().add(zero_const);
        let optimized = original.clone().optimize();

        let original_result = original.eval(&[5.0]);
        let optimized_result = optimized.eval(&[5.0]);

        assert_eq!(original_result, optimized_result);
        assert_eq!(optimized_result, 5.0);
    }
}
