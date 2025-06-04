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
//! // Define f(x) = x² in scope 0
//! let x = scoped_var::<0, 0>();
//! let f = x.clone().mul(x);
//!
//! // Define g(y) = 2y in scope 1 (no collision!)
//! let y = scoped_var::<0, 1>();
//! let g = y.mul(scoped_constant::<1>(2.0));
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

// Re-export the scoped variables system (recommended)
pub use scoped::{
    ScopedConst, ScopedMathExpr, ScopedVar, ScopedVarArray, compose, scoped_constant, scoped_var,
};

// Re-export the procedural macro for compile-time optimization
pub use dslcompile_macros::optimize_compile_time;

// ============================================================================
// MINIMAL LEGACY COMPATIBILITY (for procedural macro only)
// ============================================================================
// These types are kept minimal and only for procedural macro parsing.
// Users should use scoped variables for new code.

/// Legacy trait for procedural macro compatibility - prefer ScopedMathExpr
pub trait MathExpr: Clone + Sized {
    fn eval(&self, vars: &[f64]) -> f64;
    fn add<T: MathExpr>(self, other: T) -> Add<Self, T> {
        Add { left: self, right: other }
    }
    fn mul<T: MathExpr>(self, other: T) -> Mul<Self, T> {
        Mul { left: self, right: other }
    }
    fn sub<T: MathExpr>(self, other: T) -> Sub<Self, T> {
        Sub { left: self, right: other }
    }
    fn div<T: MathExpr>(self, other: T) -> Div<Self, T> {
        Div { left: self, right: other }
    }
    fn pow<T: MathExpr>(self, exponent: T) -> Pow<Self, T> {
        Pow { base: self, exponent }
    }
    fn exp(self) -> Exp<Self> { Exp { inner: self } }
    fn ln(self) -> Ln<Self> { Ln { inner: self } }
    fn sin(self) -> Sin<Self> { Sin { inner: self } }
    fn cos(self) -> Cos<Self> { Cos { inner: self } }
    fn sqrt(self) -> Sqrt<Self> { Sqrt { inner: self } }
    fn neg(self) -> Neg<Self> { Neg { inner: self } }
}

/// Legacy variable for procedural macro - prefer scoped_var
#[derive(Clone, Debug)]
pub struct Var<const ID: usize>;

impl<const ID: usize> MathExpr for Var<ID> {
    fn eval(&self, vars: &[f64]) -> f64 {
        vars.get(ID).copied().unwrap_or(0.0)
    }
}

/// Legacy constant for procedural macro - prefer scoped_constant  
#[derive(Clone, Debug)]
pub struct Const<const BITS: u64>;

impl<const BITS: u64> Const<BITS> {
    pub fn value(&self) -> f64 { f64::from_bits(BITS) }
}

impl<const BITS: u64> MathExpr for Const<BITS> {
    fn eval(&self, _vars: &[f64]) -> f64 { self.value() }
}

/// Legacy runtime constant for procedural macro
#[derive(Clone, Debug)]
pub struct ConstantValue { value: f64 }

impl MathExpr for ConstantValue {
    fn eval(&self, _vars: &[f64]) -> f64 { self.value }
}

// Operation types for procedural macro
#[derive(Clone, Debug)]
pub struct Add<L: MathExpr, R: MathExpr> { left: L, right: R }
#[derive(Clone, Debug)]  
pub struct Mul<L: MathExpr, R: MathExpr> { left: L, right: R }
#[derive(Clone, Debug)]
pub struct Sub<L: MathExpr, R: MathExpr> { left: L, right: R }
#[derive(Clone, Debug)]
pub struct Div<L: MathExpr, R: MathExpr> { left: L, right: R }
#[derive(Clone, Debug)]
pub struct Pow<B: MathExpr, E: MathExpr> { base: B, exponent: E }
#[derive(Clone, Debug)]
pub struct Exp<T: MathExpr> { inner: T }
#[derive(Clone, Debug)]
pub struct Ln<T: MathExpr> { inner: T }
#[derive(Clone, Debug)]
pub struct Sin<T: MathExpr> { inner: T }
#[derive(Clone, Debug)]
pub struct Cos<T: MathExpr> { inner: T }
#[derive(Clone, Debug)]
pub struct Sqrt<T: MathExpr> { inner: T }
#[derive(Clone, Debug)]
pub struct Neg<T: MathExpr> { inner: T }

// Implementations for operation types
impl<L: MathExpr, R: MathExpr> MathExpr for Add<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 { self.left.eval(vars) + self.right.eval(vars) }
}
impl<L: MathExpr, R: MathExpr> MathExpr for Mul<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 { self.left.eval(vars) * self.right.eval(vars) }
}
impl<L: MathExpr, R: MathExpr> MathExpr for Sub<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 { self.left.eval(vars) - self.right.eval(vars) }
}
impl<L: MathExpr, R: MathExpr> MathExpr for Div<L, R> {
    fn eval(&self, vars: &[f64]) -> f64 { self.left.eval(vars) / self.right.eval(vars) }
}
impl<B: MathExpr, E: MathExpr> MathExpr for Pow<B, E> {
    fn eval(&self, vars: &[f64]) -> f64 { self.base.eval(vars).powf(self.exponent.eval(vars)) }
}
impl<T: MathExpr> MathExpr for Exp<T> {
    fn eval(&self, vars: &[f64]) -> f64 { self.inner.eval(vars).exp() }
}
impl<T: MathExpr> MathExpr for Ln<T> {
    fn eval(&self, vars: &[f64]) -> f64 { self.inner.eval(vars).ln() }
}
impl<T: MathExpr> MathExpr for Sin<T> {
    fn eval(&self, vars: &[f64]) -> f64 { self.inner.eval(vars).sin() }
}
impl<T: MathExpr> MathExpr for Cos<T> {
    fn eval(&self, vars: &[f64]) -> f64 { self.inner.eval(vars).cos() }
}
impl<T: MathExpr> MathExpr for Sqrt<T> {
    fn eval(&self, vars: &[f64]) -> f64 { self.inner.eval(vars).sqrt() }
}
impl<T: MathExpr> MathExpr for Neg<T> {
    fn eval(&self, vars: &[f64]) -> f64 { -self.inner.eval(vars) }
}

// Legacy convenience functions for procedural macro
pub const fn var<const ID: usize>() -> Var<ID> { Var }
pub fn constant(value: f64) -> impl MathExpr + optimized::ToAst {
    ConstantValue { value }
}

impl optimized::ToAst for ConstantValue {
    fn to_ast(&self) -> crate::ast::ASTRepr<f64> {
        crate::ast::ASTRepr::Constant(self.value)
    }
}
