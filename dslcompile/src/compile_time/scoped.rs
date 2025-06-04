//! Type-Level Scoped Variables for Compile-Time System
//!
//! This module implements type-level variable scoping that prevents variable collisions
//! at compile time while maintaining zero runtime overhead.
//!
//! The automatic scope builder system requires nightly Rust with
//! #![`feature(generic_const_exprs)`] for full ergonomic usage, but also provides
//! a stable Rust compatible API.

use crate::ast::ASTRepr;
use crate::ast::NumericType;
use num_traits::Float;
use std::marker::PhantomData;

// Import our type-level logic system
use super::type_level_logic::{False, TypeLevelBool};

/// Conditional trait: only implement if condition is False  
pub trait WhenFalse<Condition: TypeLevelBool> {}
impl WhenFalse<False> for () {}

// Note: We can't implement the False case directly due to Rust's limitations
// with const generics, but we can work around this with blanket impls

/// Scoped variable with compile-time scope and ID tracking
#[derive(Clone, Debug)]
pub struct ScopedVar<T, const ID: usize, const SCOPE: usize>(PhantomData<T>)
where
    T: NumericType;

/// Scoped constant with compile-time scope tracking
#[derive(Clone, Debug)]
pub struct ScopedConst<T, const BITS: u64, const SCOPE: usize>(PhantomData<T>)
where
    T: NumericType;

/// Core trait for scoped mathematical expressions
pub trait ScopedMathExpr<T, const SCOPE: usize>: Clone + Sized
where
    T: NumericType,
{
    /// Evaluate the expression with scoped variable values
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T;

    /// Convert to AST representation
    fn to_ast(&self) -> ASTRepr<T>;

    /// Add two expressions in the same scope
    fn add<U: ScopedMathExpr<T, SCOPE>>(self, other: U) -> ScopedAdd<T, Self, U, SCOPE> {
        ScopedAdd {
            left: self,
            right: other,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Multiply two expressions in the same scope
    fn mul<U: ScopedMathExpr<T, SCOPE>>(self, other: U) -> ScopedMul<T, Self, U, SCOPE> {
        ScopedMul {
            left: self,
            right: other,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Subtract two expressions in the same scope
    fn sub<U: ScopedMathExpr<T, SCOPE>>(self, other: U) -> ScopedSub<T, Self, U, SCOPE> {
        ScopedSub {
            left: self,
            right: other,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Divide two expressions in the same scope
    fn div<U: ScopedMathExpr<T, SCOPE>>(self, other: U) -> ScopedDiv<T, Self, U, SCOPE> {
        ScopedDiv {
            left: self,
            right: other,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Power function
    fn pow<U: ScopedMathExpr<T, SCOPE>>(self, exponent: U) -> ScopedPow<T, Self, U, SCOPE> {
        ScopedPow {
            base: self,
            exponent,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Natural exponential
    fn exp(self) -> ScopedExp<T, Self, SCOPE> {
        ScopedExp {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Natural logarithm
    fn ln(self) -> ScopedLn<T, Self, SCOPE> {
        ScopedLn {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Sine function
    fn sin(self) -> ScopedSin<T, Self, SCOPE> {
        ScopedSin {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Cosine function
    fn cos(self) -> ScopedCos<T, Self, SCOPE> {
        ScopedCos {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Square root
    fn sqrt(self) -> ScopedSqrt<T, Self, SCOPE> {
        ScopedSqrt {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Negation
    fn neg(self) -> ScopedNeg<T, Self, SCOPE> {
        ScopedNeg {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }
}

/// Trait for composing expressions across different scopes
pub trait ScopeCompose<T, Other, const OTHER_SCOPE: usize>: Sized
where
    T: NumericType,
{
    type Output;

    /// Compose expressions from different scopes with automatic variable remapping
    fn compose_with<F>(self, other: Other, combiner: F) -> Self::Output
    where
        F: FnOnce(Self, Other) -> Self::Output;
}

/// Variable array for a specific scope
pub struct ScopedVarArray<T, const SCOPE: usize>
where
    T: NumericType,
{
    vars: Vec<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedVarArray<T, SCOPE>
where
    T: NumericType,
{
    /// Create a new scoped variable array
    #[must_use]
    pub fn new(vars: Vec<T>) -> Self {
        Self {
            vars,
            _scope: PhantomData,
        }
    }

    /// Get variable value by ID
    #[must_use]
    pub fn get(&self, id: usize) -> T
    where
        T: Default + Copy,
    {
        self.vars.get(id).copied().unwrap_or_default()
    }
}

// ============================================================================
// VARIABLE AND CONSTANT IMPLEMENTATIONS
// ============================================================================

impl<T, const ID: usize, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedVar<T, ID, SCOPE>
where
    T: NumericType + Default + Copy,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        vars.get(ID)
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Variable(ID)
    }
}

impl<T, const BITS: u64, const SCOPE: usize> ScopedMathExpr<T, SCOPE>
    for ScopedConst<T, BITS, SCOPE>
where
    T: NumericType + Copy,
{
    fn eval(&self, _vars: &ScopedVarArray<T, SCOPE>) -> T {
        // For now, we'll need a way to convert from bits representation
        // This is a limitation we'll need to address - BITS encoding is f64-specific
        // TODO: Make this generic properly
        unsafe { std::mem::transmute_copy(&BITS) }
    }

    fn to_ast(&self) -> ASTRepr<T> {
        // Same issue here - we need a better way to handle constants
        // For now, using unsafe transmute as placeholder
        ASTRepr::Constant(unsafe { std::mem::transmute_copy(&BITS) })
    }
}

// ============================================================================
// OPERATION IMPLEMENTATIONS
// ============================================================================

#[derive(Clone, Debug)]
pub struct ScopedAdd<T, L, R, const SCOPE: usize>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedAdd<T, L, R, SCOPE>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.left.eval(vars) + self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Add(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedMul<T, L, R, const SCOPE: usize>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedMul<T, L, R, SCOPE>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.left.eval(vars) * self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Mul(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedSub<T, L, R, const SCOPE: usize>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedSub<T, L, R, SCOPE>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.left.eval(vars) - self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Sub(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedDiv<T, L, R, const SCOPE: usize>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedDiv<T, L, R, SCOPE>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.left.eval(vars) / self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Div(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedPow<T, B, E, const SCOPE: usize>
where
    T: NumericType,
    B: ScopedMathExpr<T, SCOPE>,
    E: ScopedMathExpr<T, SCOPE>,
{
    base: B,
    exponent: E,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, B, E, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedPow<T, B, E, SCOPE>
where
    T: NumericType + Float,
    B: ScopedMathExpr<T, SCOPE>,
    E: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.base.eval(vars).powf(self.exponent.eval(vars))
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Pow(
            Box::new(self.base.to_ast()),
            Box::new(self.exponent.to_ast()),
        )
    }
}

// ============================================================================
// TRANSCENDENTAL FUNCTIONS
// ============================================================================

#[derive(Clone, Debug)]
pub struct ScopedExp<T, B, const SCOPE: usize>
where
    T: NumericType,
    B: ScopedMathExpr<T, SCOPE>,
{
    inner: B,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, B, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedExp<T, B, SCOPE>
where
    T: NumericType + Float,
    B: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.inner.eval(vars).exp()
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Exp(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedLn<T, B, const SCOPE: usize>
where
    T: NumericType,
    B: ScopedMathExpr<T, SCOPE>,
{
    inner: B,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, B, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedLn<T, B, SCOPE>
where
    T: NumericType + Float,
    B: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.inner.eval(vars).ln()
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Ln(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedSin<T, B, const SCOPE: usize>
where
    T: NumericType,
    B: ScopedMathExpr<T, SCOPE>,
{
    inner: B,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, B, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedSin<T, B, SCOPE>
where
    T: NumericType + Float,
    B: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.inner.eval(vars).sin()
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Sin(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedCos<T, B, const SCOPE: usize>
where
    T: NumericType,
    B: ScopedMathExpr<T, SCOPE>,
{
    inner: B,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, B, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedCos<T, B, SCOPE>
where
    T: NumericType + Float,
    B: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.inner.eval(vars).cos()
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Cos(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedSqrt<T, B, const SCOPE: usize>
where
    T: NumericType,
    B: ScopedMathExpr<T, SCOPE>,
{
    inner: B,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, B, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedSqrt<T, B, SCOPE>
where
    T: NumericType + Float,
    B: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.inner.eval(vars).sqrt()
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Sqrt(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedNeg<T, B, const SCOPE: usize>
where
    T: NumericType,
    B: ScopedMathExpr<T, SCOPE>,
{
    inner: B,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, B, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedNeg<T, B, SCOPE>
where
    T: NumericType + std::ops::Neg<Output = T>,
    B: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        -self.inner.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Neg(Box::new(self.inner.to_ast()))
    }
}

// ============================================================================
// SCOPE COMPOSITION IMPLEMENTATIONS
// ============================================================================

/// Composed expression from two different scopes
#[derive(Clone, Debug)]
pub struct ComposedExpr<T, L, R, const SCOPE1: usize, const SCOPE2: usize>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE1>,
    R: ScopedMathExpr<T, SCOPE2>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope1: PhantomData<[(); SCOPE1]>,
    _scope2: PhantomData<[(); SCOPE2]>,
}

impl<T, L, R, const SCOPE1: usize, const SCOPE2: usize> ComposedExpr<T, L, R, SCOPE1, SCOPE2>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE1>,
    R: ScopedMathExpr<T, SCOPE2>,
{
    /// Create a new composed expression
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
            _type: PhantomData,
            _scope1: PhantomData,
            _scope2: PhantomData,
        }
    }

    /// Evaluate with variables from both scopes
    pub fn eval(
        &self,
        vars1: &ScopedVarArray<T, SCOPE1>,
        vars2: &ScopedVarArray<T, SCOPE2>,
    ) -> (T, T) {
        (self.left.eval(vars1), self.right.eval(vars2))
    }

    /// Add the two scoped expressions (returns a composed expression with combined scope)
    pub fn add(self) -> ComposedAdd<T>
    where
        T: Copy,
    {
        let left_ast = self.left.to_ast();
        // Count variables in left AST to determine proper offset
        let max_left_var = find_max_variable_index(&left_ast);
        let offset = max_left_var + 1;

        // Remap right AST variables to avoid collision
        let right_ast = remap_ast_variables(&self.right.to_ast(), offset);

        ComposedAdd {
            left_ast,
            right_ast,
        }
    }

    /// Multiply the two scoped expressions (returns a composed expression with combined scope)
    pub fn mul(self) -> ComposedMul<T>
    where
        T: Copy,
    {
        let left_ast = self.left.to_ast();
        // Count variables in left AST to determine proper offset
        let max_left_var = find_max_variable_index(&left_ast);
        let offset = max_left_var + 1;

        // Remap right AST variables to avoid collision
        let right_ast = remap_ast_variables(&self.right.to_ast(), offset);

        ComposedMul {
            left_ast,
            right_ast,
        }
    }
}

/// Helper struct for composed addition
#[derive(Clone, Debug)]
pub struct ComposedAdd<T>
where
    T: NumericType,
{
    left_ast: ASTRepr<T>,
    right_ast: ASTRepr<T>,
}

impl<T> ComposedAdd<T>
where
    T: NumericType + Copy + std::ops::Add<Output = T> + Float,
{
    #[must_use]
    pub fn eval(&self, vars: &[T]) -> T {
        let left_val = eval_ast(&self.left_ast, vars);
        let right_val = eval_ast(&self.right_ast, vars);
        left_val + right_val
    }

    #[must_use]
    pub fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Add(
            Box::new(self.left_ast.clone()),
            Box::new(self.right_ast.clone()),
        )
    }
}

/// Helper struct for composed multiplication
#[derive(Clone, Debug)]
pub struct ComposedMul<T>
where
    T: NumericType,
{
    left_ast: ASTRepr<T>,
    right_ast: ASTRepr<T>,
}

impl<T> ComposedMul<T>
where
    T: NumericType + Copy + std::ops::Mul<Output = T> + Float,
{
    #[must_use]
    pub fn eval(&self, vars: &[T]) -> T {
        let left_val = eval_ast(&self.left_ast, vars);
        let right_val = eval_ast(&self.right_ast, vars);
        left_val * right_val
    }

    #[must_use]
    pub fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Mul(
            Box::new(self.left_ast.clone()),
            Box::new(self.right_ast.clone()),
        )
    }
}

// ============================================================================
// HELPER FUNCTIONS FOR COMPOSITION
// ============================================================================

/// Compose two expressions from different scopes
pub fn compose<T, L, R, const SCOPE1: usize, const SCOPE2: usize>(
    left: L,
    right: R,
) -> ComposedExpr<T, L, R, SCOPE1, SCOPE2>
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE1>,
    R: ScopedMathExpr<T, SCOPE2>,
{
    ComposedExpr::new(left, right)
}

// ============================================================================
// INTERNAL HELPER FUNCTIONS
// ============================================================================

/// Find the maximum variable index in an AST
fn find_max_variable_index<T: NumericType>(ast: &ASTRepr<T>) -> usize {
    match ast {
        ASTRepr::Constant(_) => 0,
        ASTRepr::Variable(idx) => *idx,
        ASTRepr::Add(left, right) => {
            let left_max = find_max_variable_index(left);
            let right_max = find_max_variable_index(right);
            left_max.max(right_max)
        }
        ASTRepr::Sub(left, right) => {
            let left_max = find_max_variable_index(left);
            let right_max = find_max_variable_index(right);
            left_max.max(right_max)
        }
        ASTRepr::Mul(left, right) => {
            let left_max = find_max_variable_index(left);
            let right_max = find_max_variable_index(right);
            left_max.max(right_max)
        }
        ASTRepr::Div(left, right) => {
            let left_max = find_max_variable_index(left);
            let right_max = find_max_variable_index(right);
            left_max.max(right_max)
        }
        ASTRepr::Pow(base, exp) => {
            let base_max = find_max_variable_index(base);
            let exp_max = find_max_variable_index(exp);
            base_max.max(exp_max)
        }
        ASTRepr::Neg(inner) => find_max_variable_index(inner),
        ASTRepr::Ln(inner) => find_max_variable_index(inner),
        ASTRepr::Exp(inner) => find_max_variable_index(inner),
        ASTRepr::Sin(inner) => find_max_variable_index(inner),
        ASTRepr::Cos(inner) => find_max_variable_index(inner),
        ASTRepr::Sqrt(inner) => find_max_variable_index(inner),
    }
}

/// Remap AST variables by adding an offset
fn remap_ast_variables<T: NumericType>(ast: &ASTRepr<T>, offset: usize) -> ASTRepr<T>
where
    T: Clone,
{
    match ast {
        ASTRepr::Constant(val) => ASTRepr::Constant(val.clone()),
        ASTRepr::Variable(idx) => ASTRepr::Variable(idx + offset),
        ASTRepr::Add(left, right) => ASTRepr::Add(
            Box::new(remap_ast_variables(left, offset)),
            Box::new(remap_ast_variables(right, offset)),
        ),
        ASTRepr::Sub(left, right) => ASTRepr::Sub(
            Box::new(remap_ast_variables(left, offset)),
            Box::new(remap_ast_variables(right, offset)),
        ),
        ASTRepr::Mul(left, right) => ASTRepr::Mul(
            Box::new(remap_ast_variables(left, offset)),
            Box::new(remap_ast_variables(right, offset)),
        ),
        ASTRepr::Div(left, right) => ASTRepr::Div(
            Box::new(remap_ast_variables(left, offset)),
            Box::new(remap_ast_variables(right, offset)),
        ),
        ASTRepr::Pow(left, right) => ASTRepr::Pow(
            Box::new(remap_ast_variables(left, offset)),
            Box::new(remap_ast_variables(right, offset)),
        ),
        ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(remap_ast_variables(inner, offset))),
        ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(remap_ast_variables(inner, offset))),
        ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(remap_ast_variables(inner, offset))),
        ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(remap_ast_variables(inner, offset))),
        ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(remap_ast_variables(inner, offset))),
        ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(remap_ast_variables(inner, offset))),
    }
}

/// Simple AST evaluator
fn eval_ast<T: NumericType + Copy>(ast: &ASTRepr<T>, vars: &[T]) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + Float,
{
    match ast {
        ASTRepr::Constant(val) => *val,
        ASTRepr::Variable(idx) => vars.get(*idx).copied().unwrap_or_else(T::zero),
        ASTRepr::Add(left, right) => eval_ast(left, vars) + eval_ast(right, vars),
        ASTRepr::Sub(left, right) => eval_ast(left, vars) - eval_ast(right, vars),
        ASTRepr::Mul(left, right) => eval_ast(left, vars) * eval_ast(right, vars),
        ASTRepr::Div(left, right) => eval_ast(left, vars) / eval_ast(right, vars),
        ASTRepr::Pow(base, exp) => eval_ast(base, vars).powf(eval_ast(exp, vars)),
        ASTRepr::Neg(inner) => -eval_ast(inner, vars),
        ASTRepr::Ln(inner) => eval_ast(inner, vars).ln(),
        ASTRepr::Exp(inner) => eval_ast(inner, vars).exp(),
        ASTRepr::Sin(inner) => eval_ast(inner, vars).sin(),
        ASTRepr::Cos(inner) => eval_ast(inner, vars).cos(),
        ASTRepr::Sqrt(inner) => eval_ast(inner, vars).sqrt(),
    }
}

// ===============================
// AUTOMATIC SCOPE BUILDER SYSTEM (NIGHTLY ONLY)
// ===============================

// NOTE: This section requires nightly Rust with #![feature(generic_const_exprs)] for ergonomic scope builders.

/// Type-level scope builder that creates variables and tracks their IDs
#[derive(Clone, Debug)]
pub struct ScopeBuilder<T, const SCOPE: usize, const NEXT_ID: usize>
where
    T: NumericType,
{
    _type: PhantomData<T>,
}

impl<T, const SCOPE: usize, const NEXT_ID: usize> ScopeBuilder<T, SCOPE, NEXT_ID>
where
    T: NumericType,
{
    /// Create a new variable in this scope and return the updated builder
    #[must_use]
    pub fn auto_var(
        self,
    ) -> (
        ScopedVar<T, NEXT_ID, SCOPE>,
        ScopeBuilder<T, SCOPE, { NEXT_ID + 1 }>,
    ) {
        (ScopedVar(PhantomData), ScopeBuilder { _type: PhantomData })
    }

    /// Create a constant in this scope
    #[must_use]
    pub fn constant(self, value: T) -> ScopedConstValue<T, SCOPE> {
        ScopedConstValue {
            value,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }
}

/// Top-level builder for managing unique scopes
#[derive(Clone, Debug, Default)]
pub struct Context<T, const NEXT_SCOPE: usize>
where
    T: NumericType,
{
    _type: PhantomData<T>,
}

impl<T> Context<T, 0>
where
    T: NumericType,
{
    /// Create a new builder (starts at scope 0)
    #[must_use]
    pub fn new() -> Self {
        Self { _type: PhantomData }
    }
}

impl<T, const NEXT_SCOPE: usize> Context<T, NEXT_SCOPE>
where
    T: NumericType,
{
    /// Create a new scope, passing a fresh `ScopeBuilder` to the closure
    pub fn new_scope<F, R>(&mut self, f: F) -> R
    where
        F: for<'a> FnOnce(ScopeBuilder<T, NEXT_SCOPE, 0>) -> R,
    {
        f(ScopeBuilder { _type: PhantomData })
    }

    /// Advance to the next scope
    #[must_use]
    pub fn next(self) -> Context<T, { NEXT_SCOPE + 1 }> {
        Context { _type: PhantomData }
    }
}

/// Runtime constant that can hold any numeric value in a specific scope
#[derive(Clone, Debug)]
pub struct ScopedConstValue<T, const SCOPE: usize>
where
    T: NumericType,
{
    value: T,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedConstValue<T, SCOPE>
where
    T: NumericType + Copy,
{
    fn eval(&self, _vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.value
    }

    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Constant(self.value)
    }
}

// Add a convenience function for creating f64 builders
impl Context<f64, 0> {
    /// Create a new f64 builder (convenience function)
    #[must_use]
    pub fn new_f64() -> Self {
        Self::new()
    }
}

// ============================================================================
// OPERATOR OVERLOADING - PHASE 1: API UNIFICATION COMPLETE
// ============================================================================

// Single unified implementation that handles both same-ID and different-ID cases
// using type-level dispatch - no trait coherence conflicts!

impl<T, const ID1: usize, const ID2: usize, const SCOPE: usize>
    std::ops::Add<ScopedVar<T, ID2, SCOPE>> for ScopedVar<T, ID1, SCOPE>
where
    T: NumericType + std::ops::Add<Output = T> + Default + Copy,
{
    type Output = ScopedAdd<T, Self, ScopedVar<T, ID2, SCOPE>, SCOPE>;

    fn add(self, rhs: ScopedVar<T, ID2, SCOPE>) -> Self::Output {
        ScopedMathExpr::add(self, rhs)
    }
}

impl<T, const ID1: usize, const ID2: usize, const SCOPE: usize>
    std::ops::Mul<ScopedVar<T, ID2, SCOPE>> for ScopedVar<T, ID1, SCOPE>
where
    T: NumericType + std::ops::Mul<Output = T> + Default + Copy,
{
    type Output = ScopedMul<T, Self, ScopedVar<T, ID2, SCOPE>, SCOPE>;

    fn mul(self, rhs: ScopedVar<T, ID2, SCOPE>) -> Self::Output {
        ScopedMathExpr::mul(self, rhs)
    }
}

impl<T, const ID1: usize, const ID2: usize, const SCOPE: usize>
    std::ops::Sub<ScopedVar<T, ID2, SCOPE>> for ScopedVar<T, ID1, SCOPE>
where
    T: NumericType + std::ops::Sub<Output = T> + Default + Copy,
{
    type Output = ScopedSub<T, Self, ScopedVar<T, ID2, SCOPE>, SCOPE>;

    fn sub(self, rhs: ScopedVar<T, ID2, SCOPE>) -> Self::Output {
        ScopedMathExpr::sub(self, rhs)
    }
}

impl<T, const ID1: usize, const ID2: usize, const SCOPE: usize>
    std::ops::Div<ScopedVar<T, ID2, SCOPE>> for ScopedVar<T, ID1, SCOPE>
where
    T: NumericType + std::ops::Div<Output = T> + Default + Copy,
{
    type Output = ScopedDiv<T, Self, ScopedVar<T, ID2, SCOPE>, SCOPE>;

    fn div(self, rhs: ScopedVar<T, ID2, SCOPE>) -> Self::Output {
        ScopedMathExpr::div(self, rhs)
    }
}

// Unary negation for variables
impl<T, const ID: usize, const SCOPE: usize> std::ops::Neg for ScopedVar<T, ID, SCOPE>
where
    T: NumericType + std::ops::Neg<Output = T> + Default + Copy,
{
    type Output = ScopedNeg<T, Self, SCOPE>;

    fn neg(self) -> Self::Output {
        ScopedMathExpr::neg(self)
    }
}

// ============================================================================
// CONSTANT AND CROSS-TYPE OPERATIONS
// ============================================================================

// Operator overloading for ScopedConstValue
impl<T, const SCOPE: usize> std::ops::Add for ScopedConstValue<T, SCOPE>
where
    T: NumericType + std::ops::Add<Output = T> + Copy,
{
    type Output = ScopedAdd<T, Self, Self, SCOPE>;

    fn add(self, rhs: Self) -> Self::Output {
        ScopedMathExpr::add(self, rhs)
    }
}

impl<T, const SCOPE: usize> std::ops::Mul for ScopedConstValue<T, SCOPE>
where
    T: NumericType + std::ops::Mul<Output = T> + Copy,
{
    type Output = ScopedMul<T, Self, Self, SCOPE>;

    fn mul(self, rhs: Self) -> Self::Output {
        ScopedMathExpr::mul(self, rhs)
    }
}

impl<T, const SCOPE: usize> std::ops::Sub for ScopedConstValue<T, SCOPE>
where
    T: NumericType + std::ops::Sub<Output = T> + Copy,
{
    type Output = ScopedSub<T, Self, Self, SCOPE>;

    fn sub(self, rhs: Self) -> Self::Output {
        ScopedMathExpr::sub(self, rhs)
    }
}

impl<T, const SCOPE: usize> std::ops::Div for ScopedConstValue<T, SCOPE>
where
    T: NumericType + std::ops::Div<Output = T> + Copy,
{
    type Output = ScopedDiv<T, Self, Self, SCOPE>;

    fn div(self, rhs: Self) -> Self::Output {
        ScopedMathExpr::div(self, rhs)
    }
}

impl<T, const SCOPE: usize> std::ops::Neg for ScopedConstValue<T, SCOPE>
where
    T: NumericType + std::ops::Neg<Output = T> + Copy,
{
    type Output = ScopedNeg<T, Self, SCOPE>;

    fn neg(self) -> Self::Output {
        ScopedMathExpr::neg(self)
    }
}

// Cross-type operator overloading: Variable + Constant
impl<T, const ID: usize, const SCOPE: usize> std::ops::Add<ScopedConstValue<T, SCOPE>>
    for ScopedVar<T, ID, SCOPE>
where
    T: NumericType + std::ops::Add<Output = T> + Default + Copy,
{
    type Output = ScopedAdd<T, Self, ScopedConstValue<T, SCOPE>, SCOPE>;

    fn add(self, rhs: ScopedConstValue<T, SCOPE>) -> Self::Output {
        ScopedMathExpr::add(self, rhs)
    }
}

impl<T, const ID: usize, const SCOPE: usize> std::ops::Add<ScopedVar<T, ID, SCOPE>>
    for ScopedConstValue<T, SCOPE>
where
    T: NumericType + std::ops::Add<Output = T> + Default + Copy,
{
    type Output = ScopedAdd<T, Self, ScopedVar<T, ID, SCOPE>, SCOPE>;

    fn add(self, rhs: ScopedVar<T, ID, SCOPE>) -> Self::Output {
        ScopedMathExpr::add(self, rhs)
    }
}

impl<T, const ID: usize, const SCOPE: usize> std::ops::Mul<ScopedConstValue<T, SCOPE>>
    for ScopedVar<T, ID, SCOPE>
where
    T: NumericType + std::ops::Mul<Output = T> + Default + Copy,
{
    type Output = ScopedMul<T, Self, ScopedConstValue<T, SCOPE>, SCOPE>;

    fn mul(self, rhs: ScopedConstValue<T, SCOPE>) -> Self::Output {
        ScopedMathExpr::mul(self, rhs)
    }
}

impl<T, const ID: usize, const SCOPE: usize> std::ops::Mul<ScopedVar<T, ID, SCOPE>>
    for ScopedConstValue<T, SCOPE>
where
    T: NumericType + std::ops::Mul<Output = T> + Default + Copy,
{
    type Output = ScopedMul<T, Self, ScopedVar<T, ID, SCOPE>, SCOPE>;

    fn mul(self, rhs: ScopedVar<T, ID, SCOPE>) -> Self::Output {
        ScopedMathExpr::mul(self, rhs)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automatic_scoped_variables_no_collision() {
        let mut builder = Context::new_f64();

        // Define f(x) = 2x in scope 0
        let f = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            x.mul(scope.constant(2.0))
        });

        // Advance to next scope
        let mut builder = builder.next();

        // Define g(y) = 3y in scope 1 - no collision!
        let g = builder.new_scope(|scope| {
            let (y, scope) = scope.auto_var();
            y.mul(scope.constant(3.0))
        });

        // Evaluate independently
        let f_vars = ScopedVarArray::<f64, 0>::new(vec![4.0]);
        let g_vars = ScopedVarArray::<f64, 1>::new(vec![5.0]);

        assert_eq!(f.eval(&f_vars), 8.0); // 2 * 4 = 8
        assert_eq!(g.eval(&g_vars), 15.0); // 3 * 5 = 15
    }

    #[test]
    fn test_scope_composition() {
        let mut builder = Context::new_f64();

        // Define f(x) = x² in scope 0
        let f = builder.new_scope(|scope| {
            let (x, _scope) = scope.auto_var();
            x.clone().mul(x)
        });

        // Advance to next scope
        let mut builder = builder.next();

        // Define g(y) = 2y in scope 1
        let g = builder.new_scope(|scope| {
            let (y, scope) = scope.auto_var();
            y.mul(scope.constant(2.0))
        });

        // Compose h = f + g
        let composed = compose(f, g);
        let h = composed.add();

        // Evaluate h(3, 4) = f(3) + g(4) = 9 + 8 = 17
        let vars = vec![3.0, 4.0]; // Combined variable array
        assert_eq!(h.eval(&vars), 17.0);
    }

    #[test]
    fn test_complex_scoped_expression() {
        let mut builder = Context::new_f64();

        // Build sin(x) + cos(y) in scope 0
        let expr = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, _scope) = scope.auto_var();
            x.sin().add(y.cos())
        });

        let vars = ScopedVarArray::<f64, 0>::new(vec![std::f64::consts::PI / 2.0, 0.0]);
        let result = expr.eval(&vars);

        // sin(π/2) + cos(0) = 1 + 1 = 2
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ast_conversion() {
        let mut builder = Context::new_f64();

        // Build x + y in scope 0
        let expr = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, _scope) = scope.auto_var();
            x.add(y)
        });

        let ast = expr.to_ast();

        // Should create Add(Variable(0), Variable(1))
        match ast {
            ASTRepr::Add(left, right) => {
                assert!(matches!(*left, ASTRepr::Variable(0)));
                assert!(matches!(*right, ASTRepr::Variable(1)));
            }
            _ => panic!("Expected Add expression"),
        }
    }

    #[test]
    fn test_complex_composition_variable_remapping() {
        // Test the specific bug that was fixed: ensuring proper variable offset calculation
        let mut builder = Context::new_f64();

        // Define quadratic(x,y) = x² + xy + y² in scope 0 (uses variables 0, 1)
        let quadratic = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, _scope) = scope.auto_var();
            x.clone()
                .mul(x.clone())
                .add(x.mul(y.clone()))
                .add(y.clone().mul(y))
        });

        // Advance to next scope
        let mut builder = builder.next();

        // Define linear(a,b) = 2a + 3b in scope 1 (uses variables 0, 1)
        let linear = builder.new_scope(|scope| {
            let (a, scope) = scope.auto_var();
            let (b, scope) = scope.auto_var();
            a.mul(scope.clone().constant(2.0))
                .add(b.mul(scope.constant(3.0)))
        });

        // Test individual evaluations
        let quad_vars = ScopedVarArray::<f64, 0>::new(vec![1.0, 2.0]);
        let quad_result = quadratic.eval(&quad_vars); // 1² + 1*2 + 2² = 7
        assert_eq!(quad_result, 7.0);

        let lin_vars = ScopedVarArray::<f64, 1>::new(vec![3.0, 4.0]);
        let lin_result = linear.eval(&lin_vars); // 2*3 + 3*4 = 18
        assert_eq!(lin_result, 18.0);

        // Compose and test: this was the failing case before the fix
        let composed = compose(quadratic, linear);
        let combined = composed.add();

        // Test with combined variable array [x, y, a, b] = [1, 2, 3, 4]
        // Should evaluate to quadratic(1,2) + linear(3,4) = 7 + 18 = 25
        let test_values = [1.0, 2.0, 3.0, 4.0];
        let result = combined.eval(&test_values);

        assert_eq!(
            result, 25.0,
            "Variable remapping should correctly map linear variables to indices [2,3]"
        );
    }

    #[test]
    fn test_variable_offset_calculation() {
        // Test the find_max_variable_index function works correctly
        let mut builder = Context::new_f64();

        // Single variable expression: x (var 0)
        let expr1 = builder.new_scope(|scope| {
            let (x, _scope) = scope.auto_var();
            x
        });
        assert_eq!(find_max_variable_index(&expr1.to_ast()), 0);

        // Advance to next scope for clean test
        let mut builder = builder.next();

        // Two variable expression: x + y (vars 0, 1)
        let expr2 = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, _scope) = scope.auto_var();
            x.add(y)
        });
        assert_eq!(find_max_variable_index(&expr2.to_ast()), 1);

        // Advance to next scope for clean test
        let mut builder = builder.next();

        // Complex expression: x² + xy + y² (vars 0, 1)
        let expr3 = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, _scope) = scope.auto_var();
            x.clone()
                .mul(x.clone())
                .add(x.mul(y.clone()))
                .add(y.clone().mul(y))
        });
        assert_eq!(find_max_variable_index(&expr3.to_ast()), 1);

        // Advance to next scope for clean test
        let mut builder = builder.next();

        // Test constant expression (no variables)
        let constant_expr = builder.new_scope(|scope| scope.constant(5.0));
        assert_eq!(find_max_variable_index(&constant_expr.to_ast()), 0);
    }

    #[cfg(feature = "nightly-tests")] // Enable only for nightly testing
    #[test]
    fn test_ergonomic_scope_builder() {
        // This test demonstrates the ergonomic API that works on nightly Rust
        // with #![feature(generic_const_exprs)]

        // Create a builder and first scope
        let mut builder = Context::new_f64();

        let part1 = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var(); // Auto ID assignment!
            let (y, scope) = scope.auto_var(); // Auto ID assignment!
            x.mul(y).add(scope.constant(1.0))
        });

        // Advance to next scope
        let mut builder = builder.next();

        let part2 = builder.new_scope(|scope| {
            let (z, _scope) = scope.auto_var(); // Auto ID assignment!
            z.mul(scope.constant(2.0))
        });

        // Test composition
        let composed = compose(part1, part2);
        let combined = composed.add();

        let result = combined.eval(&[3.0, 4.0, 5.0]);
        // part1: x*y + 1 = 3*4 + 1 = 13
        // part2: z*2 = 5*2 = 10
        // combined: 13 + 10 = 23
        assert_eq!(result, 23.0);
    }

    #[test]
    fn test_operator_overloading_phase1() {
        // Test Phase 1: Operator overloading for compile-time API
        // Note: Currently supports basic operations on variables and constants only
        let mut builder = Context::new_f64();

        // Test that operator syntax works for basic operations
        let expr = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, scope) = scope.auto_var();
            let c = scope.constant(2.0);

            // Basic operator syntax: x + c, then multiply with y
            (x + c).mul(y) // Mix operators and methods for complex expressions
        });

        let vars = ScopedVarArray::<f64, 0>::new(vec![3.0, 4.0]);
        let result = expr.eval(&vars);

        // (x + c) * y = (3 + 2) * 4 = 5 * 4 = 20
        assert_eq!(result, 20.0);
    }

    #[test]
    fn test_operator_overloading_comprehensive() {
        // Test basic operators: +, -, *, /, - on variables and constants
        let mut builder = Context::new_f64();

        let expr = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, scope) = scope.auto_var();
            let c1 = scope.clone().constant(2.0);
            let c2 = scope.clone().constant(3.0);

            // Test basic constant operations (these work fine)
            let prod = c1 * c2; // Constant * Constant ✅

            // Test variable + constant (this works)
            let var_const = x + scope.constant(4.0);

            // Combine using method syntax for complex expressions
            var_const.add(prod).add(y) // Mix operators and methods
        });

        let vars = ScopedVarArray::<f64, 0>::new(vec![4.0, 5.0]);
        let result = expr.eval(&vars);

        // var_const = x + 4 = 4 + 4 = 8
        // prod = c1 * c2 = 2 * 3 = 6
        // result = var_const + prod + y = 8 + 6 + 5 = 19
        assert_eq!(result, 19.0);
    }

    #[test]
    fn test_negation_operator() {
        // Test unary negation operator
        let mut builder = Context::new_f64();

        let expr = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let c = scope.constant(5.0);

            // Test negation on variable and constant
            (-x).add(-c) // -x + -c
        });

        let vars = ScopedVarArray::<f64, 0>::new(vec![7.0]);
        let result = expr.eval(&vars);

        // -x + -c = -7 + -5 = -12
        assert_eq!(result, -12.0);
    }

    #[test]
    fn test_variable_constant_mixing() {
        // Test mixing variables and constants with operators
        let mut builder = Context::new_f64();

        let expr = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let c = scope.constant(10.0);

            // Test different combinations: Variable op Constant
            x + c // Variable + Constant ✅
        });

        let vars = ScopedVarArray::<f64, 0>::new(vec![3.0]);
        let result = expr.eval(&vars);

        // x + c = 3 + 10 = 13
        assert_eq!(result, 13.0);
    }

    #[test]
    fn test_operator_overloading_documentation() {
        // This test documents the current operator overloading capabilities
        let mut builder = Context::new_f64();

        let _result = builder.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (y, scope) = scope.auto_var();
            let c1 = scope.clone().constant(2.0);
            let c2 = scope.clone().constant(3.0);

            // ✅ SUPPORTED: Basic operations between same types
            let _const_times_const = c1 * c2; // ScopedConstValue * ScopedConstValue

            // ✅ SUPPORTED: Cross-type operations (create fresh variables to avoid moves)
            let (a, scope) = scope.auto_var();
            let d = scope.clone().constant(1.0);
            let _var_plus_const = a + d; // ScopedVar + ScopedConstValue

            // ✅ SUPPORTED: Unary operations
            let (b, scope) = scope.auto_var();
            let e = scope.clone().constant(4.0);
            let _neg_var = -b; // -ScopedVar
            let _neg_const = -e; // -ScopedConstValue

            // 🔄 WORKAROUND: Use method syntax for complex expressions
            let (final_x, final_scope) = scope.auto_var();
            let final_c = final_scope.constant(1.0);
            let complex = final_x + final_c;

            // ✅ DOCUMENTED LIMITATION: Variables with different IDs require method syntax
            // let _var_plus_var = x + y;  // ❌ Type mismatch: different const IDs
            // Workaround: use method syntax
            let _var_plus_var_method = x.add(y); // ✅ Works with method syntax

            complex
        });

        // The fact that this compiles demonstrates the current capabilities
        println!("✅ Phase 1 operator overloading: Basic operations implemented");
        println!("🔄 Complex expressions: Use method syntax as documented");
        println!("📝 Variable + Variable: Use .add() method syntax due to type system constraints");
    }
}
