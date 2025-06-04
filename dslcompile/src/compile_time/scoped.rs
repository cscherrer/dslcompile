//! Type-Level Scoped Variables for Compile-Time System
//!
//! This module implements type-level variable scoping that prevents variable collisions
//! at compile time while maintaining zero runtime overhead.

use crate::ast::ASTRepr;
use std::marker::PhantomData;

/// Scoped variable with compile-time scope and ID tracking
#[derive(Clone, Debug)]
pub struct ScopedVar<const ID: usize, const SCOPE: usize>;

/// Scoped constant with compile-time scope tracking
#[derive(Clone, Debug)]
pub struct ScopedConst<const BITS: u64, const SCOPE: usize>;

/// Core trait for scoped mathematical expressions
pub trait ScopedMathExpr<const SCOPE: usize>: Clone + Sized {
    /// Evaluate the expression with scoped variable values
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64;

    /// Convert to AST representation
    fn to_ast(&self) -> ASTRepr<f64>;

    /// Add two expressions in the same scope
    fn add<T: ScopedMathExpr<SCOPE>>(self, other: T) -> ScopedAdd<Self, T, SCOPE> {
        ScopedAdd {
            left: self,
            right: other,
            _scope: PhantomData,
        }
    }

    /// Multiply two expressions in the same scope
    fn mul<T: ScopedMathExpr<SCOPE>>(self, other: T) -> ScopedMul<Self, T, SCOPE> {
        ScopedMul {
            left: self,
            right: other,
            _scope: PhantomData,
        }
    }

    /// Subtract two expressions in the same scope
    fn sub<T: ScopedMathExpr<SCOPE>>(self, other: T) -> ScopedSub<Self, T, SCOPE> {
        ScopedSub {
            left: self,
            right: other,
            _scope: PhantomData,
        }
    }

    /// Divide two expressions in the same scope
    fn div<T: ScopedMathExpr<SCOPE>>(self, other: T) -> ScopedDiv<Self, T, SCOPE> {
        ScopedDiv {
            left: self,
            right: other,
            _scope: PhantomData,
        }
    }

    /// Power function
    fn pow<T: ScopedMathExpr<SCOPE>>(self, exponent: T) -> ScopedPow<Self, T, SCOPE> {
        ScopedPow {
            base: self,
            exponent,
            _scope: PhantomData,
        }
    }

    /// Natural exponential
    fn exp(self) -> ScopedExp<Self, SCOPE> {
        ScopedExp {
            inner: self,
            _scope: PhantomData,
        }
    }

    /// Natural logarithm
    fn ln(self) -> ScopedLn<Self, SCOPE> {
        ScopedLn {
            inner: self,
            _scope: PhantomData,
        }
    }

    /// Sine function
    fn sin(self) -> ScopedSin<Self, SCOPE> {
        ScopedSin {
            inner: self,
            _scope: PhantomData,
        }
    }

    /// Cosine function
    fn cos(self) -> ScopedCos<Self, SCOPE> {
        ScopedCos {
            inner: self,
            _scope: PhantomData,
        }
    }

    /// Square root
    fn sqrt(self) -> ScopedSqrt<Self, SCOPE> {
        ScopedSqrt {
            inner: self,
            _scope: PhantomData,
        }
    }

    /// Negation
    fn neg(self) -> ScopedNeg<Self, SCOPE> {
        ScopedNeg {
            inner: self,
            _scope: PhantomData,
        }
    }
}

/// Trait for composing expressions across different scopes
pub trait ScopeCompose<Other, const OTHER_SCOPE: usize>: Sized {
    type Output;

    /// Compose expressions from different scopes with automatic variable remapping
    fn compose_with<F>(self, other: Other, combiner: F) -> Self::Output
    where
        F: FnOnce(Self, Other) -> Self::Output;
}

/// Variable array for a specific scope
pub struct ScopedVarArray<const SCOPE: usize> {
    vars: Vec<f64>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize> ScopedVarArray<SCOPE> {
    /// Create a new scoped variable array
    #[must_use]
    pub fn new(vars: Vec<f64>) -> Self {
        Self {
            vars,
            _scope: PhantomData,
        }
    }

    /// Get variable value by ID
    #[must_use]
    pub fn get(&self, id: usize) -> f64 {
        self.vars.get(id).copied().unwrap_or(0.0)
    }
}

// ============================================================================
// VARIABLE AND CONSTANT IMPLEMENTATIONS
// ============================================================================

impl<const ID: usize, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedVar<ID, SCOPE> {
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        vars.get(ID)
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Variable(ID)
    }
}

impl<const BITS: u64, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedConst<BITS, SCOPE> {
    fn eval(&self, _vars: &ScopedVarArray<SCOPE>) -> f64 {
        f64::from_bits(BITS)
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Constant(f64::from_bits(BITS))
    }
}

// ============================================================================
// OPERATION IMPLEMENTATIONS
// ============================================================================

#[derive(Clone, Debug)]
pub struct ScopedAdd<L, R, const SCOPE: usize> {
    left: L,
    right: R,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<L, R, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedAdd<L, R, SCOPE>
where
    L: ScopedMathExpr<SCOPE>,
    R: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.left.eval(vars) + self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Add(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedMul<L, R, const SCOPE: usize> {
    left: L,
    right: R,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<L, R, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedMul<L, R, SCOPE>
where
    L: ScopedMathExpr<SCOPE>,
    R: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.left.eval(vars) * self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Mul(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedSub<L, R, const SCOPE: usize> {
    left: L,
    right: R,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<L, R, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedSub<L, R, SCOPE>
where
    L: ScopedMathExpr<SCOPE>,
    R: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.left.eval(vars) - self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Sub(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedDiv<L, R, const SCOPE: usize> {
    left: L,
    right: R,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<L, R, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedDiv<L, R, SCOPE>
where
    L: ScopedMathExpr<SCOPE>,
    R: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.left.eval(vars) / self.right.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Div(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedPow<B, E, const SCOPE: usize> {
    base: B,
    exponent: E,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<B, E, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedPow<B, E, SCOPE>
where
    B: ScopedMathExpr<SCOPE>,
    E: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.base.eval(vars).powf(self.exponent.eval(vars))
    }

    fn to_ast(&self) -> ASTRepr<f64> {
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
pub struct ScopedExp<T, const SCOPE: usize> {
    inner: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedExp<T, SCOPE>
where
    T: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.inner.eval(vars).exp()
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Exp(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedLn<T, const SCOPE: usize> {
    inner: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedLn<T, SCOPE>
where
    T: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.inner.eval(vars).ln()
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Ln(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedSin<T, const SCOPE: usize> {
    inner: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedSin<T, SCOPE>
where
    T: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.inner.eval(vars).sin()
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Sin(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedCos<T, const SCOPE: usize> {
    inner: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedCos<T, SCOPE>
where
    T: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.inner.eval(vars).cos()
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Cos(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedSqrt<T, const SCOPE: usize> {
    inner: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedSqrt<T, SCOPE>
where
    T: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.inner.eval(vars).sqrt()
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Sqrt(Box::new(self.inner.to_ast()))
    }
}

#[derive(Clone, Debug)]
pub struct ScopedNeg<T, const SCOPE: usize> {
    inner: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedNeg<T, SCOPE>
where
    T: ScopedMathExpr<SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64 {
        -self.inner.eval(vars)
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Neg(Box::new(self.inner.to_ast()))
    }
}

// ============================================================================
// SCOPE COMPOSITION IMPLEMENTATIONS
// ============================================================================

/// Composed expression from two different scopes
#[derive(Clone, Debug)]
pub struct ComposedExpr<L, R, const SCOPE1: usize, const SCOPE2: usize> {
    left: L,
    right: R,
    _scope1: PhantomData<[(); SCOPE1]>,
    _scope2: PhantomData<[(); SCOPE2]>,
}

impl<L, R, const SCOPE1: usize, const SCOPE2: usize> ComposedExpr<L, R, SCOPE1, SCOPE2>
where
    L: ScopedMathExpr<SCOPE1>,
    R: ScopedMathExpr<SCOPE2>,
{
    /// Create a new composed expression
    pub fn new(left: L, right: R) -> Self {
        Self {
            left,
            right,
            _scope1: PhantomData,
            _scope2: PhantomData,
        }
    }

    /// Evaluate with variables from both scopes
    pub fn eval(
        &self,
        vars1: &ScopedVarArray<SCOPE1>,
        vars2: &ScopedVarArray<SCOPE2>,
    ) -> (f64, f64) {
        (self.left.eval(vars1), self.right.eval(vars2))
    }

    /// Add the two scoped expressions (returns a composed expression with combined scope)
    pub fn add(self) -> ComposedAdd {
        let left_ast = self.left.to_ast();
        // Count variables in left AST to determine proper offset
        let max_var_in_left = find_max_variable_index(&left_ast);
        let offset = max_var_in_left + 1;
        let right_ast = remap_ast_variables(&self.right.to_ast(), offset);

        ComposedAdd {
            left_ast,
            right_ast,
        }
    }

    /// Multiply the two scoped expressions (returns a composed expression with combined scope)
    pub fn mul(self) -> ComposedMul {
        let left_ast = self.left.to_ast();
        // Count variables in left AST to determine proper offset
        let max_var_in_left = find_max_variable_index(&left_ast);
        let offset = max_var_in_left + 1;
        let right_ast = remap_ast_variables(&self.right.to_ast(), offset);

        ComposedMul {
            left_ast,
            right_ast,
        }
    }
}

/// Helper struct for composed addition
#[derive(Clone, Debug)]
pub struct ComposedAdd {
    left_ast: ASTRepr<f64>,
    right_ast: ASTRepr<f64>,
}

impl ComposedAdd {
    #[must_use]
    pub fn eval(&self, vars: &[f64]) -> f64 {
        let left_val = eval_ast(&self.left_ast, vars);
        let right_val = eval_ast(&self.right_ast, vars);
        left_val + right_val
    }

    #[must_use]
    pub fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Add(
            Box::new(self.left_ast.clone()),
            Box::new(self.right_ast.clone()),
        )
    }
}

/// Helper struct for composed multiplication
#[derive(Clone, Debug)]
pub struct ComposedMul {
    left_ast: ASTRepr<f64>,
    right_ast: ASTRepr<f64>,
}

impl ComposedMul {
    #[must_use]
    pub fn eval(&self, vars: &[f64]) -> f64 {
        let left_val = eval_ast(&self.left_ast, vars);
        let right_val = eval_ast(&self.right_ast, vars);
        left_val * right_val
    }

    #[must_use]
    pub fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Mul(
            Box::new(self.left_ast.clone()),
            Box::new(self.right_ast.clone()),
        )
    }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Create a scoped variable
#[must_use]
pub const fn scoped_var<const ID: usize, const SCOPE: usize>() -> ScopedVar<ID, SCOPE> {
    ScopedVar
}

/// Create a scoped constant
#[must_use]
pub fn scoped_constant<const SCOPE: usize>(value: f64) -> ScopedConstValue<SCOPE> {
    ScopedConstValue {
        value,
        _scope: PhantomData,
    }
}

/// Runtime constant that can hold any f64 value in a specific scope
#[derive(Clone, Debug)]
pub struct ScopedConstValue<const SCOPE: usize> {
    value: f64,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize> ScopedMathExpr<SCOPE> for ScopedConstValue<SCOPE> {
    fn eval(&self, _vars: &ScopedVarArray<SCOPE>) -> f64 {
        self.value
    }

    fn to_ast(&self) -> ASTRepr<f64> {
        ASTRepr::Constant(self.value)
    }
}

/// Compose two expressions from different scopes
pub fn compose<L, R, const SCOPE1: usize, const SCOPE2: usize>(
    left: L,
    right: R,
) -> ComposedExpr<L, R, SCOPE1, SCOPE2>
where
    L: ScopedMathExpr<SCOPE1>,
    R: ScopedMathExpr<SCOPE2>,
{
    ComposedExpr::new(left, right)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Find the maximum variable index in an AST
fn find_max_variable_index(ast: &ASTRepr<f64>) -> usize {
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
fn remap_ast_variables(ast: &ASTRepr<f64>, offset: usize) -> ASTRepr<f64> {
    match ast {
        ASTRepr::Constant(val) => ASTRepr::Constant(*val),
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
        ASTRepr::Pow(base, exp) => ASTRepr::Pow(
            Box::new(remap_ast_variables(base, offset)),
            Box::new(remap_ast_variables(exp, offset)),
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
fn eval_ast(ast: &ASTRepr<f64>, vars: &[f64]) -> f64 {
    match ast {
        ASTRepr::Constant(val) => *val,
        ASTRepr::Variable(idx) => vars.get(*idx).copied().unwrap_or(0.0),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_variables_no_collision() {
        // Define f(x) = 2x in scope 0
        let x_f = scoped_var::<0, 0>();
        let f = x_f.mul(scoped_constant::<0>(2.0));

        // Define g(y) = 3y in scope 1 - no collision!
        let y_g = scoped_var::<0, 1>();
        let g = y_g.mul(scoped_constant::<1>(3.0));

        // Evaluate independently
        let f_vars = ScopedVarArray::<0>::new(vec![4.0]);
        let g_vars = ScopedVarArray::<1>::new(vec![5.0]);

        assert_eq!(f.eval(&f_vars), 8.0); // 2 * 4 = 8
        assert_eq!(g.eval(&g_vars), 15.0); // 3 * 5 = 15
    }

    #[test]
    fn test_scope_composition() {
        // Define f(x) = x² in scope 0
        let x_f = scoped_var::<0, 0>();
        let f = x_f.clone().mul(x_f);

        // Define g(y) = 2y in scope 1
        let y_g = scoped_var::<0, 1>();
        let g = y_g.mul(scoped_constant::<1>(2.0));

        // Compose h = f + g
        let composed = compose(f, g);
        let h = composed.add();

        // Evaluate h(3, 4) = f(3) + g(4) = 9 + 8 = 17
        let vars = vec![3.0, 4.0]; // Combined variable array
        assert_eq!(h.eval(&vars), 17.0);
    }

    #[test]
    fn test_complex_scoped_expression() {
        // Build sin(x) + cos(y) in scope 0
        let x = scoped_var::<0, 0>();
        let y = scoped_var::<1, 0>();
        let expr = x.sin().add(y.cos());

        let vars = ScopedVarArray::<0>::new(vec![std::f64::consts::PI / 2.0, 0.0]);
        let result = expr.eval(&vars);

        // sin(π/2) + cos(0) = 1 + 1 = 2
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ast_conversion() {
        // Build x + y in scope 0
        let x = scoped_var::<0, 0>();
        let y = scoped_var::<1, 0>();
        let expr = x.add(y);

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

        // Define quadratic(x,y) = x² + xy + y² in scope 0 (uses variables 0, 1)
        let x = scoped_var::<0, 0>();
        let y = scoped_var::<1, 0>();
        let quadratic = x
            .clone()
            .mul(x.clone())
            .add(x.mul(y.clone()))
            .add(y.clone().mul(y));

        // Define linear(a,b) = 2a + 3b in scope 1 (uses variables 0, 1)
        let a = scoped_var::<0, 1>();
        let b = scoped_var::<1, 1>();
        let linear = a
            .mul(scoped_constant::<1>(2.0))
            .add(b.mul(scoped_constant::<1>(3.0)));

        // Test individual evaluations
        let quad_vars = ScopedVarArray::<0>::new(vec![1.0, 2.0]);
        let quad_result = quadratic.eval(&quad_vars); // 1² + 1*2 + 2² = 7
        assert_eq!(quad_result, 7.0);

        let lin_vars = ScopedVarArray::<1>::new(vec![3.0, 4.0]);
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

        // Single variable expression: x (var 0)
        let x = scoped_var::<0, 0>();
        let expr1 = x.to_ast();
        assert_eq!(find_max_variable_index(&expr1), 0);

        // Two variable expression: x + y (vars 0, 1)
        let x = scoped_var::<0, 0>();
        let y = scoped_var::<1, 0>();
        let expr2 = x.add(y).to_ast();
        assert_eq!(find_max_variable_index(&expr2), 1);

        // Complex expression: x² + xy + y² (vars 0, 1)
        let x = scoped_var::<0, 0>();
        let y = scoped_var::<1, 0>();
        let expr3 = x
            .clone()
            .mul(x.clone())
            .add(x.mul(y.clone()))
            .add(y.clone().mul(y))
            .to_ast();
        assert_eq!(find_max_variable_index(&expr3), 1);

        // Test constant expression (no variables)
        let constant_expr = scoped_constant::<0>(5.0).to_ast();
        assert_eq!(find_max_variable_index(&constant_expr), 0);
    }
}
