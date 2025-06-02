//! Typed Expression Builder
//!
//! This module provides a typed expression builder that enables natural mathematical syntax
//! and expressions while maintaining intuitive operator overloading syntax.

use super::typed_registry::{TypedVar, VariableRegistry};
use crate::ast::ASTRepr;
use crate::final_tagless::traits::NumericType;
use num_traits::Float;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

/// Type-safe expression builder with typed variables
#[derive(Debug, Clone)]
pub struct ExpressionBuilder {
    registry: Arc<RefCell<VariableRegistry>>,
}

impl ExpressionBuilder {
    /// Create a new typed expression builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RefCell::new(VariableRegistry::new())),
        }
    }

    /// Create a typed variable
    #[must_use]
    pub fn typed_var<T: NumericType + 'static>(&self) -> TypedVar<T> {
        self.registry.borrow_mut().register_typed_variable()
    }

    /// Create an expression from a typed variable
    #[must_use]
    pub fn expr_from<T: NumericType>(&self, var: TypedVar<T>) -> TypedBuilderExpr<T> {
        TypedBuilderExpr::new(ASTRepr::Variable(var.index()), self.registry.clone())
    }

    /// Create a constant expression
    pub fn constant<T: NumericType>(&self, value: T) -> TypedBuilderExpr<T> {
        TypedBuilderExpr::new(ASTRepr::Constant(value), self.registry.clone())
    }

    /// Backward compatibility: create untyped variable (defaults to f64)
    #[must_use]
    pub fn var(&self) -> TypedBuilderExpr<f64> {
        let typed_var = self.typed_var::<f64>();
        self.expr_from(typed_var)
    }

    /// Get the registry for evaluation
    #[must_use]
    pub fn registry(&self) -> Arc<RefCell<VariableRegistry>> {
        self.registry.clone()
    }

    /// Evaluate an expression with indexed variables (simple interface)
    #[must_use]
    pub fn eval(&self, expr: &TypedBuilderExpr<f64>, variables: &[f64]) -> f64 {
        let registry = self.registry.borrow();
        let var_array = registry.create_variable_map(variables);
        expr.as_ast().eval_with_vars(&var_array)
    }

    // ============================================================================
    // High-Level Mathematical Functions (from ergonomics system)
    // ============================================================================

    /// Create a polynomial expression using Horner's method for efficient evaluation
    ///
    /// Coefficients are in ascending order of powers: [c₀, c₁, c₂, ...] represents c₀ + c₁x + c₂x² + ...
    #[must_use]
    pub fn poly(
        &self,
        coefficients: &[f64],
        variable: &TypedBuilderExpr<f64>,
    ) -> TypedBuilderExpr<f64> {
        if coefficients.is_empty() {
            return self.constant(0.0);
        }

        if coefficients.len() == 1 {
            return self.constant(coefficients[0]);
        }

        // Use Horner's method for efficient evaluation
        let mut result = self.constant(coefficients[coefficients.len() - 1]);

        for &coeff in coefficients.iter().rev().skip(1) {
            result = result.clone() * variable.clone() + self.constant(coeff);
        }

        result
    }

    /// Create a linear expression: ax + b
    #[must_use]
    pub fn linear(
        &self,
        a: f64,
        b: f64,
        variable: &TypedBuilderExpr<f64>,
    ) -> TypedBuilderExpr<f64> {
        self.poly(&[b, a], variable)
    }

    /// Create a quadratic expression: ax² + bx + c
    #[must_use]
    pub fn quadratic(
        &self,
        a: f64,
        b: f64,
        c: f64,
        variable: &TypedBuilderExpr<f64>,
    ) -> TypedBuilderExpr<f64> {
        self.poly(&[c, b, a], variable)
    }

    /// Create a Gaussian (normal) distribution function
    ///
    /// Creates: (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    #[must_use]
    pub fn gaussian(
        &self,
        mean: f64,
        std_dev: f64,
        variable: &TypedBuilderExpr<f64>,
    ) -> TypedBuilderExpr<f64> {
        let pi = self.constant(std::f64::consts::PI);
        let two = self.constant(2.0);
        let sigma_squared = self.constant(std_dev * std_dev);

        // Normalization factor: 1/√(2πσ²)
        let norm_factor = self.constant(1.0) / ((&two * &pi * &sigma_squared).sqrt());

        // Exponent: -(x-μ)²/(2σ²)
        let x_minus_mu = variable.clone() - self.constant(mean);
        let x_minus_mu_squared = x_minus_mu.clone().pow(self.constant(2.0));
        let exponent = -(x_minus_mu_squared / (&two * &sigma_squared));

        norm_factor * exponent.exp()
    }

    /// Create a logistic (sigmoid) function: 1/(1 + exp(-x))
    #[must_use]
    pub fn logistic(&self, variable: &TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64> {
        let one = self.constant(1.0);
        let neg_x = -variable.clone();
        let exp_neg_x = neg_x.exp();
        &one / (&one + exp_neg_x)
    }

    /// Create a hyperbolic tangent function: tanh(x) = (exp(2x) - 1)/(exp(2x) + 1)
    #[must_use]
    pub fn tanh(&self, variable: &TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64> {
        let two = self.constant(2.0);
        let one = self.constant(1.0);
        let two_x = &two * variable;
        let exp_2x = two_x.exp();

        let numerator = &exp_2x - &one;
        let denominator = &exp_2x + &one;

        numerator / denominator
    }
}

impl Default for ExpressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-safe expression wrapper that preserves type information and enables operator overloading
#[derive(Debug, Clone)]
pub struct TypedBuilderExpr<T> {
    ast: ASTRepr<T>,
    registry: Arc<RefCell<VariableRegistry>>,
    _phantom: PhantomData<T>,
}

impl<T: NumericType> TypedBuilderExpr<T> {
    /// Create a new typed expression
    pub fn new(ast: ASTRepr<T>, registry: Arc<RefCell<VariableRegistry>>) -> Self {
        Self {
            ast,
            registry,
            _phantom: PhantomData,
        }
    }

    /// Get the underlying AST
    pub fn as_ast(&self) -> &ASTRepr<T> {
        &self.ast
    }

    /// Convert to the underlying AST (consuming)
    pub fn into_ast(self) -> ASTRepr<T> {
        self.ast
    }

    /// Get the registry
    pub fn registry(&self) -> Arc<RefCell<VariableRegistry>> {
        self.registry.clone()
    }
}

// Type conversion methods
impl<T: NumericType> TypedBuilderExpr<T> {
    /// Convert to f64 expression (for types that can be promoted)
    pub fn as_f64(self) -> TypedBuilderExpr<f64>
    where
        T: Into<f64> + Copy,
    {
        TypedBuilderExpr::new(self.ast.convert_to_f64(), self.registry)
    }

    /// Convert to f32 expression (for types that can be demoted safely)
    pub fn as_f32(self) -> TypedBuilderExpr<f32>
    where
        T: Into<f32> + Copy,
    {
        TypedBuilderExpr::new(self.ast.convert_to_f32(), self.registry)
    }
}

// Transcendental functions for Float types
impl<T: NumericType + Float> TypedBuilderExpr<T> {
    /// Sine function
    pub fn sin(self) -> Self {
        Self::new(self.ast.sin(), self.registry)
    }

    /// Cosine function
    pub fn cos(self) -> Self {
        Self::new(self.ast.cos(), self.registry)
    }

    /// Natural logarithm
    pub fn ln(self) -> Self {
        Self::new(self.ast.ln(), self.registry)
    }

    /// Exponential function
    pub fn exp(self) -> Self {
        Self::new(self.ast.exp(), self.registry)
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        Self::new(self.ast.sqrt(), self.registry)
    }

    /// Power function
    pub fn pow(self, exp: Self) -> Self {
        Self::new(
            ASTRepr::Pow(Box::new(self.ast), Box::new(exp.ast)),
            self.registry,
        )
    }
}

// Same-type arithmetic operations
impl<T> Add for TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast + rhs.ast, self.registry)
    }
}

impl<T> Sub for TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast - rhs.ast, self.registry)
    }
}

impl<T> Mul for TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast * rhs.ast, self.registry)
    }
}

impl<T> Div for TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast / rhs.ast, self.registry)
    }
}

impl<T> Neg for TypedBuilderExpr<T>
where
    T: NumericType + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-self.ast, self.registry)
    }
}

// Reference operations for efficiency
impl<T> Add<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + &rhs.ast, self.registry.clone())
    }
}

impl<T> Add<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + rhs.ast, self.registry.clone())
    }
}

impl<T> Add<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast + &rhs.ast, self.registry)
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * &rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast * &rhs.ast, self.registry)
    }
}

// Negation for references
impl<T> Neg for &TypedBuilderExpr<T>
where
    T: NumericType + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-&self.ast, self.registry.clone())
    }
}

// Cross-type operations with automatic promotion
impl Add<TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self + promoted_rhs
    }
}

impl Add<TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self + promoted_rhs
    }
}

impl Add<&TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self + promoted_rhs
    }
}

impl Add<&TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self + promoted_rhs
    }
}

impl Mul<TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self * promoted_rhs
    }
}

impl Mul<TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self * promoted_rhs
    }
}

impl Mul<&TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self * promoted_rhs
    }
}

impl Mul<&TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self * promoted_rhs
    }
}

// Scalar operations (constants)
impl<T> Add<T> for TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self + rhs_expr
    }
}

impl<T> Add<T> for &TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self + rhs_expr
    }
}

impl<T> Mul<T> for TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self * rhs_expr
    }
}

impl<T> Mul<T> for &TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self * rhs_expr
    }
}

// Reverse scalar operations
impl Add<TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr + rhs
    }
}

impl Add<&TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr + rhs
    }
}

impl Mul<TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr * rhs
    }
}

impl Mul<&TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr * rhs
    }
}

// Extension methods for ASTRepr to support type conversion
impl<T: NumericType> ASTRepr<T> {
    /// Convert to f64 AST (placeholder - would need proper implementation)
    pub fn convert_to_f64(&self) -> ASTRepr<f64>
    where
        T: Into<f64> + Copy,
    {
        match self {
            ASTRepr::Constant(val) => ASTRepr::Constant((*val).into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(base.convert_to_f64()),
                Box::new(exp.convert_to_f64()),
            ),
            ASTRepr::Neg(expr) => ASTRepr::Neg(Box::new(expr.convert_to_f64())),
            ASTRepr::Sin(expr) => ASTRepr::Sin(Box::new(expr.convert_to_f64())),
            ASTRepr::Cos(expr) => ASTRepr::Cos(Box::new(expr.convert_to_f64())),
            ASTRepr::Ln(expr) => ASTRepr::Ln(Box::new(expr.convert_to_f64())),
            ASTRepr::Exp(expr) => ASTRepr::Exp(Box::new(expr.convert_to_f64())),
            ASTRepr::Sqrt(expr) => ASTRepr::Sqrt(Box::new(expr.convert_to_f64())),
        }
    }

    /// Convert to f32 AST (placeholder - would need proper implementation)
    pub fn convert_to_f32(&self) -> ASTRepr<f32>
    where
        T: Into<f32> + Copy,
    {
        match self {
            ASTRepr::Constant(val) => ASTRepr::Constant((*val).into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(base.convert_to_f32()),
                Box::new(exp.convert_to_f32()),
            ),
            ASTRepr::Neg(expr) => ASTRepr::Neg(Box::new(expr.convert_to_f32())),
            ASTRepr::Sin(expr) => ASTRepr::Sin(Box::new(expr.convert_to_f32())),
            ASTRepr::Cos(expr) => ASTRepr::Cos(Box::new(expr.convert_to_f32())),
            ASTRepr::Ln(expr) => ASTRepr::Ln(Box::new(expr.convert_to_f32())),
            ASTRepr::Exp(expr) => ASTRepr::Exp(Box::new(expr.convert_to_f32())),
            ASTRepr::Sqrt(expr) => ASTRepr::Sqrt(Box::new(expr.convert_to_f32())),
        }
    }
}

// Reference operations for subtraction
impl<T> Sub<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - &rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast - &rhs.ast, self.registry)
    }
}

// Reference operations for division
impl<T> Div<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / &rhs.ast, self.registry.clone())
    }
}

impl<T> Div<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / rhs.ast, self.registry.clone())
    }
}

impl<T> Div<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast / &rhs.ast, self.registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_variable_creation() {
        let builder = ExpressionBuilder::new();

        // Create typed variables
        let x: TypedVar<f64> = builder.typed_var();
        let y: TypedVar<f32> = builder.typed_var();

        assert_eq!(x.name(), "var_0");
        assert_eq!(y.name(), "var_1");
        assert_ne!(x.index(), y.index());
    }

    #[test]
    fn test_typed_expression_building() {
        let builder = ExpressionBuilder::new();

        // Create typed variables and expressions
        let x = builder.typed_var::<f64>();
        let y = builder.typed_var::<f64>();

        let x_expr = builder.expr_from(x);
        let y_expr = builder.expr_from(y);

        // Test same-type operations
        let sum = &x_expr + &y_expr;
        let product = &x_expr * &y_expr;

        // Verify the AST structure
        match sum.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }

        match product.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_cross_type_operations() {
        let builder = ExpressionBuilder::new();

        let x_f64 = builder.expr_from(builder.typed_var::<f64>());
        let y_f32 = builder.expr_from(builder.typed_var::<f32>());

        // This should work with automatic promotion
        let mixed_sum = x_f64 + y_f32;

        // Result should be f64
        match mixed_sum.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_scalar_operations() {
        let builder = ExpressionBuilder::new();

        let x = builder.expr_from(builder.typed_var::<f64>());

        // Test scalar operations
        let scaled = &x * 2.0;
        let shifted = &x + 1.0;
        let reverse_scaled = 3.0 * &x;

        match scaled.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication"),
        }

        match shifted.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }

        match reverse_scaled.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_transcendental_functions() {
        let builder = ExpressionBuilder::new();

        let x = builder.expr_from(builder.typed_var::<f64>());

        let sin_x = x.clone().sin();
        let exp_x = x.clone().exp();
        let ln_x = x.clone().ln();

        match sin_x.as_ast() {
            ASTRepr::Sin(_) => {}
            _ => panic!("Expected sine"),
        }

        match exp_x.as_ast() {
            ASTRepr::Exp(_) => {}
            _ => panic!("Expected exponential"),
        }

        match ln_x.as_ast() {
            ASTRepr::Ln(_) => {}
            _ => panic!("Expected logarithm"),
        }
    }

    #[test]
    fn test_backward_compatibility() {
        let builder = ExpressionBuilder::new();

        // Old-style variable creation should still work
        let x = builder.var(); // Should be TypedBuilderExpr<f64>
        let y = builder.var();

        let expr = &x * &x + 2.0 * &x + &y;

        // Should create a valid AST
        match expr.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition at top level"),
        }
    }

    #[test]
    fn test_complex_expression() {
        let builder = ExpressionBuilder::new();

        let x = builder.var();
        let y = builder.var();

        // Build: sin(x^2 + y) * exp(-x)
        let x_squared = x.clone().pow(builder.constant(2.0));
        let sum = x_squared + &y;
        let sin_sum = sum.sin();
        let neg_x = -&x;
        let exp_neg_x = neg_x.exp();
        let result = sin_sum * exp_neg_x;

        // Verify it creates a valid AST
        match result.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication at top level"),
        }
    }
}
