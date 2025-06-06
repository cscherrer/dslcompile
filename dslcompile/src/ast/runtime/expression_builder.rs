//! Dynamic Expression Builder
//!
//! This module provides a runtime expression builder that enables natural mathematical syntax
//! and expressions while maintaining intuitive operator overloading syntax.

use super::typed_registry::{TypedVar, VariableRegistry};
use crate::ast::ASTRepr;
use crate::ast::NumericType;
use num_traits::Float;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

/// Dynamic expression builder with runtime variable management
#[derive(Debug, Clone)]
pub struct DynamicContext {
    registry: Arc<RefCell<VariableRegistry>>,
}

impl DynamicContext {
    /// Create a new dynamic expression builder
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

    /// Evaluate an expression with parameter values for captured variables
    /// This is the key method that fixes parameter capture!
    #[must_use]
    pub fn eval_with_vars(&self, expr: &ASTRepr<f64>, variables: &[f64]) -> f64 {
        // Create a variable array that can handle all variables used in the expression
        let registry = self.registry.borrow();
        let max_var_needed = variables.len().max(registry.len());
        let mut var_array = vec![0.0; max_var_needed];

        // Copy the provided variable values
        for (i, &value) in variables.iter().enumerate() {
            if i < var_array.len() {
                var_array[i] = value;
            }
        }

        expr.eval_with_vars(&var_array)
    }

    // ============================================================================
    // High-Level Mathematical Functions (from ergonomics system)
    // ============================================================================

    /// Create a polynomial expression using Horner's method
    ///
    /// Creates: c[n]*x^n + c[n-1]*x^(n-1) + ... + c[1]*x + c[0]
    ///
    /// Uses Horner's method for efficient evaluation:
    /// ((c[n]*x + c[n-1])*x + c[n-2])*x + ... + c[0]
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

    // Removed domain-specific pattern extraction methods
    // All pattern recognition now handled by SummationOptimizer (domain-agnostic)

    // Removed domain-specific helper methods

    /// Generate pretty-printed string representation of an expression
    #[must_use]
    pub fn pretty_print<T: NumericType>(&self, expr: &TypedBuilderExpr<T>) -> String {
        crate::ast::pretty::pretty_ast(expr.as_ast(), &self.registry.borrow())
    }

    /// Create an AST from an expression (for compilation backends)
    #[must_use]
    pub fn to_ast<T: NumericType>(&self, expr: &TypedBuilderExpr<T>) -> crate::ast::ASTRepr<T> {
        expr.as_ast().clone()
    }

    /// Evaluate a two-variable expression (convenience method)
    #[must_use]
    pub fn eval_two_vars(&self, expr: &TypedBuilderExpr<f64>, x: f64, y: f64) -> f64 {
        expr.as_ast().eval_two_vars(x, y)
    }

    // Removed detect_summation_pattern - replaced by SummationOptimizer::recognize_pattern
    // The SummationOptimizer provides superior pattern recognition with decomposition support

    // Removed ast_equals helper - was only used by removed detect_summation_pattern

    /// Domain-agnostic summation method - handles both mathematical ranges and data iteration
    ///
    /// Creates optimized summations using mathematical decomposition:
    /// - Sum splitting: Σ(f(i) + g(i)) = Σ(f(i)) + Σ(g(i))  
    /// - Factor extraction: Σ(k * f(i)) = k * Σ(f(i))
    /// - Closed-form evaluation when possible
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    ///
    /// fn example() -> Result<()> {
    ///     let math = DynamicContext::new();
    ///     
    ///     // Mathematical summation over range 1..=10
    ///     let result1 = math.sum(1..=10, |i| {
    ///         i * math.constant(5.0)  // Σ(5*i) = 5*Σ(i) = 5*55 = 275
    ///     })?;
    ///     
    ///     // Data summation over actual values
    ///     let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    ///     let result2 = math.sum(data, |x| {
    ///         x * math.constant(2.0)  // Sum each data point times 2
    ///     })?;
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub fn sum<I, F>(&self, iterable: I, f: F) -> crate::Result<TypedBuilderExpr<f64>>
    where
        I: IntoSummableRange,
        F: Fn(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        match iterable.into_summable() {
            SummableRange::MathematicalRange { start, end } => {
                // Mathematical summation - can use closed-form optimizations
                let i_var = self.var(); // This becomes Variable(0) in the AST
                let expr = f(i_var);
                let ast = expr.into();

                let optimizer = SummationOptimizer::new();
                let result_value = optimizer.optimize_summation(start, end, ast)?;
                Ok(self.constant(result_value))
            }
            SummableRange::DataIteration { values } => {
                // Data summation - evaluate each data point
                if values.is_empty() {
                    return Ok(self.constant(0.0));
                }

                let mut total = 0.0;
                for x_val in values {
                    let x_expr = self.constant(x_val);
                    let result_expr = f(x_expr);
                    total += self.eval(&result_expr, &[]);
                }

                Ok(self.constant(total))
            }
        }
    }
}

impl Default for DynamicContext {
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

    /// Get the registry
    pub fn registry(&self) -> Arc<RefCell<VariableRegistry>> {
        self.registry.clone()
    }

    /// Generate pretty-printed string representation
    #[must_use]
    pub fn pretty_print(&self) -> String {
        crate::ast::pretty::pretty_ast(&self.ast, &self.registry.borrow())
    }

    /// Evaluate the expression with given variable values
    #[must_use]
    pub fn eval_with_vars(&self, variables: &[T]) -> T
    where
        T: Float + Copy + num_traits::FromPrimitive,
    {
        self.ast.eval_with_vars(variables)
    }

    /// Convert to f64 expression (no-op if already f64)
    pub fn to_f64(self) -> TypedBuilderExpr<f64>
    where
        T: Into<f64>,
    {
        // Convert the AST structure to f64
        let converted_ast = self.convert_ast_to_f64(&self.ast);
        TypedBuilderExpr::new(converted_ast, self.registry)
    }

    /// Convert to f32 expression  
    pub fn to_f32(self) -> TypedBuilderExpr<f32>
    where
        T: Into<f32>,
    {
        // Convert the AST structure to f32
        let converted_ast = self.convert_ast_to_f32(&self.ast);
        TypedBuilderExpr::new(converted_ast, self.registry)
    }

    /// Helper method to convert AST from T to f64
    fn convert_ast_to_f64(&self, ast: &ASTRepr<T>) -> ASTRepr<f64>
    where
        T: Into<f64> + Clone,
    {
        match ast {
            ASTRepr::Constant(val) => ASTRepr::Constant(val.clone().into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(self.convert_ast_to_f64(left)),
                Box::new(self.convert_ast_to_f64(right)),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(self.convert_ast_to_f64(left)),
                Box::new(self.convert_ast_to_f64(right)),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(self.convert_ast_to_f64(left)),
                Box::new(self.convert_ast_to_f64(right)),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(self.convert_ast_to_f64(left)),
                Box::new(self.convert_ast_to_f64(right)),
            ),
            ASTRepr::Pow(left, right) => ASTRepr::Pow(
                Box::new(self.convert_ast_to_f64(left)),
                Box::new(self.convert_ast_to_f64(right)),
            ),
            ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(self.convert_ast_to_f64(inner))),
            ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(self.convert_ast_to_f64(inner))),
            ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(self.convert_ast_to_f64(inner))),
            ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(self.convert_ast_to_f64(inner))),
            ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(self.convert_ast_to_f64(inner))),
            ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(self.convert_ast_to_f64(inner))),
            ASTRepr::Sum {
                range,
                body,
                iter_var,
            } => ASTRepr::Sum {
                range: self.convert_sum_range_to_f64(range),
                body: Box::new(self.convert_ast_to_f64(body)),
                iter_var: *iter_var,
            },
        }
    }

    /// Helper method to convert `SumRange` from T to f64
    fn convert_sum_range_to_f64(
        &self,
        range: &crate::ast::ast_repr::SumRange<T>,
    ) -> crate::ast::ast_repr::SumRange<f64>
    where
        T: Into<f64> + Clone,
    {
        match range {
            crate::ast::ast_repr::SumRange::Mathematical { start, end } => {
                crate::ast::ast_repr::SumRange::Mathematical {
                    start: Box::new(self.convert_ast_to_f64(start)),
                    end: Box::new(self.convert_ast_to_f64(end)),
                }
            }
            crate::ast::ast_repr::SumRange::DataParameter { data_var } => {
                crate::ast::ast_repr::SumRange::DataParameter {
                    data_var: *data_var,
                }
            }
        }
    }

    /// Helper method to convert AST from T to f32
    fn convert_ast_to_f32(&self, ast: &ASTRepr<T>) -> ASTRepr<f32>
    where
        T: Into<f32> + Clone,
    {
        match ast {
            ASTRepr::Constant(val) => ASTRepr::Constant(val.clone().into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(self.convert_ast_to_f32(left)),
                Box::new(self.convert_ast_to_f32(right)),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(self.convert_ast_to_f32(left)),
                Box::new(self.convert_ast_to_f32(right)),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(self.convert_ast_to_f32(left)),
                Box::new(self.convert_ast_to_f32(right)),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(self.convert_ast_to_f32(left)),
                Box::new(self.convert_ast_to_f32(right)),
            ),
            ASTRepr::Pow(left, right) => ASTRepr::Pow(
                Box::new(self.convert_ast_to_f32(left)),
                Box::new(self.convert_ast_to_f32(right)),
            ),
            ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(self.convert_ast_to_f32(inner))),
            ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(self.convert_ast_to_f32(inner))),
            ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(self.convert_ast_to_f32(inner))),
            ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(self.convert_ast_to_f32(inner))),
            ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(self.convert_ast_to_f32(inner))),
            ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(self.convert_ast_to_f32(inner))),
            ASTRepr::Sum {
                range,
                body,
                iter_var,
            } => ASTRepr::Sum {
                range: self.convert_sum_range_to_f32(range),
                body: Box::new(self.convert_ast_to_f32(body)),
                iter_var: *iter_var,
            },
        }
    }

    /// Helper method to convert `SumRange` from T to f32
    fn convert_sum_range_to_f32(
        &self,
        range: &crate::ast::ast_repr::SumRange<T>,
    ) -> crate::ast::ast_repr::SumRange<f32>
    where
        T: Into<f32> + Clone,
    {
        match range {
            crate::ast::ast_repr::SumRange::Mathematical { start, end } => {
                crate::ast::ast_repr::SumRange::Mathematical {
                    start: Box::new(self.convert_ast_to_f32(start)),
                    end: Box::new(self.convert_ast_to_f32(end)),
                }
            }
            crate::ast::ast_repr::SumRange::DataParameter { data_var } => {
                crate::ast::ast_repr::SumRange::DataParameter {
                    data_var: *data_var,
                }
            }
        }
    }
}

// From trait implementation for converting TypedBuilderExpr to ASTRepr
impl<T: NumericType> From<TypedBuilderExpr<T>> for ASTRepr<T> {
    fn from(expr: TypedBuilderExpr<T>) -> Self {
        expr.ast
    }
}

// From trait implementations for numeric types to TypedBuilderExpr
// This allows writing expr + 2.0 instead of expr + context.constant(2.0)
impl From<f64> for TypedBuilderExpr<f64> {
    fn from(value: f64) -> Self {
        // Create a minimal registry for constants - they don't need variable tracking
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl From<f32> for TypedBuilderExpr<f32> {
    fn from(value: f32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl From<i32> for TypedBuilderExpr<f64> {
    fn from(value: i32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(f64::from(value)), registry)
    }
}

impl From<i64> for TypedBuilderExpr<f64> {
    fn from(value: i64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

impl From<usize> for TypedBuilderExpr<f64> {
    fn from(value: usize) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value as f64), registry)
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
        let promoted_rhs = rhs.to_f64();
        self + promoted_rhs
    }
}

impl Add<TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.to_f64();
        self + promoted_rhs
    }
}

impl Add<&TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().to_f64();
        self + promoted_rhs
    }
}

impl Add<&TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().to_f64();
        self + promoted_rhs
    }
}

impl Mul<TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.to_f64();
        self * promoted_rhs
    }
}

impl Mul<TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.to_f64();
        self * promoted_rhs
    }
}

impl Mul<&TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().to_f64();
        self * promoted_rhs
    }
}

impl Mul<&TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().to_f64();
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
        let builder = DynamicContext::new();

        // Create typed variables
        let x: TypedVar<f64> = builder.typed_var();
        let y: TypedVar<f32> = builder.typed_var();

        assert_eq!(x.name(), "var_0");
        assert_eq!(y.name(), "var_1");
        assert_ne!(x.index(), y.index());
    }

    #[test]
    fn test_typed_expression_building() {
        let builder = DynamicContext::new();

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
        let builder = DynamicContext::new();

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
        let builder = DynamicContext::new();

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
        let builder = DynamicContext::new();

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
        let builder = DynamicContext::new();

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
        let builder = DynamicContext::new();

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

    #[test]
    fn test_from_numeric_types() {
        let builder = DynamicContext::new();
        let x = builder.var();

        // Test From implementations for numeric types
        let expr1: TypedBuilderExpr<f64> = 2.0.into();
        let expr2: TypedBuilderExpr<f64> = 3i32.into();
        let expr3: TypedBuilderExpr<f64> = 4i64.into();
        let expr4: TypedBuilderExpr<f64> = 5usize.into();
        let expr5: TypedBuilderExpr<f32> = 2.5f32.into();

        // Verify they create constant expressions
        match expr1.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 2.0),
            _ => panic!("Expected constant"),
        }

        match expr2.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 3.0),
            _ => panic!("Expected constant"),
        }

        match expr3.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 4.0),
            _ => panic!("Expected constant"),
        }

        match expr4.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 5.0),
            _ => panic!("Expected constant"),
        }

        match expr5.as_ast() {
            ASTRepr::Constant(val) => assert_eq!(*val, 2.5),
            _ => panic!("Expected constant"),
        }

        // Test that these can be used in expressions naturally
        let combined = &x + expr1 + expr2; // x + 2.0 + 3.0
        match combined.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }

        // Test evaluation works
        let result = builder.eval(&combined, &[1.0]); // x = 1.0
        assert_eq!(result, 6.0); // 1.0 + 2.0 + 3.0 = 6.0
    }

    #[test]
    fn test_ergonomic_expression_building() {
        let builder = DynamicContext::new();
        let x = builder.var();
        let y = builder.var();

        // OLD WAY (still works): using explicit constant() calls
        let old_way =
            &x * builder.constant(2.0) + builder.constant(3.0) * &y + builder.constant(1.0);

        // NEW WAY: using From implementations for automatic conversions
        let new_way = &x * 2.0 + 3.0 * &y + 1.0;

        // Both should evaluate to the same result
        let test_values = vec![
            vec![1.0, 2.0], // x=1, y=2 => 1*2 + 3*2 + 1 = 9
            vec![3.0, 4.0], // x=3, y=4 => 3*2 + 3*4 + 1 = 19
            vec![0.0, 1.0], // x=0, y=1 => 0*2 + 3*1 + 1 = 4
        ];

        for values in test_values {
            let old_result = builder.eval(&old_way, &values);
            let new_result = builder.eval(&new_way, &values);
            assert_eq!(old_result, new_result);
        }

        // Test that complex expressions work naturally
        let quadratic = &x * &x + 2.0 * &x + 1.0; // x² + 2x + 1 = (x + 1)²
        let result = builder.eval(&quadratic, &[3.0]); // (3 + 1)² = 16
        assert_eq!(result, 16.0);

        // Test what actually works in Rust - mixing types requires explicit conversion
        let mixed_types = &x + 1.0 + 2.5; // f64 + f64 + f64 works fine
        let result = builder.eval(&mixed_types, &[0.5]); // 0.5 + 1.0 + 2.5 = 4.0
        assert_eq!(result, 4.0);

        // Scalar operations work naturally with the right type
        let with_scalars = &x + 2.0 + 3.0; // This already works!
        let result_scalars = builder.eval(&with_scalars, &[0.5]); // 0.5 + 2 + 3 = 5.5
        assert_eq!(result_scalars, 5.5);

        // Or use the const_ helper for better readability with mixed types
        let with_const = &x + builder.const_(2.0) + builder.const_(3.0);
        let result_const = builder.eval(&with_const, &[0.5]); // 0.5 + 2 + 3 = 5.5
        assert_eq!(result_const, 5.5);

        // Test power operations with constants
        let power_expr = x.clone().pow(2.0.into()) + y.clone().pow(3i32.into());
        let result = builder.eval(&power_expr, &[2.0, 3.0]); // 2² + 3³ = 4 + 27 = 31
        assert_eq!(result, 31.0);
    }
}

// Removed SummationPatternType enum - replaced by SummationPattern in symbolic/summation.rs
// The unified SummationPattern supports decomposition and is more comprehensive

// Convenience methods for f64 expressions
impl TypedBuilderExpr<f64> {
    /// Evaluate a two-variable expression with specific values
    #[must_use]
    pub fn eval_two_vars(&self, x: f64, y: f64) -> f64 {
        self.ast.eval_two_vars(x, y)
    }

    /// Evaluate the expression directly (wrapper for `eval_with_vars`)
    #[must_use]
    pub fn eval(&self, variables: &[f64]) -> f64 {
        self.ast.eval_with_vars(variables)
    }
}

impl<T> Sub<T> for TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self - rhs_expr
    }
}

impl<T> Sub<T> for &TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self - rhs_expr
    }
}

// Reverse scalar operations
impl Sub<TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn sub(self, rhs: TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr - rhs
    }
}

impl Sub<&TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn sub(self, rhs: &TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr - rhs
    }
}

/// Functional summation optimizer
///
/// This provides the core mathematical optimizations for summations:
/// - Sum splitting: Σ(a + b) = Σ(a) + Σ(b)
/// - Factor extraction: Σ(k * f) = k * Σ(f)
/// - Closed-form evaluation for known patterns
pub struct SummationOptimizer;

impl Default for SummationOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SummationOptimizer {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Clean recursive optimization - returns final value directly
    pub fn optimize_summation(
        &self,
        start: i64,
        end: i64,
        expr: ASTRepr<f64>,
    ) -> crate::Result<f64> {
        match expr {
            // Sum splitting: Σ(a + b) = Σ(a) + Σ(b)
            ASTRepr::Add(left, right) => {
                let left_val = self.optimize_summation(start, end, *left)?;
                let right_val = self.optimize_summation(start, end, *right)?;
                Ok(left_val + right_val)
            }

            // Factor extraction: Σ(k * f) = k * Σ(f)
            ASTRepr::Mul(left, right) => {
                if let ASTRepr::Constant(factor) = left.as_ref() {
                    let inner_val = self.optimize_summation(start, end, *right)?;
                    Ok(factor * inner_val)
                } else if let ASTRepr::Constant(factor) = right.as_ref() {
                    let inner_val = self.optimize_summation(start, end, *left)?;
                    Ok(factor * inner_val)
                } else {
                    // No constant factor, fall back to numerical
                    self.evaluate_numerically(start, end, &ASTRepr::Mul(left, right))
                }
            }

            // Constant: Σ(c) = c * n
            ASTRepr::Constant(value) => {
                let n = (end - start + 1) as f64;
                Ok(value * n)
            }

            // Variable (index variable): Σ(i) = sum from start to end
            ASTRepr::Variable(_) => {
                // For any variable, treat as index variable: Σ(i) from start to end
                let sum = (start..=end).map(|i| i as f64).sum::<f64>();
                Ok(sum)
            }

            // Power of index variable: Σ(i^k)
            ASTRepr::Pow(base, exp) => {
                if matches!(base.as_ref(), ASTRepr::Variable(_)) {
                    if let ASTRepr::Constant(k) = exp.as_ref() {
                        self.evaluate_power_sum(start, end, *k)
                    } else {
                        self.evaluate_numerically(start, end, &ASTRepr::Pow(base, exp))
                    }
                } else {
                    self.evaluate_numerically(start, end, &ASTRepr::Pow(base, exp))
                }
            }

            // Fall back to numerical evaluation for complex expressions
            _ => self.evaluate_numerically(start, end, &expr),
        }
    }

    /// Helper method for numerical evaluation fallback
    fn evaluate_numerically(
        &self,
        start: i64,
        end: i64,
        expr: &ASTRepr<f64>,
    ) -> crate::Result<f64> {
        let mut sum = 0.0;
        for i in start..=end {
            let value = self.eval_with_vars(expr, &[i as f64]);
            sum += value;
        }
        Ok(sum)
    }

    /// Helper method for evaluating power sums Σ(i^k)
    fn evaluate_power_sum(&self, start: i64, end: i64, exponent: f64) -> crate::Result<f64> {
        if exponent == 1.0 {
            // Σ(i) from start to end
            let sum = (start..=end).map(|i| i as f64).sum::<f64>();
            Ok(sum)
        } else if exponent == 2.0 {
            // Σ(i²) from start to end
            let sum = (start..=end).map(|i| (i as f64).powi(2)).sum::<f64>();
            Ok(sum)
        } else {
            // Fall back to numerical evaluation for other powers
            let expr = ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(exponent)),
            );
            self.evaluate_numerically(start, end, &expr)
        }
    }

    /// Simple expression evaluation with variables
    fn eval_with_vars(&self, expr: &ASTRepr<f64>, vars: &[f64]) -> f64 {
        match expr {
            ASTRepr::Constant(c) => *c,
            ASTRepr::Variable(idx) => vars.get(*idx).copied().unwrap_or(0.0),
            ASTRepr::Add(left, right) => {
                self.eval_with_vars(left, vars) + self.eval_with_vars(right, vars)
            }
            ASTRepr::Sub(left, right) => {
                self.eval_with_vars(left, vars) - self.eval_with_vars(right, vars)
            }
            ASTRepr::Mul(left, right) => {
                self.eval_with_vars(left, vars) * self.eval_with_vars(right, vars)
            }
            ASTRepr::Div(left, right) => {
                self.eval_with_vars(left, vars) / self.eval_with_vars(right, vars)
            }
            ASTRepr::Pow(left, right) => {
                let base = self.eval_with_vars(left, vars);
                let exp = self.eval_with_vars(right, vars);
                base.powf(exp)
            }
            ASTRepr::Neg(inner) => -self.eval_with_vars(inner, vars),
            ASTRepr::Sqrt(inner) => self.eval_with_vars(inner, vars).sqrt(),
            ASTRepr::Sin(inner) => self.eval_with_vars(inner, vars).sin(),
            ASTRepr::Cos(inner) => self.eval_with_vars(inner, vars).cos(),
            ASTRepr::Exp(inner) => self.eval_with_vars(inner, vars).exp(),
            ASTRepr::Ln(inner) => self.eval_with_vars(inner, vars).ln(),
            ASTRepr::Sum { .. } => {
                // Fall back to full AST evaluation for Sum expressions
                expr.eval_with_vars(vars)
            }
        }
    }
}

/// Represents different types of summable ranges
#[derive(Debug, Clone)]
pub enum SummableRange {
    /// Mathematical range like 1..=10 for symbolic optimization
    MathematicalRange { start: i64, end: i64 },
    /// Data iteration for runtime values
    DataIteration { values: Vec<f64> },
}

/// Trait for converting different types into summable ranges
pub trait IntoSummableRange {
    fn into_summable(self) -> SummableRange;
}

/// Implementation for mathematical ranges
impl IntoSummableRange for std::ops::RangeInclusive<i64> {
    fn into_summable(self) -> SummableRange {
        SummableRange::MathematicalRange {
            start: *self.start(),
            end: *self.end(),
        }
    }
}

/// Implementation for data vectors
impl IntoSummableRange for Vec<f64> {
    fn into_summable(self) -> SummableRange {
        SummableRange::DataIteration { values: self }
    }
}

/// Implementation for data slices
impl IntoSummableRange for &[f64] {
    fn into_summable(self) -> SummableRange {
        SummableRange::DataIteration {
            values: self.to_vec(),
        }
    }
}

/// Implementation for data vector references
impl IntoSummableRange for &Vec<f64> {
    fn into_summable(self) -> SummableRange {
        SummableRange::DataIteration {
            values: self.clone(),
        }
    }
}

// ============================================================================
// UNIFIED SUMMATION TRAIT - Cross-Context Compatibility
// ============================================================================

/// Unified trait for summation across all contexts (Dynamic, Static, Heterogeneous)
///
/// This trait provides a common interface for summation operations that works
/// with `DynamicContext`, Context64, `HeteroContext16`, and other expression builders.
pub trait SummationContext {
    /// Expression type for this context
    type Expr;

    /// Mathematical index summation: Σᵢ₌ₛᵗᵃʳᵗᵉⁿᵈ f(i)
    ///
    /// Creates symbolic expressions with closed-form optimizations for mathematical ranges.
    /// The index variable `i` takes integer values from start to end (inclusive).
    fn sum_range<F>(&self, range: std::ops::RangeInclusive<i64>, f: F) -> crate::Result<Self::Expr>
    where
        F: Fn(Self::Expr) -> Self::Expr;

    /// Create a variable for use in summation expressions
    fn variable(&self) -> Self::Expr;

    /// Create a constant for use in summation expressions  
    fn constant(&self, value: f64) -> Self::Expr;
}

/// Implementation for `DynamicContext`
impl SummationContext for DynamicContext {
    type Expr = TypedBuilderExpr<f64>;

    fn sum_range<F>(&self, range: std::ops::RangeInclusive<i64>, f: F) -> crate::Result<Self::Expr>
    where
        F: Fn(Self::Expr) -> Self::Expr,
    {
        let start = *range.start();
        let end = *range.end();

        // Mathematical summation - can use closed-form optimizations
        let i_var = self.var(); // This becomes Variable(0) in the AST
        let expr = f(i_var);
        let ast = expr.into();

        let optimizer = SummationOptimizer::new();
        let result_value = optimizer.optimize_summation(start, end, ast)?;
        Ok(self.constant(result_value))
    }

    fn variable(&self) -> Self::Expr {
        self.var()
    }

    fn constant(&self, value: f64) -> Self::Expr {
        DynamicContext::constant(self, value)
    }
}

// Simple helper method for creating constants easily
impl DynamicContext {
    /// Create a constant expression (shorthand helper)
    pub fn const_<T: NumericType>(&self, value: T) -> TypedBuilderExpr<T> {
        self.constant(value)
    }
}
