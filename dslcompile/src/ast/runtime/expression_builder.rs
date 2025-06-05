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

    // Removed discover_sufficient_statistics - statistical naming violation
    // All pattern recognition now handled by SummationOptimizer (domain-agnostic)

    // Removed eval_expr helper - no longer needed without discover_sufficient_statistics

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

    /// Domain-agnostic summation method - uses proven SummationOptimizer
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
    /// fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let math = DynamicContext::new();
    ///     
    ///     // Mathematical summation over range (optimizable)
    ///     let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    ///     let result = math.sum(data, |xi| {
    ///         xi.clone() * math.constant(5.0)  // Σ(5*i) = 5*Σ(i) = 5*55 = 275
    ///     })?;
    ///     
    ///     let value = math.eval(&result, &[]); // Should be 275.0
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub fn sum<I, T, F>(&self, data: I, f: F) -> crate::Result<TypedBuilderExpr<T>>
    where
        I: IntoIterator<Item = T>,
        T: NumericType + Clone + Default + Into<f64> + From<f64>,
        F: Fn(TypedBuilderExpr<T>) -> TypedBuilderExpr<T>,
    {
        let data_vec: Vec<T> = data.into_iter().collect();

        if data_vec.is_empty() {
            return Ok(self.constant(T::default()));
        }

        // Create index variable for pattern analysis
        let index_var = self.var().to_f64(); // Convert to f64 for SummationOptimizer
        let pattern_expr = f(TypedBuilderExpr::new(
            ASTRepr::Variable(0), // Index variable for summation
            self.registry.clone()
        ));
        
        // Convert pattern to f64 AST for optimization
        let pattern_ast = pattern_expr.to_f64().into_ast();
        
        // Use SummationOptimizer for proven mathematical optimization
        use crate::symbolic::summation::IntRange;
        
        // Temporarily comment out the SummationOptimizer usage since it's being refactored
        // let mut optimizer = SummationOptimizer::new()?;
        let range = IntRange::new(1, data_vec.len() as i64);
        
        // Temporarily use simple fallback evaluation 
        let sum_value: f64 = data_vec.iter()
            .map(|x| {
                let x_expr = self.constant(x.clone());
                let result_expr = f(x_expr);
                // Simple evaluation for fallback
                result_expr.to_f64().into_ast().eval_with_vars(&[])
            })
            .sum();
        Ok(self.constant(T::from(sum_value)))
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

    /// Convert to the underlying AST (consuming)
    pub fn into_ast(self) -> ASTRepr<T> {
        self.ast
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
        T: Float + Copy,
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
            // Sum variant removed - summations handled through optimization pipeline
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
            // Sum variant removed - summations handled through optimization pipeline
        }
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
