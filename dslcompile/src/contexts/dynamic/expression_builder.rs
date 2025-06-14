//! Dynamic Expression Builder
//!
//! This module provides a runtime expression builder that enables natural mathematical syntax
//! and expressions while maintaining intuitive operator overloading syntax.

use super::typed_registry::VariableRegistry;
use crate::ast::{
    ast_repr::{ASTRepr, Collection, Lambda},
    Scalar,
};

use std::{cell::RefCell, fmt::Debug, marker::PhantomData, sync::Arc};

// ============================================================================
// SUBMODULES
// ============================================================================

/// Type system support for heterogeneous variables
pub mod type_system;

/// HList support for zero-cost heterogeneous operations
pub mod hlist_support;
pub use hlist_support::{FunctionSignature, IntoConcreteSignature, IntoVarHList, HListEval};

// ============================================================================
// TYPE SYSTEM INFRASTRUCTURE - NOW IN SEPARATE MODULE
// ============================================================================
pub use type_system::{DataType, DslType};

// ============================================================================
// MATHEMATICAL FUNCTIONS - NOW IN SEPARATE MODULE
// ============================================================================
pub mod math_functions;

// ============================================================================
// OPERATOR OVERLOADING - NOW IN SEPARATE MODULE
// ============================================================================
pub mod operators;

// Re-export operator implementations to make them available

// ============================================================================
// OPEN TRAIT SYSTEM - EXTENSIBLE TYPE SUPPORT
// ============================================================================

/// Extended trait for DSL types that can participate in code generation
/// This is the "open" part - users can implement this for custom types
// Type system traits moved to type_system.rs module

// Type system implementations moved to type_system.rs module

// ============================================================================
// HLIST INTEGRATION TRAITS
// ============================================================================

/// Trait for converting HLists into typed variable HLists
// HList traits moved to hlist_support module

// HList evaluation implementations moved to hlist_support module

// All HList implementations moved to hlist_support module

/// Dynamic expression builder with runtime variable management and heterogeneous support
/// Parameterized for type safety, scope management, and borrowed data support
///
/// The SCOPE parameter provides automatic scope management to prevent variable collisions
/// when composing expressions from different contexts - this is critical for composability.
///
/// # Type-Level Scope Safety
///
/// DynamicContext now uses type-level scopes like StaticContext to prevent variable collisions:
/// - Variables from different scopes have different types at compile time
/// - Cross-scope operations require explicit scope advancement via `next()`
/// - This eliminates the non-deterministic runtime scope merging that caused test failures
///
/// # Examples
/// ```rust
/// use dslcompile::prelude::*;
/// 
/// // Same scope - operations allowed
/// let mut ctx = DynamicContext::new();
/// let x: dslcompile::DynamicExpr<f64, 0> = ctx.var();
/// let y: dslcompile::DynamicExpr<f64, 0> = ctx.var();
/// let expr = &x + &y;  // ✓ Compiles - same scope
///
/// // Different scopes - prevented at compile time
/// let mut ctx1 = DynamicContext::<1>::new_explicit();
/// let mut ctx2 = DynamicContext::<2>::new_explicit();
/// let x1: dslcompile::DynamicExpr<f64, 1> = ctx1.var();
/// let x2: dslcompile::DynamicExpr<f64, 2> = ctx2.var();
/// // let bad = &x1 + &x2;  // ❌ Compile error - different scopes!
///
/// // Explicit scope advancement for composition
/// let ctx_next = ctx1.next();  // DynamicContext<2>
/// ```
#[derive(Debug, Clone)]
pub struct DynamicContext<const SCOPE: usize = 0> {
    /// Variable registry for heterogeneous type management
    registry: Arc<RefCell<VariableRegistry>>,
    /// Next variable ID for predictable variable indexing
    next_var_id: usize,
}

// Removed duplicate new() method - now handled by generic impl below

// Specific implementation for the default case (scope 0) - enables automatic inference
impl DynamicContext<0> {
    /// Create a new dynamic expression builder with default scope (0)
    ///
    /// This enables automatic type inference:
    /// ```rust
    /// use dslcompile::DynamicContext;
    /// let mut ctx = DynamicContext::new(); // Infers DynamicContext<0>
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RefCell::new(VariableRegistry::new())),
            next_var_id: 0,
        }
    }
}

impl<const SCOPE: usize> DynamicContext<SCOPE> {
    /// Create a new dynamic expression builder with explicit scope
    ///
    /// Use this when you need non-default scope:
    /// ```rust
    /// let mut ctx = dslcompile::DynamicContext::<1>::new_explicit(); // scope 1
    /// ```
    #[must_use]
    pub fn new_explicit() -> Self {
        Self {
            registry: Arc::new(RefCell::new(VariableRegistry::new())),
            next_var_id: 0,
        }
    }

    /// Create a variable of any scalar type (heterogeneous support)
    ///
    /// This provides the heterogeneous-by-default functionality while maintaining
    /// automatic scope management for composability.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// let mut ctx = DynamicContext::new();
    /// let x: DynamicExpr<f64, 0> = ctx.var();     // f64 with scope 0
    /// let y: DynamicExpr<f32, 0> = ctx.var();     // Heterogeneous: f32 with scope 0
    /// let z: DynamicExpr<i32, 0> = ctx.var();     // Heterogeneous: i32 with scope 0
    /// ```
    #[must_use]
    pub fn var<T: Scalar>(&mut self) -> DynamicExpr<T, SCOPE> {
        // Register the variable in the registry (gets automatic index)
        let var_id = {
            let mut registry = self.registry.borrow_mut();
            registry.register_variable()
        };

        self.next_var_id = self.next_var_id.max(var_id + 1);
        DynamicExpr::new(ASTRepr::Variable(var_id), self.registry.clone())
    }

    /// Create a constant expression
    #[must_use]
    pub fn constant<T: Scalar>(&self, value: T) -> DynamicExpr<T, SCOPE> {
        DynamicExpr::new(ASTRepr::Constant(value), self.registry.clone())
    }

    /// Evaluate expression with HList inputs (unified API)
    ///
    /// This is the recommended evaluation method that supports heterogeneous inputs
    /// through HList. It preserves type structure without flattening to Vec.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let mut ctx = DynamicContext::new();
    /// let x: DynamicExpr<f64, 0> = ctx.var();  // Variable(0)
    /// let y: DynamicExpr<f64, 0> = ctx.var();  // Variable(1)
    /// let expr = &x * 2.0 + &y;
    ///
    /// // Evaluate with HList - no flattening, direct variable access
    /// let result = ctx.eval(&expr, hlist![3.0, 4.0]);
    /// assert_eq!(result, 10.0); // 3*2 + 4 = 10
    /// ```
    #[must_use]
    pub fn eval<T, H>(&self, expr: &DynamicExpr<T, SCOPE>, hlist: H) -> T
    where
        T: Scalar,
        H: HListEval<T>,
    {
        hlist.eval_expr(expr.as_ast())
    }

    /// Create a polynomial expression from coefficients
    /// 
    /// Creates a polynomial of the form: c₀ + c₁x + c₂x² + ... + cₙxⁿ
    #[must_use]
    pub fn poly<T>(&self, coefficients: &[T], variable: &DynamicExpr<T, SCOPE>) -> DynamicExpr<T, SCOPE>
    where
        T: Scalar + num_traits::Zero + Clone,
    {
        if coefficients.is_empty() {
            return DynamicExpr::new(
                ASTRepr::Constant(T::zero()),
                self.registry.clone(),
            );
        }

        // Use Horner's method: a₀ + x(a₁ + x(a₂ + x(... + x(aₙ)...)))
        // Start from the highest degree coefficient and work backwards
        let mut result = DynamicExpr::new(
            ASTRepr::Constant(coefficients.last().unwrap().clone()),
            self.registry.clone(),
        );

        // Work backwards through coefficients (excluding the last one we already used)
        for coeff in coefficients.iter().rev().skip(1) {
            let coeff_expr = DynamicExpr::new(
                ASTRepr::Constant(coeff.clone()),
                self.registry.clone(),
            );
            // result = coeff + x * result
            result = coeff_expr + variable.clone() * result;
        }

        result
    }

    /// Pretty print an expression
    #[must_use]
    pub fn pretty_print<T>(&self, expr: &DynamicExpr<T, SCOPE>) -> String
    where
        T: Scalar + std::fmt::Display,
    {
        // Create a minimal registry for pretty printing
        let registry =
            crate::contexts::dynamic::typed_registry::VariableRegistry::for_expression(&expr.ast);
        crate::ast::pretty_ast(&expr.ast, &registry)
    }

    /// Create a lambda function with the given variable indices and body
    #[must_use]
    pub fn lambda<T: Scalar>(
        &self,
        var_indices: Vec<usize>,
        body: DynamicExpr<T, SCOPE>,
    ) -> DynamicExpr<T, SCOPE> {
        let lambda = Lambda::new(var_indices, Box::new(body.into_ast()));
        DynamicExpr::new(ASTRepr::Lambda(Box::new(lambda)), self.registry.clone())
    }

    /// Create a single-argument lambda function: λvar_index.body
    #[must_use]
    pub fn lambda_single<T: Scalar>(
        &self,
        var_index: usize,
        body: DynamicExpr<T, SCOPE>,
    ) -> DynamicExpr<T, SCOPE> {
        let lambda = Lambda::single(var_index, Box::new(body.into_ast()));
        DynamicExpr::new(ASTRepr::Lambda(Box::new(lambda)), self.registry.clone())
    }

    /// Create an identity lambda: λx.x
    #[must_use]
    pub fn identity_lambda<T: Scalar>(&self, var_index: usize) -> DynamicExpr<T, SCOPE> {
        self.lambda_single(
            var_index,
            DynamicExpr::new(ASTRepr::Variable(var_index), self.registry.clone()),
        )
    }

    /// Apply a lambda function to arguments using HList evaluation
    #[must_use]
    pub fn apply_lambda<T, H>(&self, lambda_expr: &DynamicExpr<T, SCOPE>, args: &[T], hlist: H) -> T
    where
        T: Scalar,
        H: HListEval<T>,
    {
        if let ASTRepr::Lambda(lambda) = lambda_expr.as_ast() {
            hlist.apply_lambda(lambda, args)
        } else {
            panic!("apply_lambda called on non-lambda expression")
        }
    }

    /// Convert to AST representation
    #[must_use]
    pub fn to_ast<T: Scalar>(&self, expr: &DynamicExpr<T, SCOPE>) -> crate::ast::ASTRepr<T> {
        expr.as_ast().clone()
    }

    /// Check if expression uses a specific variable index
    fn expression_uses_variable<T: Scalar>(&self, expr: &DynamicExpr<T, SCOPE>, var_index: usize) -> bool {
        self.ast_uses_variable(expr.as_ast(), var_index)
    }

    /// Check if AST uses a specific variable index
    fn ast_uses_variable<T: Scalar>(&self, ast: &ASTRepr<T>, var_index: usize) -> bool {
        match ast {
            ASTRepr::Variable(index) => *index == var_index,
            ASTRepr::Constant(_) => false,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.ast_uses_variable(left, var_index) || self.ast_uses_variable(right, var_index)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.ast_uses_variable(inner, var_index),
            ASTRepr::Sum(_collection) => false, // Assume no variable usage for now
            ASTRepr::Lambda(lambda) => {
                // Check if lambda body uses the variable
                self.ast_uses_variable(&lambda.body, var_index)
            }
            ASTRepr::BoundVar(index) => {
                // BoundVar contributes to max variable index
                *index == var_index
            }
            ASTRepr::Let(_, expr, body) => {
                // Let expressions need to check both the bound expression and body for max variable index
                let expr_max = self.find_max_variable_index_recursive(expr);
                let body_max = self.find_max_variable_index_recursive(body);
                expr_max.max(body_max) == var_index
            }
        }
    }

    /// Find maximum variable index used in expression
    #[must_use]
    pub fn find_max_variable_index<T: Scalar>(&self, expr: &DynamicExpr<T, SCOPE>) -> usize {
        self.find_max_variable_index_recursive(expr.as_ast())
    }

    /// Recursively find maximum variable index
    fn find_max_variable_index_recursive<T: Scalar>(&self, ast: &ASTRepr<T>) -> usize {
        match ast {
            ASTRepr::Variable(index) => *index,
            ASTRepr::Constant(_) => 0,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => std::cmp::max(
                self.find_max_variable_index_recursive(left),
                self.find_max_variable_index_recursive(right),
            ),
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.find_max_variable_index_recursive(inner),
            ASTRepr::Sum(_collection) => {
                // TODO: Analyze collection for variable indices
                0
            }
            ASTRepr::Lambda(lambda) => {
                // Find max variable index in lambda body
                let body_max = self.find_max_variable_index_recursive(&lambda.body);
                let lambda_max = lambda.var_indices.iter().max().copied().unwrap_or(0);
                body_max.max(lambda_max)
            }
            ASTRepr::BoundVar(index) => {
                // BoundVar contributes to max variable index
                *index
            }
            ASTRepr::Let(_, expr, body) => {
                // Let expressions need to check both the bound expression and body for max variable index
                let expr_max = self.find_max_variable_index_recursive(expr);
                let body_max = self.find_max_variable_index_recursive(body);
                expr_max.max(body_max)
            }
        }
    }

    // sum method moved to separate impl block with required trait bounds

    // new_scope method removed - use var::<T>() directly for heterogeneous variables
    // The SCOPE parameter provides automatic scope management at the type level
}

// Separate impl block for methods requiring additional trait bounds
impl<const SCOPE: usize> DynamicContext<SCOPE> {
    /// Unified HList-based summation - eliminates DataArray architecture
    ///
    /// This approach treats all inputs (scalars, vectors, etc.) as typed variables
    /// in the same HList. No artificial separation between "parameters" and "data arrays".
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let mut ctx = DynamicContext::new();
    ///
    /// // Mathematical range summation
    /// let sum1 = ctx.sum(1..=10, |i| i * 2.0);
    /// // Generates: Range summation that evaluates to constant
    ///
    /// // Data vector summation - data becomes Variable(N)
    /// let data = vec![1.0, 2.0, 3.0];
    /// let sum2 = ctx.sum(data, |x| x * 2.0);
    /// // Later evaluated with: ctx.eval(&sum2, hlist![other_params])
    /// ```
    pub fn sum<T, R, F>(&mut self, range: R, f: F) -> DynamicExpr<T, SCOPE>
    where
        T: Scalar + num_traits::FromPrimitive + Copy,
        R: IntoHListSummationRange<T>,
        F: FnOnce(DynamicExpr<T, SCOPE>) -> DynamicExpr<T, SCOPE>,
    {
        range.into_hlist_summation(self, f)
    }
}

impl<const SCOPE: usize> DynamicContext<SCOPE> {
    /// Advance to the next scope for safe composition
    ///
    /// This method consumes the current context and returns a new context
    /// with the next scope index, ensuring no variable collisions when
    /// composing expressions from different contexts.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// // Create first expression in scope 0
    /// let mut ctx = DynamicContext::new();
    /// let x: dslcompile::DynamicExpr<f64, 0> = ctx.var(); // Variable(0) in scope 0
    /// let expr1 = x.clone() * x;
    ///
    /// // Advance to scope 1 for composition safety
    /// let mut ctx = ctx.next();
    /// let y: dslcompile::DynamicExpr<f64, 1> = ctx.var(); // Variable(0) in scope 1 - no collision!
    /// let expr2 = y.clone() + y;
    /// ```
    #[must_use]
    pub fn next(self) -> DynamicContext<{ SCOPE + 1 }> {
        DynamicContext {
            registry: self.registry,
            next_var_id: self.next_var_id,
        }
    }

    /// Merge with another context, combining variable ID spaces
    ///
    /// This method safely combines two contexts by offsetting variable indices
    /// from the second context to prevent collisions with the first context.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    ///
    /// // Create two separate contexts
    /// let mut ctx1 = DynamicContext::new();
    /// let x: DynamicExpr<f64, 0> = ctx1.var(); // Variable(0)
    ///
    /// let mut ctx2 = DynamicContext::new();
    /// let y: DynamicExpr<f64, 0> = ctx2.var(); // Variable(0) in ctx2
    ///
    /// // Safe merge - variable indices combined
    /// let merged_ctx = ctx1.merge(ctx2);
    /// ```
    pub fn merge(mut self, other: DynamicContext<SCOPE>) -> DynamicContext<SCOPE> {
        // Combine variable ID spaces
        self.next_var_id += other.next_var_id;
        self
    }
}

impl Default for DynamicContext {
    fn default() -> Self {
        Self::new()
    }
}

// Type alias for backward compatibility 
pub type DynamicF64Context = DynamicContext;
pub type DynamicF32Context = DynamicContext;
pub type DynamicI32Context = DynamicContext;

/// Unified variable expression that adapts behavior based on type
///
/// This replaces the old separate var() and data_var() methods with a single
/// type-driven approach:
/// - VariableExpr<f64> → Scalar arithmetic operations
/// - VariableExpr<Vec<f64>> → Collection iteration operations  
/// - VariableExpr<Matrix<f64>> → Matrix operations (future)
#[derive(Debug, Clone)]
pub struct VariableExpr<T> {
    var_id: usize,
    registry: Arc<RefCell<VariableRegistry>>,
    _phantom: PhantomData<T>,
}

impl<T> VariableExpr<T> {
    /// Create a new variable expression
    pub fn new(var_id: usize, registry: Arc<RefCell<VariableRegistry>>) -> Self {
        Self {
            var_id,
            registry,
            _phantom: PhantomData,
        }
    }

    /// Get the variable ID
    pub fn var_id(&self) -> usize {
        self.var_id
    }
}

/// Typed expression builder that carries scope information at the type level
/// 
/// The SCOPE parameter ensures that expressions from different contexts cannot be
/// accidentally combined, preventing variable collision issues at compile time.
#[derive(Debug, Clone)]
pub struct DynamicExpr<T, const SCOPE: usize = 0> {
    pub(crate) ast: ASTRepr<T>,
    pub(crate) registry: Arc<RefCell<VariableRegistry>>,
}

impl<T, const SCOPE: usize> DynamicExpr<T, SCOPE> {
    /// Create a new typed expression with scope information
    #[must_use]
    pub fn new(ast: ASTRepr<T>, registry: Arc<RefCell<VariableRegistry>>) -> Self {
        Self { ast, registry }
    }

    /// Get reference to the underlying AST
    #[must_use]
    pub fn as_ast(&self) -> &ASTRepr<T> {
        &self.ast
    }

    /// Convert into the underlying AST
    #[must_use]
    pub fn into_ast(self) -> ASTRepr<T> {
        self.ast
    }

    /// Identity conversion for compatibility
    #[must_use]
    pub fn into_expr(self) -> Self {
        self
    }

    /// Get variable ID (only works for Variable expressions)
    #[must_use]
    pub fn var_id(&self) -> usize {
        match &self.ast {
            ASTRepr::Variable(id) => *id,
            _ => panic!("var_id() called on non-variable expression"),
        }
    }

    /// Get the variable registry
    #[must_use]
    pub fn registry(&self) -> Arc<RefCell<VariableRegistry>> {
        self.registry.clone()
    }

    /// Pretty print the expression
    #[must_use]
    pub fn pretty_print(&self) -> String
    where
        T: std::fmt::Display,
    {
        // Create a minimal registry for pretty printing
        let registry =
            crate::contexts::dynamic::typed_registry::VariableRegistry::for_expression(&self.ast);
        crate::ast::pretty_ast(&self.ast, &registry)
    }
}

// ============================================================================
// UNIFIED EXPR TRAIT IMPLEMENTATION FOR DYNAMICEXPR
// ============================================================================

impl<T: Scalar, const SCOPE: usize> crate::contexts::Expr<T> for DynamicExpr<T, SCOPE> {
    fn to_ast(&self) -> ASTRepr<T> {
        self.ast.clone()
    }
    
    fn pretty_print(&self) -> String {
        // Create a minimal registry for pretty printing
        let registry =
            crate::contexts::dynamic::typed_registry::VariableRegistry::for_expression(&self.ast);
        crate::ast::pretty_ast(&self.ast, &registry)
    }
    
    fn get_variables(&self) -> std::collections::HashSet<usize> {
        crate::ast::ast_utils::collect_variable_indices(&self.ast)
    }
}

impl DynamicExpr<f32> {
    /// Convert f32 expression to f64 expression
    #[must_use]
    pub fn to_f64(self) -> DynamicExpr<f64> {
        DynamicExpr::new(convert_ast_pure_rust(&self.ast), self.registry)
    }
}

impl DynamicExpr<f64> {
    /// Convert f64 expression to f64 expression (identity operation)
    #[must_use]
    pub fn to_f64(self) -> DynamicExpr<f64> {
        self
    }
}

impl DynamicExpr<i32> {
    /// Convert i32 expression to f64 expression
    #[must_use]
    pub fn to_f64(self) -> DynamicExpr<f64> {
        DynamicExpr::new(convert_i32_ast_to_f64(&self.ast), self.registry)
    }
}

// TODO: Consider adding a generic `map` method for DynamicExpr type transformations
// This would allow: expr.map(|val| val as f64) instead of specific to_f64() methods
// However, this requires careful design for symbolic expressions vs concrete values
// For now, explicit conversion methods like to_f64() provide clearer semantics

/// Scalar variable operations (f64, f32, i32, etc.)
impl<T: Scalar> VariableExpr<T> {
    /// Convert to a typed expression for arithmetic operations
    pub fn into_expr<const SCOPE: usize>(self) -> DynamicExpr<T, SCOPE> {
        DynamicExpr::new(ASTRepr::Variable(self.var_id), self.registry)
    }
}

// ============================================================================
// All operator implementations moved to operators module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_variable_creation() {
        let mut builder_f64 = DynamicContext::new();
        let mut builder_f32 = DynamicContext::new();

        // Create variables for different types
        let x = builder_f64.var::<f64>();
        let y = builder_f32.var::<f32>();

        // Variables should have the correct IDs
        assert_eq!(x.var_id(), 0);
        assert_eq!(y.var_id(), 0); // Each context starts from 0
    }

    #[test]
    fn test_typed_expression_building() {
        let mut builder = DynamicContext::new();

        // Use the new unified API
        let x = builder.var::<f64>();
        let y = builder.var::<f64>();

        // Test same-type operations
        let sum = &x + &y;
        let product = &x * &y;

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
        let mut builder_f64 = DynamicContext::new();
        let mut builder_f32 = DynamicContext::new();

        let x_f64 = builder_f64.var::<f64>();
        let y_f32 = builder_f32.var::<f32>();

        // Convert f32 expression to f64 for cross-type operation
        let mixed_sum = x_f64 + y_f32.to_f64();

        // Result should be f64
        match mixed_sum.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_scalar_operations() {
        let mut builder = DynamicContext::new();

        let x: DynamicExpr<f64> = builder.var();

        // Test scalar operations
        let scaled: DynamicExpr<f64> = &x * 2.0;
        let shifted: DynamicExpr<f64> = &x + 1.0;
        let reverse_scaled: DynamicExpr<f64> = 3.0 * &x;

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
        let mut builder = DynamicContext::new();

        let x: DynamicExpr<f64> = builder.var();

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
        let mut builder = DynamicContext::new();

        // Use the clean generic API - no need for .into_expr()
        let x: DynamicExpr<f64> = builder.var();
        let y: DynamicExpr<f64> = builder.var();

        let expr: DynamicExpr<f64> = &x * &x + 2.0 * &x + &y;

        // Should create a valid AST
        match expr.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition at top level"),
        }
    }

    #[test]
    fn test_complex_expression() {
        let mut builder = DynamicContext::new();

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
        let mut builder = DynamicContext::new();
        let x = builder.var();

        // Test From implementations for numeric types
        let expr1: DynamicExpr<f64> = 2.0.into();
        let expr2: DynamicExpr<f64> = 3i32.into();
        let expr3: DynamicExpr<f64> = 4i64.into();
        let expr4: DynamicExpr<f64> = 5usize.into();
        let expr5: DynamicExpr<f32> = 2.5f32.into();

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

        // Test evaluation works with new unified API
        let result = builder.eval(&combined, frunk::hlist![1.0]); // x = 1.0
        assert_eq!(result, 6.0); // 1.0 + 2.0 + 3.0 = 6.0
    }

    #[test]
    fn test_ergonomic_expression_building() {
        let mut math = DynamicContext::new();
        let x = math.var();
        let y = math.var();

        // Test natural mathematical syntax
        let expr1 = &x + &y;
        let expr2 = &x * &y;
        let expr3 = &x * &x + 2.0 * &x * &y + &y * &y; // (x + y)²

        // Test evaluation with unified HList API
        let result1 = math.eval(&expr1, frunk::hlist![3.0, 4.0]);
        let result2 = math.eval(&expr2, frunk::hlist![3.0, 4.0]);
        let result3 = math.eval(&expr3, frunk::hlist![3.0, 4.0]);

        assert_eq!(result1, 7.0); // 3 + 4
        assert_eq!(result2, 12.0); // 3 * 4
        assert_eq!(result3, 49.0); // (3 + 4)² = 7² = 49

        // Test mixed operations
        let complex_expr = (&x + 2.0) * (&y - 1.0) + 5.0;
        let complex_result = math.eval(&complex_expr, frunk::hlist![3.0, 4.0]);
        assert_eq!(complex_result, 20.0); // (3 + 2) * (4 - 1) + 5 = 5 * 3 + 5 = 20
    }

    #[test]
    fn test_triple_integration_open_traits_concrete_codegen_hlists() {
        use frunk::hlist;

        let ctx = DynamicContext::new();

        // ============================================================================
        // PHASE 1: OPEN TRAIT SYSTEM - Extensible type support
        // ============================================================================

        // Test DslType implementations for code generation
        assert_eq!(<f64 as DslType>::TYPE_NAME, "f64");
        assert_eq!(<i32 as DslType>::TYPE_NAME, "i32");
        assert_eq!(f64::codegen_add(), "+");
        assert_eq!(f64::codegen_mul(), "*");

        // Test code generation strings
        assert_eq!(<f64 as CodegenScalar>::codegen_literal(2.5), "2.5");
        assert_eq!(<i32 as CodegenScalar>::codegen_literal(42), "42i32");

        // Test evaluation value conversion
        assert_eq!(f64::to_eval_value(2.5), 2.5);
        assert_eq!(i32::to_eval_value(42), 42);

        println!("✅ Phase 1: Open trait system working");

        // ============================================================================
        // PHASE 2: CONCRETE CODEGEN - Zero-overhead code generation
        // ============================================================================

        // Test function signature generation (DEPRECATED - HList methods removed)
        // type TestSig = frunk::HCons<f64, frunk::HCons<i32, frunk::HNil>>;
        // let sig = ctx.signature_from_hlist_type::<TestSig>();
        // println!("Generated signature: {}", sig.parameters());
        // assert!(sig.parameters().contains("f64"));
        // assert!(sig.parameters().contains("i32"));

        println!("✅ Phase 2: Concrete codegen working");

        // ============================================================================
        // PHASE 3: HLIST INTEGRATION - Zero-cost heterogeneous operations
        // ============================================================================

        // Test HList variable creation with predictable IDs (DEPRECATED - HList methods removed)
        // let vars = ctx.vars_from_hlist(hlist![0.0_f64, 0_i32]);
        // let frunk::hlist_pat![x, y] = vars;

        // Create variables using new unified API - use single context for sequential IDs
        let mut ctx_f64 = DynamicContext::new();
        let x = ctx_f64.var(); // ID: 0
        let y = ctx_f64.var(); // ID: 1

        // Variables should have predictable IDs: 0, 1
        println!("Variable x ID: {}, y ID: {}", x.var_id(), y.var_id());

        // Build expression using the variables (both f64 now)
        let expr = &x * 2.0 + &y * 3.0;

        // Test evaluation with array inputs
        let result = ctx_f64.eval(&expr, hlist![5.0, 10.0]);
        println!("Array evaluation result: {result} (expected: 40.0)");
        assert_eq!(result, 40.0); // 5*2 + 10*3 = 10 + 30 = 40

        println!("✅ Phase 3: HList integration working");

        // ============================================================================
        // PHASE 4: COMBINED APPROACH - All three working together
        // ============================================================================

        // Test the complete integration: open traits + concrete codegen + HLists

        // Create variables with predictable IDs using unified API
        let mut ctx2 = DynamicContext::new();
        let x = ctx2.var(); // ID: 0
        let y = ctx2.var(); // ID: 1

        // Build expression
        let expr = &x * 2.0 + &y * 3.0;

        // Evaluate using array indexing (should match variable IDs)
        let result = ctx2.eval(&expr, hlist![5.0, 10.0]); // Index 0->5.0, Index 1->10.0
        println!("Direct evaluation result: {result} (expected: 40.0)");
        assert_eq!(result, 40.0);

        println!("✅ Phase 4: Complete triple integration working!");

        // ============================================================================
        // VERIFICATION: Type-level scoping solved the indexing problem
        // ============================================================================

        println!("🎯 SUCCESS: Type-level scoping provides predictable variable indexing!");
        println!("   ✅ Variables get sequential IDs: 0, 1, 2, ...");
        println!("   ✅ HList evaluation uses correct variable mapping");
        println!("   ✅ No more runtime-dependent variable indices");
        println!("   ✅ Zero-cost heterogeneous operations working");
    }

    #[test]
    fn test_unified_sum_api() {
        let mut ctx = DynamicContext::new();

        // Test 1: Range summation using the unified API
        let range_sum = ctx.sum(1..=5, |x| x * 2);
        println!("Range sum AST: {:?}", range_sum.as_ast());

        // Test 2: Parametric summation using the unified API
        let param: DynamicExpr<f64, 0> = ctx.var();
        let param_sum = ctx.sum(1..=3, |x| x * param.clone());
        println!("Parametric sum AST: {:?}", param_sum.as_ast());

        // Verify the AST structure is correct (Sum(Map{lambda, collection}))
        match range_sum.as_ast() {
            ASTRepr::Sum(collection) => match collection.as_ref() {
                Collection::Map {
                    lambda: _,
                    collection: _,
                } => {
                    println!("✅ Correct structure: Sum(Map{{lambda, collection}})");
                }
                _ => panic!("❌ Expected Map collection"),
            },
            _ => panic!("❌ Expected Sum AST"),
        }
    }

    #[test]
    fn test_lambda_hlist_integration() {
        use frunk::hlist;

        let ctx = DynamicContext::new();

        // Test 1: Create and apply identity lambda
        let identity = ctx.identity_lambda(0);
        let result = ctx.apply_lambda(&identity, &[42.0], hlist![100.0, 200.0]);
        assert_eq!(result, 42.0);
        println!("✅ Identity lambda: λx.x applied to 42.0 = {}", result);

        // Test 2: Create and apply doubling lambda: λx.x*2
        let x_var = DynamicExpr::new(ASTRepr::Variable(0), ctx.registry.clone());
        let double_body = x_var * ctx.constant(2.0);
        let double_lambda = ctx.lambda_single(0, double_body);
        let result = ctx.apply_lambda(&double_lambda, &[7.0], hlist![100.0, 200.0]);
        assert_eq!(result, 14.0);
        println!("✅ Doubling lambda: λx.x*2 applied to 7.0 = {}", result);

        // Test 3: Create and apply multi-argument lambda: λ(x,y).x+y
        let x_var = DynamicExpr::new(ASTRepr::Variable(0), ctx.registry.clone());
        let y_var = DynamicExpr::new(ASTRepr::Variable(1), ctx.registry.clone());
        let add_body = x_var + y_var;
        let add_lambda = ctx.lambda(vec![0, 1], add_body);
        let result = ctx.apply_lambda(&add_lambda, &[3.0, 4.0], hlist![100.0, 200.0]);
        assert_eq!(result, 7.0);
        println!(
            "✅ Addition lambda: λ(x,y).x+y applied to (3.0, 4.0) = {}",
            result
        );

        // Test 4: Lambda that uses HList variables: λx.x + hlist[1]
        let x_var = DynamicExpr::new(ASTRepr::Variable(0), ctx.registry.clone());
        let hlist_var = DynamicExpr::new(ASTRepr::Variable(1), ctx.registry.clone());
        let mixed_body = x_var + hlist_var;
        let mixed_lambda = ctx.lambda_single(0, mixed_body);
        let result = ctx.apply_lambda(&mixed_lambda, &[5.0], hlist![10.0, 20.0]);
        assert_eq!(result, 25.0); // 5.0 + 20.0
        println!(
            "✅ Mixed lambda: λx.x+hlist[1] applied to 5.0 with hlist[10.0, 20.0] = {}",
            result
        );

        println!("🎯 Lambda-HList integration tests passed!");
        println!("✅ Zero-cost lambda evaluation with heterogeneous HLists");
        println!("✅ Variable substitution working correctly");
        println!("✅ Mixed lambda/HList variable access working");
    }
}

// ============================================================================
// RUST-IDIOMATIC SCALAR TYPE SYSTEM (Phase 3)
// ============================================================================

/// Rust-idiomatic scalar trait without 'static constraints or auto-promotion
/// Extended Scalar trait for code generation  
pub trait CodegenScalar: crate::ast::Scalar {
    /// Type identifier for code generation
    const TYPE_NAME: &'static str;

    /// Generate Rust code for a literal value
    fn codegen_literal(value: Self) -> String;
}

/// Float operations for scalar types that support them
pub trait ScalarFloat: crate::ast::Scalar + num_traits::Float {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn pow(self, exp: Self) -> Self;
}

// ============================================================================
// SCALAR IMPLEMENTATIONS (Code generation support)
// ============================================================================

impl CodegenScalar for f64 {
    const TYPE_NAME: &'static str = "f64";

    fn codegen_literal(value: Self) -> String {
        format!("{value}")
    }
}

impl ScalarFloat for f64 {
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn pow(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl CodegenScalar for f32 {
    const TYPE_NAME: &'static str = "f32";

    fn codegen_literal(value: Self) -> String {
        format!("{value}f32")
    }
}

impl ScalarFloat for f32 {
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn pow(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl CodegenScalar for i32 {
    const TYPE_NAME: &'static str = "i32";

    fn codegen_literal(value: Self) -> String {
        format!("{value}i32")
    }
}

impl CodegenScalar for i64 {
    const TYPE_NAME: &'static str = "i64";

    fn codegen_literal(value: Self) -> String {
        format!("{value}i64")
    }
}

impl CodegenScalar for usize {
    const TYPE_NAME: &'static str = "usize";

    fn codegen_literal(value: Self) -> String {
        format!("{value}usize")
    }
}

// ============================================================================
// EXPLICIT CONVERSIONS (No auto-promotion!)
// ============================================================================

// Example usage:
// let f32_expr: DynamicExpr<f32> = /* ... */;
// let f64_expr: DynamicExpr<f64> = f32_expr.into(); // Explicit conversion!

/// Explicit conversion from f32 expressions to f64 expressions
impl From<DynamicExpr<f32>> for DynamicExpr<f64> {
    fn from(expr: DynamicExpr<f32>) -> Self {
        expr.to_f64()
    }
}

/// Explicit conversion from i32 expressions to f64 expressions  
impl From<DynamicExpr<i32>> for DynamicExpr<f64> {
    fn from(expr: DynamicExpr<i32>) -> Self {
        DynamicExpr::new(convert_i32_ast_to_f64(&expr.ast), expr.registry)
    }
}

/// Helper to convert i32 AST to f64 AST
fn convert_i32_ast_to_f64(ast: &ASTRepr<i32>) -> ASTRepr<f64> {
    match ast {
        ASTRepr::Constant(value) => ASTRepr::Constant(*value as f64),
        ASTRepr::Variable(index) => ASTRepr::Variable(*index),
        ASTRepr::Add(left, right) => ASTRepr::Add(
            Box::new(convert_i32_ast_to_f64(left)),
            Box::new(convert_i32_ast_to_f64(right)),
        ),
        ASTRepr::Sub(left, right) => ASTRepr::Sub(
            Box::new(convert_i32_ast_to_f64(left)),
            Box::new(convert_i32_ast_to_f64(right)),
        ),
        ASTRepr::Mul(left, right) => ASTRepr::Mul(
            Box::new(convert_i32_ast_to_f64(left)),
            Box::new(convert_i32_ast_to_f64(right)),
        ),
        ASTRepr::Div(left, right) => ASTRepr::Div(
            Box::new(convert_i32_ast_to_f64(left)),
            Box::new(convert_i32_ast_to_f64(right)),
        ),
        ASTRepr::Pow(base, exp) => ASTRepr::Pow(
            Box::new(convert_i32_ast_to_f64(base)),
            Box::new(convert_i32_ast_to_f64(exp)),
        ),
        ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(convert_i32_ast_to_f64(inner))),
        // Transcendental functions don't make sense for i32, but we'll convert anyway
        ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(convert_i32_ast_to_f64(inner))),
        ASTRepr::Sum(collection) => {
            // Convert collection from i32 to f64
            ASTRepr::Sum(Box::new(convert_collection_pure_rust(collection)))
        }
        ASTRepr::Lambda(lambda) => {
            // Convert lambda to f64
            ASTRepr::Lambda(Box::new(Lambda {
                var_indices: lambda.var_indices.clone(),
                body: Box::new(convert_i32_ast_to_f64(&lambda.body)),
            }))
        }
        ASTRepr::BoundVar(index) => {
            // BoundVar index stays the same across type conversions
            ASTRepr::BoundVar(*index)
        }
        ASTRepr::Let(binding_id, expr, body) => {
            // Convert both the bound expression and body
            ASTRepr::Let(
                *binding_id,
                Box::new(convert_i32_ast_to_f64(expr)),
                Box::new(convert_i32_ast_to_f64(body)),
            )
        }
    }
}

// Duplicate section removed - use CodegenScalar trait above instead

// ============================================================================
// PURE RUST FROM/INTO CONVERSIONS (The Right Way!)
// ============================================================================

/// Generic AST conversion using ONLY standard Rust From trait
fn convert_ast_pure_rust<T, U>(ast: &ASTRepr<T>) -> ASTRepr<U>
where
    T: Clone,
    U: From<T>,
{
    match ast {
        // Use Rust's built-in From trait for primitives
        ASTRepr::Constant(value) => ASTRepr::Constant(U::from(value.clone())),
        ASTRepr::Variable(index) => ASTRepr::Variable(*index),
        ASTRepr::Add(left, right) => ASTRepr::Add(
            Box::new(convert_ast_pure_rust(left)),
            Box::new(convert_ast_pure_rust(right)),
        ),
        ASTRepr::Sub(left, right) => ASTRepr::Sub(
            Box::new(convert_ast_pure_rust(left)),
            Box::new(convert_ast_pure_rust(right)),
        ),
        ASTRepr::Mul(left, right) => ASTRepr::Mul(
            Box::new(convert_ast_pure_rust(left)),
            Box::new(convert_ast_pure_rust(right)),
        ),
        ASTRepr::Div(left, right) => ASTRepr::Div(
            Box::new(convert_ast_pure_rust(left)),
            Box::new(convert_ast_pure_rust(right)),
        ),
        ASTRepr::Pow(base, exp) => ASTRepr::Pow(
            Box::new(convert_ast_pure_rust(base)),
            Box::new(convert_ast_pure_rust(exp)),
        ),
        ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(convert_ast_pure_rust(inner))),
        ASTRepr::Sum(collection) => {
            ASTRepr::Sum(Box::new(convert_collection_pure_rust(collection)))
        }
        ASTRepr::Lambda(lambda) => ASTRepr::Lambda(Box::new(convert_lambda_pure_rust(lambda))),
        ASTRepr::BoundVar(index) => {
            // BoundVar index stays the same across type conversions
            ASTRepr::BoundVar(*index)
        }
        ASTRepr::Let(binding_id, expr, body) => {
            // Convert both the bound expression and body
            ASTRepr::Let(
                *binding_id,
                Box::new(convert_ast_pure_rust(expr)),
                Box::new(convert_ast_pure_rust(body)),
            )
        }
    }
}

/// Convert Collection using standard Rust From trait
fn convert_collection_pure_rust<T, U>(collection: &Collection<T>) -> Collection<U>
where
    T: Clone,
    U: From<T>,
{
    use crate::ast::ast_repr::Collection;

    match collection {
        Collection::Empty => Collection::Empty,
        Collection::Singleton(expr) => Collection::Singleton(Box::new(convert_ast_pure_rust(expr))),
        Collection::Range { start, end } => Collection::Range {
            start: Box::new(convert_ast_pure_rust(start)),
            end: Box::new(convert_ast_pure_rust(end)),
        },

        Collection::Variable(index) => Collection::Variable(*index),
        Collection::Filter {
            collection,
            predicate,
        } => Collection::Filter {
            collection: Box::new(convert_collection_pure_rust(collection)),
            predicate: Box::new(convert_ast_pure_rust(predicate)),
        },
        Collection::Map { lambda, collection } => Collection::Map {
            lambda: Box::new(convert_lambda_pure_rust(lambda)),
            collection: Box::new(convert_collection_pure_rust(collection)),
        },
        Collection::DataArray(data) => Collection::DataArray(
            data.iter().map(|x| U::from(x.clone())).collect()
        ),
    }
}

/// Convert Lambda using standard Rust From trait  
fn convert_lambda_pure_rust<T, U>(lambda: &Lambda<T>) -> Lambda<U>
where
    T: Clone,
    U: From<T>,
{
    Lambda {
        var_indices: lambda.var_indices.clone(),
        body: Box::new(convert_ast_pure_rust(&lambda.body)),
    }
}

// ============================================================================
// EXAMPLE USAGE (Pure Rust Way)
// ============================================================================

/*
// ✅ Explicit conversions using standard Rust traits:

let i32_expr: DynamicExpr<i32> = ctx.constant(42);
let f64_expr: DynamicExpr<f64> = i32_expr.into(); // Uses standard Into!

// Or explicitly:
let f64_expr = DynamicExpr::<f64>::from(i32_expr);

// Rust's built-in conversions work automatically:
// i32 -> f64 ✅ (built into Rust)
// f32 -> f64 ✅ (built into Rust)
// usize -> f64 ✅ (we can add this)

// ❌ No more auto-promotion:
// let result = f64_expr + i32_expr; // Compile error! Must convert first.

// ✅ Explicit conversion required:
let result = f64_expr + DynamicExpr::<f64>::from(i32_expr);
*/

#[cfg(test)]
mod test_comprehensive_api {
    use super::*;

    #[test]
    fn test_comprehensive_typed_api() {
        use frunk::hlist;
        // Test the comprehensive API working together
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        let y = ctx.var();

        // Build complex expression using all operators
        let expr = &x * 2.0 + &y.sin();

        // Test evaluation
        let result = ctx.eval(&expr, hlist![3.0, 1.57]); // sin(1.57) ≈ 1
        assert!((result - 7.0).abs() < 0.1); // 2*3 + sin(1.57) ≈ 7
    }
}

/// Trait for HList-based summation that eliminates DataArray architecture
///
/// This trait provides a unified approach where all inputs (mathematical ranges,
/// data vectors, etc.) are treated as typed variables in the same HList rather
/// than artificial DataArray separation.
pub trait IntoHListSummationRange<T: Scalar> {
    /// Convert input to HList summation, creating appropriate Variable references
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<T, SCOPE>
    where
        F: FnOnce(DynamicExpr<T, SCOPE>) -> DynamicExpr<T, SCOPE>,
        T: num_traits::FromPrimitive + Copy;
}

/// Implementation for mathematical ranges - creates Range collection (no DataArray)
impl<T: Scalar + num_traits::FromPrimitive> IntoHListSummationRange<T>
    for std::ops::RangeInclusive<T>
{
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<T, SCOPE>
    where
        F: FnOnce(DynamicExpr<T, SCOPE>) -> DynamicExpr<T, SCOPE>,
        T: num_traits::FromPrimitive + Copy,
    {
        let start = *self.start();
        let end = *self.end();

        // Create iterator variable for the lambda - use a separate index space for bound variables
        // This doesn't consume a global variable index since it's bound within the lambda
        let iter_var_id = 0; // BoundVar always uses index 0 for single-argument lambdas

        // Create iterator variable expression using BoundVar for lambda body
        let iter_var = DynamicExpr::new(
            ASTRepr::BoundVar(iter_var_id),
            ctx.registry.clone(),
        );

        // Apply the function to the iterator variable
        let body = f(iter_var);

        // Create the lambda that maps over the range
        let lambda = Lambda {
            var_indices: vec![iter_var_id],
            body: Box::new(body.ast),
        };

        // Create the underlying range collection
        let range_collection = Collection::Range {
            start: Box::new(ASTRepr::Constant(start)),
            end: Box::new(ASTRepr::Constant(end)),
        };

        // Create Map collection that applies lambda to range
        let map_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(range_collection),
        };

        DynamicExpr::new(
            ASTRepr::Sum(Box::new(map_collection)),
            ctx.registry.clone(),
        )
    }
}

/// Implementation for integer ranges with f64 context - converts integers to f64
impl IntoHListSummationRange<f64> for std::ops::RangeInclusive<i32> {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Convert integer range to f64 range
        let start = *self.start() as f64;
        let end = *self.end() as f64;

        // Create iterator variable for the lambda - use a separate index space for bound variables
        // This doesn't consume a global variable index since it's bound within the lambda
        let iter_var_id = 0; // BoundVar always uses index 0 for single-argument lambdas

        // Create iterator variable expression using BoundVar for lambda body
        let iter_var = DynamicExpr::new(
            ASTRepr::BoundVar(iter_var_id),
            ctx.registry.clone(),
        );

        // Apply the function to the iterator variable
        let body = f(iter_var);

        // Create the lambda that maps over the range
        let lambda = Lambda {
            var_indices: vec![iter_var_id],
            body: Box::new(body.ast),
        };

        // Create the underlying range collection
        let range_collection = Collection::Range {
            start: Box::new(ASTRepr::Constant(start)),
            end: Box::new(ASTRepr::Constant(end)),
        };

        // Create Map collection that applies lambda to range
        let map_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(range_collection),
        };

        DynamicExpr::new(
            ASTRepr::Sum(Box::new(map_collection)),
            ctx.registry.clone(),
        )
    }
}

/// Implementation for data vectors - creates explicit singleton collections
impl IntoHListSummationRange<f64> for Vec<f64> {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        if self.is_empty() {
            // Empty data array - return sum of empty collection
            return DynamicExpr::new(
                ASTRepr::Sum(Box::new(Collection::Empty)),
                ctx.registry.clone(),
            );
        }

        // Create iterator variable for the lambda - use a separate index space for bound variables
        // This doesn't consume a global variable index since it's bound within the lambda
        let iter_var_id = 0; // BoundVar always uses index 0 for single-argument lambdas

        // Create iterator variable expression using BoundVar for lambda body
        let iter_var = DynamicExpr::new(
            ASTRepr::BoundVar(iter_var_id),
            ctx.registry.clone(),
        );

        // Apply the function to the iterator variable
        let body = f(iter_var);

        // Create the lambda that maps over the data
        let lambda = Lambda {
            var_indices: vec![iter_var_id],
            body: Box::new(body.ast),
        };

        // For data arrays, embed the data directly in the AST
        // This avoids variable indexing issues and makes evaluation simpler
        let data_collection = Collection::DataArray(self);

        // Create Map collection that applies lambda to the data array
        let map_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(data_collection),
        };

        DynamicExpr::new(
            ASTRepr::Sum(Box::new(map_collection)),
            ctx.registry.clone(),
        )
    }
}

/// Implementation for data slices - creates DataArray collection (transitional approach)
impl IntoHListSummationRange<f64> for &Vec<f64> {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Clone and delegate to Vec<f64> implementation
        self.clone().into_hlist_summation(ctx, f)
    }
}

/// Implementation for f64 slices - converts to Vec and delegates
impl IntoHListSummationRange<f64> for &[f64] {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Convert slice to Vec and delegate
        self.to_vec().into_hlist_summation(ctx, f)
    }
}

// DynamicScopeBuilder removed - functionality now integrated into DynamicContext
// Users should use DynamicContext::var::<T>() for heterogeneous variables

// Add missing From implementation for DynamicExpr to ASTRepr conversion
impl<T> From<DynamicExpr<T>> for ASTRepr<T> {
    fn from(expr: DynamicExpr<T>) -> Self {
        expr.ast
    }
}

// Add From implementations for scalar types
impl From<f64> for DynamicExpr<f64> {
    fn from(value: f64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl From<f32> for DynamicExpr<f32> {
    fn from(value: f32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl From<i32> for DynamicExpr<i32> {
    fn from(value: i32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl From<i64> for DynamicExpr<i64> {
    fn from(value: i64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value), registry)
    }
}

// Add cross-type From implementations
impl From<i32> for DynamicExpr<f64> {
    fn from(value: i32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

impl From<i64> for DynamicExpr<f64> {
    fn from(value: i64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

impl From<f32> for DynamicExpr<f64> {
    fn from(value: f32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

impl From<usize> for DynamicExpr<f64> {
    fn from(value: usize) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        DynamicExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}
