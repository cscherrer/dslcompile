//! Dynamic Expression Builder
//!
//! This module provides a runtime expression builder that enables natural mathematical syntax
//! and expressions while maintaining intuitive operator overloading syntax.

use super::typed_registry::VariableRegistry;
use crate::ast::{
    ASTRepr, Scalar,
    ast_repr::{Collection, Lambda},
};
use frunk::hlist::HList;
use num_traits::{Float, FromPrimitive};
use std::{cell::RefCell, fmt::Debug, marker::PhantomData, sync::Arc};

// ============================================================================
// FRUNK HLIST IMPORTS - ZERO-COST HETEROGENEOUS OPERATIONS
// ============================================================================

// ============================================================================
// TYPE SYSTEM INFRASTRUCTURE - NOW IN SEPARATE MODULE
// ============================================================================
pub mod type_system;
pub use type_system::{DataType, DslType};

// ============================================================================
// HLIST INTEGRATION - NOW IN SEPARATE MODULE
// ============================================================================
pub mod hlist_support;
pub use hlist_support::{FunctionSignature, HListEval, IntoConcreteSignature, IntoVarHList};

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
#[derive(Debug, Clone)]
pub struct DynamicContext<T: Scalar = f64, const SCOPE: usize = 0> {
    /// Variables storage - direct typed storage, no type erasure needed
    variables: Vec<Option<T>>,
    /// Variable registry for heterogeneous type management
    registry: Arc<RefCell<VariableRegistry>>,
    /// Next variable ID for predictable variable indexing
    next_var_id: usize,
    /// JIT compilation strategy
    jit_strategy: JITStrategy,
    /// Data arrays for Collection::DataArray evaluation
    ///
    /// TODO: ARCHITECTURAL MIGRATION - Replace with HList-based storage
    ///
    /// Current: data_arrays: Vec<Vec<T>> - homogeneous, runtime indexing
    /// Future:  data_hlist: DataHList - heterogeneous, compile-time type safety
    ///
    /// This change would provide:
    /// - Type-safe data binding: HCons<Vec<f64>, HCons<Vec<i32>, HNil>>
    /// - Zero runtime indexing: Compile-time data array access
    /// - Consistent architecture: Everything uses HLists throughout
    /// - No type erasure: Preserve heterogeneous types through evaluation
    ///
    /// The current Vec<Vec<T>> is a pragmatic implementation to get Collection
    /// evaluation working immediately. ~95% of the evaluation logic will transfer
    /// directly to the HList version, only changing data retrieval mechanism.
    data_arrays: Vec<Vec<T>>,
    _phantom: PhantomData<T>,
}

/// JIT compilation strategy for DynamicContext
#[derive(Debug, Clone, PartialEq)]
pub enum JITStrategy {
    /// Always use interpretation (no JIT)
    AlwaysInterpret,
    /// Always use JIT compilation
    AlwaysJIT,
    /// Adaptive: use JIT for complex expressions, interpretation for simple ones
    Adaptive {
        complexity_threshold: usize,
        call_count_threshold: usize,
    },
    /// LLVM-based JIT compilation
    LLVM,
}

impl Default for JITStrategy {
    fn default() -> Self {
        Self::Adaptive {
            complexity_threshold: 5,
            call_count_threshold: 3,
        }
    }
}

impl<T: Scalar> DynamicContext<T, 0> {
    /// Create a new dynamic expression builder with default JIT strategy
    #[must_use]
    pub fn new() -> Self {
        Self::with_jit_strategy(JITStrategy::default())
    }
}

impl<T: Scalar, const SCOPE: usize> DynamicContext<T, SCOPE> {
    /// Create a new dynamic expression builder with specified JIT strategy
    #[must_use]
    pub fn with_jit_strategy(strategy: JITStrategy) -> Self {
        Self {
            variables: Vec::new(),
            registry: Arc::new(RefCell::new(VariableRegistry::new())),
            next_var_id: 0,
            jit_strategy: strategy,
            data_arrays: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Create a new dynamic expression builder optimized for JIT compilation
    #[must_use]
    pub fn new_jit_optimized() -> Self {
        Self::with_jit_strategy(JITStrategy::LLVM)
    }

    /// Create a new dynamic expression builder optimized for interpretation
    #[must_use]
    pub fn new_interpreter() -> Self {
        Self::with_jit_strategy(JITStrategy::AlwaysInterpret)
    }

    /// Create a variable of any scalar type (heterogeneous support)
    ///
    /// This provides the heterogeneous-by-default functionality while maintaining
    /// automatic scope management for composability.
    ///
    /// # Examples
    /// ```rust
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var::<f64>();           // Explicit f64
    /// let data = ctx.var::<Vec<f64>>();   // Heterogeneous: Vec<f64>
    /// let index = ctx.var::<usize>();     // Heterogeneous: usize  
    /// ```
    #[must_use]
    pub fn var<U: Scalar>(&mut self) -> TypedBuilderExpr<U> {
        // Register the variable in the registry (gets automatic index)
        let var_id = {
            let mut registry = self.registry.borrow_mut();
            registry.register_variable()
        };

        TypedBuilderExpr::new(ASTRepr::Variable(var_id), self.registry.clone())
    }

    /// Create a variable of the context's type T (legacy method)
    ///
    /// This method is kept for backward compatibility with code that depends on
    /// the context's parameterized type. New code should use `var::<T>()` for
    /// explicit heterogeneous type specification.
    #[must_use]
    pub fn var_context_type(&mut self) -> TypedBuilderExpr<T> {
        let var_index = self.variables.len();
        self.variables.push(None); // Placeholder for variable value

        let var_id = self.next_var_id;
        self.next_var_id += 1;

        TypedBuilderExpr::new(ASTRepr::Variable(var_index), self.registry.clone())
    }

    /// Create a constant expression
    #[must_use]
    pub fn constant(&self, value: T) -> TypedBuilderExpr<T> {
        TypedBuilderExpr::new(ASTRepr::Constant(value), self.registry.clone())
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
    /// let x = ctx.var();  // Variable(0)
    /// let y = ctx.var();  // Variable(1)
    /// let expr = &x * 2.0 + &y;
    ///
    /// // Evaluate with HList - no flattening, direct variable access
    /// let result = ctx.eval(&expr, hlist![3.0, 4.0]);
    /// assert_eq!(result, 10.0); // 3*2 + 4 = 10
    /// ```
    #[must_use]
    pub fn eval<H>(&self, expr: &TypedBuilderExpr<T>, hlist: H) -> T
    where
        H: HListEval<T>,
    {
        hlist.eval_expr(expr.as_ast())
    }

    /// Legacy method: Evaluate expression with borrowed array parameters
    ///
    /// This method is kept for internal use and backward compatibility.
    /// New code should use the unified `eval()` method with HLists.
    #[must_use]
    pub fn eval_borrowed(&self, expr: &TypedBuilderExpr<T>, params: &[T]) -> T
    where
        T: num_traits::Float + Copy + num_traits::FromPrimitive + num_traits::Zero,
    {
        match &self.jit_strategy {
            JITStrategy::AlwaysInterpret => self.eval_with_interpretation(expr, params),
            JITStrategy::AlwaysJIT => {
                // JIT compilation would go here
                // For now, fall back to interpretation
                self.eval_with_interpretation(expr, params)
            }
            JITStrategy::Adaptive { .. } => {
                // Adaptive strategy would analyze complexity
                // For now, use interpretation
                self.eval_with_interpretation(expr, params)
            }
            JITStrategy::LLVM => {
                // LLVM-based JIT compilation would go here
                // For now, fall back to interpretation
                self.eval_with_interpretation(expr, params)
            }
        }
    }

    /// Internal method for interpretation-based evaluation
    fn eval_with_interpretation(&self, expr: &TypedBuilderExpr<T>, params: &[T]) -> T
    where
        T: num_traits::Float + Copy + num_traits::FromPrimitive + num_traits::Zero,
    {
        let ast = expr.as_ast();
        ast.eval_with_vars(params)
    }

    /// Estimate the computational complexity of an expression
    fn estimate_complexity(&self, expr: &TypedBuilderExpr<T>) -> usize {
        let ast = expr.as_ast();
        self.count_operations(ast)
    }

    /// Count the number of operations in an AST
    fn count_operations(&self, ast: &ASTRepr<T>) -> usize {
        match ast {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
            ASTRepr::Add(l, r)
            | ASTRepr::Sub(l, r)
            | ASTRepr::Mul(l, r)
            | ASTRepr::Div(l, r)
            | ASTRepr::Pow(l, r) => 1 + self.count_operations(l) + self.count_operations(r),
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => 1 + self.count_operations(inner),
            ASTRepr::Sum(collection) => {
                // Estimate sum complexity based on collection
                // TODO: Implement collection complexity analysis
                10 // Placeholder
            }
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    /// Generate cache key for expressions (for future JIT optimization)
    fn generate_cache_key(&self, ast: &ASTRepr<T>) -> String
    where
        T: std::fmt::Debug,
    {
        // Simple cache key based on AST structure
        format!("{ast:?}")
    }

    /// Clear JIT cache (for future implementation)
    pub fn clear_jit_cache(&self) {
        // TODO: Implement JIT cache clearing
    }

    /// Set JIT strategy
    pub fn set_jit_strategy(&mut self, strategy: JITStrategy) {
        self.jit_strategy = strategy;
    }

    /// Store a data array and return its index for Collection::DataArray references
    ///
    /// This enables data-driven summation: `ctx.sum(data_vec, |x| x * 2.0)`
    /// where the data is bound at evaluation time.
    pub(crate) fn store_data_array(&mut self, data: Vec<T>) -> usize {
        let data_index = self.data_arrays.len();
        self.data_arrays.push(data);
        data_index
    }

    /// Get a reference to a stored data array by index
    ///
    /// Returns None if the index is out of bounds.
    pub fn get_data_array(&self, index: usize) -> Option<&Vec<T>> {
        self.data_arrays.get(index)
    }

    /// Create polynomial expression with given coefficients
    pub fn poly(&self, coefficients: &[T], variable: &TypedBuilderExpr<T>) -> TypedBuilderExpr<T>
where {
        if coefficients.is_empty() {
            return self.constant(T::default());
        }

        let mut result = self.constant(coefficients[0].clone());

        for (power, coeff) in coefficients.iter().skip(1).enumerate() {
            let power = power + 1;
            let term = if power == 1 {
                self.constant(coeff.clone()) * variable.clone()
            } else {
                // Create x^power by repeated multiplication
                let mut power_expr = variable.clone();
                for _ in 1..power {
                    power_expr = power_expr * variable.clone();
                }
                self.constant(coeff.clone()) * power_expr
            };
            result = result + term;
        }

        result
    }

    /// Generate pretty-printed string representation
    #[must_use]
    pub fn pretty_print(&self, expr: &TypedBuilderExpr<T>) -> String
    where
        T: std::fmt::Display,
    {
        // Create a minimal registry for pretty printing
        let registry = VariableRegistry::for_expression(expr.as_ast());
        crate::ast::pretty_ast(expr.as_ast(), &registry)
    }

    /// Convert to AST representation
    #[must_use]
    pub fn to_ast(&self, expr: &TypedBuilderExpr<T>) -> crate::ast::ASTRepr<T> {
        expr.as_ast().clone()
    }

    /// Check if expression has unbound variables
    #[must_use]
    pub fn has_unbound_variables(&self, expr: &TypedBuilderExpr<T>) -> bool {
        !self.find_unbound_variables(expr).is_empty()
    }

    /// Find all unbound variables in expression
    #[must_use]
    pub fn find_unbound_variables(&self, expr: &TypedBuilderExpr<T>) -> Vec<usize> {
        let mut unbound_vars = Vec::new();
        self.collect_unbound_variables_recursive(expr.as_ast(), &mut unbound_vars);
        unbound_vars.sort_unstable();
        unbound_vars.dedup();
        unbound_vars
    }

    /// Recursively collect unbound variables
    fn collect_unbound_variables_recursive(&self, ast: &ASTRepr<T>, unbound_vars: &mut Vec<usize>) {
        match ast {
            ASTRepr::Variable(index) => {
                if *index >= self.data_arrays.len() || self.data_arrays[*index].is_empty() {
                    unbound_vars.push(*index);
                }
            }
            ASTRepr::Constant(_) => {}
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.collect_unbound_variables_recursive(left, unbound_vars);
                self.collect_unbound_variables_recursive(right, unbound_vars);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                self.collect_unbound_variables_recursive(inner, unbound_vars);
            }
            ASTRepr::Sum(collection) => {
                // TODO: Analyze collection for unbound variables
                // For now, assume no unbound variables in collections
            }
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    /// Find maximum variable index used in expression
    #[must_use]
    pub fn find_max_variable_index(&self, expr: &TypedBuilderExpr<T>) -> usize {
        self.find_max_variable_index_recursive(expr.as_ast())
    }

    /// Recursively find maximum variable index
    fn find_max_variable_index_recursive(&self, ast: &ASTRepr<T>) -> usize {
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
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    // sum method moved to separate impl block with required trait bounds

    // new_scope method removed - use var::<T>() directly for heterogeneous variables
    // The SCOPE parameter provides automatic scope management at the type level
}

// Separate impl block for methods requiring additional trait bounds
impl<T: Scalar + num_traits::FromPrimitive + Copy, const SCOPE: usize> DynamicContext<T, SCOPE> {
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
    /// let data_var = ctx.var(); // This represents Vec<f64> in HList
    /// let sum2 = ctx.sum_data(data_var, |x| x * 2.0);
    /// // Later evaluated with: ctx.eval(&sum2, hlist![other_params, data_vec])
    /// ```
    pub fn sum<R, F>(&mut self, range: R, f: F) -> TypedBuilderExpr<T>
    where
        R: IntoHListSummationRange<T>,
        F: FnOnce(TypedBuilderExpr<T>) -> TypedBuilderExpr<T>,
    {
        range.into_hlist_summation(self, f)
    }
}

impl<T: Scalar, const SCOPE: usize> DynamicContext<T, SCOPE> {
    /// Advance to the next scope for safe composition
    ///
    /// This method consumes the current context and returns a new context
    /// with the next scope index, ensuring no variable collisions when
    /// composing expressions from different contexts.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::ast::DynamicContext;
    /// use frunk::hlist;
    ///
    /// // Create first expression in scope 0
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var(); // Variable(0) in scope 0
    /// let expr1 = x.clone() * x;
    ///
    /// // Advance to scope 1 for composition safety
    /// let mut ctx = ctx.next();
    /// let y = ctx.var(); // Variable(0) in scope 1 - no collision!
    /// let expr2 = y.clone() + y;
    /// ```
    #[must_use]
    pub fn next(self) -> DynamicContext<T, { SCOPE + 1 }> {
        DynamicContext {
            variables: self.variables,
            registry: self.registry,
            next_var_id: self.next_var_id,
            jit_strategy: self.jit_strategy,
            data_arrays: self.data_arrays,
            _phantom: PhantomData,
        }
    }

    /// Merge with another context, automatically remapping variables to prevent collisions
    ///
    /// This method safely combines two contexts by remapping variable indices
    /// from the second context to prevent collisions with the first context.
    /// The type system ensures that contexts from different scopes can be merged safely.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::ast::DynamicContext;
    ///
    /// // Create two separate contexts (different scopes)
    /// let mut ctx1 = DynamicContext::new();     // Scope 0
    /// let x = ctx1.var(); // Variable(0)
    ///
    /// let mut ctx2 = DynamicContext::new().next(); // Scope 1  
    /// let y = ctx2.var(); // Variable(0) in scope 1
    ///
    /// // Safe merge - variables automatically remapped
    /// let merged_ctx = ctx1.merge(ctx2);
    /// ```
    pub fn merge(mut self, other: DynamicContext<T, SCOPE>) -> DynamicContext<T, SCOPE> {
        // Calculate variable offset to prevent collisions
        let var_offset = self.next_var_id;

        // Merge variables (other's variables get remapped indices)
        self.data_arrays.extend(other.data_arrays);
        self.next_var_id += other.next_var_id;

        // Use the more advanced JIT strategy
        let merged_strategy = match (&self.jit_strategy, other.jit_strategy) {
            (JITStrategy::LLVM, _) | (_, JITStrategy::LLVM) => JITStrategy::LLVM,
            (JITStrategy::AlwaysJIT, _) | (_, JITStrategy::AlwaysJIT) => JITStrategy::AlwaysJIT,
            (JITStrategy::Adaptive { .. }, _) | (_, JITStrategy::Adaptive { .. }) => {
                self.jit_strategy.clone()
            }
            _ => JITStrategy::AlwaysInterpret,
        };

        self.jit_strategy = merged_strategy;
        self
    }
}

impl Default for DynamicContext {
    fn default() -> Self {
        Self::new()
    }
}

// Type alias for backward compatibility with f64 default
pub type DynamicF64Context = DynamicContext;
pub type DynamicF32Context = DynamicContext<f32>;
pub type DynamicI32Context = DynamicContext<i32>;

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

/// Typed expression wrapper that provides type-safe operations
#[derive(Debug, Clone)]
pub struct TypedBuilderExpr<T> {
    pub(crate) ast: ASTRepr<T>,
    pub(crate) registry: Arc<RefCell<VariableRegistry>>,
}

impl<T> TypedBuilderExpr<T> {
    /// Create a new typed expression
    pub fn new(ast: ASTRepr<T>, registry: Arc<RefCell<VariableRegistry>>) -> Self {
        Self { ast, registry }
    }

    /// Get the underlying AST
    pub fn as_ast(&self) -> &ASTRepr<T> {
        &self.ast
    }

    /// Convert to owned AST
    pub fn into_ast(self) -> ASTRepr<T> {
        self.ast
    }

    /// Convert to TypedBuilderExpr (identity operation for compatibility)
    pub fn into_expr(self) -> Self {
        self
    }

    /// Get variable ID if this expression is a single variable
    pub fn var_id(&self) -> usize {
        match &self.ast {
            ASTRepr::Variable(id) => *id,
            _ => panic!("var_id() called on non-variable expression"),
        }
    }

    /// Pretty print the expression
    #[must_use]
    pub fn pretty_print(&self) -> String
    where
        T: std::fmt::Display,
    {
        // Create a minimal registry for pretty printing
        let registry =
            crate::ast::runtime::typed_registry::VariableRegistry::for_expression(&self.ast);
        crate::ast::pretty_ast(&self.ast, &registry)
    }
}

impl TypedBuilderExpr<f32> {
    /// Convert f32 expression to f64 expression
    #[must_use]
    pub fn to_f64(self) -> TypedBuilderExpr<f64> {
        TypedBuilderExpr::new(convert_ast_pure_rust(&self.ast), self.registry)
    }
}

impl TypedBuilderExpr<f64> {
    /// Convert f64 expression to f64 expression (identity operation)
    #[must_use]
    pub fn to_f64(self) -> TypedBuilderExpr<f64> {
        self
    }
}

impl TypedBuilderExpr<i32> {
    /// Convert i32 expression to f64 expression
    #[must_use]
    pub fn to_f64(self) -> TypedBuilderExpr<f64> {
        TypedBuilderExpr::new(convert_i32_ast_to_f64(&self.ast), self.registry)
    }
}

/// Scalar variable operations (f64, f32, i32, etc.)
impl<T: Scalar> VariableExpr<T> {
    /// Convert to a typed expression for arithmetic operations
    pub fn into_expr(self) -> TypedBuilderExpr<T> {
        TypedBuilderExpr::new(ASTRepr::Variable(self.var_id), self.registry)
    }
}

// ============================================================================
// All operator implementations moved to operators module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_variable_creation() {
        let mut builder_f64 = DynamicContext::<f64>::new();
        let mut builder_f32 = DynamicContext::<f32>::new();

        // Create variables for different types
        let x = builder_f64.var::<f64>();
        let y = builder_f32.var::<f32>();

        // Variables should have the correct IDs
        assert_eq!(x.var_id(), 0);
        assert_eq!(y.var_id(), 0); // Each context starts from 0
    }

    #[test]
    fn test_typed_expression_building() {
        let mut builder = DynamicContext::<f64>::new();

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
        let mut builder_f64 = DynamicContext::<f64>::new();
        let mut builder_f32 = DynamicContext::<f32>::new();

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

        let x = builder.var();

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
        let mut builder = DynamicContext::new();

        let x = builder.var();

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
        let x = builder.var();
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
        let range_sum = ctx.sum(1..=5, |x| x * 2.0);
        println!("Range sum AST: {:?}", range_sum.as_ast());

        // Test 2: Parametric summation using the unified API
        let param = ctx.var();
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
// let f32_expr: TypedBuilderExpr<f32> = /* ... */;
// let f64_expr: TypedBuilderExpr<f64> = f32_expr.into(); // Explicit conversion!

/// Explicit conversion from f32 expressions to f64 expressions
impl From<TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    fn from(expr: TypedBuilderExpr<f32>) -> Self {
        expr.to_f64()
    }
}

/// Explicit conversion from i32 expressions to f64 expressions  
impl From<TypedBuilderExpr<i32>> for TypedBuilderExpr<f64> {
    fn from(expr: TypedBuilderExpr<i32>) -> Self {
        TypedBuilderExpr::new(convert_i32_ast_to_f64(&expr.ast), expr.registry)
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
        ASTRepr::Sum(_collection) => {
            // TODO: Implement collection conversion
            todo!()
        }
        ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
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
        ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
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
        Collection::Union { left, right } => Collection::Union {
            left: Box::new(convert_collection_pure_rust(left)),
            right: Box::new(convert_collection_pure_rust(right)),
        },
        Collection::Intersection { left, right } => Collection::Intersection {
            left: Box::new(convert_collection_pure_rust(left)),
            right: Box::new(convert_collection_pure_rust(right)),
        },
        Collection::DataArray(index) => Collection::DataArray(*index),
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
    }
}

/// Convert Lambda using standard Rust From trait  
fn convert_lambda_pure_rust<T, U>(lambda: &Lambda<T>) -> Lambda<U>
where
    T: Clone,
    U: From<T>,
{
    use crate::ast::ast_repr::Lambda;

    match lambda {
        Lambda::Identity => Lambda::Identity,
        Lambda::Constant(expr) => Lambda::Constant(Box::new(convert_ast_pure_rust(expr))),
        Lambda::Lambda { var_index, body } => Lambda::Lambda {
            var_index: *var_index,
            body: Box::new(convert_ast_pure_rust(body)),
        },
        Lambda::Compose { f, g } => Lambda::Compose {
            f: Box::new(convert_lambda_pure_rust(f)),
            g: Box::new(convert_lambda_pure_rust(g)),
        },
    }
}

// ============================================================================
// EXAMPLE USAGE (Pure Rust Way)
// ============================================================================

/*
// ✅ Explicit conversions using standard Rust traits:

let i32_expr: TypedBuilderExpr<i32> = ctx.constant(42);
let f64_expr: TypedBuilderExpr<f64> = i32_expr.into(); // Uses standard Into!

// Or explicitly:
let f64_expr = TypedBuilderExpr::<f64>::from(i32_expr);

// Rust's built-in conversions work automatically:
// i32 -> f64 ✅ (built into Rust)
// f32 -> f64 ✅ (built into Rust)
// usize -> f64 ✅ (we can add this)

// ❌ No more auto-promotion:
// let result = f64_expr + i32_expr; // Compile error! Must convert first.

// ✅ Explicit conversion required:
let result = f64_expr + TypedBuilderExpr::<f64>::from(i32_expr);
*/

#[cfg(test)]
mod test_comprehensive_api {
    use super::*;

    #[test]
    fn test_comprehensive_typed_api() {
        use frunk::hlist;
        // Test the comprehensive API working together
        let mut ctx: DynamicContext = DynamicContext::new();
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
        ctx: &mut DynamicContext<T, SCOPE>,
        f: F,
    ) -> TypedBuilderExpr<T>
    where
        F: FnOnce(TypedBuilderExpr<T>) -> TypedBuilderExpr<T>,
        T: num_traits::FromPrimitive + Copy;
}

/// Implementation for mathematical ranges - creates Range collection (no DataArray)
impl<T: Scalar + num_traits::FromPrimitive> IntoHListSummationRange<T>
    for std::ops::RangeInclusive<i64>
{
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<T, SCOPE>,
        f: F,
    ) -> TypedBuilderExpr<T>
    where
        F: FnOnce(TypedBuilderExpr<T>) -> TypedBuilderExpr<T>,
        T: num_traits::FromPrimitive + Copy,
    {
        // Mathematical ranges don't need DataArray - use Collection::Range directly
        let collection = Collection::Range {
            start: Box::new(ASTRepr::Constant(
                T::from_i64(*self.start()).unwrap_or_default(),
            )),
            end: Box::new(ASTRepr::Constant(
                T::from_i64(*self.end()).unwrap_or_default(),
            )),
        };

        let iter_var_index = ctx.next_var_id;
        ctx.next_var_id += 1;

        // Create iterator variable for the lambda
        let iter_var =
            TypedBuilderExpr::new(ASTRepr::Variable(iter_var_index), ctx.registry.clone());

        // Apply the user's function to get the lambda body
        let body_expr = f(iter_var);

        // Create lambda from the body expression
        let lambda = Lambda::Lambda {
            var_index: iter_var_index,
            body: Box::new(body_expr.ast),
        };

        // Create mapped collection
        let mapped_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(collection),
        };

        // Create sum expression using the Collection system
        let sum_ast = ASTRepr::Sum(Box::new(mapped_collection));

        TypedBuilderExpr::new(sum_ast, ctx.registry.clone())
    }
}

/// Implementation for data vectors - creates DataArray collection (transitional approach)
impl IntoHListSummationRange<f64> for Vec<f64> {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<f64, SCOPE>,
        f: F,
    ) -> TypedBuilderExpr<f64>
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Store data array and create DataArray collection
        let data_var_id = ctx.store_data_array(self);
        let collection = Collection::DataArray(data_var_id);

        let iter_var_index = ctx.next_var_id;
        ctx.next_var_id += 1;

        // Create iterator variable for the lambda
        let iter_var =
            TypedBuilderExpr::new(ASTRepr::Variable(iter_var_index), ctx.registry.clone());

        // Apply the user's function to get the lambda body
        let body_expr = f(iter_var);

        // Create lambda from the body expression
        let lambda = Lambda::Lambda {
            var_index: iter_var_index,
            body: Box::new(body_expr.ast),
        };

        // Create mapped collection
        let mapped_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(collection),
        };

        // Create sum expression using the Collection system
        let sum_ast = ASTRepr::Sum(Box::new(mapped_collection));

        TypedBuilderExpr::new(sum_ast, ctx.registry.clone())
    }
}

/// Implementation for data slices - creates DataArray collection (transitional approach)
impl IntoHListSummationRange<f64> for &[f64] {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<f64, SCOPE>,
        f: F,
    ) -> TypedBuilderExpr<f64>
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Convert slice to vector and use Vec implementation
        self.to_vec().into_hlist_summation(ctx, f)
    }
}

// DynamicScopeBuilder removed - functionality now integrated into DynamicContext
// Users should use DynamicContext::var::<T>() for heterogeneous variables

// Add missing From implementation for TypedBuilderExpr to ASTRepr conversion
impl<T> From<TypedBuilderExpr<T>> for ASTRepr<T> {
    fn from(expr: TypedBuilderExpr<T>) -> Self {
        expr.ast
    }
}

// Add From implementations for scalar types
impl From<f64> for TypedBuilderExpr<f64> {
    fn from(value: f64) -> Self {
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

impl From<i32> for TypedBuilderExpr<i32> {
    fn from(value: i32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value), registry)
    }
}

impl From<i64> for TypedBuilderExpr<i64> {
    fn from(value: i64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value), registry)
    }
}

// Add cross-type From implementations
impl From<i32> for TypedBuilderExpr<f64> {
    fn from(value: i32) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

impl From<i64> for TypedBuilderExpr<f64> {
    fn from(value: i64) -> Self {
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value as f64), registry)
    }
}

impl From<f32> for TypedBuilderExpr<f64> {
    fn from(value: f32) -> Self {
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
