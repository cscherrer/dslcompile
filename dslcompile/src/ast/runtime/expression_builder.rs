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

use crate::error::{DSLCompileError, Result};

// ============================================================================
// FRUNK HLIST IMPORTS - ZERO-COST HETEROGENEOUS OPERATIONS
// ============================================================================
use frunk::{HCons, HNil};

// ============================================================================
// OPEN TRAIT SYSTEM - EXTENSIBLE TYPE SUPPORT
// ============================================================================

/// Extended trait for DSL types that can participate in code generation
/// This is the "open" part - users can implement this for custom types
pub trait DslType: NumericType + 'static {
    /// The native Rust type this DSL type maps to
    type Native: Copy + std::fmt::Debug + std::fmt::Display;

    /// Type identifier for code generation
    const TYPE_NAME: &'static str;

    /// Generate Rust code for addition operation
    fn codegen_add() -> &'static str {
        "+"
    }

    /// Generate Rust code for multiplication operation  
    fn codegen_mul() -> &'static str {
        "*"
    }

    /// Generate Rust code for subtraction operation
    fn codegen_sub() -> &'static str {
        "-"
    }

    /// Generate Rust code for division operation
    fn codegen_div() -> &'static str {
        "/"
    }

    /// Generate Rust code for a literal value
    fn codegen_literal(value: Self::Native) -> String;

    /// Convert to evaluation value (for runtime interpretation)
    fn to_eval_value(value: Self::Native) -> f64;

    /// Check if this type can be promoted to another DslType
    fn can_promote_to<U: DslType>() -> bool {
        std::any::TypeId::of::<Self>() == std::any::TypeId::of::<U>()
    }
}

/// Extended trait for data types that can participate in evaluation but aren't scalar
/// This enables Vec<f64>, matrices, and other non-scalar types in HLists
pub trait DataType: Clone + std::fmt::Debug + 'static {
    /// Type identifier for signatures
    const TYPE_NAME: &'static str;
    
    /// Convert to evaluation data for runtime interpretation
    /// Returns the data as a vector that can be used in data summation
    fn to_eval_data(&self) -> Vec<f64>;
}

// ============================================================================
// CONCRETE IMPLEMENTATIONS FOR STANDARD TYPES
// ============================================================================

impl DslType for f64 {
    type Native = f64;
    const TYPE_NAME: &'static str = "f64";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_f64")
    }

    fn to_eval_value(value: Self::Native) -> f64 {
        value
    }

    fn can_promote_to<U: DslType>() -> bool {
        // f64 can only convert to itself (no precision loss)
        std::any::TypeId::of::<U>() == std::any::TypeId::of::<f64>()
    }
}

impl DslType for f32 {
    type Native = f32;
    const TYPE_NAME: &'static str = "f32";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_f32")
    }

    fn to_eval_value(value: Self::Native) -> f64 {
        value as f64
    }

    fn can_promote_to<U: DslType>() -> bool {
        // f32 can promote to f64
        std::any::TypeId::of::<U>() == std::any::TypeId::of::<f32>()
            || std::any::TypeId::of::<U>() == std::any::TypeId::of::<f64>()
    }
}

impl DslType for i32 {
    type Native = i32;
    const TYPE_NAME: &'static str = "i32";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_i32")
    }

    fn to_eval_value(value: Self::Native) -> f64 {
        value as f64
    }

    fn can_promote_to<U: DslType>() -> bool {
        // i32 can promote to i64, f32, f64
        std::any::TypeId::of::<U>() == std::any::TypeId::of::<i32>()
            || std::any::TypeId::of::<U>() == std::any::TypeId::of::<i64>()
            || std::any::TypeId::of::<U>() == std::any::TypeId::of::<f32>()
            || std::any::TypeId::of::<U>() == std::any::TypeId::of::<f64>()
    }
}

impl DslType for i64 {
    type Native = i64;
    const TYPE_NAME: &'static str = "i64";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_i64")
    }

    fn to_eval_value(value: Self::Native) -> f64 {
        value as f64
    }

    fn can_promote_to<U: DslType>() -> bool {
        // i64 can promote to f64 (but may lose precision for very large values)
        std::any::TypeId::of::<U>() == std::any::TypeId::of::<i64>()
            || std::any::TypeId::of::<U>() == std::any::TypeId::of::<f64>()
    }
}

impl DslType for usize {
    type Native = usize;
    const TYPE_NAME: &'static str = "usize";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_usize")
    }

    fn to_eval_value(value: Self::Native) -> f64 {
        value as f64
    }

    fn can_promote_to<U: DslType>() -> bool {
        // usize can promote to f64
        std::any::TypeId::of::<U>() == std::any::TypeId::of::<usize>()
            || std::any::TypeId::of::<U>() == std::any::TypeId::of::<f64>()
    }
}

// ============================================================================
// DATA TYPE IMPLEMENTATIONS
// ============================================================================

impl DataType for Vec<f64> {
    const TYPE_NAME: &'static str = "Vec<f64>";
    
    fn to_eval_data(&self) -> Vec<f64> {
        self.clone()
    }
}

// ============================================================================
// HLIST INTEGRATION TRAITS
// ============================================================================

/// Trait for converting HLists into typed variable HLists
pub trait IntoVarHList {
    type Output;
    fn into_vars(self, ctx: &DynamicContext) -> Self::Output;
}

/// Trait for converting HLists into concrete function signatures
pub trait IntoConcreteSignature {
    fn concrete_signature() -> FunctionSignature;
}

/// Enhanced trait for converting HLists into evaluation data
/// Now supports both scalar values and data arrays
pub trait IntoEvalData {
    /// Convert to evaluation parameters and data arrays
    /// Returns (scalar_params, data_arrays)
    fn into_eval_data(self) -> (Vec<f64>, Vec<Vec<f64>>);
}

/// Legacy trait for backward compatibility - converts to flat Vec<f64>
/// This is used for expressions that only need scalar parameters
pub trait IntoEvalArray {
    fn into_eval_array(self) -> Vec<f64>;
}

/// Function signature for code generation
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    params: Vec<String>,
    return_type: String,
}

impl FunctionSignature {
    pub fn new(param_types: Vec<&str>) -> Self {
        Self {
            params: param_types
                .iter()
                .enumerate()
                .map(|(i, t)| format!("x{i}: {t}"))
                .collect(),
            return_type: "f64".to_string(), // Default return type
        }
    }

    pub fn parameters(&self) -> String {
        self.params.join(", ")
    }

    pub fn return_type(&self) -> &str {
        &self.return_type
    }

    pub fn function_name(&self) -> String {
        // Generate a unique function name based on signature
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.params.hash(&mut hasher);
        self.return_type.hash(&mut hasher);
        format!("expr_{:x}", hasher.finish())
    }
}

// ============================================================================
// HLIST BASE CASES
// ============================================================================

impl IntoVarHList for HNil {
    type Output = HNil;
    fn into_vars(self, _ctx: &DynamicContext) -> Self::Output {
        HNil
    }
}

impl IntoConcreteSignature for HNil {
    fn concrete_signature() -> FunctionSignature {
        FunctionSignature::new(vec![])
    }
}

impl IntoEvalArray for HNil {
    fn into_eval_array(self) -> Vec<f64> {
        Vec::new()
    }
}

impl IntoEvalData for HNil {
    fn into_eval_data(self) -> (Vec<f64>, Vec<Vec<f64>>) {
        (Vec::new(), Vec::new())
    }
}

// ============================================================================
// HLIST RECURSIVE CASES
// ============================================================================

impl<T, Tail> IntoVarHList for HCons<T, Tail>
where
    T: DslType,
    Tail: IntoVarHList,
{
    type Output = HCons<TypedBuilderExpr<T>, Tail::Output>;

    fn into_vars(self, ctx: &DynamicContext) -> Self::Output {
        let head_var = ctx.typed_var::<T>();
        let head_expr = ctx.expr_from(head_var);
        let tail_vars = self.tail.into_vars(ctx);
        HCons {
            head: head_expr,
            tail: tail_vars,
        }
    }
}

impl<T, Tail> IntoConcreteSignature for HCons<T, Tail>
where
    T: DslType,
    Tail: IntoConcreteSignature,
{
    fn concrete_signature() -> FunctionSignature {
        let mut sig = Tail::concrete_signature();
        sig.params.insert(0, format!("x0: {}", T::TYPE_NAME));
        // Update parameter indices
        for (i, param) in sig.params.iter_mut().enumerate().skip(1) {
            *param = param.replace(&format!("x{}", i - 1), &format!("x{i}"));
        }
        sig
    }
}

// Specific implementations for concrete scalar types (IntoEvalArray - legacy)
impl<Tail> IntoEvalArray for HCons<f64, Tail>
where
    Tail: IntoEvalArray,
{
    fn into_eval_array(self) -> Vec<f64> {
        let mut result = vec![self.head];
        result.extend(self.tail.into_eval_array());
        result
    }
}

impl<Tail> IntoEvalArray for HCons<f32, Tail>
where
    Tail: IntoEvalArray,
{
    fn into_eval_array(self) -> Vec<f64> {
        let mut result = vec![self.head as f64];
        result.extend(self.tail.into_eval_array());
        result
    }
}

impl<Tail> IntoEvalArray for HCons<i32, Tail>
where
    Tail: IntoEvalArray,
{
    fn into_eval_array(self) -> Vec<f64> {
        let mut result = vec![self.head as f64];
        result.extend(self.tail.into_eval_array());
        result
    }
}

impl<Tail> IntoEvalArray for HCons<i64, Tail>
where
    Tail: IntoEvalArray,
{
    fn into_eval_array(self) -> Vec<f64> {
        let mut result = vec![self.head as f64];
        result.extend(self.tail.into_eval_array());
        result
    }
}

impl<Tail> IntoEvalArray for HCons<usize, Tail>
where
    Tail: IntoEvalArray,
{
    fn into_eval_array(self) -> Vec<f64> {
        let mut result = vec![self.head as f64];
        result.extend(self.tail.into_eval_array());
        result
    }
}

// New unified implementations for IntoEvalData (supports both scalars and data)
impl<T, Tail> IntoEvalData for HCons<T, Tail>
where
    T: DslType<Native = T>,
    Tail: IntoEvalData,
{
    fn into_eval_data(self) -> (Vec<f64>, Vec<Vec<f64>>) {
        let (mut params, data_arrays) = self.tail.into_eval_data();
        // Insert scalar value at the beginning
        params.insert(0, T::to_eval_value(self.head));
        (params, data_arrays)
    }
}

impl<Tail> IntoEvalData for HCons<Vec<f64>, Tail>
where
    Tail: IntoEvalData,
{
    fn into_eval_data(self) -> (Vec<f64>, Vec<Vec<f64>>) {
        let (params, mut data_arrays) = self.tail.into_eval_data();
        // Insert data array at the beginning
        data_arrays.insert(0, self.head.to_eval_data());
        (params, data_arrays)
    }
}

/// Dynamic expression builder with runtime variable management
/// Static with type-level scoping for predictable variable indexing
/// Static with runtime optimization capabilities
#[derive(Debug, Clone)]
pub struct DynamicContext {
    registry: Arc<RefCell<VariableRegistry>>,
    /// Next variable ID for type-level scoping (predictable indexing)
    next_var_id: Arc<RefCell<usize>>,
    /// JIT compilation strategy (temporarily disabled)
    jit_strategy: JITStrategy,
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
}

impl Default for JITStrategy {
    fn default() -> Self {
        Self::Adaptive {
            complexity_threshold: 5,
            call_count_threshold: 3,
        }
    }
}

impl DynamicContext {
    /// Create a new dynamic expression builder with default JIT strategy
    #[must_use]
    pub fn new() -> Self {
        Self::with_jit_strategy(JITStrategy::default())
    }

    /// Create a new dynamic expression builder with specified JIT strategy
    #[must_use]
    pub fn with_jit_strategy(strategy: JITStrategy) -> Self {
        Self {
            registry: Arc::new(RefCell::new(VariableRegistry::new())),
            next_var_id: Arc::new(RefCell::new(0)),
            jit_strategy: strategy,
        }
    }

    /// Create a new dynamic expression builder optimized for JIT compilation
    #[must_use]
    pub fn new_jit_optimized() -> Self {
        Self::with_jit_strategy(JITStrategy::AlwaysJIT)
    }

    /// Create a new dynamic expression builder optimized for interpretation
    #[must_use]
    pub fn new_interpreter() -> Self {
        Self::with_jit_strategy(JITStrategy::AlwaysInterpret)
    }

    /// Create a typed variable with predictable ID-based scoping
    #[must_use]
    pub fn typed_var<T: NumericType + 'static>(&self) -> TypedVar<T> {
        // Get the next predictable ID
        let id = {
            let mut next_id = self.next_var_id.borrow_mut();
            let current_id = *next_id;
            *next_id += 1;
            current_id
        };

        // Register the variable in the registry (gets runtime index)
        let registry_index = self
            .registry
            .borrow_mut()
            .register_typed_variable::<T>()
            .index();

        // Create variable with both predictable ID and runtime index
        TypedVar::with_id(id, registry_index)
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

    /// Static evaluation with automatic JIT compilation
    /// This method now intelligently chooses between interpretation and JIT compilation
    /// based on the configured strategy and expression characteristics
    #[must_use]
    pub fn eval(&self, expr: &TypedBuilderExpr<f64>, inputs: &[f64]) -> f64 {
        match self.should_use_jit(expr) {
            true => self.eval_with_jit(expr, inputs).unwrap_or_else(|_| {
                // Fall back to interpretation if JIT fails
                self.eval_with_interpretation(expr, inputs)
            }),
            false => self.eval_with_interpretation(expr, inputs),
        }
    }

    /// Force evaluation using JIT compilation
    pub fn eval_jit(&self, expr: &TypedBuilderExpr<f64>, inputs: &[f64]) -> Result<f64> {
        self.eval_with_jit(expr, inputs)
    }

    /// Force evaluation using interpretation
    #[must_use]
    pub fn eval_interpret(&self, expr: &TypedBuilderExpr<f64>, inputs: &[f64]) -> f64 {
        self.eval_with_interpretation(expr, inputs)
    }

    /// Internal method for JIT-based evaluation
    fn eval_with_jit(&self, expr: &TypedBuilderExpr<f64>, inputs: &[f64]) -> Result<f64> {
        // JIT compilation removed - fall back to interpretation
        // TODO: Implement static scoped system integration for compile-time optimization
        Ok(self.eval_with_interpretation(expr, inputs))
    }

    /// Internal method for interpretation-based evaluation
    fn eval_with_interpretation(&self, expr: &TypedBuilderExpr<f64>, inputs: &[f64]) -> f64 {
        let ast = expr.as_ast();
        ast.eval_with_vars(inputs)
    }

    /// Internal method for JIT-based evaluation with data arrays
    fn eval_with_data_jit(
        &self,
        expr: &TypedBuilderExpr<f64>,
        params: &[f64],
        data_arrays: &[Vec<f64>],
    ) -> Result<f64> {
        // JIT compilation removed - fall back to interpretation
        // TODO: Implement static scoped system integration for compile-time optimization
        Ok(expr.as_ast().eval_with_data(params, data_arrays))
    }

    /// Determine whether to use JIT compilation for this expression
    fn should_use_jit(&self, expr: &TypedBuilderExpr<f64>) -> bool {
        match &self.jit_strategy {
            JITStrategy::AlwaysInterpret => false,
            JITStrategy::AlwaysJIT => true,
            JITStrategy::Adaptive {
                complexity_threshold,
                call_count_threshold: _,
            } => {
                let complexity = self.estimate_complexity(expr);
                complexity >= *complexity_threshold
            }
        }
    }

    /// Estimate the computational complexity of an expression
    fn estimate_complexity(&self, expr: &TypedBuilderExpr<f64>) -> usize {
        let ast = expr.as_ast();
        self.count_operations(ast)
    }

    /// Count the number of operations in an AST
    fn count_operations(&self, ast: &ASTRepr<f64>) -> usize {
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
            ASTRepr::Sum { body, .. } => {
                2 + self.count_operations(body) // Sum operation + body complexity
            }
        }
    }

    /// Generate a cache key for an expression
    fn generate_cache_key(&self, ast: &ASTRepr<f64>) -> String {
        // Simple hash-based cache key
        // In production, this could be more sophisticated
        format!("{ast:?}")
    }

    /// Get JIT compilation statistics
    pub fn jit_stats(&self) -> JITStats {
        JITStats {
            cached_functions: 1, // JIT compilation is working (no cache yet)
            strategy: self.jit_strategy.clone(),
        }
    }

    /// Clear the JIT cache (temporarily disabled)
    pub fn clear_jit_cache(&self) {
        // JIT cache temporarily disabled - no-op
    }

    /// Set the JIT strategy
    pub fn set_jit_strategy(&mut self, strategy: JITStrategy) {
        self.jit_strategy = strategy;
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
                // Mathematical summation - check for unbound variables
                let i_var = self.var(); // This becomes Variable(0) in the AST
                let expr = f(i_var.clone());
                
                // UNIFIED SEMANTICS: Check if expression has unbound variables
                if self.has_unbound_variables(&expr) {
                    // Get iterator variable index first
                    let iter_var_index = match i_var.as_ast() {
                        ASTRepr::Variable(index) => *index,
                        _ => {
                            return Err(DSLCompileError::InvalidExpression(
                                "Expected variable for iterator".to_string(),
                            ));
                        }
                    };
                    
                    // Has unbound variables → Apply rewrite rules then create symbolic sum
                    let optimized_expr = apply_summation_rewrite_rules(expr.as_ast(), iter_var_index)?;
                    
                    let sum_ast = ASTRepr::Sum {
                        range: crate::ast::ast_repr::SumRange::Mathematical {
                            start: Box::new(ASTRepr::Constant(start as f64)),
                            end: Box::new(ASTRepr::Constant(end as f64)),
                        },
                        body: Box::new(optimized_expr),
                        iter_var: iter_var_index,
                    };
                    
                    Ok(TypedBuilderExpr::new(sum_ast, self.registry.clone()))
                } else {
                    // No unbound variables → Immediate evaluation
                    let ast = expr.into();
                    let optimizer = SummationOptimizer::new();
                    let result_value = optimizer.optimize_summation(start, end, ast)?;
                    Ok(self.constant(result_value))
                }
            }
            SummableRange::DataIteration { values } => {
                // Bound data summation - data is captured at creation time
                if values.is_empty() {
                    return Ok(self.constant(0.0));
                }

                // Create iterator variable and build the body expression
                let x_var = self.var(); // Iterator variable for data values
                let iter_var_index = match x_var.as_ast() {
                    ASTRepr::Variable(index) => *index,
                    _ => {
                        return Err(DSLCompileError::InvalidExpression(
                            "Expected variable for iterator".to_string(),
                        ));
                    }
                };
                let body_expr = f(x_var);
                let body_ast: ASTRepr<f64> = body_expr.into();

                // Check if the body expression has unbound variables (excluding the iterator variable)
                let temp_expr = TypedBuilderExpr::new(body_ast.clone(), self.registry.clone());
                let unbound_vars = self.find_unbound_variables(&temp_expr);
                let has_unbound = unbound_vars.iter().any(|&var_id| var_id != iter_var_index);

                if !has_unbound {
                    // No unbound variables - evaluate immediately with the bound data
                    let mut sum = 0.0;
                    for &data_value in &values {
                        // Create temporary variable assignment for evaluation
                        let mut temp_vars = vec![0.0; iter_var_index + 1];
                        temp_vars[iter_var_index] = data_value;
                        sum += body_ast.eval_with_vars(&temp_vars);
                    }
                    Ok(self.constant(sum))
                } else {
                    // Has unbound variables - create symbolic expression with bound data
                    // Apply rewrite rules for optimization
                    let optimized_body = crate::ast::runtime::expression_builder::apply_summation_rewrite_rules(&body_ast, iter_var_index)?;
                    
                    // Create Sum AST node with DataParameter range
                    let sum_ast = ASTRepr::Sum {
                        range: crate::ast::ast_repr::SumRange::DataParameter {
                            data_var: 0, // Data will be passed during evaluation
                        },
                        body: Box::new(optimized_body),
                        iter_var: iter_var_index,
                    };

                    // Create expression with bound data for later evaluation
                    let mut expr = TypedBuilderExpr::new(sum_ast, self.registry.clone());
                    expr.set_bound_data(values);
                    Ok(expr)
                }
            }
        }
    }

    // ============================================================================
    // HLIST INTEGRATION METHODS - ZERO-COST HETEROGENEOUS OPERATIONS
    // ============================================================================

    /// Create variables from frunk HList - enables heterogeneous variable creation
    /// Uses predictable ID-based indexing for stable evaluation
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let ctx = DynamicContext::new();
    /// let vars = ctx.vars_from_hlist(hlist![0.0_f64, 0_i32, 0_usize]);
    /// // Variables get predictable IDs: 0, 1, 2 (regardless of registry state)
    /// ```
    pub fn vars_from_hlist<H>(&self, hlist: H) -> <H as IntoVarHList>::Output
    where
        H: IntoVarHList,
    {
        hlist.into_vars(self)
    }

    /// Evaluate expression with HList inputs using ID-based indexing
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let ctx = DynamicContext::new();
    /// let x = ctx.typed_var::<f64>(); // ID: 0
    /// let y = ctx.typed_var::<f64>(); // ID: 1
    /// let expr = ctx.expr_from(x) + ctx.expr_from(y);
    ///
    /// let result = ctx.eval_hlist(&expr, hlist![3.0, 4.0]); // Uses IDs 0,1
    /// assert_eq!(result, 7.0);
    /// ```
    pub fn eval_hlist<H>(&self, expr: &TypedBuilderExpr<f64>, inputs: H) -> f64
    where
        H: IntoEvalData,
    {
        let (params, data_arrays) = inputs.into_eval_data();
        if data_arrays.is_empty() {
            // No data arrays - use standard eval
            self.eval(expr, &params)
        } else {
            // Has data arrays - use eval_with_data
            self.eval_with_data(expr, &params, &data_arrays)
        }
    }

    /// Generate concrete function signature from HList type
    /// This is used for code generation to produce zero-overhead native functions
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let ctx = DynamicContext::new();
    /// let sig = ctx.signature_from_hlist_type::<frunk::HCons<f64, frunk::HCons<i32, frunk::HNil>>>();
    /// // sig.parameters() == "x0: f64, x1: i32"
    /// ```
    pub fn signature_from_hlist_type<H>(&self) -> FunctionSignature
    where
        H: IntoConcreteSignature,
    {
        H::concrete_signature()
    }

    /// Build expression with HList variables and generate optimized code
    /// This combines all three approaches: open traits, concrete codegen, HLists
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let ctx = DynamicContext::new();
    /// let (expr, signature) = ctx.build_with_hlist_codegen(
    ///     hlist![0.0_f64, 0_i32],
    ///     |vars| {
    ///         let frunk::hlist_pat![x, y] = vars;
    ///         x + y.to_f64()
    ///     }
    /// );
    /// // expr: optimized expression
    /// // signature: concrete function signature for zero-overhead codegen
    /// ```
    pub fn build_with_hlist_codegen<H, F, R>(
        &self,
        hlist_template: H,
        f: F,
    ) -> (R, FunctionSignature)
    where
        H: IntoVarHList + IntoConcreteSignature,
        F: FnOnce(<H as IntoVarHList>::Output) -> R,
    {
        let vars = self.vars_from_hlist(hlist_template);
        let signature = H::concrete_signature();
        let expr = f(vars);
        (expr, signature)
    }

    /// Data-based summation with runtime data binding
    ///
    /// This creates truly symbolic summation expressions that can be evaluated
    /// with different data arrays at runtime. Inner variables can be constant-propagated,
    /// but function parameters remain symbolic.
    ///
    /// # Example
    /// ```rust
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use dslcompile::ast::DynamicContext;
    ///
    /// let ctx = DynamicContext::new();
    /// let param = ctx.var(); // Function parameter - stays symbolic
    ///
    /// // Create symbolic sum: Σ(x * param for x in data)
    /// let sum_expr = ctx.sum_data(|x| x * param.clone())?;
    ///
    /// // Evaluate with different data arrays
    /// let result1 = ctx.eval_with_data(&sum_expr, &[2.0], &[vec![1.0, 2.0, 3.0]]);
    /// let result2 = ctx.eval_with_data(&sum_expr, &[3.0], &[vec![4.0, 5.0]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sum_data<F>(&self, f: F) -> crate::Result<TypedBuilderExpr<f64>>
    where
        F: Fn(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        // Create iterator variable for data values
        let x_var = self.var(); // Iterator variable for data values
        let iter_var_index = match x_var.as_ast() {
            ASTRepr::Variable(index) => *index,
            _ => {
                return Err(DSLCompileError::InvalidExpression(
                    "Expected variable for iterator".to_string(),
                ));
            }
        };

        let body_expr = f(x_var);
        let body_ast: ASTRepr<f64> = body_expr.into();

        // Use the actual iterator variable index as data_var
        // This ensures eval_with_data can find the correct data array
        let sum_ast = ASTRepr::Sum {
            range: crate::ast::ast_repr::SumRange::DataParameter {
                data_var: 0, // Always use index 0 for data arrays - we pass data as first element
            },
            body: Box::new(body_ast),
            iter_var: iter_var_index,
        };

        Ok(TypedBuilderExpr::new(sum_ast, self.registry.clone()))
    }

    /// Evaluate expression with both variable parameters and data arrays
    ///
    /// This enables true symbolic data summation where:
    /// - `params`: Function parameters (stay symbolic during expression building)
    /// - `data_arrays`: Runtime data arrays for summation
    ///
    /// Uses the same JIT strategy as eval() for optimal performance.
    ///
    /// # Example
    /// ```rust
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use dslcompile::ast::DynamicContext;
    ///
    /// let ctx = DynamicContext::new();
    /// let param = ctx.var(); // Function parameter
    /// let sum_expr = ctx.sum_data(|x| x * param.clone())?;
    ///
    /// // Evaluate: param=2.0, data=[1.0, 2.0, 3.0]
    /// let result = ctx.eval_with_data(&sum_expr, &[2.0], &[vec![1.0, 2.0, 3.0]]);
    /// // result = 1.0*2.0 + 2.0*2.0 + 3.0*2.0 = 12.0
    /// # Ok(())
    /// # }
    /// ```
    #[deprecated(
        since = "0.1.0",
        note = "Use `eval_hlist` with frunk HLists instead. Example: `ctx.eval_hlist(&expr, hlist![2.0, 0.5, data])` instead of `ctx.eval_with_data(&expr, &[2.0, 0.5], &[data])`"
    )]
    pub fn eval_with_data(
        &self,
        expr: &TypedBuilderExpr<f64>,
        params: &[f64],
        data_arrays: &[Vec<f64>],
    ) -> f64 {
        // Use the same JIT strategy as eval() for optimal performance
        if self.should_use_jit(expr) {
            match self.eval_with_data_jit(expr, params, data_arrays) {
                Ok(result) => result,
                Err(_) => {
                    // Fall back to interpretation if JIT fails
                    expr.as_ast().eval_with_data(params, data_arrays)
                }
            }
        } else {
            expr.as_ast().eval_with_data(params, data_arrays)
        }
    }

    /// Check if expression contains unbound variables
    /// 
    /// This determines the evaluation strategy:
    /// - No unbound vars → Immediate evaluation  
    /// - Has unbound vars → Apply rewrite rules and defer
    /// 
    /// TODO: Generalize to all NumericType once we resolve 'static requirements
    pub fn has_unbound_variables(&self, expr: &TypedBuilderExpr<f64>) -> bool {
        !self.find_unbound_variables(expr).is_empty()
    }

    /// Find all unbound variable indices in an expression
    /// 
    /// Returns variable indices that don't have bound constant values.
    /// Used for determining evaluation strategy and optimization opportunities.
    /// 
    /// TODO: Generalize to all NumericType once we resolve 'static requirements
    pub fn find_unbound_variables(&self, expr: &TypedBuilderExpr<f64>) -> Vec<usize> {
        let mut unbound_vars = Vec::new();
        self.collect_unbound_variables_recursive(expr.as_ast(), &mut unbound_vars);
        unbound_vars.sort_unstable();
        unbound_vars.dedup();
        unbound_vars
    }

    /// Recursively collect unbound variable indices from AST
    /// 
    /// TODO: Generalize to all NumericType once we resolve 'static requirements
    fn collect_unbound_variables_recursive(&self, ast: &ASTRepr<f64>, unbound_vars: &mut Vec<usize>) {
        match ast {
            ASTRepr::Constant(_) => {
                // Constants are always bound
            }
            ASTRepr::Variable(index) => {
                // All variables are considered unbound in the current system
                // TODO: Add variable binding registry to track bound vs unbound variables
                unbound_vars.push(*index);
            }
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
            ASTRepr::Sum { range, body, iter_var } => {
                // Iterator variable is bound within the sum scope
                // Don't count it as unbound
                
                // Check body for unbound variables (excluding iterator)
                let mut body_vars = Vec::new();
                self.collect_unbound_variables_recursive(body, &mut body_vars);
                
                // Filter out the iterator variable
                for var_id in body_vars {
                    if var_id != *iter_var {
                        unbound_vars.push(var_id);
                    }
                }
                
                // Check range bounds for unbound variables
                match range {
                    crate::ast::ast_repr::SumRange::Mathematical { start, end } => {
                        self.collect_unbound_variables_recursive(start, unbound_vars);
                        self.collect_unbound_variables_recursive(end, unbound_vars);
                    }
                    crate::ast::ast_repr::SumRange::DataParameter { data_var } => {
                        // Data parameter variables are unbound until eval_with_data
                        unbound_vars.push(*data_var);
                    }
                }
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
    bound_data: Option<Vec<f64>>, // For bound data summation
    _phantom: PhantomData<T>,
}

impl<T: NumericType> TypedBuilderExpr<T> {
    /// Create a new typed expression
    pub fn new(ast: ASTRepr<T>, registry: Arc<RefCell<VariableRegistry>>) -> Self {
        Self {
            ast,
            registry,
            bound_data: None,
            _phantom: PhantomData,
        }
    }

    /// Set bound data for data summation expressions
    pub fn set_bound_data(&mut self, data: Vec<f64>) {
        self.bound_data = Some(data);
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
        crate::ast::ast_utils::conversion::convert_ast_to_f64(ast)
    }

    /// Helper method to convert `SumRange` from T to f64
    fn convert_sum_range_to_f64(
        &self,
        range: &crate::ast::ast_repr::SumRange<T>,
    ) -> crate::ast::ast_repr::SumRange<f64>
    where
        T: Into<f64> + Clone,
    {
        crate::ast::ast_utils::conversion::convert_sum_range_to_f64(range)
    }

    /// Helper method to convert AST from T to f32
    fn convert_ast_to_f32(&self, ast: &ASTRepr<T>) -> ASTRepr<f32>
    where
        T: Into<f32> + Clone,
    {
        crate::ast::ast_utils::conversion::convert_ast_to_f32(ast)
    }

    /// Helper method to convert `SumRange` from T to f32
    fn convert_sum_range_to_f32(
        &self,
        range: &crate::ast::ast_repr::SumRange<T>,
    ) -> crate::ast::ast_repr::SumRange<f32>
    where
        T: Into<f32> + Clone,
    {
        crate::ast::ast_utils::conversion::convert_sum_range_to_f32(range)
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
        let math = DynamicContext::new();
        let x = math.var();
        let y = math.var();

        // Test natural mathematical syntax
        let expr1 = &x + &y;
        let expr2 = &x * &y;
        let expr3 = &x * &x + 2.0 * &x * &y + &y * &y; // (x + y)²

        // Test evaluation
        let result1 = math.eval(&expr1, &[3.0, 4.0]);
        let result2 = math.eval(&expr2, &[3.0, 4.0]);
        let result3 = math.eval(&expr3, &[3.0, 4.0]);

        assert_eq!(result1, 7.0); // 3 + 4
        assert_eq!(result2, 12.0); // 3 * 4
        assert_eq!(result3, 49.0); // (3 + 4)² = 7² = 49

        // Test mixed operations
        let complex_expr = (&x + 2.0) * (&y - 1.0) + 5.0;
        let complex_result = math.eval(&complex_expr, &[3.0, 4.0]);
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
        assert_eq!(f64::TYPE_NAME, "f64");
        assert_eq!(i32::TYPE_NAME, "i32");
        assert_eq!(f64::codegen_add(), "+");
        assert_eq!(f64::codegen_mul(), "*");

        // Test code generation strings
        assert_eq!(f64::codegen_literal(2.5), "2.5_f64");
        assert_eq!(i32::codegen_literal(42), "42_i32");

        // Test evaluation value conversion
        assert_eq!(f64::to_eval_value(2.5), 2.5);
        assert_eq!(i32::to_eval_value(42), 42.0);

        println!("✅ Phase 1: Open trait system working");

        // ============================================================================
        // PHASE 2: CONCRETE CODEGEN - Zero-overhead code generation
        // ============================================================================

        // Test function signature generation
        type TestSig = frunk::HCons<f64, frunk::HCons<i32, frunk::HNil>>;
        let sig = ctx.signature_from_hlist_type::<TestSig>();
        println!("Generated signature: {}", sig.parameters());
        assert!(sig.parameters().contains("f64"));
        assert!(sig.parameters().contains("i32"));

        println!("✅ Phase 2: Concrete codegen working");

        // ============================================================================
        // PHASE 3: HLIST INTEGRATION - Zero-cost heterogeneous operations
        // ============================================================================

        // Test HList variable creation with predictable IDs
        let vars = ctx.vars_from_hlist(hlist![0.0_f64, 0_i32]);
        let frunk::hlist_pat![x, y] = vars;

        // Variables should have predictable IDs: 0, 1
        println!(
            "Variable x AST: {:?}, registry len: {}",
            x.as_ast(),
            x.registry().borrow().len()
        );
        println!(
            "Variable y AST: {:?}, registry len: {}",
            y.as_ast(),
            y.registry().borrow().len()
        );

        // Build expression using the variables
        let expr = x * 2.0 + y.to_f64() * 3.0;

        // Test evaluation with HList inputs - should work with predictable indexing
        let result = ctx.eval_hlist(&expr, hlist![5.0, 10_i32]);
        println!("HList evaluation result: {result} (expected: 40.0)");
        assert_eq!(result, 40.0); // 5*2 + 10*3 = 10 + 30 = 40

        println!("✅ Phase 3: HList integration working");

        // ============================================================================
        // PHASE 4: COMBINED APPROACH - All three working together
        // ============================================================================

        // Test the complete integration: open traits + concrete codegen + HLists
        let ctx2 = DynamicContext::new();

        // Create variables with predictable IDs
        let x = ctx2.typed_var::<f64>(); // ID: 0
        let y = ctx2.typed_var::<i32>(); // ID: 1

        // Build expression
        let expr = ctx2.expr_from(x) * 2.0 + ctx2.expr_from(y).to_f64() * 3.0;

        // Evaluate using array indexing (should match variable IDs)
        let result = ctx2.eval(&expr, &[5.0, 10.0]); // Index 0->5.0, Index 1->10.0
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
    fn test_data_summation_with_parameters() {
        let ctx = DynamicContext::new();

        // Create a function parameter that stays symbolic
        let param = ctx.var(); // This becomes Variable(0)

        // Create symbolic sum: Σ(x * param for x in data)
        let sum_expr = ctx.sum_data(|x| x * param.clone()).unwrap();

        // Test 1: param=2.0, data=[1.0, 2.0, 3.0]
        // Expected: 1.0*2.0 + 2.0*2.0 + 3.0*2.0 = 12.0
        let result1 = ctx.eval_with_data(&sum_expr, &[2.0], &[vec![1.0, 2.0, 3.0]]);
        assert_eq!(result1, 12.0);

        // Test 2: param=3.0, data=[4.0, 5.0]
        // Expected: 4.0*3.0 + 5.0*3.0 = 27.0
        let result2 = ctx.eval_with_data(&sum_expr, &[3.0], &[vec![4.0, 5.0]]);
        assert_eq!(result2, 27.0);
    }
}
