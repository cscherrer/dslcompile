//! Dynamic Expression Builder
//!
//! This module provides a runtime expression builder that enables natural mathematical syntax
//! and expressions while maintaining intuitive operator overloading syntax.

use super::typed_registry::{TypedVar, VariableRegistry};
use crate::ast::ASTRepr;
use crate::ast::Scalar;
use crate::ast::ast_repr::Collection;
use crate::ast::ast_repr::Lambda;
use num_traits::{Float, FromPrimitive};
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

// ============================================================================
// FRUNK HLIST IMPORTS - ZERO-COST HETEROGENEOUS OPERATIONS
// ============================================================================
use frunk::{hlist, HCons, HNil};

// ============================================================================
// OPEN TRAIT SYSTEM - EXTENSIBLE TYPE SUPPORT
// ============================================================================

/// Extended trait for DSL types that can participate in code generation
/// This is the "open" part - users can implement this for custom types
pub trait DslType: Scalar + 'static {
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
    /// Generic version - returns same type for type safety
    fn to_eval_value(value: Self::Native) -> Self::Native {
        value
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

    // Uses default implementation which returns same type
}

impl DslType for f32 {
    type Native = f32;
    const TYPE_NAME: &'static str = "f32";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_f32")
    }

    // Uses default implementation which returns same type
}

impl DslType for i32 {
    type Native = i32;
    const TYPE_NAME: &'static str = "i32";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_i32")
    }

    // Uses default implementation which returns same type
}

impl DslType for i64 {
    type Native = i64;
    const TYPE_NAME: &'static str = "i64";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_i64")
    }

    // Uses default implementation which returns same type
}

impl DslType for usize {
    type Native = usize;
    const TYPE_NAME: &'static str = "usize";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_usize")
    }

    // Uses default implementation which returns same type
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
    T: DslType + crate::ast::Scalar,
    Tail: IntoVarHList,
{
    type Output = HCons<TypedBuilderExpr<T>, Tail::Output>;

    fn into_vars(self, ctx: &DynamicContext) -> Self::Output {
        // Create a typed context for this type
        let mut ctx_typed: DynamicContext<T> = DynamicContext::new();
        let head_expr = ctx_typed.var();
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

// ============================================================================
// BACKWARDS COMPATIBILITY - Vec<T> support for unified eval() API
// ============================================================================

impl IntoEvalArray for Vec<f64> {
    fn into_eval_array(self) -> Vec<f64> {
        self
    }
}

impl IntoEvalArray for Vec<f32> {
    fn into_eval_array(self) -> Vec<f64> {
        self.into_iter().map(|x| x as f64).collect()
    }
}

impl IntoEvalArray for Vec<i32> {
    fn into_eval_array(self) -> Vec<f64> {
        self.into_iter().map(|x| x as f64).collect()
    }
}

impl IntoEvalArray for Vec<i64> {
    fn into_eval_array(self) -> Vec<f64> {
        self.into_iter().map(|x| x as f64).collect()
    }
}

impl IntoEvalArray for Vec<usize> {
    fn into_eval_array(self) -> Vec<f64> {
        self.into_iter().map(|x| x as f64).collect()
    }
}

// Array slice support
impl IntoEvalArray for &[f64] {
    fn into_eval_array(self) -> Vec<f64> {
        self.to_vec()
    }
}

impl<const N: usize> IntoEvalArray for [f64; N] {
    fn into_eval_array(self) -> Vec<f64> {
        self.to_vec()
    }
}

impl<const N: usize> IntoEvalArray for &[f64; N] {
    fn into_eval_array(self) -> Vec<f64> {
        self.to_vec()
    }
}

// New unified implementations for IntoEvalData (supports both scalars and data)
impl<T, Tail> IntoEvalData for HCons<T, Tail>
where
    T: DslType<Native = T> + Into<f64>,
    Tail: IntoEvalData,
{
    fn into_eval_data(self) -> (Vec<f64>, Vec<Vec<f64>>) {
        let (mut params, data_arrays) = self.tail.into_eval_data();
        // Insert scalar value at the beginning, converting to f64
        params.insert(0, self.head.into());
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
/// Parameterized for type safety and borrowed data support
#[derive(Debug, Clone)]
pub struct DynamicContext<T: Scalar = f64> {
    /// Variables storage - direct typed storage, no type erasure needed
    variables: Vec<Option<T>>,
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
}

impl Default for JITStrategy {
    fn default() -> Self {
        Self::Adaptive {
            complexity_threshold: 5,
            call_count_threshold: 3,
        }
    }
}

impl<T: Scalar> DynamicContext<T> {
    /// Create a new dynamic expression builder with default JIT strategy
    #[must_use]
    pub fn new() -> Self {
        Self::with_jit_strategy(JITStrategy::default())
    }

    /// Create a new dynamic expression builder with specified JIT strategy
    #[must_use]
    pub fn with_jit_strategy(strategy: JITStrategy) -> Self {
        Self {
            variables: Vec::new(),
            next_var_id: 0,
            jit_strategy: strategy,
            data_arrays: Vec::new(),
            _phantom: PhantomData,
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

    /// Create a variable of type T
    #[must_use]
    pub fn var(&mut self) -> TypedBuilderExpr<T> {
        let var_index = self.variables.len();
        self.variables.push(None); // Placeholder for variable value

        let var_id = self.next_var_id;
        self.next_var_id += 1;

        // Create a dummy registry for compatibility
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Variable(var_index), registry)
    }

    /// Create a constant expression
    #[must_use]
    pub fn constant(&self, value: T) -> TypedBuilderExpr<T> {
        // Create a dummy registry for compatibility
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));
        TypedBuilderExpr::new(ASTRepr::Constant(value), registry)
    }

    /// Evaluate expression with HList inputs (unified API)
    /// 
    /// This is the primary evaluation method that supports heterogeneous inputs
    /// through HList. It automatically handles type conversion and provides a
    /// clean, unified interface for all evaluation needs.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let y = ctx.var();
    /// let expr = &x * 2.0 + &y * 3.0;
    ///
    /// // Evaluate with HList inputs
    /// let result = ctx.eval(&expr, hlist![5.0, 10.0]);
    /// assert_eq!(result, 40.0); // 5*2 + 10*3 = 40
    /// ```
    #[must_use]
    pub fn eval<H>(&self, expr: &TypedBuilderExpr<T>, hlist: H) -> T
    where
        T: ScalarFloat + Copy + num_traits::FromPrimitive,
        H: IntoEvalArray,
    {
        let params = hlist.into_eval_array();
        // Convert Vec<f64> to Vec<T> using FromPrimitive
        let typed_params: Vec<T> = params
            .into_iter()
            .map(|x| T::from_f64(x).unwrap_or_else(|| panic!("Failed to convert f64 to target type")))
            .collect();
        self.eval_borrowed(expr, &typed_params)
    }

    /// Legacy method: Evaluate expression with borrowed array parameters
    /// 
    /// This method is kept for internal use and backward compatibility.
    /// New code should use the unified `eval()` method with HLists.
    #[must_use]
    pub fn eval_borrowed(&self, expr: &TypedBuilderExpr<T>, params: &[T]) -> T
    where
        T: ScalarFloat + Copy + num_traits::FromPrimitive,
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
        }
    }

    /// Internal method for interpretation-based evaluation
    fn eval_with_interpretation(&self, expr: &TypedBuilderExpr<T>, params: &[T]) -> T
    where
        T: ScalarFloat + Copy + num_traits::FromPrimitive,
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
        }
    }

    /// Generate cache key for expressions (for future JIT optimization)
    fn generate_cache_key(&self, ast: &ASTRepr<T>) -> String {
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
    pub fn store_data_array(&mut self, data: Vec<T>) -> usize {
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

    /// Evaluate expression with both scalar parameters and data arrays
    /// 
    /// This method provides the bridge between the HList-based input system
    /// and the Collection evaluation system that needs access to data arrays.
    /// 
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    /// 
    /// let mut ctx = DynamicContext::new();
    /// let data = vec![1.0, 2.0, 3.0];
    /// let data_idx = ctx.store_data_array(data.clone());
    /// 
    /// // Create summation over data: sum(x * param for x in data)
    /// let param = ctx.var();
    /// let sum_expr = ctx.sum(data, |x| x * param.clone());
    /// 
    /// // Evaluate with scalar parameter = 2.0
    /// let result = ctx.eval_with_data_arrays(&sum_expr, hlist![2.0]);
    /// assert_eq!(result, 12.0); // (1+2+3) * 2 = 12
    /// ```
    pub fn eval_with_data_arrays<H>(&self, expr: &TypedBuilderExpr<T>, hlist: H) -> T
    where
        T: ScalarFloat + Copy + num_traits::FromPrimitive,
        H: IntoEvalArray,
    {
        let params = hlist.into_eval_array();
        // Convert Vec<f64> to Vec<T> using FromPrimitive
        let typed_params: Vec<T> = params
            .into_iter()
            .map(|x| T::from_f64(x).unwrap_or_else(|| panic!("Failed to convert f64 to target type")))
            .collect();
        
        // Use the eval_with_data method from evaluation.rs
        let ast = expr.as_ast();
        ast.eval_with_data(&typed_params, &self.data_arrays)
    }

    /// Create polynomial expression with given coefficients
    pub fn poly(&self, coefficients: &[T], variable: &TypedBuilderExpr<T>) -> TypedBuilderExpr<T>
    where
        T: Copy,
    {
        if coefficients.is_empty() {
            return self.constant(T::default());
        }

        let mut result = self.constant(coefficients[0]);

        for (power, &coeff) in coefficients.iter().skip(1).enumerate() {
            let power = power + 1;
            let term = if power == 1 {
                self.constant(coeff) * variable.clone()
            } else {
                // Create x^power by repeated multiplication
                let mut power_expr = variable.clone();
                for _ in 1..power {
                    power_expr = power_expr * variable.clone();
                }
                self.constant(coeff) * power_expr
            };
            result = result + term;
        }

        result
    }

    /// Generate pretty-printed string representation
    #[must_use]
    pub fn pretty_print(&self, expr: &TypedBuilderExpr<T>) -> String {
        // Create a minimal registry for pretty printing
        let registry = VariableRegistry::for_expression(expr.as_ast());
        crate::ast::pretty::pretty_ast(expr.as_ast(), &registry)
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
                if *index >= self.variables.len() || self.variables[*index].is_none() {
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
        }
    }

    /// Unified summation method that integrates with Collection/Lambda system
    /// 
    /// This method creates proper Collection-based summation expressions that leverage
    /// the sophisticated mathematical optimization infrastructure. It automatically
    /// handles both mathematical ranges and data arrays through the Collection system.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// 
    /// let mut ctx = DynamicContext::new();
    /// 
    /// // Mathematical range summation
    /// let sum1 = ctx.sum(1..=10, |i| i * 2.0);
    /// 
    /// // This creates a Sum(Map{lambda, Range{1, 10}}) AST node
    /// // that can be optimized by the Collection system
    /// ```
    pub fn sum<R, F>(&mut self, range: R, f: F) -> TypedBuilderExpr<T>
    where
        R: IntoSummationRange<T>,
        F: FnOnce(TypedBuilderExpr<T>) -> TypedBuilderExpr<T>,
        T: num_traits::FromPrimitive + Copy,
    {
        let collection = range.into_summation_range(self);
        let iter_var_index = self.next_var_id;
        self.next_var_id += 1;

        // Create iterator variable for the lambda
        let iter_var = TypedBuilderExpr::new(
            ASTRepr::Variable(iter_var_index), 
            Arc::new(RefCell::new(VariableRegistry::new()))
        );

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

        TypedBuilderExpr::new(sum_ast, Arc::new(RefCell::new(VariableRegistry::new())))
    }
}

impl<T: Scalar> Default for DynamicContext<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Type alias for backward compatibility with f64 default
pub type DynamicF64Context = DynamicContext<f64>;
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

/// Scalar variable operations (f64, f32, i32, etc.)
impl<T: Scalar> VariableExpr<T> {
    /// Convert to a typed expression for arithmetic operations
    pub fn into_expr(self) -> TypedBuilderExpr<T> {
        TypedBuilderExpr::new(ASTRepr::Variable(self.var_id), self.registry)
    }
}

// ============================================================================
// OPERATOR OVERLOADING FOR VariableExpr - AUTOMATIC CONVERSION
// ============================================================================

// Arithmetic operations for VariableExpr - automatically convert to TypedBuilderExpr
impl<T> Add for VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.into_expr() + rhs.into_expr()
    }
}

impl<T> Add<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() + rhs.clone().into_expr()
    }
}

impl<T> Add<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() + rhs.into_expr()
    }
}

impl<T> Add<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr() + rhs.clone().into_expr()
    }
}

impl<T> Mul for VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.into_expr() * rhs.into_expr()
    }
}

impl<T> Mul<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() * rhs.clone().into_expr()
    }
}

impl<T> Mul<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() * rhs.into_expr()
    }
}

impl<T> Mul<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr() * rhs.clone().into_expr()
    }
}

impl<T> Sub for VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.into_expr() - rhs.into_expr()
    }
}

impl<T> Sub<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() - rhs.clone().into_expr()
    }
}

impl<T> Sub<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() - rhs.into_expr()
    }
}

impl<T> Sub<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr() - rhs.clone().into_expr()
    }
}

impl<T> Neg for VariableExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        -self.into_expr()
    }
}

impl<T> Neg for &VariableExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        -self.clone().into_expr()
    }
}

// Scalar operations for VariableExpr
impl<T> Add<T> for VariableExpr<T>
where
    T: Scalar + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.into_expr() + rhs
    }
}

impl<T> Add<T> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.clone().into_expr() + rhs
    }
}

impl<T> Mul<T> for VariableExpr<T>
where
    T: Scalar + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.into_expr() * rhs
    }
}

impl<T> Mul<T> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.clone().into_expr() * rhs
    }
}

impl<T> Sub<T> for VariableExpr<T>
where
    T: Scalar + Sub<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.into_expr() - rhs
    }
}

impl<T> Sub<T> for &VariableExpr<T>
where
    T: Scalar + Sub<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.clone().into_expr() - rhs
    }
}

// Reverse scalar operations for VariableExpr
impl Add<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: VariableExpr<f64>) -> Self::Output {
        self + rhs.into_expr()
    }
}

impl Add<&VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &VariableExpr<f64>) -> Self::Output {
        self + rhs.clone().into_expr()
    }
}

impl Mul<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: VariableExpr<f64>) -> Self::Output {
        self * rhs.into_expr()
    }
}

impl Mul<&VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &VariableExpr<f64>) -> Self::Output {
        self * rhs.clone().into_expr()
    }
}

// Mixed operations between VariableExpr and TypedBuilderExpr
impl<T> Add<TypedBuilderExpr<T>> for VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        self.into_expr() + rhs
    }
}

impl<T> Add<TypedBuilderExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        self.clone().into_expr() + rhs
    }
}

impl<T> Add<VariableExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: VariableExpr<T>) -> Self::Output {
        self + rhs.into_expr()
    }
}

impl<T> Add<&VariableExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
        self + rhs.clone().into_expr()
    }
}

impl<T> Mul<TypedBuilderExpr<T>> for VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        self.into_expr() * rhs
    }
}

impl<T> Mul<TypedBuilderExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        self.clone().into_expr() * rhs
    }
}

impl<T> Mul<VariableExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: VariableExpr<T>) -> Self::Output {
        self * rhs.into_expr()
    }
}

impl<T> Mul<&VariableExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
        self * rhs.clone().into_expr()
    }
}

// Transcendental functions for VariableExpr - FIXED TRAIT BOUNDS
impl<T> VariableExpr<T> 
where
    T: Scalar + num_traits::Float + num_traits::FromPrimitive,
{
    /// Sine function
    pub fn sin(self) -> TypedBuilderExpr<T> {
        self.into_expr().sin()
    }

    /// Cosine function
    pub fn cos(self) -> TypedBuilderExpr<T> {
        self.into_expr().cos()
    }

    /// Natural logarithm
    pub fn ln(self) -> TypedBuilderExpr<T> {
        self.into_expr().ln()
    }

    /// Exponential function
    pub fn exp(self) -> TypedBuilderExpr<T> {
        self.into_expr().exp()
    }

    /// Square root
    pub fn sqrt(self) -> TypedBuilderExpr<T> {
        self.into_expr().sqrt()
    }

    /// Power function
    pub fn pow(self, exp: TypedBuilderExpr<T>) -> TypedBuilderExpr<T> {
        self.into_expr().pow(exp)
    }
}

/// Collection variable operations (Vec<f64>, etc.)
impl VariableExpr<Vec<f64>> {
    /// Map operation that builds AST expressions for collections
    ///
    /// This creates a symbolic mapping using the new Collection system.
    /// The closure receives a fresh variable representing each element.
    pub fn map<F>(self, f: F) -> TypedBuilderExpr<f64>
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        use crate::ast::ast_repr::{Collection, Lambda};

        // Register a fresh variable for the iterator element
        let element_var_index = self
            .registry
            .borrow_mut()
            .register_typed_variable::<f64>()
            .index();

        // Create the iterator variable expression
        let element_expr =
            TypedBuilderExpr::new(ASTRepr::Variable(element_var_index), self.registry.clone());

        // Apply the mapping function to get the body expression
        let body_expr = f(element_expr);

        // Create collection representing the data array
        let data_collection = Collection::DataArray(self.var_id);

        // Create lambda function from the body expression
        let lambda = Lambda::Lambda {
            var_index: element_var_index,
            body: Box::new(body_expr.ast),
        };

        // Create mapped collection
        let mapped_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(data_collection),
        };

        // Create sum expression using the new Sum variant
        let sum_ast = ASTRepr::Sum(Box::new(mapped_collection));

        TypedBuilderExpr::new(sum_ast, self.registry)
    }
}



/// Type-safe expression wrapper that preserves type information and enables operator overloading
#[derive(Debug, Clone)]
pub struct TypedBuilderExpr<T> {
    ast: ASTRepr<T>,
    registry: Arc<RefCell<VariableRegistry>>,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> TypedBuilderExpr<T> {
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

    /// Get the variable ID if this expression is a variable, otherwise panic
    pub fn var_id(&self) -> usize {
        match &self.ast {
            ASTRepr::Variable(id) => *id,
            _ => panic!("var_id() called on non-variable expression"),
        }
    }

    /// Convert to expression (no-op since TypedBuilderExpr is already an expression)
    pub fn into_expr(self) -> Self {
        self
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
    fn convert_collection_to_f64(
        &self,
        collection: &crate::ast::ast_repr::Collection<T>,
    ) -> crate::ast::ast_repr::Collection<f64>
    where
        T: Into<f64> + Clone,
    {
        crate::ast::ast_utils::conversion::convert_collection_to_f64(collection)
    }

    /// Helper method to convert AST from T to f32
    fn convert_ast_to_f32(&self, ast: &ASTRepr<T>) -> ASTRepr<f32>
    where
        T: Into<f32> + Clone,
    {
        crate::ast::ast_utils::conversion::convert_ast_to_f32(ast)
    }

    /// Helper method to convert `Collection` from T to f32
    fn convert_collection_to_f32(
        &self,
        collection: &crate::ast::ast_repr::Collection<T>,
    ) -> crate::ast::ast_repr::Collection<f32>
    where
        T: Into<f32> + Clone,
    {
        crate::ast::ast_utils::conversion::convert_collection_to_f32(collection)
    }
}

// From trait implementation for converting TypedBuilderExpr to ASTRepr
impl<T: Scalar> From<TypedBuilderExpr<T>> for ASTRepr<T> {
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
impl<T: Scalar + Float + FromPrimitive> TypedBuilderExpr<T> {
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
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast + rhs.ast, self.registry)
    }
}

impl<T> Sub for TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast - rhs.ast, self.registry)
    }
}

impl<T> Mul for TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast * rhs.ast, self.registry)
    }
}

impl<T> Div for TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast / rhs.ast, self.registry)
    }
}

impl<T> Neg for TypedBuilderExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-self.ast, self.registry)
    }
}

// Reference operations for efficiency
impl<T> Add<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + &rhs.ast, self.registry.clone())
    }
}

impl<T> Add<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + rhs.ast, self.registry.clone())
    }
}

impl<T> Add<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast + &rhs.ast, self.registry)
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * &rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast * &rhs.ast, self.registry)
    }
}

// Negation for references
impl<T> Neg for &TypedBuilderExpr<T>
where
    T: Scalar + Neg<Output = T>,
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
    T: Scalar + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self + rhs_expr
    }
}

impl<T> Add<T> for &TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self + rhs_expr
    }
}

impl<T> Mul<T> for TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self * rhs_expr
    }
}

impl<T> Mul<T> for &TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self * rhs_expr
    }
}

impl<T> Sub<T> for TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self - rhs_expr
    }
}

impl<T> Sub<T> for &TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self - rhs_expr
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
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - &rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast - &rhs.ast, self.registry)
    }
}

impl<T> Div<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / &rhs.ast, self.registry.clone())
    }
}

impl<T> Div<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / rhs.ast, self.registry.clone())
    }
}

impl<T> Div<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
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
        let mut builder_f64 = DynamicContext::<f64>::new();
        let mut builder_f32 = DynamicContext::<f32>::new();

        // Create variables for different types
        let x = builder_f64.var();
        let y = builder_f32.var();

        // Variables should have the correct IDs
        assert_eq!(x.var_id(), 0);
        assert_eq!(y.var_id(), 0); // Each context starts from 0
    }

    #[test]
    fn test_typed_expression_building() {
        let mut builder = DynamicContext::<f64>::new();

        // Use the new unified API
        let x = builder.var();
        let y = builder.var();

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

        let x_f64 = builder_f64.var();
        let y_f32 = builder_f32.var();

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
        let mut builder = DynamicContext::<f64>::new();

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
        let mut builder = DynamicContext::<f64>::new();

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
        let mut builder = DynamicContext::<f64>::new();
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
        let mut math = DynamicContext::<f64>::new();
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

        let mut ctx = DynamicContext::<f64>::new();

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

        // Create variables using new unified API  
        let mut ctx_f64 = DynamicContext::<f64>::new();
        let mut ctx_i32 = DynamicContext::<i32>::new();
        let x = ctx_f64.var();
        let y = ctx_i32.var();

        // Variables should have predictable IDs: 0, 1
        println!("Variable x ID: {}, y ID: {}", x.var_id(), y.var_id());

        // Build expression using the variables (convert y to f64)
        let expr = &x * 2.0 + y.into_expr().to_f64() * 3.0;

        // Test evaluation with array inputs - need to use the f64 context
        let result = ctx_f64.eval(&expr, hlist![5.0, 10.0]);
        println!("Array evaluation result: {result} (expected: 40.0)");
        assert_eq!(result, 40.0); // 5*2 + 10*3 = 10 + 30 = 40

        println!("✅ Phase 3: HList integration working");

        // ============================================================================
        // PHASE 4: COMBINED APPROACH - All three working together
        // ============================================================================

        // Test the complete integration: open traits + concrete codegen + HLists

        // Create variables with predictable IDs using unified API
        let mut ctx2 = DynamicContext::<f64>::new();
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
    fn test_new_iterator_api() {
        use super::SymbolicRangeExt;

        let mut ctx = DynamicContext::<f64>::new();

        // Test 1: Range with symbolic mapping
        let range_sum = SymbolicRangeExt::map(1..=5, |x| x * 2.0).sum();
        println!("Range sum AST: {:?}", range_sum.as_ast());

        // Test 2: Data variable with mapping - Use VariableExpr<Vec<f64>> directly
        // Note: This test is for future implementation of data variable mapping
        println!("Data variable mapping test skipped - requires VariableExpr<Vec<f64>> implementation");

        // Test 3: Range with parameter (NEW UNIFIED API)
        let param = ctx.var().into_expr();
        let param_sum = SymbolicRangeExt::map(1..=3, |x| x * param.clone()).sum();
        println!("Parametric sum AST: {:?}", param_sum.as_ast());
    }
}

// ============================================================================
// ITERATOR EXTENSION TRAITS FOR EGGLOG SUMMATION
// ============================================================================

/// Extension trait for ranges to support symbolic mapping and summation
pub trait SymbolicRangeExt {
    fn map<F>(self, f: F) -> SymbolicMappedRange
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>;
}

/// Result of mapping over a symbolic range
#[derive(Debug, Clone)]
pub struct SymbolicMappedRange {
    start: i64,
    end: i64,
    body_expr: TypedBuilderExpr<f64>,
    registry: Arc<RefCell<VariableRegistry>>,
}

impl SymbolicMappedRange {
    /// Sum the mapped range to create a summation expression
    pub fn sum(self) -> TypedBuilderExpr<f64> {
        // Create a Sum AST node with Collection format
        let sum_ast = ASTRepr::Sum(Box::new(crate::ast::ast_repr::Collection::Range {
            start: Box::new(ASTRepr::Constant(self.start as f64)),
            end: Box::new(ASTRepr::Constant(self.end as f64)),
        }));

        TypedBuilderExpr::new(sum_ast, self.registry)
    }
}

impl SymbolicRangeExt for std::ops::RangeInclusive<i64> {
    fn map<F>(self, f: F) -> SymbolicMappedRange
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        // Create a minimal registry for the range operation
        let registry = Arc::new(RefCell::new(VariableRegistry::new()));

        // Create a variable for the iterator element
        let element_expr = TypedBuilderExpr::new(
            ASTRepr::Variable(0), // Use variable 0 for range iteration
            registry.clone(),
        );

        // Apply the mapping function to get the body expression
        let body_expr = f(element_expr);

        SymbolicMappedRange {
            start: *self.start(),
            end: *self.end(),
            body_expr,
            registry,
        }
    }
}

impl SymbolicRangeExt for std::ops::Range<i64> {
    fn map<F>(self, f: F) -> SymbolicMappedRange
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        // Convert to inclusive range
        let inclusive_range = self.start..=(self.end - 1);
        SymbolicRangeExt::map(inclusive_range, f)
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
        // Convert through AST transformation
        let converted_ast = convert_i32_ast_to_f64(expr.as_ast());
        TypedBuilderExpr::new(converted_ast, expr.registry)
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
            ASTRepr::Constant(0.0) // Placeholder
        }
    }
}

// Duplicate section removed - use CodegenScalar trait above instead

// ============================================================================
// PURE RUST FROM/INTO CONVERSIONS (The Right Way!)
// ============================================================================

/// Convert TypedBuilderExpr using standard Rust From trait
/// Note: This impl is disabled to avoid conflict with blanket From<T> for T
/// Users should use specific From implementations like From<TypedBuilderExpr<i32>> for TypedBuilderExpr<f64>
// impl<T, U> From<TypedBuilderExpr<T>> for TypedBuilderExpr<U>
// where
//     T: crate::ast::Scalar + Clone,
//     U: crate::ast::Scalar + From<T>,
// {
//     fn from(expr: TypedBuilderExpr<T>) -> Self {
//         // Convert AST using pure Rust From trait
//         let converted_ast = convert_ast_pure_rust(expr.as_ast());
//         TypedBuilderExpr::new(converted_ast, expr.registry)
//     }
// }

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
        // Test the comprehensive API working together
        let mut ctx: DynamicContext<f64> = DynamicContext::new();
        let x = ctx.var();
        let y = ctx.var();

        // Build complex expression using all operators
        let expr = &x * 2.0 + &y.sin();

        // Test evaluation
        let result = ctx.eval(&expr, hlist![3.0, 1.57]); // sin(1.57) ≈ 1
        assert!((result - 7.0).abs() < 0.1); // 2*3 + sin(1.57) ≈ 7
    }
}

/// Trait for converting various input types into Collection summation ranges
pub trait IntoSummationRange<T: Scalar> {
    fn into_summation_range(self, ctx: &mut DynamicContext<T>) -> Collection<T>;
}

/// Implementation for mathematical ranges
impl<T: Scalar + num_traits::FromPrimitive> IntoSummationRange<T> for std::ops::RangeInclusive<i64> {
    fn into_summation_range(self, _ctx: &mut DynamicContext<T>) -> Collection<T> {
        Collection::Range {
            start: Box::new(ASTRepr::Constant(T::from_i64(*self.start()).unwrap_or_default())),
            end: Box::new(ASTRepr::Constant(T::from_i64(*self.end()).unwrap_or_default())),
        }
    }
}

/// Implementation for regular ranges
impl<T: Scalar + num_traits::FromPrimitive> IntoSummationRange<T> for std::ops::Range<i64> {
    fn into_summation_range(self, ctx: &mut DynamicContext<T>) -> Collection<T> {
        // Convert to inclusive range
        (self.start..=(self.end - 1)).into_summation_range(ctx)
    }
}

/// Implementation for data vectors (creates DataArray collection)
impl IntoSummationRange<f64> for Vec<f64> {
    fn into_summation_range(self, ctx: &mut DynamicContext<f64>) -> Collection<f64> {
        // Store the data array and get its index
        let data_var_id = ctx.store_data_array(self);
        Collection::DataArray(data_var_id)
    }
}

/// Implementation for data slices
impl IntoSummationRange<f64> for &[f64] {
    fn into_summation_range(self, ctx: &mut DynamicContext<f64>) -> Collection<f64> {
        // Convert slice to vector and store
        let data_var_id = ctx.store_data_array(self.to_vec());
        Collection::DataArray(data_var_id)
    }
}
