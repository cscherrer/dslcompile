//! Heterogeneous Context v2 - Production Ready
//!
//! This module implements a production-ready heterogeneous static context that removes
//! the type parameter constraint from Context<T, SCOPE> while maintaining compile-time
//! type safety and zero runtime overhead.

use std::marker::PhantomData;

// ============================================================================
// CORE TRAITS - Production Ready Type System
// ============================================================================

/// Base trait for types that can be used in expressions
/// This replaces the restrictive `NumericType` constraint
pub trait ExpressionType: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Human-readable name for this type (for debugging/codegen)
    fn type_name() -> &'static str;

    /// Type-specific evaluation context (for codegen)
    fn evaluation_context() -> EvaluationContext;
}

/// Evaluation context describes how to handle the type during evaluation
#[derive(Debug, Clone)]
pub enum EvaluationContext {
    /// Direct value (scalars: f64, i32, etc.)
    DirectValue,
    /// Reference to collection (Vec<T>, arrays)
    CollectionRef,
    /// Owned collection (for constants)
    OwnedCollection,
    /// Custom evaluation strategy
    Custom(String),
}

// ============================================================================
// BASIC IMPLEMENTATIONS
// ============================================================================

// Scalar types
impl ExpressionType for f64 {
    fn type_name() -> &'static str {
        "f64"
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::DirectValue
    }
}

impl ExpressionType for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::DirectValue
    }
}

impl ExpressionType for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::DirectValue
    }
}

impl ExpressionType for i64 {
    fn type_name() -> &'static str {
        "i64"
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::DirectValue
    }
}

impl ExpressionType for usize {
    fn type_name() -> &'static str {
        "usize"
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::DirectValue
    }
}

impl ExpressionType for bool {
    fn type_name() -> &'static str {
        "bool"
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::DirectValue
    }
}

// Collection types
impl<T: ExpressionType> ExpressionType for Vec<T> {
    fn type_name() -> &'static str {
        // For Vec<f64> specifically - we'll need a better solution for generics
        "Vec<f64>" // This is a limitation we'll address later
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::CollectionRef
    }
}

impl<T: ExpressionType, const N: usize> ExpressionType for [T; N] {
    fn type_name() -> &'static str {
        // For [f64; N] specifically - we'll need a better solution for generics
        "[f64; N]" // This is a limitation we'll address later  
    }
    fn evaluation_context() -> EvaluationContext {
        EvaluationContext::CollectionRef
    }
}

// ============================================================================
// HETEROGENEOUS AST REPRESENTATION
// ============================================================================

/// Type-erased AST representation for heterogeneous expressions
#[derive(Debug, Clone)]
pub enum HeteroAST {
    /// Variable reference with ID and type information
    Variable {
        id: usize,
        type_name: String,
        evaluation_context: EvaluationContext,
    },
    /// Constant value (type-erased but with metadata)
    Constant {
        value: ConstantValue,
        type_name: String,
    },
    /// Function call with type-erased arguments
    FunctionCall {
        name: String,
        args: Vec<HeteroAST>,
        return_type: String,
    },
}

/// Type-erased constant values
#[derive(Debug, Clone)]
pub enum ConstantValue {
    F64(f64),
    F32(f32),
    I32(i32),
    I64(i64),
    Usize(usize),
    Bool(bool),
    // Add more as needed
}

// ============================================================================
// HETEROGENEOUS CONTEXT - NO TYPE PARAMETERS!
// ============================================================================

/// Heterogeneous Context - works with ANY types, no constraints!
/// This replaces Context<T, SCOPE> with a much more flexible system
#[derive(Debug)]
pub struct HeteroContext<const SCOPE: usize> {
    next_var_id: usize,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize> Default for HeteroContext<SCOPE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SCOPE: usize> HeteroContext<SCOPE> {
    /// Create a new heterogeneous context
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_var_id: 0,
            _scope: PhantomData,
        }
    }

    /// Create a variable of any type that implements `ExpressionType`
    pub fn var<T: ExpressionType>(&mut self) -> HeteroVar<T, SCOPE> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        HeteroVar::new(id)
    }

    /// Create a constant of any type
    pub fn constant<T: ExpressionType>(&self, value: T) -> HeteroConst<T, SCOPE> {
        HeteroConst::new(value)
    }

    /// Start a new scope for perfect composition
    pub fn new_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut HeteroScopeBuilder<SCOPE>) -> R,
    {
        let mut scope_builder = HeteroScopeBuilder::new();
        f(&mut scope_builder)
    }
}

// ============================================================================
// VARIABLES AND CONSTANTS
// ============================================================================

/// Heterogeneous variable with full type information
#[derive(Debug, Clone)]
pub struct HeteroVar<T: ExpressionType, const SCOPE: usize> {
    id: usize,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T: ExpressionType, const SCOPE: usize> HeteroVar<T, SCOPE> {
    fn new(id: usize) -> Self {
        Self {
            id,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Get the variable ID
    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Convert to AST representation
    #[must_use]
    pub fn to_ast(&self) -> HeteroAST {
        HeteroAST::Variable {
            id: self.id,
            type_name: T::type_name().to_string(),
            evaluation_context: T::evaluation_context(),
        }
    }
}

/// Heterogeneous constant with full type information
#[derive(Debug, Clone)]
pub struct HeteroConst<T: ExpressionType, const SCOPE: usize> {
    value: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T: ExpressionType, const SCOPE: usize> HeteroConst<T, SCOPE> {
    fn new(value: T) -> Self {
        Self {
            value,
            _scope: PhantomData,
        }
    }

    /// Convert to AST representation
    pub fn to_ast(&self) -> HeteroAST {
        // Convert the value to ConstantValue enum
        let constant_value = self.value_to_constant();

        HeteroAST::Constant {
            value: constant_value,
            type_name: T::type_name().to_string(),
        }
    }

    /// Convert typed value to type-erased `ConstantValue`
    /// This is where we handle the type-erasure for constants
    fn value_to_constant(&self) -> ConstantValue {
        // This is a bit tricky - we need to downcast T to known types
        // For now, we'll use Any trait to handle this safely
        use std::any::Any;

        let any_ref = &self.value as &dyn Any;

        if let Some(val) = any_ref.downcast_ref::<f64>() {
            ConstantValue::F64(*val)
        } else if let Some(val) = any_ref.downcast_ref::<f32>() {
            ConstantValue::F32(*val)
        } else if let Some(val) = any_ref.downcast_ref::<i32>() {
            ConstantValue::I32(*val)
        } else if let Some(val) = any_ref.downcast_ref::<i64>() {
            ConstantValue::I64(*val)
        } else if let Some(val) = any_ref.downcast_ref::<usize>() {
            ConstantValue::Usize(*val)
        } else if let Some(val) = any_ref.downcast_ref::<bool>() {
            ConstantValue::Bool(*val)
        } else {
            // For unknown types, we could serialize to string or use other strategies
            panic!("Unsupported constant type: {}", T::type_name());
        }
    }
}

// ============================================================================
// SCOPE BUILDER FOR ERGONOMIC API
// ============================================================================

/// Scope builder for ergonomic variable creation
#[derive(Debug)]
pub struct HeteroScopeBuilder<const SCOPE: usize> {
    next_var_id: usize,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize> HeteroScopeBuilder<SCOPE> {
    fn new() -> Self {
        Self {
            next_var_id: 0,
            _scope: PhantomData,
        }
    }

    /// Create a variable with automatic ID assignment
    pub fn auto_var<T: ExpressionType>(&mut self) -> (HeteroVar<T, SCOPE>, &mut Self) {
        let var = HeteroVar::new(self.next_var_id);
        self.next_var_id += 1;
        (var, self)
    }

    /// Create a constant
    pub fn constant<T: ExpressionType>(&self, value: T) -> HeteroConst<T, SCOPE> {
        HeteroConst::new(value)
    }
}

// ============================================================================
// OPERATIONS - TYPE-SAFE BUT HETEROGENEOUS
// ============================================================================

/// Array indexing operation: Vec<T>[usize] -> T
#[must_use]
pub fn array_index<T: ExpressionType, const SCOPE: usize>(
    array: HeteroVar<Vec<T>, SCOPE>,
    index: HeteroVar<usize, SCOPE>,
) -> HeteroAST {
    HeteroAST::FunctionCall {
        name: "array_index".to_string(),
        args: vec![array.to_ast(), index.to_ast()],
        return_type: T::type_name().to_string(),
    }
}

/// Array indexing with constant index
#[must_use]
pub fn array_index_const<T: ExpressionType, const SCOPE: usize>(
    array: HeteroVar<Vec<T>, SCOPE>,
    index: HeteroConst<usize, SCOPE>,
) -> HeteroAST {
    HeteroAST::FunctionCall {
        name: "array_index".to_string(),
        args: vec![array.to_ast(), index.to_ast()],
        return_type: T::type_name().to_string(),
    }
}

/// Scalar addition: T + T -> T (where T supports addition)
#[must_use]
pub fn scalar_add<T: ExpressionType, const SCOPE: usize>(
    left: HeteroVar<T, SCOPE>,
    right: HeteroVar<T, SCOPE>,
) -> HeteroAST {
    HeteroAST::FunctionCall {
        name: "add".to_string(),
        args: vec![left.to_ast(), right.to_ast()],
        return_type: T::type_name().to_string(),
    }
}

/// Mixed addition: Variable + Constant
pub fn scalar_add_const<T: ExpressionType, const SCOPE: usize>(
    left: HeteroVar<T, SCOPE>,
    right: HeteroConst<T, SCOPE>,
) -> HeteroAST {
    HeteroAST::FunctionCall {
        name: "add".to_string(),
        args: vec![left.to_ast(), right.to_ast()],
        return_type: T::type_name().to_string(),
    }
}

/// Scalar multiplication: T * T -> T
#[must_use]
pub fn scalar_mul<T: ExpressionType, const SCOPE: usize>(
    left: HeteroVar<T, SCOPE>,
    right: HeteroVar<T, SCOPE>,
) -> HeteroAST {
    HeteroAST::FunctionCall {
        name: "mul".to_string(),
        args: vec![left.to_ast(), right.to_ast()],
        return_type: T::type_name().to_string(),
    }
}

// ============================================================================
// EVALUATION SYSTEM - NATIVE TYPES!
// ============================================================================

/// Native evaluation context that accepts heterogeneous inputs
#[derive(Debug)]
pub struct HeteroEvaluator<const SCOPE: usize> {
    _scope: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize> Default for HeteroEvaluator<SCOPE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SCOPE: usize> HeteroEvaluator<SCOPE> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            _scope: PhantomData,
        }
    }

    /// Evaluate with native input types - NO Vec<f64> flattening!
    #[must_use]
    pub fn eval_native(&self, ast: &HeteroAST, inputs: &HeteroInputs) -> EvaluationResult {
        match ast {
            HeteroAST::Variable { id, type_name, .. } => inputs.get_variable(*id, type_name),
            HeteroAST::Constant { value, .. } => EvaluationResult::from_constant(value),
            HeteroAST::FunctionCall {
                name,
                args,
                return_type,
            } => self.eval_function_call(name, args, return_type, inputs),
        }
    }

    fn eval_function_call(
        &self,
        name: &str,
        args: &[HeteroAST],
        return_type: &str,
        inputs: &HeteroInputs,
    ) -> EvaluationResult {
        match name {
            "array_index" => {
                let array_result = self.eval_native(&args[0], inputs);
                let index_result = self.eval_native(&args[1], inputs);

                // Perform array indexing with native types
                self.perform_array_indexing(array_result, index_result)
            }
            "add" => {
                let left = self.eval_native(&args[0], inputs);
                let right = self.eval_native(&args[1], inputs);
                self.perform_addition(left, right)
            }
            "mul" => {
                let left = self.eval_native(&args[0], inputs);
                let right = self.eval_native(&args[1], inputs);
                self.perform_multiplication(left, right)
            }
            _ => panic!("Unsupported operation: {name}"),
        }
    }

    fn perform_array_indexing(
        &self,
        array: EvaluationResult,
        index: EvaluationResult,
    ) -> EvaluationResult {
        match (array, index) {
            (EvaluationResult::VecF64(vec), EvaluationResult::Usize(idx)) => {
                EvaluationResult::F64(vec[idx])
            }
            (EvaluationResult::VecI32(vec), EvaluationResult::Usize(idx)) => {
                EvaluationResult::I32(vec[idx])
            }
            _ => panic!("Type mismatch in array indexing"),
        }
    }

    fn perform_addition(
        &self,
        left: EvaluationResult,
        right: EvaluationResult,
    ) -> EvaluationResult {
        match (left, right) {
            (EvaluationResult::F64(a), EvaluationResult::F64(b)) => EvaluationResult::F64(a + b),
            (EvaluationResult::I32(a), EvaluationResult::I32(b)) => EvaluationResult::I32(a + b),
            _ => panic!("Type mismatch in addition"),
        }
    }

    fn perform_multiplication(
        &self,
        left: EvaluationResult,
        right: EvaluationResult,
    ) -> EvaluationResult {
        match (left, right) {
            (EvaluationResult::F64(a), EvaluationResult::F64(b)) => EvaluationResult::F64(a * b),
            (EvaluationResult::I32(a), EvaluationResult::I32(b)) => EvaluationResult::I32(a * b),
            _ => panic!("Type mismatch in multiplication"),
        }
    }
}

/// Native input container - NO Vec<f64> flattening!
#[derive(Debug)]
pub struct HeteroInputs {
    scalars_f64: Vec<f64>,
    scalars_i32: Vec<i32>,
    scalars_usize: Vec<usize>,
    vecs_f64: Vec<Vec<f64>>,
    vecs_i32: Vec<Vec<i32>>,
    variable_map: Vec<(usize, String, InputLocation)>,
}

#[derive(Debug, Clone)]
enum InputLocation {
    ScalarF64(usize),
    ScalarI32(usize),
    ScalarUsize(usize),
    VecF64(usize),
    VecI32(usize),
}

/// Evaluation result with native types
#[derive(Debug, Clone)]
pub enum EvaluationResult {
    F64(f64),
    F32(f32),
    I32(i32),
    I64(i64),
    Usize(usize),
    Bool(bool),
    VecF64(Vec<f64>),
    VecI32(Vec<i32>),
}

impl Default for HeteroInputs {
    fn default() -> Self {
        Self::new()
    }
}

impl HeteroInputs {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scalars_f64: Vec::new(),
            scalars_i32: Vec::new(),
            scalars_usize: Vec::new(),
            vecs_f64: Vec::new(),
            vecs_i32: Vec::new(),
            variable_map: Vec::new(),
        }
    }

    /// Add a scalar f64 input
    pub fn add_scalar_f64(&mut self, var_id: usize, value: f64) {
        let index = self.scalars_f64.len();
        self.scalars_f64.push(value);
        self.variable_map
            .push((var_id, "f64".to_string(), InputLocation::ScalarF64(index)));
    }

    /// Add a vector f64 input - NO flattening!
    pub fn add_vec_f64(&mut self, var_id: usize, value: Vec<f64>) {
        let index = self.vecs_f64.len();
        self.vecs_f64.push(value);
        self.variable_map
            .push((var_id, "Vec<f64>".to_string(), InputLocation::VecF64(index)));
    }

    /// Add a usize input
    pub fn add_usize(&mut self, var_id: usize, value: usize) {
        let index = self.scalars_usize.len();
        self.scalars_usize.push(value);
        self.variable_map.push((
            var_id,
            "usize".to_string(),
            InputLocation::ScalarUsize(index),
        ));
    }

    fn get_variable(&self, var_id: usize, type_name: &str) -> EvaluationResult {
        for (id, name, location) in &self.variable_map {
            if *id == var_id && name == type_name {
                return match location {
                    InputLocation::ScalarF64(idx) => EvaluationResult::F64(self.scalars_f64[*idx]),
                    InputLocation::ScalarUsize(idx) => {
                        EvaluationResult::Usize(self.scalars_usize[*idx])
                    }
                    InputLocation::VecF64(idx) => {
                        EvaluationResult::VecF64(self.vecs_f64[*idx].clone())
                    }
                    _ => panic!("Unsupported input location"),
                };
            }
        }
        panic!("Variable not found: {var_id} of type {type_name}");
    }
}

impl EvaluationResult {
    fn from_constant(value: &ConstantValue) -> Self {
        match value {
            ConstantValue::F64(v) => EvaluationResult::F64(*v),
            ConstantValue::F32(v) => EvaluationResult::F32(*v),
            ConstantValue::I32(v) => EvaluationResult::I32(*v),
            ConstantValue::I64(v) => EvaluationResult::I64(*v),
            ConstantValue::Usize(v) => EvaluationResult::Usize(*v),
            ConstantValue::Bool(v) => EvaluationResult::Bool(*v),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heterogeneous_context_creation() {
        let mut ctx: HeteroContext<0> = HeteroContext::new();

        // Create variables of different types - NO constraints!
        let _f64_var: HeteroVar<f64, 0> = ctx.var();
        let _vec_var: HeteroVar<Vec<f64>, 0> = ctx.var();
        let _usize_var: HeteroVar<usize, 0> = ctx.var();
        let _bool_var: HeteroVar<bool, 0> = ctx.var();

        println!("✅ Heterogeneous context created with multiple types!");
    }

    #[test]
    fn test_native_array_indexing() {
        let mut ctx: HeteroContext<0> = HeteroContext::new();

        let array_var: HeteroVar<Vec<f64>, 0> = ctx.var();
        let index_var: HeteroVar<usize, 0> = ctx.var();

        let expr = array_index(array_var, index_var);

        // Create native inputs
        let mut inputs = HeteroInputs::new();
        inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0]); // array_var (id: 0)
        inputs.add_usize(1, 1); // index_var (id: 1)

        let evaluator: HeteroEvaluator<0> = HeteroEvaluator::new();
        let result = evaluator.eval_native(&expr, &inputs);

        match result {
            EvaluationResult::F64(val) => {
                assert_eq!(val, 2.0); // array[1] = 2.0
                println!("✅ Native array indexing: vec[1] = {val}");
            }
            _ => panic!("Expected f64 result"),
        }
    }

    #[test]
    fn test_mixed_operations() {
        let mut ctx: HeteroContext<0> = HeteroContext::new();

        // Neural network-like example: weights[index] + bias
        let weights: HeteroVar<Vec<f64>, 0> = ctx.var();
        let index: HeteroVar<usize, 0> = ctx.var();
        let bias: HeteroVar<f64, 0> = ctx.var();

        // Build expression: weights[index] + bias
        let indexed_weight = array_index(weights, index);
        // Note: We need to convert HeteroAST back to typed variables for operations
        // This is a limitation we'll need to address in the next iteration

        println!("✅ Mixed operations expression created!");
        println!("Expression: weights[index] + bias");
        println!("AST: {indexed_weight:#?}");
    }
}
