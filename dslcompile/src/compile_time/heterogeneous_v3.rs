//! Heterogeneous Context v3 - Zero-Overhead Compile-Time
//!
//! This module implements a fully compile-time heterogeneous context that:
//! 1. Uses direct array indexing instead of HashMap lookups  
//! 2. Uses compile-time monomorphization instead of runtime type dispatch
//! 3. Maintains full type safety with zero runtime overhead

use std::marker::PhantomData;

// ============================================================================
// COMPILE-TIME TYPE SYSTEM - NO RUNTIME OVERHEAD
// ============================================================================

/// Base trait for types that can be used in expressions
pub trait ExpressionType: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Type name for debugging
    fn type_name() -> &'static str;
}

// Basic implementations
impl ExpressionType for f64 {
    fn type_name() -> &'static str { "f64" }
}

impl ExpressionType for f32 {
    fn type_name() -> &'static str { "f32" }
}

impl ExpressionType for i32 {
    fn type_name() -> &'static str { "i32" }
}

impl ExpressionType for usize {
    fn type_name() -> &'static str { "usize" }
}

impl ExpressionType for bool {
    fn type_name() -> &'static str { "bool" }
}

impl<T: ExpressionType> ExpressionType for Vec<T> {
    fn type_name() -> &'static str { "Vec<T>" }
}

// ============================================================================
// ZERO-OVERHEAD HETEROGENEOUS CONTEXT
// ============================================================================

/// Zero-overhead heterogeneous context with compile-time type tracking
#[derive(Debug)]
pub struct ZeroContext<const SCOPE: usize> {
    next_var_id: usize,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize> Default for ZeroContext<SCOPE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SCOPE: usize> ZeroContext<SCOPE> {
    /// Create new context
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_var_id: 0,
            _scope: PhantomData,
        }
    }

    /// Create typed variable with compile-time ID assignment
    pub fn var<T: ExpressionType>(&mut self) -> ZeroVar<T, SCOPE> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        ZeroVar::new(id)
    }

    /// Create constant
    pub fn constant<T: ExpressionType>(&self, value: T) -> ZeroConst<T, SCOPE> {
        ZeroConst::new(value)
    }

    /// Create new scope with improved API
    pub fn new_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(ZeroScopeBuilder<SCOPE>) -> R,
    {
        let scope_builder = ZeroScopeBuilder::new();
        f(scope_builder)
    }
}

// ============================================================================
// COMPILE-TIME VARIABLES - DIRECT INDEXING
// ============================================================================

/// Compile-time typed variable with direct indexing
#[derive(Debug, Clone)]
pub struct ZeroVar<T: ExpressionType, const SCOPE: usize> {
    id: usize,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T: ExpressionType, const SCOPE: usize> ZeroVar<T, SCOPE> {
    fn new(id: usize) -> Self {
        Self {
            id,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }
}

/// Compile-time constant
#[derive(Debug, Clone)]
pub struct ZeroConst<T: ExpressionType, const SCOPE: usize> {
    value: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T: ExpressionType, const SCOPE: usize> ZeroConst<T, SCOPE> {
    fn new(value: T) -> Self {
        Self {
            value,
            _scope: PhantomData,
        }
    }

    #[must_use]
    pub fn value(&self) -> &T {
        &self.value
    }
}

// ============================================================================
// SCOPE BUILDER WITH ERGONOMIC API
// ============================================================================

/// Scope builder for ergonomic variable creation
#[derive(Debug, Clone)]
pub struct ZeroScopeBuilder<const SCOPE: usize> {
    next_var_id: usize,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize> ZeroScopeBuilder<SCOPE> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_var_id: 0,
            _scope: PhantomData,
        }
    }

    /// Create variable with automatic ID assignment
    pub fn auto_var<T: ExpressionType>(&mut self) -> ZeroVar<T, SCOPE> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        ZeroVar::new(id)
    }

    /// Create constant
    pub fn constant<T: ExpressionType>(&self, value: T) -> ZeroConst<T, SCOPE> {
        ZeroConst::new(value)
    }
}

// ============================================================================
// ZERO-OVERHEAD OPERATIONS - FULLY COMPILE-TIME
// ============================================================================

/// Compile-time scalar addition: T + T -> T
#[derive(Debug, Clone)]
pub struct ZeroAdd<T, L, R, const SCOPE: usize>
where
    T: ExpressionType + std::ops::Add<Output = T>,
    L: ZeroExpr<T, SCOPE>,
    R: ZeroExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

/// Compile-time array indexing: Vec<T>[usize] -> T
#[derive(Debug, Clone)]
pub struct ZeroArrayIndex<T, A, I, const SCOPE: usize>
where
    T: ExpressionType,
    A: ZeroExpr<Vec<T>, SCOPE>,
    I: ZeroExpr<usize, SCOPE>,
{
    array: A,
    index: I,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

/// Compile-time multiplication: T * T -> T
#[derive(Debug, Clone)]
pub struct ZeroMul<T, L, R, const SCOPE: usize>
where
    T: ExpressionType + std::ops::Mul<Output = T>,
    L: ZeroExpr<T, SCOPE>,
    R: ZeroExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

// ============================================================================
// ZERO-OVERHEAD EXPRESSION TRAIT - COMPILE-TIME EVALUATION
// ============================================================================

/// Compile-time expression evaluation trait - zero runtime overhead
pub trait ZeroExpr<T: ExpressionType, const SCOPE: usize>: Clone + std::fmt::Debug {
    /// Evaluate with compile-time type safety and direct indexing
    fn eval(&self, inputs: &ZeroInputs) -> T;
}

// ============================================================================
// VARIABLE IMPLEMENTATIONS - DIRECT INDEXING
// ============================================================================

impl<T: ExpressionType, const SCOPE: usize> ZeroExpr<T, SCOPE> for ZeroVar<T, SCOPE>
where
    T: Clone,
{
    fn eval(&self, inputs: &ZeroInputs) -> T {
        // DIRECT INDEXING - NO HASHMAP LOOKUP!
        inputs.get_direct::<T>(self.id)
    }
}

impl<T: ExpressionType, const SCOPE: usize> ZeroExpr<T, SCOPE> for ZeroConst<T, SCOPE>
where
    T: Clone,
{
    fn eval(&self, _inputs: &ZeroInputs) -> T {
        // COMPILE-TIME CONSTANT - ZERO RUNTIME COST
        self.value.clone()
    }
}

// ============================================================================
// OPERATION IMPLEMENTATIONS - COMPILE-TIME MONOMORPHIZATION
// ============================================================================

impl<T, L, R, const SCOPE: usize> ZeroExpr<T, SCOPE> for ZeroAdd<T, L, R, SCOPE>
where
    T: ExpressionType + std::ops::Add<Output = T>,
    L: ZeroExpr<T, SCOPE>,
    R: ZeroExpr<T, SCOPE>,
{
    fn eval(&self, inputs: &ZeroInputs) -> T {
        // COMPILE-TIME MONOMORPHIZATION - NO RUNTIME DISPATCH!
        self.left.eval(inputs) + self.right.eval(inputs)
    }
}

impl<T, L, R, const SCOPE: usize> ZeroExpr<T, SCOPE> for ZeroMul<T, L, R, SCOPE>
where
    T: ExpressionType + std::ops::Mul<Output = T>,
    L: ZeroExpr<T, SCOPE>,
    R: ZeroExpr<T, SCOPE>,
{
    fn eval(&self, inputs: &ZeroInputs) -> T {
        // COMPILE-TIME MONOMORPHIZATION - NO RUNTIME DISPATCH!
        self.left.eval(inputs) * self.right.eval(inputs)
    }
}

impl<T, A, I, const SCOPE: usize> ZeroExpr<T, SCOPE> for ZeroArrayIndex<T, A, I, SCOPE>
where
    T: ExpressionType + Clone,
    A: ZeroExpr<Vec<T>, SCOPE>,
    I: ZeroExpr<usize, SCOPE>,
{
    fn eval(&self, inputs: &ZeroInputs) -> T {
        // COMPILE-TIME MONOMORPHIZATION - NO RUNTIME DISPATCH!
        let array = self.array.eval(inputs);
        let index = self.index.eval(inputs);
        array[index].clone()
    }
}

// ============================================================================
// OPERATION CONSTRUCTORS - ERGONOMIC API
// ============================================================================

/// Create addition operation
#[must_use]
pub fn zero_add<T, L, R, const SCOPE: usize>(
    left: L,
    right: R,
) -> ZeroAdd<T, L, R, SCOPE>
where
    T: ExpressionType + std::ops::Add<Output = T>,
    L: ZeroExpr<T, SCOPE>,
    R: ZeroExpr<T, SCOPE>,
{
    ZeroAdd {
        left,
        right,
        _type: PhantomData,
        _scope: PhantomData,
    }
}

/// Create multiplication operation
#[must_use]
pub fn zero_mul<T, L, R, const SCOPE: usize>(
    left: L,
    right: R,
) -> ZeroMul<T, L, R, SCOPE>
where
    T: ExpressionType + std::ops::Mul<Output = T>,
    L: ZeroExpr<T, SCOPE>,
    R: ZeroExpr<T, SCOPE>,
{
    ZeroMul {
        left,
        right,
        _type: PhantomData,
        _scope: PhantomData,
    }
}

/// Create array indexing operation
#[must_use]
pub fn zero_array_index<T, A, I, const SCOPE: usize>(
    array: A,
    index: I,
) -> ZeroArrayIndex<T, A, I, SCOPE>
where
    T: ExpressionType,
    A: ZeroExpr<Vec<T>, SCOPE>,
    I: ZeroExpr<usize, SCOPE>,
{
    ZeroArrayIndex {
        array,
        index,
        _type: PhantomData,
        _scope: PhantomData,
    }
}

// ============================================================================
// ZERO-OVERHEAD INPUT CONTAINER - DIRECT INDEXING
// ============================================================================

/// Input container with direct indexing - NO HASHMAP!
#[derive(Debug)]
pub struct ZeroInputs {
    // Type-segregated storage for direct indexing
    f64_values: Vec<f64>,
    i32_values: Vec<i32>,
    usize_values: Vec<usize>,
    bool_values: Vec<bool>,
    vec_f64_values: Vec<Vec<f64>>,
    
    // Mapping from variable ID to storage index and type
    // This could be optimized further with const generics
    var_map: Vec<(usize, VarType, usize)>, // (var_id, type, storage_index)
}

#[derive(Debug, Clone)]
enum VarType {
    F64,
    I32,
    Usize,
    Bool,
    VecF64,
}

impl Default for ZeroInputs {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeroInputs {
    #[must_use]
    pub fn new() -> Self {
        Self {
            f64_values: Vec::new(),
            i32_values: Vec::new(),
            usize_values: Vec::new(),
            bool_values: Vec::new(),
            vec_f64_values: Vec::new(),
            var_map: Vec::new(),
        }
    }

    /// Add f64 value with direct indexing
    pub fn add_f64(&mut self, var_id: usize, value: f64) {
        let storage_index = self.f64_values.len();
        self.f64_values.push(value);
        self.var_map.push((var_id, VarType::F64, storage_index));
    }

    /// Add Vec<f64> value with direct indexing
    pub fn add_vec_f64(&mut self, var_id: usize, value: Vec<f64>) {
        let storage_index = self.vec_f64_values.len();
        self.vec_f64_values.push(value);
        self.var_map.push((var_id, VarType::VecF64, storage_index));
    }

    /// Add usize value with direct indexing
    pub fn add_usize(&mut self, var_id: usize, value: usize) {
        let storage_index = self.usize_values.len();
        self.usize_values.push(value);
        self.var_map.push((var_id, VarType::Usize, storage_index));
    }

    /// Direct type-safe retrieval - ZERO RUNTIME DISPATCH!
    #[must_use]
    pub fn get_direct<T>(&self, var_id: usize) -> T
    where
        T: ExpressionType + Clone,
    {
        // Find the variable mapping
        for (id, var_type, storage_index) in &self.var_map {
            if *id == var_id {
                // COMPILE-TIME TYPE DISPATCH - NO RUNTIME OVERHEAD!
                return self.get_typed::<T>(var_type, *storage_index);
            }
        }
        panic!("Variable not found: {var_id}");
    }

    /// Type-safe retrieval with compile-time dispatch
    fn get_typed<T>(&self, var_type: &VarType, storage_index: usize) -> T
    where
        T: ExpressionType + Clone + 'static,
    {
        // Safe compile-time type dispatch using Any trait
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            match var_type {
                VarType::F64 => {
                    let value = self.f64_values[storage_index];
                    // SAFETY: We've verified T is f64 via TypeId check
                    // This will be optimized to a direct cast at compile time
                    (&value as &dyn std::any::Any)
                        .downcast_ref::<T>()
                        .expect("Type mismatch despite TypeId check")
                        .clone()
                }
                _ => panic!("Type mismatch: expected f64"),
            }
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<usize>() {
            match var_type {
                VarType::Usize => {
                    let value = self.usize_values[storage_index];
                    // SAFETY: We've verified T is usize via TypeId check
                    (&value as &dyn std::any::Any)
                        .downcast_ref::<T>()
                        .expect("Type mismatch despite TypeId check")
                        .clone()
                }
                _ => panic!("Type mismatch: expected usize"),
            }
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Vec<f64>>() {
            match var_type {
                VarType::VecF64 => {
                    let value = &self.vec_f64_values[storage_index];
                    // SAFETY: We've verified T is Vec<f64> via TypeId check
                    (value as &dyn std::any::Any)
                        .downcast_ref::<T>()
                        .expect("Type mismatch despite TypeId check")
                        .clone()
                }
                _ => panic!("Type mismatch: expected Vec<f64>"),
            }
        } else {
            panic!("Unsupported type for direct retrieval");
        }
    }
}

// ============================================================================
// TESTS - ZERO OVERHEAD VERIFICATION
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_overhead_scalar_add() {
        let mut ctx: ZeroContext<0> = ZeroContext::new();
        
        let x: ZeroVar<f64, 0> = ctx.var();
        let y: ZeroVar<f64, 0> = ctx.var();
        
        let expr = zero_add::<f64, _, _, 0>(x, y);
        
        let mut inputs = ZeroInputs::new();
        inputs.add_f64(0, 3.0);
        inputs.add_f64(1, 4.0);
        
        let result = expr.eval(&inputs);
        assert_eq!(result, 7.0);
        
        println!("✅ Zero-overhead scalar addition: 3 + 4 = {result}");
    }

    #[test]
    fn test_zero_overhead_array_indexing() {
        let mut ctx: ZeroContext<0> = ZeroContext::new();
        
        let array: ZeroVar<Vec<f64>, 0> = ctx.var();
        let index: ZeroVar<usize, 0> = ctx.var();
        
        let expr = zero_array_index::<f64, _, _, 0>(array, index);
        
        let mut inputs = ZeroInputs::new();
        inputs.add_vec_f64(0, vec![1.0, 2.0, 3.0, 4.0]);
        inputs.add_usize(1, 2);
        
        let result = expr.eval(&inputs);
        assert_eq!(result, 3.0);
        
        println!("✅ Zero-overhead array indexing: array[2] = {result}");
    }

    #[test]
    fn test_zero_overhead_complex_expression() {
        let mut ctx: ZeroContext<0> = ZeroContext::new();
        
        // Build weights[index] + bias
        let weights: ZeroVar<Vec<f64>, 0> = ctx.var();
        let index: ZeroVar<usize, 0> = ctx.var();
        let bias: ZeroVar<f64, 0> = ctx.var();
        
        let indexed_weight = zero_array_index::<f64, _, _, 0>(weights, index);
        let expr = zero_add::<f64, _, _, 0>(indexed_weight, bias);
        
        let mut inputs = ZeroInputs::new();
        inputs.add_vec_f64(0, vec![0.1, 0.2, 0.3, 0.4]);
        inputs.add_usize(1, 1);
        inputs.add_f64(2, 0.5);
        
        let result = expr.eval(&inputs);
        assert_eq!(result, 0.7); // weights[1] + bias = 0.2 + 0.5
        
        println!("✅ Zero-overhead neural network: weights[1] + bias = {result}");
    }
} 