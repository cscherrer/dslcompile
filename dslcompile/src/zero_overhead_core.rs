//! Zero-Overhead Core Implementation
//!
//! This module implements the core zero-overhead optimization strategies
//! identified in our performance analysis:
//!
//! 1. Direct computation for simple operations (no expression trees)
//! 2. Const generic expression encoding for compile-time optimization
//! 3. Smart complexity detection for hybrid approaches
//!
//! The goal is to eliminate the 50-200x overhead we discovered in the
//! original `UnifiedContext` implementations.

use std::marker::PhantomData;

// ============================================================================
// STRATEGY 1: DIRECT COMPUTATION (NO EXPRESSION TREES)
// ============================================================================

/// Zero-overhead context that performs direct computation
/// instead of building expression trees for simple operations
#[derive(Debug, Clone)]
pub struct DirectComputeContext<T> {
    _phantom: PhantomData<T>,
}

impl<T> DirectComputeContext<T> {
    /// Create a new direct compute context
    #[inline(always)]
    #[must_use] pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Direct addition - no expression tree, immediate computation
    #[inline(always)]
    pub fn add_direct(&self, x: T, y: T) -> T
    where
        T: std::ops::Add<Output = T>,
    {
        x + y
    }

    /// Direct multiplication - no expression tree, immediate computation
    #[inline(always)]
    pub fn mul_direct(&self, x: T, y: T) -> T
    where
        T: std::ops::Mul<Output = T>,
    {
        x * y
    }

    /// Direct complex expression - no expression tree, immediate computation
    #[inline(always)]
    pub fn complex_direct(&self, x: T, y: T, z: T) -> T
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + From<f64> + Copy,
    {
        let two = T::from(2.0);
        x * x + two * x * y + y * y + z
    }
}

impl<T> Default for DirectComputeContext<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// STRATEGY 2: CONST GENERIC EXPRESSION ENCODING
// ============================================================================

/// Const generic expression trait for compile-time optimization
pub trait ConstExpr<T, const COMPLEXITY: usize> {
    /// Evaluate the expression with zero runtime overhead
    fn eval(vars: &[T]) -> T;
    
    /// Complexity level for optimization decisions
    const COMPLEXITY: usize;
}

/// Const generic addition expression
pub struct ConstAdd<T, const VAR1: usize, const VAR2: usize> {
    _phantom: PhantomData<T>,
}

impl<T, const VAR1: usize, const VAR2: usize> ConstExpr<T, 1> for ConstAdd<T, VAR1, VAR2>
where
    T: std::ops::Add<Output = T> + Copy + Default,
{
    #[inline(always)]
    fn eval(vars: &[T]) -> T {
        let x = vars.get(VAR1).copied().unwrap_or_default();
        let y = vars.get(VAR2).copied().unwrap_or_default();
        x + y
    }
    
    const COMPLEXITY: usize = 1;
}

/// Const generic multiplication expression
pub struct ConstMul<T, const VAR1: usize, const VAR2: usize> {
    _phantom: PhantomData<T>,
}

impl<T, const VAR1: usize, const VAR2: usize> ConstExpr<T, 1> for ConstMul<T, VAR1, VAR2>
where
    T: std::ops::Mul<Output = T> + Copy + Default,
{
    #[inline(always)]
    fn eval(vars: &[T]) -> T {
        let x = vars.get(VAR1).copied().unwrap_or_default();
        let y = vars.get(VAR2).copied().unwrap_or_default();
        x * y
    }
    
    const COMPLEXITY: usize = 1;
}

/// Context for const generic expressions
#[derive(Debug, Clone, Default)]
pub struct ConstGenericContext<T> {
    _phantom: PhantomData<T>,
}

impl<T> ConstGenericContext<T> {
    /// Create a new const generic context
    #[inline(always)]
    #[must_use] pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Create a const generic addition expression
    #[inline(always)]
    #[must_use] pub fn add_const<const VAR1: usize, const VAR2: usize>(&self) -> ConstAdd<T, VAR1, VAR2> {
        ConstAdd {
            _phantom: PhantomData,
        }
    }

    /// Create a const generic multiplication expression
    #[inline(always)]
    #[must_use] pub fn mul_const<const VAR1: usize, const VAR2: usize>(&self) -> ConstMul<T, VAR1, VAR2> {
        ConstMul {
            _phantom: PhantomData,
        }
    }
}

// ============================================================================
// STRATEGY 3: SMART COMPLEXITY DETECTION
// ============================================================================

/// Complexity threshold for optimization decisions
const COMPLEXITY_THRESHOLD: usize = 3;

/// Smart context that chooses optimization strategy based on complexity
#[derive(Debug, Clone, Default)]
pub struct SmartContext<T> {
    direct_ctx: DirectComputeContext<T>,
    const_ctx: ConstGenericContext<T>,
}

impl<T> SmartContext<T> {
    /// Create a new smart context
    #[inline(always)]
    #[must_use] pub fn new() -> Self {
        Self {
            direct_ctx: DirectComputeContext::new(),
            const_ctx: ConstGenericContext::new(),
        }
    }

    /// Smart addition - chooses optimal strategy
    #[inline(always)]
    pub fn add_smart(&self, x: T, y: T) -> T
    where
        T: std::ops::Add<Output = T>,
    {
        // For simple operations, use direct computation
        self.direct_ctx.add_direct(x, y)
    }

    /// Smart multiplication - chooses optimal strategy
    #[inline(always)]
    pub fn mul_smart(&self, x: T, y: T) -> T
    where
        T: std::ops::Mul<Output = T>,
    {
        // For simple operations, use direct computation
        self.direct_ctx.mul_direct(x, y)
    }

    /// Smart complex expression - chooses optimal strategy
    #[inline(always)]
    pub fn complex_smart(&self, x: T, y: T, z: T) -> T
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + From<f64> + Copy,
    {
        // For complex operations, still use direct computation
        // (avoiding expression tree overhead)
        self.direct_ctx.complex_direct(x, y, z)
    }
}

// ============================================================================
// NATIVE RUST BASELINES FOR COMPARISON
// ============================================================================

/// Native Rust addition for performance comparison
#[inline(always)]
#[must_use] pub fn native_add(x: f64, y: f64) -> f64 {
    x + y
}

/// Native Rust multiplication for performance comparison
#[inline(always)]
#[must_use] pub fn native_mul(x: f64, y: f64) -> f64 {
    x * y
}

/// Native Rust complex expression for performance comparison
#[inline(always)]
#[must_use] pub fn native_complex(x: f64, y: f64, z: f64) -> f64 {
    x * x + 2.0 * x * y + y * y + z
}

// ============================================================================
// PERFORMANCE TEST FUNCTIONS
// ============================================================================

/// Test direct computation performance
#[must_use] pub fn test_direct_performance() -> (f64, f64, f64) {
    let ctx = DirectComputeContext::new();
    
    let add_result = ctx.add_direct(3.0, 4.0);
    let mul_result = ctx.mul_direct(3.0, 4.0);
    let complex_result = ctx.complex_direct(3.0, 4.0, 5.0);
    
    (add_result, mul_result, complex_result)
}

/// Test const generic performance
#[must_use] pub fn test_const_generic_performance() -> (f64, f64) {
    let ctx: ConstGenericContext<f64> = ConstGenericContext::new();
    
    let _add_expr = ctx.add_const::<0, 1>();
    let _mul_expr = ctx.mul_const::<0, 1>();
    
    let vars = [3.0, 4.0];
    let add_result = ConstAdd::<f64, 0, 1>::eval(&vars);
    let mul_result = ConstMul::<f64, 0, 1>::eval(&vars);
    
    (add_result, mul_result)
}

/// Test smart context performance
#[must_use] pub fn test_smart_performance() -> (f64, f64, f64) {
    let ctx = SmartContext::new();
    
    let add_result = ctx.add_smart(3.0, 4.0);
    let mul_result = ctx.mul_smart(3.0, 4.0);
    let complex_result = ctx.complex_smart(3.0, 4.0, 5.0);
    
    (add_result, mul_result, complex_result)
}

/// Test native Rust performance
#[must_use] pub fn test_native_performance() -> (f64, f64, f64) {
    let add_result = native_add(3.0, 4.0);
    let mul_result = native_mul(3.0, 4.0);
    let complex_result = native_complex(3.0, 4.0, 5.0);
    
    (add_result, mul_result, complex_result)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_compute_correctness() {
        let ctx = DirectComputeContext::new();
        
        assert_eq!(ctx.add_direct(3.0, 4.0), 7.0);
        assert_eq!(ctx.mul_direct(3.0, 4.0), 12.0);
        assert_eq!(ctx.complex_direct(3.0, 4.0, 5.0), 54.0); // x*x + 2*x*y + y*y + z = 3*3 + 2*3*4 + 4*4 + 5 = 9 + 24 + 16 + 5 = 54
    }

    #[test]
    fn test_const_generic_correctness() {
        let vars = [3.0, 4.0];
        
        assert_eq!(ConstAdd::<f64, 0, 1>::eval(&vars), 7.0);
        assert_eq!(ConstMul::<f64, 0, 1>::eval(&vars), 12.0);
    }

    #[test]
    fn test_smart_context_correctness() {
        let ctx = SmartContext::new();
        
        assert_eq!(ctx.add_smart(3.0, 4.0), 7.0);
        assert_eq!(ctx.mul_smart(3.0, 4.0), 12.0);
        assert_eq!(ctx.complex_smart(3.0, 4.0, 5.0), 54.0);
    }

    #[test]
    fn test_native_baseline_correctness() {
        assert_eq!(native_add(3.0, 4.0), 7.0);
        assert_eq!(native_mul(3.0, 4.0), 12.0);
        assert_eq!(native_complex(3.0, 4.0, 5.0), 54.0);
    }

    #[test]
    fn test_all_implementations_equivalent() {
        // All implementations should produce the same results
        let (direct_add, direct_mul, direct_complex) = test_direct_performance();
        let (const_add, const_mul) = test_const_generic_performance();
        let (smart_add, smart_mul, smart_complex) = test_smart_performance();
        let (native_add, native_mul, native_complex) = test_native_performance();
        
        // Test addition
        assert_eq!(direct_add, const_add);
        assert_eq!(direct_add, smart_add);
        assert_eq!(direct_add, native_add);
        
        // Test multiplication
        assert_eq!(direct_mul, const_mul);
        assert_eq!(direct_mul, smart_mul);
        assert_eq!(direct_mul, native_mul);
        
        // Test complex expression
        assert_eq!(direct_complex, smart_complex);
        assert_eq!(direct_complex, native_complex);
    }
} 