//! True Heterogeneous Evaluation using Tuples
//!
//! This module provides truly heterogeneous evaluation where variables can have
//! different types, using Rust's tuple system with automatic type conversion.

use crate::ast::{Scalar, ast_repr::ASTRepr};
use num_traits::{Float, FromPrimitive};

/// Trait for truly heterogeneous evaluation storage
/// 
/// This allows variables of different types to be stored and accessed
/// with automatic type conversion when needed.
pub trait HeterogeneousEval {
    /// Get a variable by index, converting to the requested type T
    fn get_var<T: Scalar + FromPrimitive>(&self, index: usize) -> Option<T>;
    
    /// Get the number of variables stored
    fn var_count(&self) -> usize;
    
    /// Evaluate an expression using heterogeneous variable storage
    fn eval_expr<T: Scalar + Float + FromPrimitive>(&self, ast: &ASTRepr<T>) -> T {
        match ast {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => {
                self.get_var(*index).unwrap_or_else(|| {
                    panic!("Variable index {} out of bounds for heterogeneous evaluation", index)
                })
            },
            ASTRepr::Add(terms) => {
                use num_traits::Zero;
                terms.elements()
                    .map(|term| self.eval_expr(term))
                    .fold(T::zero(), |acc, x| acc + x)
            },
            ASTRepr::Sub(left, right) => self.eval_expr(left) - self.eval_expr(right),
            ASTRepr::Mul(factors) => {
                use num_traits::One;
                factors.elements()
                    .map(|factor| self.eval_expr(factor))
                    .fold(T::one(), |acc, x| acc * x)
            },
            ASTRepr::Div(left, right) => self.eval_expr(left) / self.eval_expr(right),
            ASTRepr::Pow(base, exp) => self.eval_expr(base).powf(self.eval_expr(exp)),
            ASTRepr::Neg(inner) => -self.eval_expr(inner),
            ASTRepr::Ln(inner) => self.eval_expr(inner).ln(),
            ASTRepr::Exp(inner) => self.eval_expr(inner).exp(),
            ASTRepr::Sin(inner) => self.eval_expr(inner).sin(),
            ASTRepr::Cos(inner) => self.eval_expr(inner).cos(),
            ASTRepr::Sqrt(inner) => self.eval_expr(inner).sqrt(),
            // TODO: Handle Sum, Lambda, Let, BoundVar
            _ => panic!("Heterogeneous evaluation not yet implemented for {:?}", std::any::type_name::<T>()),
        }
    }
}

/// Helper trait for converting values to f64 for type conversion
trait ToF64 {
    fn to_f64_lossy(&self) -> f64;
}

impl ToF64 for f64 { fn to_f64_lossy(&self) -> f64 { *self } }
impl ToF64 for f32 { fn to_f64_lossy(&self) -> f64 { *self as f64 } }
impl ToF64 for i32 { fn to_f64_lossy(&self) -> f64 { *self as f64 } }
impl ToF64 for i64 { fn to_f64_lossy(&self) -> f64 { *self as f64 } }
impl ToF64 for u32 { fn to_f64_lossy(&self) -> f64 { *self as f64 } }
impl ToF64 for u64 { fn to_f64_lossy(&self) -> f64 { *self as f64 } }

// Implementation for 1-tuple (single value)
impl<T1> HeterogeneousEval for (T1,)
where
    T1: ToF64 + Copy,
{
    fn get_var<T: Scalar + FromPrimitive>(&self, index: usize) -> Option<T> {
        match index {
            0 => T::from_f64(self.0.to_f64_lossy()),
            _ => None,
        }
    }
    
    fn var_count(&self) -> usize { 1 }
}

// Implementation for 2-tuple
impl<T1, T2> HeterogeneousEval for (T1, T2)
where
    T1: ToF64 + Copy,
    T2: ToF64 + Copy,
{
    fn get_var<T: Scalar + FromPrimitive>(&self, index: usize) -> Option<T> {
        match index {
            0 => T::from_f64(self.0.to_f64_lossy()),
            1 => T::from_f64(self.1.to_f64_lossy()),
            _ => None,
        }
    }
    
    fn var_count(&self) -> usize { 2 }
}

// Implementation for 3-tuple
impl<T1, T2, T3> HeterogeneousEval for (T1, T2, T3)
where
    T1: ToF64 + Copy,
    T2: ToF64 + Copy,
    T3: ToF64 + Copy,
{
    fn get_var<T: Scalar + FromPrimitive>(&self, index: usize) -> Option<T> {
        match index {
            0 => T::from_f64(self.0.to_f64_lossy()),
            1 => T::from_f64(self.1.to_f64_lossy()),
            2 => T::from_f64(self.2.to_f64_lossy()),
            _ => None,
        }
    }
    
    fn var_count(&self) -> usize { 3 }
}

// Implementation for 4-tuple  
impl<T1, T2, T3, T4> HeterogeneousEval for (T1, T2, T3, T4)
where
    T1: ToF64 + Copy,
    T2: ToF64 + Copy,
    T3: ToF64 + Copy,
    T4: ToF64 + Copy,
{
    fn get_var<T: Scalar + FromPrimitive>(&self, index: usize) -> Option<T> {
        match index {
            0 => T::from_f64(self.0.to_f64_lossy()),
            1 => T::from_f64(self.1.to_f64_lossy()),
            2 => T::from_f64(self.2.to_f64_lossy()),
            3 => T::from_f64(self.3.to_f64_lossy()),
            _ => None,
        }
    }
    
    fn var_count(&self) -> usize { 4 }
}

// We can easily extend this to more tuple sizes as needed...

/// Extension trait for ASTRepr to enable heterogeneous evaluation
pub trait HeterogeneousEvalExt<T: Scalar + Float + FromPrimitive> {
    /// Evaluate the expression with heterogeneous variable storage
    fn eval_heterogeneous<H: HeterogeneousEval>(&self, storage: &H) -> T;
}

impl<T: Scalar + Float + FromPrimitive> HeterogeneousEvalExt<T> for ASTRepr<T> {
    fn eval_heterogeneous<H: HeterogeneousEval>(&self, storage: &H) -> T {
        storage.eval_expr(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ASTRepr;

    #[test]
    fn test_heterogeneous_tuple_evaluation() {
        // Create heterogeneous storage: (f64, i32, f32)
        let storage = (3.14_f64, 42_i32, 2.5_f32);
        
        // Create expressions that use different variables
        let x = ASTRepr::<f64>::Variable(0); // Will get 3.14 as f64
        let y = ASTRepr::<f64>::Variable(1); // Will get 42.0 as f64 (converted from i32)
        let z = ASTRepr::<f64>::Variable(2); // Will get 2.5 as f64 (converted from f32)
        
        // Test individual variable access
        assert_eq!(x.eval_heterogeneous(&storage), 3.14);
        assert_eq!(y.eval_heterogeneous(&storage), 42.0);
        assert_eq!(z.eval_heterogeneous(&storage), 2.5);
        
        // Test mixed operations
        let sum = &x + &y; // 3.14 + 42.0 = 45.14
        assert!((sum.eval_heterogeneous(&storage) - 45.14).abs() < 1e-10);
    }

    #[test]
    fn test_type_conversion() {
        let storage = (1_i32, 2.5_f32);
        
        // Evaluate as f64 (should convert)
        let expr = ASTRepr::<f64>::Variable(0);
        assert_eq!(expr.eval_heterogeneous(&storage), 1.0);
        
        // Evaluate as f32 (should convert)
        let expr = ASTRepr::<f32>::Variable(1);
        assert_eq!(expr.eval_heterogeneous(&storage), 2.5);
    }

    #[test]
    #[should_panic(expected = "Variable index 3 out of bounds")]
    fn test_out_of_bounds() {
        let storage = (1.0, 2.0);
        let expr = ASTRepr::<f64>::Variable(3);
        expr.eval_heterogeneous(&storage);
    }
}