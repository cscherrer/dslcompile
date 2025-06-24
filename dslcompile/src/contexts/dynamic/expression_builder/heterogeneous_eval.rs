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
    
    /// Get a Vec<f64> variable by index (for collection operations)
    fn get_collection_var(&self, index: usize) -> Option<&Vec<f64>>;
    
    /// Get the number of variables stored
    fn var_count(&self) -> usize;
    
    /// Evaluate an expression using heterogeneous variable storage
    fn eval_expr<T: Scalar + Float + FromPrimitive + 'static>(&self, ast: &ASTRepr<T>) -> T {
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
            ASTRepr::Sum(collection) => self.eval_collection_sum(collection),
            // TODO: Handle Lambda, Let, BoundVar
            _ => panic!("Heterogeneous evaluation not yet implemented for {:?}", std::any::type_name::<T>()),
        }
    }
    
    /// Evaluate collection summation with heterogeneous support
    fn eval_collection_sum<T: Scalar + Float + FromPrimitive + 'static>(&self, collection: &crate::ast::ast_repr::Collection<T>) -> T {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Variable(index) => {
                // Try to get as Vec<f64> for collection operations
                if let Some(vec_data) = self.get_collection_var(*index) {
                    use num_traits::Zero;
                    vec_data.iter()
                        .map(|&x| T::from_f64(x).unwrap_or(T::zero()))
                        .fold(T::zero(), |acc, x| acc + x)
                } else {
                    // Fall back to scalar variable
                    self.get_var(*index).unwrap_or(T::zero())
                }
            },
            Collection::Map { lambda, collection: inner_collection } => {
                self.eval_map_collection(lambda, inner_collection)
            },
            Collection::Range { start, end } => {
                let start_val = self.eval_expr(start);
                let end_val = self.eval_expr(end);
                let start_int = start_val.to_f64().unwrap_or(0.0) as i32;
                let end_int = end_val.to_f64().unwrap_or(0.0) as i32;
                
                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    sum = sum + i_val;
                }
                sum
            },
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => self.eval_expr(expr),
            Collection::Constant(data) => {
                data.iter().fold(T::zero(), |acc, x| acc + *x)
            },
            Collection::Filter { .. } => {
                // TODO: Implement filtered collection evaluation
                T::zero()
            }
        }
    }
    
    /// Evaluate map collection operations with heterogeneous support  
    fn eval_map_collection<T: Scalar + Float + FromPrimitive + 'static>(
        &self, 
        lambda: &crate::ast::ast_repr::Lambda<T>, 
        collection: &crate::ast::ast_repr::Collection<T>
    ) -> T {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Variable(index) => {
                // Try to get as Vec<f64> for map operations
                if let Some(vec_data) = self.get_collection_var(*index) {
                    // TODO: Implement proper lambda evaluation for map operations
                    // For now, just return the sum of the collection (simplified)
                    use num_traits::Zero;
                    vec_data.iter()
                        .map(|&x| T::from_f64(x).unwrap_or(T::zero()))
                        .fold(T::zero(), |acc, x| acc + x)
                } else {
                    // Fall back to scalar variable
                    self.get_var(*index).unwrap_or(T::zero())
                }
            },
            _ => {
                // For other collection types, return zero for now
                // TODO: Implement full collection evaluation
                T::zero()
            }
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
    
    fn get_collection_var(&self, _index: usize) -> Option<&Vec<f64>> {
        None // Single scalars don't provide collection variables
    }
    
    fn var_count(&self) -> usize { 1 }
}

// Implementation for single Vec<f64> (collection-only tuple)
impl HeterogeneousEval for (Vec<f64>,) {
    fn get_var<T: Scalar + FromPrimitive>(&self, _index: usize) -> Option<T> {
        None // Vec<f64> can't be converted to scalar
    }
    
    fn get_collection_var(&self, index: usize) -> Option<&Vec<f64>> {
        match index {
            0 => Some(&self.0), // Vec<f64> at position 0
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
    
    fn get_collection_var(&self, _index: usize) -> Option<&Vec<f64>> {
        None // Scalars don't provide collection variables
    }
    
    fn var_count(&self) -> usize { 2 }
}

// Implementation for 3-tuple with Vec<f64> support
impl<T1, T2> HeterogeneousEval for (T1, T2, Vec<f64>)
where
    T1: ToF64 + Copy,
    T2: ToF64 + Copy,
{
    fn get_var<T: Scalar + FromPrimitive>(&self, index: usize) -> Option<T> {
        match index {
            0 => T::from_f64(self.0.to_f64_lossy()),
            1 => T::from_f64(self.1.to_f64_lossy()),
            2 => None, // Vec<f64> can't be converted to scalar
            _ => None,
        }
    }
    
    fn get_collection_var(&self, index: usize) -> Option<&Vec<f64>> {
        match index {
            2 => Some(&self.2), // Vec<f64> at position 2
            _ => None,
        }
    }
    
    fn var_count(&self) -> usize { 3 }
}

// Implementation for regular 3-tuple (all scalars)
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
    
    fn get_collection_var(&self, _index: usize) -> Option<&Vec<f64>> {
        None // All scalars
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
    
    fn get_collection_var(&self, _index: usize) -> Option<&Vec<f64>> {
        None // All scalars  
    }
    
    fn var_count(&self) -> usize { 4 }
}

/// Extension trait for ASTRepr to enable heterogeneous evaluation
pub trait HeterogeneousEvalExt<T: Scalar + Float + FromPrimitive + 'static> {
    /// Evaluate the expression with heterogeneous variable storage
    fn eval_heterogeneous<H: HeterogeneousEval>(&self, storage: &H) -> T;
}

impl<T: Scalar + Float + FromPrimitive + 'static> HeterogeneousEvalExt<T> for ASTRepr<T> {
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
    fn test_vec_f64_collection() {
        let storage = (1.0_f64, 2.0_f64, vec![1.0, 2.0, 3.0]);
        
        // Test collection variable access
        assert_eq!(storage.get_collection_var(2), Some(&vec![1.0, 2.0, 3.0]));
        assert_eq!(storage.get_collection_var(0), None); // Scalar, not collection
        
        // Test that Vec<f64> can't be accessed as scalar
        assert_eq!(storage.get_var::<f64>(2), None);
    }

    #[test]
    #[should_panic(expected = "Variable index 3 out of bounds")]
    fn test_out_of_bounds() {
        let storage = (1.0, 2.0);
        let expr = ASTRepr::<f64>::Variable(3);
        expr.eval_heterogeneous(&storage);
    }
}