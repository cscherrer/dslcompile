//! Runtime Expression Building
//!
//! This module provides the runtime expression building system that enables
//! data-aware expression construction with pattern recognition and optimization.

pub mod expression_builder;
pub mod typed_registry;

// Re-export the main types
pub use expression_builder::{DynamicContext, DynamicExpr};
pub use typed_registry::{TypeCategory, TypedVar, VariableRegistry};

// Legacy type aliases removed - use DynamicContext directly for runtime expression building
// Use StaticContext for compile-time optimized expressions

#[cfg(test)]
mod tests {
    use super::*;
    use frunk::hlist;

    #[test]
    fn test_simple_sum_evaluation() {
        // Test 1: Simple sum Σ(x) for x in [1,2,3] should equal 6
        let mut ctx1 = DynamicContext::new();
        let data1 = vec![1.0, 2.0, 3.0];
        let sum_expr1 = ctx1.sum(data1.clone(), |x| x);
        
        println!("Test 1 AST: {:?}", sum_expr1.as_ast());
        
        // The AST shows Collection::Variable(1), so put a placeholder at index 0 and data at index 1
        let result1 = ctx1.eval(&sum_expr1, hlist![0.0, data1.clone()]);
        assert_eq!(result1, 6.0, "Simple sum Σ(x) for [1,2,3] should equal 6, got {}", result1);
    }

    #[test]
    fn test_parameterized_sum_evaluation() {
        // Test 2: Parameterized sum Σ(a * x) for a=2, x in [1,2,3] should equal 12
        let mut ctx2 = DynamicContext::new();
        let a = ctx2.var::<f64>();
        let data2 = vec![1.0, 2.0, 3.0];
        let sum_expr2 = ctx2.sum(data2.clone(), |x| a.clone() * x);
        
        println!("Test 2 AST: {:?}", sum_expr2.as_ast());
        
        // The AST should show Variable(0) for 'a' and Collection::Variable(2) for data
        // So we need: [a_value, placeholder, data]
        let result2 = ctx2.eval(&sum_expr2, hlist![2.0, 0.0, data2.clone()]);
        assert_eq!(result2, 12.0, "Parameterized sum Σ(a * x) for a=2, x in [1,2,3] should equal 12, got {}", result2);
    }

    #[test]
    fn test_sum_evaluation_bug_reproduction() {
        // Test that reproduces the bug where sum expressions evaluate to 0
        let mut ctx = DynamicContext::new();
        let data = vec![1.0, 2.0, 3.0];
        let sum_expr = ctx.sum(data.clone(), |x| x);
        
        // This should just work - no manual index management needed
        let result = ctx.eval(&sum_expr, hlist![data]);
        assert_eq!(result, 6.0, "Simple sum Σ(x) for [1,2,3] should equal 6, got {}", result);
    }

    #[test]
    fn test_parameterized_sum_evaluation_bug() {
        // Test parameterized sum: Σ(a * x) for a=2, x=[1,2,3] should equal 2*(1+2+3) = 12
        let mut ctx = DynamicContext::new();
        let a = ctx.var::<f64>();  // Variable 0
        let data = vec![1.0, 2.0, 3.0];
        let sum_expr = ctx.sum(data, |x| a.clone() * x);
        
        let result = ctx.eval(&sum_expr, hlist![2.0]);
        assert_eq!(result, 12.0, "Parameterized sum Σ(a * x) for a=2, x=[1,2,3] should equal 12, got {}", result);
    }
}
