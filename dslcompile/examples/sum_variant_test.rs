//! Test for the new Sum variant in ASTRepr
//!
//! This demonstrates the basic functionality of the Sum variant for 
//! mathematical range summation.

use dslcompile::ast::{ASTRepr, ast_repr::SumRange};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sum Variant Test ===");
    
    // Test 1: Simple mathematical range summation: Σ(i) from 1 to 5
    // Should equal 1 + 2 + 3 + 4 + 5 = 15
    let sum_expr = ASTRepr::Sum {
        range: SumRange::Mathematical {
            start: Box::new(ASTRepr::Constant(1.0)),
            end: Box::new(ASTRepr::Constant(5.0)),
        },
        body: Box::new(ASTRepr::Variable(0)), // i
        iter_var: 0,
    };
    
    println!("Expression: Σ(i) from 1 to 5");
    let result: f64 = sum_expr.eval_with_vars(&[]);
    println!("Result: {}", result);
    println!("Expected: 15.0");
    assert!((result - 15.0).abs() < 1e-10);
    
    // Test 2: Summation with a body expression: Σ(2*i) from 1 to 3
    // Should equal 2*1 + 2*2 + 2*3 = 2 + 4 + 6 = 12
    let sum_expr2 = ASTRepr::Sum {
        range: SumRange::Mathematical {
            start: Box::new(ASTRepr::Constant(1.0)),
            end: Box::new(ASTRepr::Constant(3.0)),
        },
        body: Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(2.0)),
            Box::new(ASTRepr::Variable(0)), // i
        )),
        iter_var: 0,
    };
    
    println!("\nExpression: Σ(2*i) from 1 to 3");
    let result2: f64 = sum_expr2.eval_with_vars(&[]);
    println!("Result: {}", result2);
    println!("Expected: 12.0");
    assert!((result2 - 12.0).abs() < 1e-10);
    
    // Test 3: Summation with variable range: Σ(i) from 1 to n
    let sum_expr3 = ASTRepr::Sum {
        range: SumRange::Mathematical {
            start: Box::new(ASTRepr::Constant(1.0)),
            end: Box::new(ASTRepr::Variable(1)), // n
        },
        body: Box::new(ASTRepr::Variable(0)), // i
        iter_var: 0,
    };
    
    println!("\nExpression: Σ(i) from 1 to n, where n=4");
    let result3: f64 = sum_expr3.eval_with_vars(&[0.0, 4.0]); // i=0 (unused), n=4
    println!("Result: {}", result3);
    println!("Expected: 10.0 (1+2+3+4)");
    assert!((result3 - 10.0).abs() < 1e-10);
    
    // Test 4: Data parameter summation (placeholder - returns 0 for now)
    let sum_expr4 = ASTRepr::Sum {
        range: SumRange::DataParameter { data_var: 0 },
        body: Box::new(ASTRepr::Variable(1)), // x
        iter_var: 1,
    };
    
    println!("\nExpression: Σ(x in data) - data parameter (placeholder)");
    let result4: f64 = sum_expr4.eval_with_vars(&[]);
    println!("Result: {} (placeholder - should be 0)", result4);
    assert_eq!(result4, 0.0f64);
    
    println!("\n✅ All Sum variant tests passed!");
    Ok(())
}

/// Helper trait to make eval_with_vars work with our test
trait EvalHelper<T> {
    fn eval_with_vars(&self, vars: &[T]) -> Result<T, Box<dyn std::error::Error>>;
}

impl EvalHelper<f64> for ASTRepr<f64> {
    fn eval_with_vars(&self, vars: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
        Ok(ASTRepr::eval_with_vars(self, vars))
    }
} 