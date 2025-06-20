// Quick debug test for StaticContext sum NaN issue
use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("Debug test for StaticContext sum NaN issue");
    
    let mut ctx = StaticContext::new();
    
    // Test individual components first
    println!("Testing individual components:");
    
    // Test simple bound var access
    let simple_test = ctx.new_scope(|scope| {
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| x.clone());
        sum_expr
    });
    let simple_result = simple_test.eval(hlist![]);
    println!("Simple sum: {}", simple_result);
    
    // Test bound var with free var
    let mixed_test = ctx.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| x - mu.clone());
        sum_expr
    });
    let mixed_result = mixed_test.eval(hlist![0.5]);
    println!("Mixed sum (x - mu): {}", mixed_result);
    
    // Test ln function
    let ln_test = ctx.new_scope(|scope| {
        let (sigma, _) = scope.auto_var::<f64>();
        sigma.ln()
    });
    let ln_result = ln_test.eval(hlist![1.0]);
    println!("ln(1.0): {}", ln_result);
    
    // Test the problematic expression step by step
    let step_by_step = ctx.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();
        let (sigma, scope) = scope.auto_var::<f64>();
        
        let log_2pi = scope.constant((2.0 * std::f64::consts::PI).ln());
        let neg_half = scope.constant(-0.5);
        
        let (sum_expr, _) = scope.sum(vec![1.0], |x| {
            println!("In lambda - x value should be bound variable");
            let centered = x - mu.clone();
            let standardized = centered / sigma.clone();
            let squared = standardized.clone() * standardized;
            let ln_sigma = sigma.clone().ln();
            
            let term1 = neg_half.clone() * log_2pi.clone();
            let term2 = ln_sigma;  
            let term3 = neg_half.clone() * squared;
            
            term1 - term2 + term3
        });
        sum_expr
    });
    
    let step_result = step_by_step.eval(hlist![0.0, 1.0]);
    println!("Step by step result: {}", step_result);
}