// Quick debug test for StaticContext sum NaN issue
use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("Debug test for StaticContext sum NaN issue");
    
    // Test simple bound var access
    let mut ctx1 = StaticContext::new();
    let simple_test = ctx1.new_scope(|scope| {
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| x.clone());
        sum_expr
    });
    let simple_result = simple_test.eval(hlist![]);
    println!("Simple sum: {}", simple_result);
    
    // Test basic variable access outside of summation
    let mut ctx_basic = StaticContext::new();
    let basic_test = ctx_basic.new_scope(|scope| {
        let (x, _) = scope.auto_var::<f64>();
        x
    });
    
    println!("Basic variable access test:");
    let basic_result = basic_test.eval(hlist![42.0]);
    println!("   x = {} (should be 42.0)", basic_result);
    
    // Test bound var with free var - this is where the issue might be
    let mut ctx2 = StaticContext::new();
    let mixed_test = ctx2.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| x - mu.clone());
        sum_expr
    });
    let mixed_result = mixed_test.eval(hlist![0.5]);
    println!("Mixed sum (x - mu): {} (should be (1-0.5) + (2-0.5) = 2)", mixed_result);
    
    // Test the more complex case that was failing
    let mut ctx3 = StaticContext::new();
    let complex_test = ctx3.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();
        let (sigma, scope) = scope.auto_var::<f64>();
        
        let log_2pi = scope.constant((2.0 * std::f64::consts::PI).ln());
        let neg_half = scope.constant(-0.5);
        
        let (sum_expr, _) = scope.sum(vec![1.0], |x| {
            let centered = x - mu.clone();
            let standardized = centered / sigma.clone();
            let squared = standardized.clone() * standardized;
            
            neg_half.clone() * log_2pi.clone() - sigma.clone().ln() + neg_half.clone() * squared
        });
        sum_expr
    });
    
    let complex_result = complex_test.eval(hlist![0.0, 1.0]);
    println!("Complex result: {}", complex_result);
}