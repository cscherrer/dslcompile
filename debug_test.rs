use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
// use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

fn main() {
    #[cfg(feature = "optimization")]
    {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        
        // Test 1: Simple sum with collection identity
        let data = vec![1.0, 2.0, 3.0];
        let sum_expr = ctx.sum(data.clone(), |item| &x * item);
        let original_ast = ctx.to_ast(&sum_expr);
        
        println!("Original AST: {:?}", original_ast);
        
        // Perform round-trip: AST → MathLang → AST
        // Optimization functionality removed
        let result: Result<_, ()> = Ok(original_ast.clone());
        if let Ok(optimized_ast) = result {
            println!("Optimized AST: {:?}", optimized_ast);
            
            // Test semantic equivalence
            let test_x = 2.0;
            let original_result = ctx.eval(&sum_expr, hlist![test_x]);
            let optimized_result = optimized_ast.eval_with_vars(&[test_x]);
            
            let expected = test_x * (1.0 + 2.0 + 3.0);
            
            println!("Original result: {}", original_result);
            println!("Optimized result: {}", optimized_result);
            println!("Expected: {}", expected);
            
            println!("Original diff from expected: {}", (original_result - expected).abs());
            println!("Optimized diff from expected: {}", (optimized_result - expected).abs());
        } else {
            println!("Optimization failed: {:?}", result);
        }
    }
    
    #[cfg(not(feature = "optimization"))]
    {
        println!("Optimization feature not enabled");
    }
}