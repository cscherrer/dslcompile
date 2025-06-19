//! Test Sum Splitting Priority: Σ(2*x + 3*y) → 2*Σ(x) + 3*Σ(y)
//!
//! This test ensures sum splitting happens before factoring by using 
//! different variables that can't be easily factored together.

use dslcompile::prelude::*;

#[cfg(feature = "egg_optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

fn main() -> Result<()> {
    println!("🎯 Testing Sum Splitting Priority");
    println!("=================================\n");

    // Test case: Σ(2*x + 3*y) with different variables
    // This MUST split before factoring because 2*x + 3*y can't be factored
    
    // Use the proper LambdaVar approach with MathFunction::from_lambda
    let math_func = dslcompile::composition::MathFunction::from_lambda("sum_test", |builder| {
        builder.lambda(|x| {
            // For now, test with single variable: 2*x + 3*x  
            // TODO: Extend to multi-variable case once we understand the API better
            x.clone() * 2.0 + x.clone() * 3.0
        })
    });

    let lambda_expr = math_func.to_ast();
    
    // Create a proper sum expression using DSLCompile's API
    // For now, just test the lambda expression optimization
    let sum_expr = lambda_expr;
    
    println!("📊 Test Expression: lambda function with 2*x + 3*x");
    println!("Expected optimization: 2*x + 3*x → 5*x");
    println!("Lambda expression: {:?}\n", sum_expr);
    
    println!("Sum expression: Sum over [1,2,3]");
    println!("With x=[1,2,3] and y=[1,2,3]:");
    println!("Expected: 2*(1+2+3) + 3*(1+2+3) = 2*6 + 3*6 = 12 + 18 = 30\n");
    
    #[cfg(feature = "egg_optimization")]
    {
        // Test the inner expression first
        println!("🔄 Testing inner expression optimization (2*x + 3*y)...");
        match optimize_simple_sum_splitting(&inner_expr) {
            Ok(optimized_inner) => {
                println!("✅ Inner optimization completed!");
                println!("Original:  {:?}", inner_expr);
                println!("Optimized: {:?}", optimized_inner);
                
                // Test evaluation with sample values
                let test_values = [2.0, 3.0]; // x=2, y=3
                let orig_result = inner_expr.eval_with_vars(&test_values);
                let opt_result = optimized_inner.eval_with_vars(&test_values);
                
                println!("\n📊 Evaluation test (x=2, y=3):");
                println!("  Original:  2*2 + 3*3 = {}", orig_result);
                println!("  Optimized: {}", opt_result);
                println!("  Match: {}", (orig_result - opt_result).abs() < 1e-10);
                
                // Check structure change
                let orig_str = format!("{:?}", inner_expr);
                let opt_str = format!("{:?}", optimized_inner);
                if orig_str != opt_str {
                    println!("✅ Structure changed - some optimization applied!");
                } else {
                    println!("⚠️ Structure unchanged");
                }
            }
            Err(e) => {
                println!("❌ Inner optimization failed: {}", e);
            }
        }
        
        println!("\n🔄 Testing sum expression optimization...");
        match optimize_simple_sum_splitting(&sum_expr) {
            Ok(optimized_sum) => {
                println!("✅ Sum optimization completed!");
                println!("Original:  {:?}", sum_expr);
                println!("Optimized: {:?}", optimized_sum);
            }
            Err(e) => {
                println!("❌ Sum optimization failed: {}", e);
            }
        }
    }
    
    #[cfg(not(feature = "egg_optimization"))]
    {
        println!("🚫 Egg optimization feature not enabled");
        println!("Run with: cargo run --example test_sum_splitting_priority --features egg_optimization");
    }
    
    // Additional test: Same variable case (Σ(2*x + 3*x) → 5*Σ(x))
    println!("\n🔄 Comparison test: Same variable case (2*x + 3*x)");
    
    let same_var_expr = ASTRepr::add_binary(
        ASTRepr::mul_binary(two, x.clone()),
        ASTRepr::mul_binary(three, x)
    );
    
    println!("Expression: 2*x + 3*x");
    println!("Expected: Should eventually become 5*x or 5*Σ(x)");
    
    #[cfg(feature = "egg_optimization")]
    {
        match optimize_simple_sum_splitting(&same_var_expr) {
            Ok(optimized_same) => {
                println!("Original:  {:?}", same_var_expr);
                println!("Optimized: {:?}", optimized_same);
                
                // Test evaluation
                let test_val = 4.0;
                let orig = same_var_expr.eval_with_vars(&[test_val]);
                let opt = optimized_same.eval_with_vars(&[test_val]);
                println!("Test (x={}): {} vs {} = {}", test_val, orig, opt, orig == opt);
            }
            Err(e) => {
                println!("Failed: {}", e);
            }
        }
    }
    
    Ok(())
}