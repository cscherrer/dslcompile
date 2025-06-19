//! Test Sum Factoring: Σ(2*x + 3*x) → 5*Σ(x)  
//!
//! This demonstrates the key sum splitting optimization where constants
//! are factored out of summations to reduce computational complexity.

use dslcompile::prelude::*;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

fn main() -> Result<()> {
    println!("🧮 Testing Sum Factoring Optimization");
    println!("====================================\n");

    // Test 1: Create expression that should factor - Σ(2*x + 3*x) 
    println!("📊 Test 1: Sum with additive constants");
    
    // Build the inner expression: 2*x + 3*x
    let x = ASTRepr::Variable(0);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);
    
    let inner = ASTRepr::add_binary(
        ASTRepr::mul_binary(two.clone(), x.clone()),
        ASTRepr::mul_binary(three.clone(), x.clone())
    );
    
    println!("Inner expression: {:?}", inner);
    println!("Expected after factoring: 5*x");
    
    // Create sum over this expression
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let collection = dslcompile::ast::ast_repr::Collection::DataArray(data);
    let sum_expr = ASTRepr::Sum(Box::new(collection));
    
    println!("Sum expression: Sum over [1,2,3,4,5]");
    println!("Expected result: Σ(2*x + 3*x) = 5*Σ(x) = 5*(1+2+3+4+5) = 5*15 = 75\n");
    
    #[cfg(feature = "optimization")]
    {
        println!("🔄 Applying egg optimization to sum...");
        match optimize_simple_sum_splitting(&sum_expr) {
            Ok(optimized_sum) => {
                println!("✅ Sum optimization completed!");
                println!("Optimized: {:?}\n", optimized_sum);
            }
            Err(e) => {
                println!("❌ Sum optimization failed: {}\n", e);
            }
        }
        
        println!("🔄 Applying egg optimization to inner expression...");
        match optimize_simple_sum_splitting(&inner) {
            Ok(optimized_inner) => {
                println!("✅ Inner optimization completed!");
                println!("Original:  {:?}", inner);
                println!("Optimized: {:?}", optimized_inner);
                
                // Check if optimization actually changed the structure
                let orig_str = format!("{:?}", inner);
                let opt_str = format!("{:?}", optimized_inner);
                
                if orig_str != opt_str {
                    println!("✅ Structure changed - optimization applied!");
                } else {
                    println!("⚠️ Structure unchanged - need better rules");
                }
                
                // Test mathematical equivalence
                let test_value = 2.0;
                let orig_result = inner.eval_with_vars(&[test_value]);
                let opt_result = optimized_inner.eval_with_vars(&[test_value]);
                
                println!("\n📊 Evaluation test (x = {}):", test_value);
                println!("  Original:  2*{} + 3*{} = {}", test_value, test_value, orig_result);
                println!("  Optimized: {}", opt_result);
                println!("  Match: {}", (orig_result - opt_result).abs() < 1e-10);
                
            }
            Err(e) => {
                println!("❌ Inner optimization failed: {}", e);
            }
        }
    }
    
    #[cfg(not(feature = "optimization"))]
    {
        println!("🚫 Egg optimization feature not enabled");
        println!("Run with: cargo run --example test_sum_factoring --features egg_optimization");
    }
    
    // Test 2: Simple constant multiplication that should definitely optimize
    println!("\n🔍 Test 2: Simple constant multiplication");
    let simple_mul = ASTRepr::mul_binary(
        ASTRepr::Constant(6.0),
        ASTRepr::Variable(0)
    );
    
    println!("Expression: 6*x");
    
    #[cfg(feature = "optimization")]
    {
        match optimize_simple_sum_splitting(&simple_mul) {
            Ok(opt_simple) => {
                println!("Optimized: {:?}", opt_simple);
                
                let test_val = 3.0;
                let orig = simple_mul.eval_with_vars(&[test_val]);
                let opt = opt_simple.eval_with_vars(&[test_val]);
                println!("Test: 6*{} = {} vs {}", test_val, orig, opt);
            }
            Err(e) => {
                println!("Failed: {}", e);
            }
        }
    }
    
    Ok(())
}