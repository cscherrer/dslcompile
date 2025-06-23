//! Test Egg Sum Splitting: Σ(a*x + b*x) → (a+b)*Σ(x)
//!
//! This example demonstrates our egg-based sum splitting optimization
//! using constant factoring rules.

use dslcompile::prelude::*;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

fn main() -> Result<()> {
    println!("🥚 Testing Egg Sum Splitting Optimization");
    println!("========================================\n");

    // Create test expression: 2*x + 3*x
    let x = ASTRepr::Variable(0);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    // Build: 2*x + 3*x
    let inner_expr = ASTRepr::add_binary(
        ASTRepr::mul_binary(two, x.clone()),
        ASTRepr::mul_binary(three, x),
    );

    println!("📊 Test Expression: 2*x + 3*x");
    println!("Expected optimization: (2+3)*x = 5*x");
    println!("Original: {inner_expr:?}\n");

    #[cfg(feature = "optimization")]
    {
        println!("🔄 Applying egg optimization...");
        match optimize_simple_sum_splitting(&inner_expr) {
            Ok(optimized) => {
                println!("✅ Optimization completed!");
                println!("Optimized: {optimized:?}\n");

                // Test that they evaluate to the same result
                let original_result = inner_expr.eval_with_vars(&[2.0]); // x = 2
                let optimized_result = optimized.eval_with_vars(&[2.0]);

                println!("📊 Evaluation test (x = 2.0):");
                println!("  Original:  2*2 + 3*2 = {original_result}");
                println!("  Optimized: {optimized_result} ");
                println!(
                    "  Match: {}",
                    (original_result - optimized_result).abs() < 1e-10
                );

                // Check if the optimization actually changed the structure
                let orig_str = format!("{inner_expr:?}");
                let opt_str = format!("{optimized:?}");

                if orig_str == opt_str {
                    println!("⚠️ Structure unchanged - optimization may not have applied");
                } else {
                    println!("✅ Structure changed - optimization applied!");
                }
            }
            Err(e) => {
                println!("❌ Optimization failed: {e}");
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("🚫 Egg optimization feature not enabled");
        println!(
            "Run with: cargo run --example test_egg_sum_splitting --features egg_optimization"
        );
    }

    // Now test sum splitting with a proper sum
    println!("\n🔍 Testing with actual Sum expression...");

    // Create: Sum(2*x + 3*x) over data
    let data_array = vec![1.0, 2.0, 3.0];
    let collection = dslcompile::ast::ast_repr::Collection::Constant(data_array);
    let sum_expr = ASTRepr::Sum(Box::new(collection));

    println!("Sum expression: {sum_expr:?}");

    #[cfg(feature = "optimization")]
    {
        match optimize_simple_sum_splitting(&sum_expr) {
            Ok(optimized_sum) => {
                println!("✅ Sum optimization completed!");
                println!("Optimized sum: {optimized_sum:?}");
            }
            Err(e) => {
                println!("❌ Sum optimization failed: {e}");
            }
        }
    }

    Ok(())
}
