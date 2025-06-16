//! Test Sum Splitting Optimization Status
//!
//! This is a focused test to check if sum splitting (Σ(a*x + b*x) → (a+b)*Σ(x)) 
//! is working in the DSLCompile optimization system.

use dslcompile::prelude::*;
use frunk::hlist;
use dslcompile::contexts::dynamic::DynamicContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 SUM SPLITTING TEST: Σ(x + 1) → Σ(x) + Σ(1)");
    println!("========================================\n");

    // Create a dynamic context
    let mut ctx = DynamicContext::new();

    // Create variables
    let x = ctx.var::<f64>();  // This will be our summation variable

    // Create the expression inside the sum: x + 1
    let inner_expr = x + 1.0;

    println!("📋 Test Expression: Σ(x + 1) over some range");
    println!("Expected optimization: Σ(x + 1) → Σ(x) + Σ(1)");

    // Create a sum over a simple range (1..=3)
    let sum_expr = ctx.sum(1..=3, |_iter_var| {
        // Note: In real implementation, we'd use the iter_var
        // For now, let's create a simple test expression
        inner_expr.clone()
    });

    println!("\n🔍 Expression Structure:");
    println!("Sum expression: {sum_expr:?}");

    // Try basic evaluation to see current behavior
    println!("\n📊 Current Evaluation (before optimization):");
    let params = dslcompile::hlist![];
    match sum_expr.eval_hlist(&params) {
        Ok(result) => {
            println!("Sum result: {result}");
            println!("Expected: 1+1 + 2+1 + 3+1 = 3 + 6 = 9");
        }
        Err(e) => println!("Evaluation error: {e:?}"),
    }

    // TODO: Test egglog optimization once we get the basic structure working
    #[cfg(feature = "optimization")]
    {
        println!("\n🔧 Testing Egglog Optimization...");
        use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

        let mut optimizer = NativeEgglogOptimizer::new()?;
        match optimizer.optimize(&sum_expr) {
            Ok(optimized) => {
                println!("✅ Optimized expression: {optimized:?}");

                // Test if optimization worked
                match optimized.eval_hlist(&params) {
                    Ok(opt_result) => {
                        println!("Optimized result: {opt_result}");
                        println!("Result matches: {}", opt_result == 9.0);
                    }
                    Err(e) => println!("Optimized evaluation error: {e:?}"),
                }
            }
            Err(e) => println!("❌ Optimization error: {e:?}"),
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n⚠️  Optimization disabled (missing 'optimization' feature)");
        println!("To test optimization, run with: cargo run --example test_sum_splitting --features optimization");
    }

         println!("\n=== Sum splitting test completed ===");
     Ok(())
} 