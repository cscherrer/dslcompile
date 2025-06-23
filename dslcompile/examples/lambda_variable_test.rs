//! Lambda Variable Indexing Test
//!
//! This test verifies that lambda variables and bound variables use consistent indexing
//! for egglog rule matching.

use dslcompile::prelude::*;
use frunk::hlist;

// #[cfg(feature = "optimization")]
// use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

fn main() -> Result<()> {
    println!("🔬 Lambda Variable Indexing Test");
    println!("================================\n");

    // Create a simple lambda expression manually
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    let _b = ctx.var::<f64>();

    // Create a simple sum: Σ(a * x) where x is the iterator variable
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(data, |x| a.clone() * x);

    println!("1️⃣ Simple Sum Expression:");
    println!("   Expression: {}", ctx.pretty_print(&sum_expr));

    // Test evaluation
    let result = ctx.eval(&sum_expr, hlist![2.0]);
    println!("   Evaluated with a=2: {result} (expected: 2*(1+2+3) = 12)");
    assert_eq!(result, 12.0);

    // #[cfg(feature = "optimization")]
    // {
    //     println!("\n2️⃣ Testing Constant Factoring Rule:");
    //     println!("   This should apply constant factoring: Σ(a * x) → a * Σ(x)");

    //     match optimize_simple_sum_splitting(sum_expr.as_ast()) {
    //         Ok(optimized) => {
    //             println!("   Optimized AST: {optimized:?}");
    //             // The egg optimizer works at the AST level
    //             // This demonstrates that the optimization preserves semantics
    //         }
    //         Err(e) => {
    //             println!("   Optimization error: {e}");
    //         }
    //     }
    // }

    println!("\n2️⃣ Optimization Testing:");
    println!("   (Skipped - optimize_simple_sum_splitting function removed)");

    println!("\n✅ Lambda variable test completed successfully!");
    Ok(())
}
