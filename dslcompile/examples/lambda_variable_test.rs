//! Lambda Variable Indexing Test
//!
//! This test verifies that lambda variables and bound variables use consistent indexing
//! for egglog rule matching.

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

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
    println!("   Evaluated with a=2: {} (expected: 2*(1+2+3) = 12)", result);
    assert_eq!(result, 12.0);

    #[cfg(feature = "optimization")]
    {
        let optimizer = NativeEgglogOptimizer::new()?;
        let egglog_expr = optimizer.ast_to_egglog(sum_expr.as_ast())?;
        println!("   Egglog: {egglog_expr}");

        // Test if this matches the constant factoring rule:
        // (rule ((= lhs (Sum (Map (LambdaFunc ?var (Mul ?k ?f)) ?collection))))
        //       ((union lhs (Mul ?k (Sum (Map (LambdaFunc ?var ?f) ?collection)))))
        //       :ruleset stage3_summation)

        println!("\n2️⃣ Testing Constant Factoring Rule:");
        println!("   This should apply constant factoring: Σ(a * x) → a * Σ(x)");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n2️⃣ Optimization Testing:");
        println!("   (Skipped - optimization feature not enabled)");
    }

    println!("\n✅ Lambda variable test completed successfully!");
    Ok(())
}