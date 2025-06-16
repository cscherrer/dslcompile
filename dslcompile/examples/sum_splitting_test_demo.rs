//! Sum Splitting Test Demo
//!
//! This demo specifically tests whether sum splitting optimization is working correctly.
//! Sum splitting is the rule: Σ(f + g) = Σ(f) + Σ(g)
//!
//! We'll create expressions that should benefit from sum splitting and verify
//! that the optimization is actually being applied.

use dslcompile::{
    ast::ast_utils::visitors::{OperationCountVisitor, SummationAwareCostVisitor},
    prelude::*,
};
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🔄 Sum Splitting Test Demo");
    println!("==========================\n");

    // =======================================================================
    // 1. Create Expression That Should Benefit from Sum Splitting
    // =======================================================================

    println!("1️⃣ Creating Expression for Sum Splitting Test");
    println!("----------------------------------------------");

    let mut ctx = DynamicContext::new();

    // Create variables
    let a = ctx.var::<f64>(); // Coefficient
    let b = ctx.var::<f64>(); // Another coefficient
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Create expression: Σ(a * x_i + b * x_i) for x_i in data
    // This should split into: Σ(a * x_i) + Σ(b * x_i) = a * Σ(x_i) + b * Σ(x_i) = (a + b) * Σ(x_i)
    let sum_expr = ctx.sum(&data, |x_i| &a * &x_i + &b * &x_i);

    println!(
        "✅ Created expression: Σ(a * x_i + b * x_i) for x_i in {:?}",
        data
    );
    println!(
        "   Expected optimization: Σ(a * x_i + b * x_i) → Σ(a * x_i) + Σ(b * x_i) → a * Σ(x_i) + b * Σ(x_i) → (a + b) * Σ(x_i)"
    );

    // =======================================================================
    // 2. Analyze Original Expression
    // =======================================================================

    println!("\n2️⃣ Analyzing Original Expression");
    println!("---------------------------------");

    let original_ast = ctx.to_ast(&sum_expr);
    let original_ops = OperationCountVisitor::count_operations(&original_ast);
    let original_cost = SummationAwareCostVisitor::compute_cost(&original_ast);

    println!("Original expression analysis:");
    println!("   Operations: {}", original_ops);
    println!("   Cost (new model): {}", original_cost);
    println!("   AST structure: {:?}", original_ast);

    // =======================================================================
    // 3. Test Evaluation Before Optimization
    // =======================================================================

    println!("\n3️⃣ Testing Evaluation Before Optimization");
    println!("------------------------------------------");

    // Test with a=2, b=3, so (a+b) = 5
    // Σ(x_i) for [1,2,3,4,5] = 15
    // Expected result: 5 * 15 = 75
    let test_values = [2.0, 3.0]; // a=2, b=3
    let original_result = ctx.eval(&sum_expr, hlist![2.0, 3.0]);
    println!("Test evaluation (a=2, b=3): {}", original_result);
    println!("Expected result: (2+3) * (1+2+3+4+5) = 5 * 15 = 75");

    // =======================================================================
    // 4. Apply Optimization
    // =======================================================================

    #[cfg(feature = "optimization")]
    {
        println!("\n4️⃣ Applying Sum Splitting Optimization");
        println!("---------------------------------------");

        let mut optimizer = NativeEgglogOptimizer::new()?;

        match optimizer.optimize(&original_ast) {
            Ok(optimized_ast) => {
                let optimized_ops = OperationCountVisitor::count_operations(&optimized_ast);
                let optimized_cost = SummationAwareCostVisitor::compute_cost(&optimized_ast);

                println!("✅ Optimization completed!");
                println!("Optimized expression analysis:");
                println!(
                    "   Operations: {} → {} (change: {})",
                    original_ops,
                    optimized_ops,
                    optimized_ops as i32 - original_ops as i32
                );
                println!(
                    "   Cost: {} → {} (change: {})",
                    original_cost,
                    optimized_cost,
                    optimized_cost as i32 - original_cost as i32
                );
                println!("   AST structure: {:?}", optimized_ast);

                // Test that optimization preserves semantics
                let optimized_result = optimized_ast.eval_with_vars(&test_values);
                println!("\nSemantic preservation test:");
                println!("   Original result:  {}", original_result);
                println!("   Optimized result: {}", optimized_result);
                println!(
                    "   Difference: {:.2e}",
                    (original_result - optimized_result).abs()
                );

                if (original_result - optimized_result).abs() < 1e-10 {
                    println!("   ✅ Semantics preserved!");
                } else {
                    println!("   ❌ Semantics NOT preserved!");
                }

                // Check if sum splitting actually occurred
                if optimized_ops < original_ops {
                    println!("\n🎉 Sum splitting optimization successful!");
                    println!(
                        "   Reduced operations from {} to {}",
                        original_ops, optimized_ops
                    );
                } else if optimized_ops == original_ops {
                    println!("\n⚠️  No operation reduction detected");
                    println!("   This might mean:");
                    println!("   • Sum splitting rule didn't match the pattern");
                    println!("   • Expression was already optimal");
                    println!("   • Optimization created equivalent but not simpler form");
                } else {
                    println!("\n❌ Operation count increased - unexpected!");
                }
            }
            Err(e) => {
                println!("❌ Optimization failed: {}", e);
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n4️⃣ Optimization Test Skipped");
        println!("-----------------------------");
        println!("⚠️  Optimization features disabled");
        println!(
            "   Run with: cargo run --features optimization --example sum_splitting_test_demo"
        );
    }

    // =======================================================================
    // 5. Manual Sum Splitting Test
    // =======================================================================

    println!("\n5️⃣ Manual Sum Splitting Verification");
    println!("-------------------------------------");

    // Create the manually split version: a * Σ(x_i) + b * Σ(x_i)
    let sum_data = ctx.sum(&data, |x_i| x_i.clone());
    let manual_split = &a * &sum_data + &b * &sum_data;

    // Further simplify to: (a + b) * Σ(x_i)
    let manual_factored = (&a + &b) * &sum_data;

    println!("Manual transformations:");

    let split_result = ctx.eval(&manual_split, hlist![2.0, 3.0]);
    println!("   Split form a*Σ(x) + b*Σ(x): {}", split_result);

    let factored_result = ctx.eval(&manual_factored, hlist![2.0, 3.0]);
    println!("   Factored form (a+b)*Σ(x): {}", factored_result);

    println!("   All should equal: {}", original_result);

    // Verify all forms are equivalent
    let split_correct = (original_result - split_result).abs() < 1e-10;
    let factored_correct = (original_result - factored_result).abs() < 1e-10;

    if split_correct && factored_correct {
        println!("   ✅ Manual transformations are mathematically correct!");
    } else {
        println!("   ❌ Manual transformations have errors!");
    }

    // =======================================================================
    // 6. More Complex Sum Splitting Test
    // =======================================================================

    println!("\n6️⃣ Complex Sum Splitting Test");
    println!("------------------------------");

    let c = ctx.var::<f64>();

    // Create: Σ(a * x_i + b + c * x_i²)
    // Should split into: Σ(a * x_i) + Σ(b) + Σ(c * x_i²)
    // = a * Σ(x_i) + b * n + c * Σ(x_i²)
    let complex_sum = ctx.sum(&data, |x_i| &a * &x_i + &b + &c * (&x_i * &x_i));

    println!("Complex expression: Σ(a * x_i + b + c * x_i²)");

    let complex_ast = ctx.to_ast(&complex_sum);
    let complex_ops = OperationCountVisitor::count_operations(&complex_ast);
    let complex_cost = SummationAwareCostVisitor::compute_cost(&complex_ast);

    println!("   Operations: {}", complex_ops);
    println!("   Cost: {}", complex_cost);

    let test_values_3 = [2.0, 3.0, 0.5]; // a=2, b=3, c=0.5
    let complex_result = ctx.eval(&complex_sum, hlist![2.0, 3.0, 0.5]);
    println!("   Test result (a=2, b=3, c=0.5): {}", complex_result);

    // Manual calculation for verification:
    // Σ(x_i) = 1+2+3+4+5 = 15
    // Σ(b) = b * 5 = 3 * 5 = 15
    // Σ(x_i²) = 1+4+9+16+25 = 55
    // Result = 2*15 + 15 + 0.5*55 = 30 + 15 + 27.5 = 72.5
    println!("   Expected: 2*15 + 3*5 + 0.5*55 = 30 + 15 + 27.5 = 72.5");

    println!("\n🎉 Sum Splitting Test Complete!");
    println!("================================");
    println!("Key findings:");
    println!("• Created expressions that should benefit from sum splitting");
    println!("• Verified mathematical correctness of manual transformations");
    println!("• Tested both simple and complex sum splitting patterns");

    #[cfg(feature = "optimization")]
    println!("• Applied egglog optimization to test automatic sum splitting");

    #[cfg(not(feature = "optimization"))]
    println!("• Run with --features optimization to test automatic sum splitting");

    Ok(())
}
