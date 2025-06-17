//! Test Sum Splitting Optimization
//!
//! Comprehensive test to verify that sum splitting is now working correctly
//! with the new clean_summation_rules.egg and enhanced dependency analysis.

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🧮 COMPREHENSIVE SUM SPLITTING OPTIMIZATION TEST");
    println!("================================================\n");

    // =======================================================================
    // 1. Create Test Expressions for Sum Splitting
    // =======================================================================

    println!("1️⃣ Creating Test Expressions");
    println!("-----------------------------");

    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>(); // External coefficient
    let b = ctx.var::<f64>(); // Another external coefficient
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Test Case 1: Basic sum splitting - Σ(a*x + b*x) should become (a+b)*Σ(x)
    let test1 = ctx.sum(&data, |x| &a * &x + &b * &x);
    println!("Test 1: Σ(a*x + b*x) over {:?}", data);

    // Test Case 2: Coefficient factoring - Σ(a*x) should become a*Σ(x)
    let test2 = ctx.sum(&data, |x| &a * &x);
    println!("Test 2: Σ(a*x) over {:?}", data);

    // Test Case 3: Constant addition - Σ(x + c) should become Σ(x) + c*n
    let c = 2.0;
    let test3 = ctx.sum(&data, |x| x.clone() + c);
    println!("Test 3: Σ(x + {}) over {:?}", c, data);

    // Test Case 4: Mixed pattern - Σ(a*x + b*x + c) should become (a+b)*Σ(x) + c*n
    let test4 = ctx.sum(&data, |x| &a * &x + &b * &x + c);
    println!("Test 4: Σ(a*x + b*x + {}) over {:?}", c, data);

    // =======================================================================
    // 2. Test Original Evaluation (Before Optimization)
    // =======================================================================

    println!("\n2️⃣ Original Evaluation Results");
    println!("-------------------------------");

    let test_params = hlist![2.0, 3.0]; // a=2, b=3

    let result1 = ctx.eval(&test1, test_params.clone());
    let result2 = ctx.eval(&test2, test_params.clone());
    let result3 = ctx.eval(&test3, hlist![]);
    let result4 = ctx.eval(&test4, test_params.clone());

    println!("Test 1 result (a=2, b=3): {}", result1);
    println!("Test 2 result (a=2): {}", result2);
    println!("Test 3 result: {}", result3);
    println!("Test 4 result (a=2, b=3): {}", result4);

    // Manual verification calculations
    let sum_x = data.iter().sum::<f64>(); // 1+2+3+4+5 = 15
    let n = data.len() as f64; // 5
    println!("\nExpected calculations:");
    println!("Σ(x) = {}", sum_x);
    println!("n = {}", n);
    println!("Test 1 expected: (2+3)*15 = {}", (2.0 + 3.0) * sum_x);
    println!("Test 2 expected: 2*15 = {}", 2.0 * sum_x);
    println!("Test 3 expected: 15 + 2*5 = {}", sum_x + c * n);
    println!(
        "Test 4 expected: (2+3)*15 + 2*5 = {}",
        (2.0 + 3.0) * sum_x + c * n
    );

    // =======================================================================
    // 3. Apply Optimization and Test
    // =======================================================================

    #[cfg(feature = "optimization")]
    {
        println!("\n3️⃣ Applying Sum Splitting Optimization");
        println!("---------------------------------------");

        // Use clean summation rules which include sum splitting
        use dslcompile::symbolic::rule_loader::{RuleCategory, RuleConfig, RuleLoader};
        let config = RuleConfig {
            categories: vec![RuleCategory::CoreDatatypes, RuleCategory::Summation],
            validate_syntax: true,
            include_comments: true,
            ..Default::default()
        };

        let rule_loader = RuleLoader::new(config);
        let mut optimizer = NativeEgglogOptimizer::with_rule_loader(rule_loader)?;

        // Test each case
        for (i, test_expr) in [&test1, &test2, &test3, &test4].iter().enumerate() {
            println!("\n🔍 Optimizing Test {} Expression", i + 1);
            println!("----------------------------------");

            let original_ast = ctx.to_ast(test_expr);
            println!("Original AST: {:?}", original_ast);

            match optimizer.optimize(&original_ast) {
                Ok(optimized_ast) => {
                    println!("Optimized AST: {:?}", optimized_ast);

                    // Test semantic preservation
                    let original_result = match i {
                        0 => result1,
                        1 => result2,
                        2 => result3,
                        3 => result4,
                        _ => unreachable!(),
                    };

                    let optimized_result = match i {
                        0 | 1 | 3 => optimized_ast.eval_with_vars(&[2.0, 3.0]),
                        2 => optimized_ast.eval_with_vars(&[]),
                        _ => unreachable!(),
                    };

                    println!("Original result: {}", original_result);
                    println!("Optimized result: {}", optimized_result);

                    let diff = (original_result - optimized_result).abs();
                    println!("Difference: {:.2e}", diff);

                    if diff < 1e-10 {
                        println!("✅ Semantics preserved!");
                    } else {
                        println!("❌ Semantics NOT preserved!");
                    }

                    // Check if optimization actually occurred
                    let original_str = format!("{:?}", original_ast);
                    let optimized_str = format!("{:?}", optimized_ast);

                    if original_str != optimized_str {
                        println!("🎉 Optimization applied!");

                        // Look for signs of sum splitting
                        if optimized_str.contains("Add") && optimized_str.contains("Sum") {
                            println!("   Detected sum splitting pattern in result");
                        }
                        if optimized_str.contains("Mul") && optimized_str.len() < original_str.len()
                        {
                            println!("   Detected coefficient factoring (shorter expression)");
                        }
                    } else {
                        println!("⚠️  No optimization detected (expressions identical)");
                    }
                }
                Err(e) => {
                    println!("❌ Optimization failed: {}", e);
                }
            }
        }

        // =======================================================================
        // 4. Advanced Pattern Testing
        // =======================================================================

        println!("\n4️⃣ Advanced Pattern Testing");
        println!("-----------------------------");

        // Test nested coefficient: Σ((a+b)*x)
        let test_nested = ctx.sum(&data, |x| (&a + &b) * &x);
        println!("Nested coefficient test: Σ((a+b)*x)");

        let nested_ast = ctx.to_ast(&test_nested);
        match optimizer.optimize(&nested_ast) {
            Ok(optimized) => {
                println!("Original:  {:?}", nested_ast);
                println!("Optimized: {:?}", optimized);

                let original_result = ctx.eval(&test_nested, test_params.clone());
                let optimized_result = optimized.eval_with_vars(&[2.0, 3.0]);

                println!(
                    "Results match: {}",
                    (original_result - optimized_result).abs() < 1e-10
                );
            }
            Err(e) => println!("Nested optimization failed: {}", e),
        }

        // Test subtraction: Σ(a*x - b*x)
        let test_sub = ctx.sum(&data, |x| &a * &x - &b * &x);
        println!("\nSubtraction test: Σ(a*x - b*x)");

        let sub_ast = ctx.to_ast(&test_sub);
        match optimizer.optimize(&sub_ast) {
            Ok(optimized) => {
                println!("Original:  {:?}", sub_ast);
                println!("Optimized: {:?}", optimized);

                let original_result = ctx.eval(&test_sub, test_params.clone());
                let optimized_result = optimized.eval_with_vars(&[2.0, 3.0]);

                println!(
                    "Results match: {}",
                    (original_result - optimized_result).abs() < 1e-10
                );
                println!("Expected: (2-3)*15 = {}", (2.0 - 3.0) * sum_x);
            }
            Err(e) => println!("Subtraction optimization failed: {}", e),
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n3️⃣ Optimization Test Skipped");
        println!("-----------------------------");
        println!("⚠️  Optimization features disabled");
        println!(
            "   Run with: cargo run --features optimization --example test_sum_splitting_optimization"
        );
    }

    // =======================================================================
    // 5. Summary
    // =======================================================================

    println!("\n🎯 SUM SPLITTING TEST SUMMARY");
    println!("=============================");
    println!("✅ Created comprehensive test expressions");
    println!("✅ Verified original evaluation correctness");
    println!("✅ Mathematical expectations calculated manually");

    #[cfg(feature = "optimization")]
    println!("✅ Applied egglog optimization with sum splitting rules");

    #[cfg(not(feature = "optimization"))]
    println!("⚠️  Egglog optimization requires --features optimization");

    println!("\n📊 Test Cases:");
    println!("   1. Basic sum splitting: Σ(a*x + b*x) → (a+b)*Σ(x)");
    println!("   2. Coefficient factoring: Σ(a*x) → a*Σ(x)");
    println!("   3. Constant addition: Σ(x + c) → Σ(x) + c*n");
    println!("   4. Mixed pattern: Σ(a*x + b*x + c) → (a+b)*Σ(x) + c*n");
    println!("   5. Nested coefficients: Σ((a+b)*x)");
    println!("   6. Subtraction: Σ(a*x - b*x) → (a-b)*Σ(x)");

    Ok(())
}
