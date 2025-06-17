//! Test Extraction and Costs - Understanding Why Optimization Isn't Being Extracted

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🔍 EXTRACTION COST ANALYSIS");
    println!("===========================\n");

    // Test the simplest case that should definitely optimize: Σ(a*x)
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>(); // Variable(0)
    let data = vec![1.0, 2.0, 3.0];

    let simple_factoring = ctx.sum(&data, |x| &a * &x);

    println!("📊 SIMPLE COEFFICIENT FACTORING TEST: Σ(a*x)");
    println!("Expected: a * Σ(x) = a * 6 = 2 * 6 = 12");

    let original_result = ctx.eval(&simple_factoring, hlist![2.0]);
    println!("✅ Original evaluation: {}", original_result);

    // Convert to AST
    let original_ast = ctx.to_ast(&simple_factoring);
    println!("\n🏗️  ORIGINAL AST:");
    println!("{:#?}", original_ast);

    #[cfg(feature = "optimization")]
    {
        use dslcompile::symbolic::rule_loader::{RuleCategory, RuleConfig, RuleLoader};
        let config = RuleConfig {
            categories: vec![RuleCategory::CoreDatatypes, RuleCategory::Summation],
            validate_syntax: true,
            include_comments: true,
            ..Default::default()
        };

        let rule_loader = RuleLoader::new(config.clone());
        let mut optimizer = NativeEgglogOptimizer::with_rule_loader(rule_loader)?;

        println!("\n🧮 APPLYING OPTIMIZATION TO SIMPLE CASE...");
        match optimizer.optimize(&original_ast) {
            Ok(optimized_ast) => {
                println!("✅ Optimization completed");

                println!("\n🏗️  OPTIMIZED AST:");
                println!("{:#?}", optimized_ast);

                // Test evaluation
                let optimized_result = optimized_ast.eval_with_vars(&[2.0]);
                println!("\n📊 EVALUATION:");
                println!("Original:  {}", original_result);
                println!("Optimized: {}", optimized_result);
                println!(
                    "Match: {}",
                    (original_result - optimized_result).abs() < 1e-10
                );

                // Analyze difference
                let orig_str = format!("{:?}", original_ast);
                let opt_str = format!("{:?}", optimized_ast);

                println!("\n🔍 STRUCTURAL ANALYSIS:");
                println!("Original:  {}", orig_str);
                println!("Optimized: {}", opt_str);

                if orig_str == opt_str {
                    println!("❌ NO EXTRACTION OCCURRED");
                    println!("   Possible causes:");
                    println!("   1. Rules aren't firing (check statistics)");
                    println!("   2. Optimized forms have higher cost than original");
                    println!("   3. Extraction is not finding optimized forms");
                } else {
                    println!("✅ EXTRACTION SUCCESSFUL");

                    // Check for expected patterns
                    if opt_str.contains("Mul")
                        && opt_str.contains("Sum")
                        && !opt_str.contains("Map")
                    {
                        println!("🎉 FOUND EXPECTED PATTERN: Mul(coefficient, Sum(...))");
                    } else if opt_str.contains("Add") && opt_str.len() < orig_str.len() {
                        println!("🎉 FOUND SUM SPLITTING: Expression shortened");
                    } else {
                        println!("⚠️  Optimized but not in expected pattern");
                    }
                }
            }
            Err(e) => {
                println!("❌ Optimization failed: {}", e);
            }
        }

        // Now test the more complex case: Σ(a*x + b*x)
        println!("\n{}", "=".repeat(50));
        println!("📊 COMPLEX SUM SPLITTING TEST: Σ(a*x + b*x)");
        println!("Expected: (a+b) * Σ(x) = (2+3) * 6 = 30");

        let b = ctx.var::<f64>(); // Variable(1)
        let complex_expr = ctx.sum(&data, |x| &a * &x + &b * &x);
        let complex_original = ctx.eval(&complex_expr, hlist![2.0, 3.0]);
        println!("✅ Original evaluation: {}", complex_original);

        let complex_ast = ctx.to_ast(&complex_expr);
        println!("\n🏗️  ORIGINAL COMPLEX AST:");
        println!("{:#?}", complex_ast);

        // Create a fresh optimizer for the complex case
        let rule_loader2 = RuleLoader::new(config.clone());
        let mut optimizer2 = NativeEgglogOptimizer::with_rule_loader(rule_loader2)?;

        match optimizer2.optimize(&complex_ast) {
            Ok(complex_opt) => {
                let complex_opt_result = complex_opt.eval_with_vars(&[2.0, 3.0]);

                println!("\n🏗️  OPTIMIZED COMPLEX AST:");
                println!("{:#?}", complex_opt);

                println!("\n📊 COMPLEX EVALUATION:");
                println!("Original:  {}", complex_original);
                println!("Optimized: {}", complex_opt_result);
                println!(
                    "Match: {}",
                    (complex_original - complex_opt_result).abs() < 1e-10
                );

                let complex_orig_str = format!("{:?}", complex_ast);
                let complex_opt_str = format!("{:?}", complex_opt);

                println!("\n🔍 COMPLEX STRUCTURAL ANALYSIS:");
                println!("Original chars:  {}", complex_orig_str.len());
                println!("Optimized chars: {}", complex_opt_str.len());

                if complex_orig_str == complex_opt_str {
                    println!("❌ NO COMPLEX EXTRACTION OCCURRED");
                } else {
                    println!("✅ COMPLEX EXTRACTION SUCCESSFUL");
                    if complex_opt_str.len() < complex_orig_str.len() {
                        println!("🎉 Expression was simplified!");
                    }
                }
            }
            Err(e) => {
                println!("❌ Complex optimization failed: {}", e);
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n⚠️  Optimization features not enabled");
        println!("Run with: cargo run --features optimization --example test_extraction_costs");
    }

    Ok(())
}
