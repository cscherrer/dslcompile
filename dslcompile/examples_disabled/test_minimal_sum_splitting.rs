//! Minimal test for sum splitting - isolate the egglog parsing issue

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🔍 MINIMAL SUM SPLITTING TEST");
    println!("=============================\n");

    // Create a simple expression that should trigger sum splitting
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    let data = vec![1.0, 2.0, 3.0];

    // Simple test: Σ(a*x) - coefficient factoring
    let test_expr = ctx.sum(&data, |x| &a * &x);
    println!("Test expression: Σ(a*x) over {data:?}");

    let result = ctx.eval(&test_expr, hlist![2.0]); // a = 2
    println!("Original result (a=2): {result}");

    #[cfg(feature = "optimization")]
    {
        println!("\n🔧 Testing Basic Egglog Loading...");

        // Try with minimal rule set - just core datatypes
        use dslcompile::symbolic::rule_loader::{RuleCategory, RuleConfig, RuleLoader};
        let config = RuleConfig {
            categories: vec![RuleCategory::CoreDatatypes],
            validate_syntax: true,
            include_comments: false,
            ..Default::default()
        };

        println!("Creating rule loader with core datatypes only...");
        let rule_loader = RuleLoader::new(config);

        println!("Initializing egglog optimizer...");
        match NativeEgglogOptimizer::with_rule_loader(rule_loader) {
            Ok(mut optimizer) => {
                println!("✅ Basic egglog loading successful!");

                // Try to optimize
                let ast = ctx.to_ast(&test_expr);
                println!("Original AST: {ast:?}");

                match optimizer.optimize(&ast) {
                    Ok(optimized) => {
                        println!("✅ Basic optimization successful!");
                        println!("Optimized AST: {optimized:?}");
                    }
                    Err(e) => {
                        println!("❌ Optimization failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("❌ Basic egglog loading failed: {e}");
            }
        }

        // Now try with dependency analysis
        println!("\n🔧 Testing with Dependency Analysis...");
        let config_with_deps = RuleConfig {
            categories: vec![
                RuleCategory::CoreDatatypes,
                RuleCategory::DependencyAnalysis,
            ],
            validate_syntax: true,
            include_comments: false,
            ..Default::default()
        };

        match NativeEgglogOptimizer::with_rule_loader(RuleLoader::new(config_with_deps)) {
            Ok(_) => println!("✅ Dependency analysis loading successful!"),
            Err(e) => println!("❌ Dependency analysis loading failed: {e}"),
        }

        // Skip summation rules for now - test them separately once dependency analysis works
        println!("\n⏭️  Skipping summation rules test until dependency analysis is fixed");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("⚠️  Optimization features disabled");
        println!("   Run with: cargo run --bin test_minimal_sum_splitting --features optimization");
    }

    Ok(())
}
