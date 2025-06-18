//! Test true multiset simplification to verify it solves our problems

use dslcompile::prelude::*;
use dslcompile::symbolic::rule_loader::{RuleConfig, RuleLoader};
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🧪 TESTING TRUE MULTISET SIMPLIFICATION");
    println!("=====================================\n");

    // Create test expression: (x + 0) * 1 + (x * 0)
    let mut ctx = DynamicContext::new();
    let x = ctx.var();
    let expr: Expr<f64> = (&x + 0.0) * 1.0 + (&x * 0.0);
    let ast: ASTRepr<f64> = expr.into();
    
    println!("📝 Original expression: (x + 0) * 1 + (x * 0)");
    println!("🔍 Original AST: {:?}", ast);
    println!("📊 Operations count: {}", ast.count_operations());
    
    // Test with basic rules only (no multisets)
    println!("\n📌 Testing with BASIC arithmetic rules only:");
    test_with_rules(&ast, vec!["staged_core_math.egg"])?;
    
    // Test with NEW true multiset rules
    println!("\n✅ Testing with NEW true multiset rules:");
    test_with_rules(&ast, vec!["staged_core_math.egg", "true_multiset_simplification.egg"])?;
    
    // Test a more complex expression to show multiset benefits
    println!("\n🧪 TESTING COMPLEX EXPRESSION");
    println!("=====================================\n");
    
    let y = ctx.var();
    let z = ctx.var();
    // This should simplify to x + y + z + 6
    let complex: Expr<f64> = (&x + 1.0) + ((&y + 2.0) + (&z + 3.0));
    let complex_ast: ASTRepr<f64> = complex.into();
    
    println!("📝 Complex expression: (x + 1) + ((y + 2) + (z + 3))");
    println!("🔍 Original AST: {:?}", complex_ast);
    
    println!("\n✅ Testing complex expression with true multisets:");
    test_with_rules(&complex_ast, vec!["staged_core_math.egg", "true_multiset_simplification.egg"])?;
    
    Ok(())
}

fn test_with_rules(ast: &ASTRepr<f64>, rule_files: Vec<&str>) -> Result<()> {
    let mut config = RuleConfig::default();
    config.search_paths = vec!["dslcompile/src/egglog_rules".into()];
    config.rule_files = rule_files.into_iter().map(Into::into).collect();
    config.categories = vec![]; // Only load specified files
    config.include_comments = true;
    
    let rule_loader = RuleLoader::new(config);
    let mut optimizer = NativeEgglogOptimizer::with_rule_loader(rule_loader)?;
    
    let start = std::time::Instant::now();
    
    match optimizer.optimize(ast) {
        Ok(optimized) => {
            let elapsed = start.elapsed();
            println!("⏱️  Optimization time: {:.3}s", elapsed.as_secs_f64());
            println!("🎯 Optimized AST: {:?}", optimized);
            println!("📊 Operations count: {} -> {}", 
                     ast.count_operations(), 
                     optimized.count_operations());
            
            // Check if we got the expected simplification
            let original_str = format!("{:?}", ast);
            let optimized_str = format!("{:?}", optimized);
            
            if original_str == optimized_str {
                println!("⚠️  No optimization occurred!");
            } else if optimized_str.len() < original_str.len() / 2 {
                println!("🎉 Significant simplification achieved!");
            } else {
                println!("✅ Some optimization occurred");
            }
            
            // Verify correctness
            let test_val = 2.5;
            let original_result = ast.eval_with_vars(&[test_val]);
            let optimized_result = optimized.eval_with_vars(&[test_val]);
            
            if (original_result - optimized_result).abs() < 1e-10 {
                println!("✅ Results match: {} = {}", original_result, optimized_result);
            } else {
                println!("❌ Results differ: {} != {}", original_result, optimized_result);
            }
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("❌ Optimization failed after {:.3}s: {}", elapsed.as_secs_f64(), e);
            
            // Check if it's a timeout/memory issue
            if elapsed.as_secs() > 1 {
                println!("💥 Likely memory/performance issue - took too long!");
            }
        }
    }
    
    println!();
    Ok(())
}

#[cfg(not(feature = "optimization"))]
fn main() {
    println!("⚠️  This example requires the 'optimization' feature");
    println!("Run with: cargo run --features optimization --example test_true_multisets");
}