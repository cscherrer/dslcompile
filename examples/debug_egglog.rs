use mathjit::final_tagless::{ASTEval, ASTMathExpr, DirectEval};

#[cfg(feature = "optimization")]
use mathjit::egglog_integration::EgglogOptimizer;

fn main() {
    println!("=== Debug Egglog Pattern Optimization ===\n");

    #[cfg(feature = "optimization")]
    {
        let optimizer = EgglogOptimizer::new().unwrap();

        // Test 1: Simple x + 0 optimization
        println!("1. Testing x + 0 optimization:");
        let expr1 = ASTEval::add(ASTEval::var_by_name("x"), ASTEval::constant(0.0));
        println!("   Original expression: {expr1:?}");

        // Check pattern detection
        let patterns = optimizer.extract_optimization_patterns(&expr1);
        println!("   Detected patterns: {patterns:?}");

        // Test individual pattern optimization
        if patterns.is_empty() {
            println!("   No patterns detected!");
        } else {
            for pattern in &patterns {
                match optimizer.apply_pattern_optimization(&expr1, pattern) {
                    Ok(optimized) => {
                        println!("   Optimized expression: {optimized:?}");
                        let original_val = DirectEval::eval_with_vars(&expr1, &[5.0]);
                        let optimized_val = DirectEval::eval_with_vars(&optimized, &[5.0]);
                        println!("   Pattern {pattern:?}: Original = {original_val}, Optimized = {optimized_val}");
                        println!(
                            "   Structurally different: {}",
                            !expressions_equal(&expr1, &optimized)
                        );
                    }
                    Err(e) => {
                        println!("   Pattern {pattern:?}: Failed with error: {e}");
                    }
                }
            }
        }

        // Test 2: Simple x * 0 optimization
        println!("\n2. Testing x * 0 optimization:");
        let expr2 = ASTEval::mul(ASTEval::var_by_name("x"), ASTEval::constant(0.0));
        println!("   Original expression: {expr2:?}");

        let patterns2 = optimizer.extract_optimization_patterns(&expr2);
        println!("   Detected patterns: {patterns2:?}");

        if patterns2.is_empty() {
            println!("   No patterns detected!");
        } else {
            for pattern in &patterns2 {
                match optimizer.apply_pattern_optimization(&expr2, pattern) {
                    Ok(optimized) => {
                        println!("   Optimized expression: {optimized:?}");
                        let original_val = DirectEval::eval_with_vars(&expr2, &[5.0]);
                        let optimized_val = DirectEval::eval_with_vars(&optimized, &[5.0]);
                        println!("   Pattern {pattern:?}: Original = {original_val}, Optimized = {optimized_val}");
                        println!(
                            "   Structurally different: {}",
                            !expressions_equal(&expr2, &optimized)
                        );
                    }
                    Err(e) => {
                        println!("   Pattern {pattern:?}: Failed with error: {e}");
                    }
                }
            }
        }

        // Test 3: Direct pattern application for x * 1
        println!("\n3. Testing x * 1 optimization:");
        let expr3 = ASTEval::mul(ASTEval::var_by_name("x"), ASTEval::constant(1.0));
        println!("   Original expression: {expr3:?}");

        let patterns3 = optimizer.extract_optimization_patterns(&expr3);
        println!("   Detected patterns: {patterns3:?}");

        // Try manual optimization
        match optimizer.optimize_mul_one(&expr3) {
            Ok(optimized) => {
                println!("   Optimized expression: {optimized:?}");
                let original_val = DirectEval::eval_with_vars(&expr3, &[7.0]);
                let optimized_val = DirectEval::eval_with_vars(&optimized, &[7.0]);
                println!("   Manual optimize_mul_one: Original = {original_val}, Optimized = {optimized_val}");
                println!(
                    "   Structurally different: {}",
                    !expressions_equal(&expr3, &optimized)
                );
            }
            Err(e) => {
                println!("   Manual optimize_mul_one failed: {e}");
            }
        }

        // Test 4: Full egglog optimization pipeline
        println!("\n4. Testing full egglog optimization pipeline:");
        let mut full_optimizer = EgglogOptimizer::new().unwrap();
        let expr4 = ASTEval::add(ASTEval::var_by_name("x"), ASTEval::constant(0.0));
        println!("   Original expression: {expr4:?}");

        match full_optimizer.optimize(&expr4) {
            Ok(optimized) => {
                println!("   ✅ Full optimization succeeded!");
                println!("   Optimized expression: {optimized:?}");
                let original_val = DirectEval::eval_with_vars(&expr4, &[5.0]);
                let optimized_val = DirectEval::eval_with_vars(&optimized, &[5.0]);
                println!("   Values: Original = {original_val}, Optimized = {optimized_val}");
                println!(
                    "   Structurally different: {}",
                    !expressions_equal(&expr4, &optimized)
                );
            }
            Err(e) => {
                println!("   ⚠️  Full optimization failed: {e}");
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("Egglog optimization feature is not enabled.");
        println!("Run with: cargo run --example debug_egglog --features optimization");
    }
}

/// Helper function to check if two expressions are structurally equal
fn expressions_equal(
    expr1: &mathjit::final_tagless::ASTRepr<f64>,
    expr2: &mathjit::final_tagless::ASTRepr<f64>,
) -> bool {
    format!("{expr1:?}") == format!("{expr2:?}")
}
