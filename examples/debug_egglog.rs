use mathcompile::final_tagless::{ASTEval, ASTMathExpr, DirectEval};

#[cfg(feature = "optimization")]
use mathcompile::egglog_integration::EgglogOptimizer;

fn main() {
    println!("=== Debug Egglog Optimization ===\n");

    #[cfg(feature = "optimization")]
    {
        let mut optimizer = EgglogOptimizer::new().unwrap();

        // Test 1: Simple x + 0 optimization
        println!("1. Testing x + 0 optimization:");
        let expr1 = ASTEval::add(ASTEval::var_by_name("x"), ASTEval::constant(0.0));
        println!("   Original expression: {expr1:?}");

        match optimizer.optimize(&expr1) {
            Ok(optimized) => {
                println!("   ✅ Optimization succeeded!");
                println!("   Optimized expression: {optimized:?}");
                let original_val = DirectEval::eval_with_vars(&expr1, &[5.0]);
                let optimized_val = DirectEval::eval_with_vars(&optimized, &[5.0]);
                println!("   Values: Original = {original_val}, Optimized = {optimized_val}");
                println!(
                    "   Structurally different: {}",
                    !expressions_equal(&expr1, &optimized)
                );
            }
            Err(e) => {
                println!("   ⚠️  Optimization failed: {e}");
            }
        }

        // Test 2: Simple x * 0 optimization
        println!("\n2. Testing x * 0 optimization:");
        let expr2 = ASTEval::mul(ASTEval::var_by_name("x"), ASTEval::constant(0.0));
        println!("   Original expression: {expr2:?}");

        match optimizer.optimize(&expr2) {
            Ok(optimized) => {
                println!("   ✅ Optimization succeeded!");
                println!("   Optimized expression: {optimized:?}");
                let original_val = DirectEval::eval_with_vars(&expr2, &[5.0]);
                let optimized_val = DirectEval::eval_with_vars(&optimized, &[5.0]);
                println!("   Values: Original = {original_val}, Optimized = {optimized_val}");
                println!(
                    "   Structurally different: {}",
                    !expressions_equal(&expr2, &optimized)
                );
            }
            Err(e) => {
                println!("   ⚠️  Optimization failed: {e}");
            }
        }

        // Test 3: x * 1 optimization
        println!("\n3. Testing x * 1 optimization:");
        let expr3 = ASTEval::mul(ASTEval::var_by_name("x"), ASTEval::constant(1.0));
        println!("   Original expression: {expr3:?}");

        match optimizer.optimize(&expr3) {
            Ok(optimized) => {
                println!("   ✅ Optimization succeeded!");
                println!("   Optimized expression: {optimized:?}");
                let original_val = DirectEval::eval_with_vars(&expr3, &[7.0]);
                let optimized_val = DirectEval::eval_with_vars(&optimized, &[7.0]);
                println!("   Values: Original = {original_val}, Optimized = {optimized_val}");
                println!(
                    "   Structurally different: {}",
                    !expressions_equal(&expr3, &optimized)
                );
            }
            Err(e) => {
                println!("   ⚠️  Optimization failed: {e}");
            }
        }

        // Test 4: Complex expression optimization
        println!("\n4. Testing complex expression optimization:");
        let expr4 = ASTEval::add(
            ASTEval::mul(ASTEval::var_by_name("x"), ASTEval::constant(1.0)),
            ASTEval::mul(ASTEval::constant(0.0), ASTEval::var_by_name("y")),
        );
        println!("   Original expression: {expr4:?}");

        match optimizer.optimize(&expr4) {
            Ok(optimized) => {
                println!("   ✅ Optimization succeeded!");
                println!("   Optimized expression: {optimized:?}");
                let original_val = DirectEval::eval_with_vars(&expr4, &[5.0, 3.0]);
                let optimized_val = DirectEval::eval_with_vars(&optimized, &[5.0, 3.0]);
                println!("   Values: Original = {original_val}, Optimized = {optimized_val}");
                println!(
                    "   Structurally different: {}",
                    !expressions_equal(&expr4, &optimized)
                );
            }
            Err(e) => {
                println!("   ⚠️  Optimization failed: {e}");
            }
        }

        // Test 5: Algebraic simplification
        println!("\n5. Testing algebraic simplification:");
        let expr5 = ASTEval::add(
            ASTEval::add(ASTEval::var_by_name("x"), ASTEval::var_by_name("x")),
            ASTEval::constant(0.0),
        );
        println!("   Original expression: {expr5:?}");

        match optimizer.optimize(&expr5) {
            Ok(optimized) => {
                println!("   ✅ Optimization succeeded!");
                println!("   Optimized expression: {optimized:?}");
                let original_val = DirectEval::eval_with_vars(&expr5, &[4.0]);
                let optimized_val = DirectEval::eval_with_vars(&optimized, &[4.0]);
                println!("   Values: Original = {original_val}, Optimized = {optimized_val}");
                println!(
                    "   Structurally different: {}",
                    !expressions_equal(&expr5, &optimized)
                );
            }
            Err(e) => {
                println!("   ⚠️  Optimization failed: {e}");
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
    expr1: &mathcompile::final_tagless::ASTRepr<f64>,
    expr2: &mathcompile::final_tagless::ASTRepr<f64>,
) -> bool {
    format!("{expr1:?}") == format!("{expr2:?}")
}
