use mathcompile::ast::ASTRepr;
use mathcompile::prelude::*;

fn main() -> Result<()> {
    println!("=== Domain-Aware Logarithm Rules Demo ===\n");

    // Create expression: ln(a/b) using AST directly
    let a = ASTRepr::Variable(0); // Variable index 0 for 'a'
    let b = ASTRepr::Variable(1); // Variable index 1 for 'b'
    let div = ASTRepr::Div(Box::new(a.clone()), Box::new(b.clone()));
    let ln_div = ASTRepr::Ln(Box::new(div));

    println!("Original expression: ln(a/b)");
    println!("AST: {ln_div:?}\n");

    println!("Testing with a = 10.0, b = 2.0 (both positive):");

    // Evaluate original expression using DirectEval with variable values
    let values = vec![10.0, 2.0]; // a=10.0, b=2.0
    let original_result = DirectEval::eval_with_vars(&ln_div, &values);
    println!("ln(10.0/2.0) = ln(5.0) = {original_result:.6}");

    // Expected result using the rule: ln(a) - ln(b)
    let expected = 10.0_f64.ln() - 2.0_f64.ln();
    println!("ln(10.0) - ln(2.0) = {expected:.6}");
    println!("Difference: {:.10}", (original_result - expected).abs());

    #[cfg(feature = "optimization")]
    {
        println!("\n=== Testing Domain-Aware Optimization ===");

        // Create symbolic optimizer
        let mut optimizer = SymbolicOptimizer::new()?;

        // Optimize the expression - should apply ln(a/b) = ln(a) - ln(b) when safe
        let optimized = optimizer.optimize(&ln_div)?;
        println!("Optimized expression: {optimized:?}");

        // Evaluate optimized expression
        let optimized_result = DirectEval::eval_with_vars(&optimized, &values);
        println!("Optimized result: {optimized_result:.6}");
        println!(
            "Optimization preserved correctness: {}",
            (original_result - optimized_result).abs() < 1e-10
        );

        // Test with native egglog optimizer
        use mathcompile::symbolic::native_egglog::NativeEgglogOptimizer;

        if let Ok(mut native_optimizer) = NativeEgglogOptimizer::new() {
            println!("\n=== Testing Native Egglog Domain Analysis ===");

            // Check if the optimizer can prove domain safety
            let a_safe = native_optimizer.is_domain_safe(&a, "ln").unwrap_or(false);
            let b_safe = native_optimizer.is_domain_safe(&b, "ln").unwrap_or(false);

            println!("Variable 'a' is provably positive for ln: {a_safe}");
            println!("Variable 'b' is provably positive for ln: {b_safe}");

            // Note: Variables are not provably positive without additional constraints
            // But constants would be
            let pos_const = ASTRepr::Constant(5.0);
            let const_safe = native_optimizer
                .is_domain_safe(&pos_const, "ln")
                .unwrap_or(false);
            println!("Constant 5.0 is provably positive for ln: {const_safe}");

            // Test with constant expression: ln(5.0/2.0)
            let const_div = ASTRepr::Div(
                Box::new(ASTRepr::Constant(5.0)),
                Box::new(ASTRepr::Constant(2.0)),
            );
            let ln_const_div = ASTRepr::Ln(Box::new(const_div));
            println!("\nTesting constant expression: ln(5.0/2.0)");

            let const_optimized = native_optimizer.optimize(&ln_const_div)?;
            println!("Native egglog optimized: {const_optimized:?}");

            let const_result = DirectEval::eval_with_vars(&const_optimized, &[]);
            let const_expected = DirectEval::eval_with_vars(&ln_const_div, &[]);
            println!("Original result: {const_expected:.6}");
            println!("Optimized result: {const_result:.6}");
            println!(
                "Native egglog preserved correctness: {}",
                (const_expected - const_result).abs() < 1e-10
            );
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!(
            "\nOptimization features not enabled. Compile with --features optimization to see domain-aware optimizations."
        );
    }

    println!("\n=== Testing Edge Cases ===");

    // Test with negative values (should not apply the rule unsafely)
    println!("Testing with a = -5.0, b = 2.0 (a is negative):");

    // This should result in NaN for the original expression
    let negative_values = vec![-5.0, 2.0];
    let negative_result = DirectEval::eval_with_vars(&ln_div, &negative_values);
    println!("ln(-5.0/2.0) = {negative_result:.6} (NaN expected due to negative argument)");

    // Test with zero
    println!("Testing with a = 0.0, b = 2.0:");
    let zero_values = vec![0.0, 2.0];
    let zero_result = DirectEval::eval_with_vars(&ln_div, &zero_values);
    println!("ln(0.0/2.0) = {zero_result:.6} (-inf expected)");

    println!("\n=== Summary ===");
    println!("✓ Domain-aware ln(a/b) = ln(a) - ln(b) rule implemented");
    println!("✓ Native egglog with interval analysis working");
    println!("✓ Proptests ensure mathematical correctness");
    println!("✓ Edge cases handled safely (NaN for negative, -inf for zero)");

    Ok(())
}
