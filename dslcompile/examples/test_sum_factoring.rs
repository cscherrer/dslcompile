//! Test Sum Factoring: Σ(2*x + 3*x) → 5*Σ(x)  
//!
//! This demonstrates the key sum splitting optimization where constants
//! are factored out of summations to reduce computational complexity.

use dslcompile::prelude::*;

// #[cfg(feature = "optimization")]
// use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

fn main() -> Result<()> {
    println!("🧮 Testing Sum Factoring Optimization");
    println!("====================================\n");

    // Test 1: Create expression that should factor - Σ(2*x + 3*x)
    println!("📊 Test 1: Sum with additive constants");

    // Build the inner expression: 2*x + 3*x
    let x = ASTRepr::Variable(0);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    let inner = ASTRepr::add_binary(
        ASTRepr::mul_binary(two.clone(), x.clone()),
        ASTRepr::mul_binary(three.clone(), x.clone()),
    );

    println!("Inner expression: {inner:?}");
    println!("Expected after factoring: 5*x");

    // Create sum over this expression
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let collection = dslcompile::ast::ast_repr::Collection::Constant(data);
    let sum_expr = ASTRepr::Sum(Box::new(collection));

    println!("Sum expression: Sum over [1,2,3,4,5]");
    println!("Expected result: Σ(2*x + 3*x) = 5*Σ(x) = 5*(1+2+3+4+5) = 5*15 = 75\n");

    // Optimization functionality removed
    {
        println!("🔄 Testing inner expression evaluation...");
        println!("✅ Expression created successfully!");
        println!("Expression: {inner:?}\n");

        // Test mathematical evaluation
        let test_value = 2.0;
        let result = inner.eval_with_vars(&[test_value]);

        println!("\n📊 Evaluation test (x = {test_value}):");
        println!("  Expression: 2*{test_value} + 3*{test_value} = {result}");
        println!("  Expected: {} (2*2 + 3*2 = 10)", 2.0 * test_value + 3.0 * test_value);
        println!("  Match: {}", (result - 10.0_f64).abs() < 1e-10_f64);
    }

    println!("🚫 Optimization functionality removed");
    println!("Expression evaluation completed successfully.");

    // Test 2: Simple constant multiplication that should definitely optimize
    println!("\n🔍 Test 2: Simple constant multiplication");
    let simple_mul = ASTRepr::mul_binary(ASTRepr::Constant(6.0), ASTRepr::Variable(0));

    println!("Expression: 6*x");

    // Optimization functionality removed
    {
        println!("Created: {simple_mul:?}");

        let test_val = 3.0;
        let result = simple_mul.eval_with_vars(&[test_val]);
        println!("Test: 6*{test_val} = {result}");
        println!("Expected: {}", 6.0 * test_val);
        println!("Match: {}", (result - 18.0_f64).abs() < 1e-10_f64);
    }

    Ok(())
}
