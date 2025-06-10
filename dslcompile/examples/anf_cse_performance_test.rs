//! ANF-CSE Performance Test
//!
//! This test specifically targets the Gaussian performance issue:
//! ((x - mu) / sigma)^2 should have (x - mu) / sigma cached in a let binding.

use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;
use std::time::Instant;

/// Count the number of operations in an AST
fn count_operations(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
        ASTRepr::Add(l, r)
        | ASTRepr::Sub(l, r)
        | ASTRepr::Mul(l, r)
        | ASTRepr::Div(l, r)
        | ASTRepr::Pow(l, r) => 1 + count_operations(l) + count_operations(r),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner) => 1 + count_operations(inner),
        _ => 0, // For other variants
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 ANF-CSE Performance Test");
    println!("===========================");
    println!("Testing Common Subexpression Elimination for Gaussian pattern");
    println!("Expression: -0.5 * ((x - mu) / sigma)²");
    println!("CSE opportunity: (x - mu) / sigma appears twice");
    println!();

    // Create the Gaussian log-likelihood expression: -0.5 * ((x - mu) / sigma)²
    let mut ctx = DynamicContext::<f64>::new();
    let x = ctx.var();
    let mu = ctx.var();
    let sigma = ctx.var();

    // Build: ((x - mu) / sigma)²
    let standardized = (x - mu) / sigma;
    let squared = standardized.clone() * standardized.clone();
    let expr_typed = ctx.constant(-0.5) * squared;

    // Convert to ASTRepr for the optimizer
    let expression: ASTRepr<f64> = expr_typed.into();

    // Test regular egglog optimization
    println!("🔧 Testing Regular Egglog Optimization:");
    let start = Instant::now();
    let mut optimizer = NativeEgglogOptimizer::new().expect("Failed to create optimizer");
    let regular_result = optimizer
        .optimize(&expression)
        .expect("Regular optimization failed");
    let regular_time = start.elapsed();
    println!("   Time: {regular_time:.2?}");
    println!("   Result: {regular_result:?}");
    println!();

    // Test ANF-CSE optimization
    println!("🚀 Testing ANF-CSE Optimization:");
    let start = Instant::now();
    let mut cse_optimizer = NativeEgglogOptimizer::new().expect("Failed to create CSE optimizer");
    let cse_result = cse_optimizer
        .optimize_with_anf_cse(&expression)
        .expect("ANF-CSE optimization failed");
    let cse_time = start.elapsed();
    println!("   Time: {cse_time:.2?}");
    println!("   Result: {cse_result:?}");
    println!();

    // Debug: Print the egglog representation
    println!("🔍 Debug: Egglog representation");
    let egglog_repr = cse_optimizer
        .ast_to_egglog(&expression)
        .expect("Failed to convert to egglog");
    println!("   Egglog format: {egglog_repr}");
    println!();

    // Debug: Let's create a simple test case to see if our CSE rules work
    println!("🧪 Testing Simple CSE Pattern:");
    let simple_x = ASTRepr::Variable(0);
    let simple_squared = ASTRepr::Mul(Box::new(simple_x.clone()), Box::new(simple_x.clone()));
    let simple_egglog = cse_optimizer
        .ast_to_egglog(&simple_squared)
        .expect("Failed to convert simple");
    println!("   Simple pattern: {simple_egglog}");

    // Test if this simpler pattern gets optimized
    let simple_result = cse_optimizer
        .optimize_with_anf_cse(&simple_squared)
        .expect("Simple CSE failed");
    println!("   Simple result: {simple_result:?}");
    println!();

    // Check if optimization actually changed the expression
    let expressions_identical = format!("{regular_result:?}") == format!("{cse_result:?}");
    if expressions_identical {
        println!("⚠️  Results are identical - ANF-CSE didn't change the expression");
    } else {
        println!("✅ ANF-CSE successfully optimized the expression!");
    }
    println!();

    // Test evaluation correctness
    println!("📊 Testing Evaluation Correctness:");
    let test_values = [2.0, 1.0, 0.5]; // x=2.0, mu=1.0, sigma=0.5

    let original_result = expression.eval_with_vars(&test_values);
    let regular_eval = regular_result.eval_with_vars(&test_values);
    let cse_eval = cse_result.eval_with_vars(&test_values);

    println!("   Original:  {original_result:.10}");
    println!("   Regular:   {regular_eval:.10}");
    println!("   ANF-CSE:   {cse_eval:.10}");

    let regular_diff = (original_result - regular_eval).abs();
    let cse_diff = (original_result - cse_eval).abs();

    println!("   Regular diff:  {regular_diff:.2e}");
    println!("   ANF-CSE diff:  {cse_diff:.2e}");

    if regular_diff < 1e-10 && cse_diff < 1e-10 {
        println!("   ✅ All results are mathematically equivalent");
    } else {
        println!("   ❌ Results differ significantly!");
    }
    println!();

    // Summary
    println!("🎯 Test Summary:");
    println!("   Regular optimization: {regular_time:.2?}");
    println!("   ANF-CSE optimization: {cse_time:.2?}");
    if expressions_identical {
        println!("   Expected: ANF-CSE should detect and eliminate ((x-mu)/sigma) duplication");
    } else {
        println!("   ✅ ANF-CSE successfully applied optimizations!");
    }

    // Debug: Let's create the exact Gaussian pattern test
    println!("🎯 Testing Exact Gaussian Pattern:");
    let x = ASTRepr::Variable(0); // x
    let mu = ASTRepr::Variable(1); // mu  
    let sigma = ASTRepr::Variable(2); // sigma

    // Build: (x - mu) / sigma
    let x_minus_mu = ASTRepr::Sub(Box::new(x.clone()), Box::new(mu.clone()));
    let standardized = ASTRepr::Div(Box::new(x_minus_mu.clone()), Box::new(sigma.clone()));

    // Build: ((x - mu) / sigma) * ((x - mu) / sigma) - the exact duplication!
    let gaussian_squared = ASTRepr::Mul(
        Box::new(standardized.clone()),
        Box::new(standardized.clone()),
    );

    let gaussian_egglog = cse_optimizer
        .ast_to_egglog(&gaussian_squared)
        .expect("Failed to convert Gaussian");
    println!("   Gaussian pattern: {gaussian_egglog}");

    // Count operations in original
    let original_ops = count_operations(&gaussian_squared);
    println!("   Original operations: {original_ops}");

    // What our CSE should produce: Let 1004 = (x-mu)/sigma in 1004 * 1004
    let expected_cse = ASTRepr::Sub(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Variable(1)),
    ); // temp for counting
    let cse_inner = ASTRepr::Div(Box::new(expected_cse), Box::new(ASTRepr::Variable(2)));
    let expected_ops = count_operations(&cse_inner) + 1; // +1 for the final multiplication of bound vars
    println!("   Expected CSE operations: {expected_ops}");
    println!(
        "   CSE savings: {} operations ({:.1}x reduction)",
        original_ops.saturating_sub(expected_ops),
        original_ops as f64 / expected_ops.max(1) as f64
    );

    // Test if this gets optimized
    let gaussian_result = cse_optimizer
        .optimize_with_anf_cse(&gaussian_squared)
        .expect("Gaussian CSE failed");
    println!("   Gaussian result: {gaussian_result:?}");

    // Check if it actually changed
    let gaussian_changed = format!("{gaussian_squared:?}") != format!("{gaussian_result:?}");
    println!(
        "   Pattern changed: {}",
        if gaussian_changed {
            "✅ YES"
        } else {
            "❌ NO"
        }
    );
    println!();

    Ok(())
}
