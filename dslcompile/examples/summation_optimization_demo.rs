/// Priority Summation Optimizations Demo
/// 
/// This example demonstrates the two core optimizations that beat naive Rust:
/// 1. Sum splitting: Σ(f(i) + g(i)) = Σ(f(i)) + Σ(g(i))
/// 2. Constant factor distribution: Σ(k * f(i)) = k * Σ(f(i))

use dslcompile::ast::ASTRepr;
use dslcompile::symbolic::summation::{SummationOptimizer, IntRange};
use dslcompile::Result;

fn main() -> Result<()> {
    println!("🚀 DSLCompile Summation Optimization Demo");
    println!("==========================================");

    demo_sum_splitting()?;
    println!();
    demo_constant_factor_distribution()?;
    println!();
    demo_combined_optimizations()?;

    Ok(())
}

/// Demo 1: Sum Splitting - Σ(f(i) + g(i)) = Σ(f(i)) + Σ(g(i))
fn demo_sum_splitting() -> Result<()> {
    println!("📊 Priority Optimization #1: Sum Splitting");
    println!("Σ(f(i) + g(i)) = Σ(f(i)) + Σ(g(i))");
    println!("---------------------------------------------");

    let mut optimizer = SummationOptimizer::new()?;
    let range = IntRange::new(1, 10);
    
    // Test: Σ(i + i²) should split into Σ(i) + Σ(i²)
    // Create AST: Variable(0) + Variable(0)^2
    let i_var = ASTRepr::Variable(0);
    let i_squared = ASTRepr::Pow(
        Box::new(i_var.clone()),
        Box::new(ASTRepr::Constant(2.0))
    );
    let summand = ASTRepr::Add(Box::new(i_var), Box::new(i_squared));
    
    println!("Expression: Σ(i + i²) for i = 1..10");
    
    let result = optimizer.optimize_summation(range, summand)?;
    
    println!("✅ Is optimized: {}", result.is_optimized);
    println!("🔍 Pattern: {:?}", result.pattern);
    println!("📈 Extracted factors: {:?}", result.extracted_factors);
    
    // Calculate expected: Σ(i) + Σ(i²) for i=1..10
    let n = 10.0;
    let sum_i = n * (n + 1.0) / 2.0;           // 55
    let sum_i2 = n * (n + 1.0) * (2.0*n + 1.0) / 6.0; // 385
    let expected = sum_i + sum_i2;             // 440
    
    let actual = result.evaluate(&[])?;
    println!("📊 Expected: {}, Actual: {}", expected, actual);
    
    let accuracy = (actual - expected).abs();
    println!("🎯 Accuracy: {:.2e} (should be < 1e-6)", accuracy);
    
    if accuracy < 1e-6 {
        println!("✅ SUCCESS: Sum splitting optimization working perfectly!");
    } else {
        println!("❌ FAILED: Optimization accuracy not sufficient");
    }

    Ok(())
}

/// Demo 2: Constant Factor Distribution - Σ(k * f(i)) = k * Σ(f(i))
fn demo_constant_factor_distribution() -> Result<()> {
    println!("📊 Priority Optimization #2: Constant Factor Distribution");
    println!("Σ(k * f(i)) = k * Σ(f(i))");
    println!("-----------------------------------------------------------");

    let mut optimizer = SummationOptimizer::new()?;
    let range = IntRange::new(1, 10);
    
    // Test: Σ(5 * i) should become 5 * Σ(i)
    // Create AST: 5.0 * Variable(0)
    let summand = ASTRepr::Mul(
        Box::new(ASTRepr::Constant(5.0)),
        Box::new(ASTRepr::Variable(0))
    );
    
    println!("Expression: Σ(5 * i) for i = 1..10");
    
    let result = optimizer.optimize_summation(range, summand)?;
    
    println!("✅ Is optimized: {}", result.is_optimized);
    println!("🔍 Pattern: {:?}", result.pattern);
    println!("📈 Extracted factors: {:?}", result.extracted_factors);
    
    // Should extract factor 5.0
    let has_factor_5 = result.extracted_factors.contains(&5.0);
    println!("🔢 Contains factor 5.0: {}", has_factor_5);
    
    // Expected: 5 * Σ(i) = 5 * n(n+1)/2 = 5 * 10*11/2 = 275
    let expected = 5.0 * 10.0 * 11.0 / 2.0;
    let actual = result.evaluate(&[])?;
    println!("📊 Expected: {}, Actual: {}", expected, actual);
    
    let accuracy = (actual - expected).abs();
    println!("🎯 Accuracy: {:.2e} (should be < 1e-6)", accuracy);
    
    if accuracy < 1e-6 && has_factor_5 {
        println!("✅ SUCCESS: Constant factor optimization working perfectly!");
    } else {
        println!("❌ FAILED: Optimization not working as expected");
    }

    Ok(())
}

/// Demo 3: Combined Optimizations - Both working together
fn demo_combined_optimizations() -> Result<()> {
    println!("📊 Combined Demo: Both Optimizations Together");
    println!("Σ(3*i + 2*i²) should split AND extract factors");
    println!("-----------------------------------------------");

    let mut optimizer = SummationOptimizer::new()?;
    let range = IntRange::new(1, 5);
    
    // Create AST: (3.0 * Variable(0)) + (2.0 * Variable(0)²)
    let left = ASTRepr::Mul(
        Box::new(ASTRepr::Constant(3.0)),
        Box::new(ASTRepr::Variable(0))
    );
    let right = ASTRepr::Mul(
        Box::new(ASTRepr::Constant(2.0)),
        Box::new(ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0))
        ))
    );
    let summand = ASTRepr::Add(Box::new(left), Box::new(right));
    
    println!("Expression: Σ(3*i + 2*i²) for i = 1..5");
    
    let result = optimizer.optimize_summation(range, summand)?;
    
    println!("✅ Is optimized: {}", result.is_optimized);
    println!("🔍 Pattern: {:?}", result.pattern);
    println!("📈 Extracted factors: {:?}", result.extracted_factors);
    
    // Expected: 3*Σ(i) + 2*Σ(i²) = 3*15 + 2*55 = 45 + 110 = 155
    let n = 5.0;
    let sum_i = n * (n + 1.0) / 2.0;           // 15
    let sum_i2 = n * (n + 1.0) * (2.0*n + 1.0) / 6.0; // 55
    let expected = 3.0 * sum_i + 2.0 * sum_i2; // 155
    
    let actual = result.evaluate(&[])?;
    println!("📊 Expected: {}, Actual: {}", expected, actual);
    
    let accuracy = (actual - expected).abs();
    println!("🎯 Accuracy: {:.2e} (should be < 1e-6)", accuracy);
    
    if accuracy < 1e-6 {
        println!("✅ SUCCESS: Combined optimizations working perfectly!");
        println!("🎉 Both sum splitting AND factor extraction are functional!");
    } else {
        println!("❌ FAILED: Combined optimization accuracy not sufficient");
    }

    Ok(())
} 