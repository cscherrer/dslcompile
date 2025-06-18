//! Simple Dynamic Cost Assignment Demo
//!
//! This example demonstrates the basic dynamic cost functionality from egglog-experimental.

use dslcompile::{ast::ASTRepr, prelude::*};

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🎯 Simple Dynamic Cost Assignment Demo");
    println!("=====================================");

    // Example 1: Basic expression optimization with dynamic costs
    println!("\n📊 Example 1: Basic Dynamic Cost");

    // Create a simple expression: x + 0
    let simple_expr = ASTRepr::Add(vec![ASTRepr::Variable(0), ASTRepr::Constant(0.0)]);

    println!("   Original: {simple_expr:?}");

    // Create symbolic optimizer with dynamic cost support
    #[cfg(feature = "optimization")]
    let mut optimizer = NativeEgglogOptimizer::new()?;

    // Set a specific cost for the constant zero
    let zero_expr = ASTRepr::Constant(0.0);
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&zero_expr, 1000)?; // High cost for zero constants

    // Optimize the expression
    #[cfg(feature = "optimization")]
    let optimized = optimizer.optimize(&simple_expr)?;
    #[cfg(feature = "optimization")]
    println!("   Optimized: {optimized:?}");

    // Example 2: Power expression with dynamic costs
    println!("\n📊 Example 2: Power Expression Dynamic Cost");

    // Create a power expression: x^2
    let power_expr = ASTRepr::Pow(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Constant(2.0)),
    );

    println!("   Original: {power_expr:?}");

    // Set cost for power operations
    let generic_pow = ASTRepr::Pow(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Variable(1)),
    );
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&generic_pow, 50)?; // Lower cost for power operations

    #[cfg(feature = "optimization")]
    let optimized_power = optimizer.optimize(&power_expr)?;
    #[cfg(feature = "optimization")]
    println!("   Optimized: {optimized_power:?}");

    // Example 3: Multiplication with different costs
    println!("\n📊 Example 3: Multiplication Cost Preferences");

    // Create multiplication: x * 1
    let mult_expr = ASTRepr::Mul(vec![ASTRepr::Variable(0), ASTRepr::Constant(1.0)]);

    println!("   Original: {mult_expr:?}");

    // Set high cost for multiplying by 1 (should simplify to just x)
    let mult_by_one = ASTRepr::Mul(vec![ASTRepr::Variable(0), ASTRepr::Constant(1.0)]);
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&mult_by_one, 2000)?;

    #[cfg(feature = "optimization")]
    let optimized_mult = optimizer.optimize(&mult_expr)?;
    #[cfg(feature = "optimization")]
    println!("   Optimized: {optimized_mult:?}");

    println!("\n✅ Simple Dynamic Cost Demo Complete!");
    println!("   Dynamic costs allow fine-grained control over optimization preferences");

    Ok(())
}
