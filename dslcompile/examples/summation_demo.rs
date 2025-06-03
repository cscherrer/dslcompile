//! # Advanced Summation Demo
//!
//! This example demonstrates the advanced summation capabilities of `DSLCompile`,
//! including pattern recognition, factor extraction, and closed-form evaluation.

use dslcompile::Result;
use dslcompile::final_tagless::{ExpressionBuilder, IntRange};
use dslcompile::symbolic::summation::{SummationConfig, SummationProcessor};

fn main() -> Result<()> {
    println!("🧮 DSLCompile Advanced Summation Demo");
    println!("=====================================\n");

    // Create a summation processor
    let mut processor = SummationProcessor::new()?;

    // Demo 1: Constant summation
    println!("📊 Demo 1: Constant Summation");
    println!("Σ(i=1 to 10) 5 = ?");

    let range = IntRange::new(1, 10);
    let result = processor.sum(range, |_i| {
        let math = ExpressionBuilder::new();
        math.constant(5.0)
    })?;

    println!("Pattern recognized: {:?}", result.pattern);
    if let Some(closed_form) = &result.closed_form {
        println!("Closed form available");
        let value = result.evaluate(&[])?;
        println!("Result: {value}");
    }
    println!();

    // Demo 2: Arithmetic series  
    println!("📊 Demo 2: Arithmetic Series");
    println!("Σ(i=1 to 10) (2*i + 3) = ?");

    let range = IntRange::new(1, 10);
    let result = processor.sum(range, |i| {
        let math = ExpressionBuilder::new();
        math.constant(2.0) * i + math.constant(3.0)
    })?;

    println!("Pattern recognized: {:?}", result.pattern);
    if let Some(closed_form) = &result.closed_form {
        println!("Closed form available");
        let value = result.evaluate(&[])?;
        println!("Result: {value}");
    }
    println!();

    // Demo 3: Geometric series  
    println!("📊 Demo 3: Geometric Series");
    println!("Σ(i=1 to 10) 3 * 2^i = ?");

    let range = IntRange::new(1, 10);
    let result = processor.sum(range, |i| {
        let math = ExpressionBuilder::new();
        math.constant(3.0) * math.constant(2.0).pow(i)
    })?;

    println!("Pattern recognized: {:?}", result.pattern);
    if let Some(closed_form) = &result.closed_form {
        println!("Closed form available");
        let value = result.evaluate(&[])?;
        println!("Result: {value}");
    }
    println!();

    // Demo 4: Configuration options
    println!("📊 Demo 4: Configuration Options");
    println!("Demonstrating different optimization levels...");

    let conservative_config = SummationConfig {
        enable_pattern_recognition: true,
        enable_closed_form: false,
        enable_factor_extraction: true,
        tolerance: 1e-10,
    };

    let conservative_processor = SummationProcessor::with_config(conservative_config)?;
    println!("Conservative configuration created successfully!");

    let aggressive_config = SummationConfig {
        enable_pattern_recognition: true,
        enable_closed_form: true,
        enable_factor_extraction: true,
        tolerance: 1e-12,
    };

    let aggressive_processor = SummationProcessor::with_config(aggressive_config)?;
    println!("Aggressive configuration created successfully!");
    println!();

    println!("✅ All summation demos completed successfully!");
    println!("\n🎯 Key Features Demonstrated:");
    println!("   • Pattern recognition for constant, arithmetic, and geometric series");
    println!("   • Automatic closed-form evaluation");
    println!("   • Configurable optimization levels");
    println!("   • Type-safe closure-based API");

    Ok(())
}
