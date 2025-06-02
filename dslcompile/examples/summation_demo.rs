//! # Advanced Summation Demo
//!
//! This example demonstrates the advanced summation capabilities of `DSLCompile`,
//! including pattern recognition, factor extraction, and closed-form evaluation.

use dslcompile::Result;
use dslcompile::final_tagless::{ASTFunction, IntRange};
use dslcompile::symbolic::summation::{SummationConfig, SummationSimplifier};

fn main() -> Result<()> {
    println!("🧮 DSLCompile Advanced Summation Demo");
    println!("=====================================\n");

    // Create a summation simplifier
    let mut simplifier = SummationSimplifier::new();

    // Demo 1: Constant summation
    println!("📊 Demo 1: Constant Summation");
    println!("Σ(i=1 to 10) 5 = ?");

    let range = IntRange::new(1, 10);
    let constant_func = ASTFunction::constant_func("i", 5.0);

    let result = simplifier.simplify_finite_sum(&range, &constant_func)?;

    println!("Pattern recognized: {:?}", result.recognized_pattern);
    if let Some(closed_form) = &result.closed_form {
        println!("Closed form: {closed_form:?}");
        if let dslcompile::final_tagless::ASTRepr::Constant(value) = closed_form {
            println!("Result: {value}");
        }
    }
    println!();

    // Demo 2: Arithmetic series
    println!("📊 Demo 2: Arithmetic Series");
    println!("Σ(i=1 to 10) (2*i + 3) = ?");

    let arithmetic_func = ASTFunction::poly("i", &[3.0, 2.0]); // 3 + 2*i
    let result = simplifier.simplify_finite_sum(&range, &arithmetic_func)?;

    println!("Pattern recognized: {:?}", result.recognized_pattern);
    if let Some(closed_form) = &result.closed_form {
        println!("Closed form: {closed_form:?}");
        if let dslcompile::final_tagless::ASTRepr::Constant(value) = closed_form {
            println!("Result: {value}");
        }
    }
    println!();

    // Demo 3: Geometric series
    println!("📊 Demo 3: Geometric Series");
    println!("Σ(i=1 to 10) 3 * 2^i = ?");

    // Create a geometric function: 3 * 2^i
    let geometric_func = ASTFunction::new(
        "i",
        dslcompile::final_tagless::ASTRepr::Mul(
            Box::new(dslcompile::final_tagless::ASTRepr::Constant(3.0)),
            Box::new(dslcompile::final_tagless::ASTRepr::Pow(
                Box::new(dslcompile::final_tagless::ASTRepr::Constant(2.0)),
                Box::new(dslcompile::final_tagless::ASTRepr::Variable(0)),
            )),
        ),
    );

    let result = simplifier.simplify_finite_sum(&range, &geometric_func)?;

    println!("Pattern recognized: {:?}", result.recognized_pattern);
    if let Some(closed_form) = &result.closed_form {
        println!("Closed form: {closed_form:?}");
        if let dslcompile::final_tagless::ASTRepr::Constant(value) = closed_form {
            println!("Result: {value}");
        }
    }
    println!();

    // Demo 4: Configuration options
    println!("📊 Demo 4: Configuration Options");
    println!("Demonstrating different optimization levels...");

    let conservative_config = SummationConfig {
        extract_factors: true,
        recognize_patterns: true,
        closed_form_evaluation: false,
        telescoping_detection: false,
        max_polynomial_degree: 3,
        tolerance: 1e-10,
    };

    let conservative_simplifier = SummationSimplifier::with_config(conservative_config);
    println!("Conservative configuration created successfully!");

    let aggressive_config = SummationConfig {
        extract_factors: true,
        recognize_patterns: true,
        closed_form_evaluation: true,
        telescoping_detection: true,
        max_polynomial_degree: 10,
        tolerance: 1e-12,
    };

    let aggressive_simplifier = SummationSimplifier::with_config(aggressive_config);
    println!("Aggressive configuration created successfully!");
    println!();

    println!("✅ All summation demos completed successfully!");
    println!("\n🎯 Key Features Demonstrated:");
    println!("   • Pattern recognition for constant, arithmetic, and geometric series");
    println!("   • Automatic closed-form evaluation");
    println!("   • Configurable optimization levels");
    println!("   • Integration with final tagless architecture");

    Ok(())
}
