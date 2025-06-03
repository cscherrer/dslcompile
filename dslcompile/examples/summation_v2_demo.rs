//! # Next-Generation Summation System Demo
//!
//! This example demonstrates the new string-free summation system that eliminates
//! the fragile string-based variable naming in favor of closure-based scoping.
//!
//! Key improvements:
//! - No string-based variable names
//! - Closure-based variable scoping prevents index variable escape
//! - Direct AST manipulation
//! - Type-safe variable binding
//! - Same optimization capabilities as the old system

use dslcompile::Result;
use dslcompile::final_tagless::{ExpressionBuilder, IntRange, RangeType};
use dslcompile::symbolic::summation::{SummationPattern, SummationProcessor};

fn main() -> Result<()> {
    println!("🚀 Next-Generation Summation System Demo");
    println!("========================================\n");

    demo_constant_summation()?;
    demo_linear_summation()?;
    demo_factor_extraction()?;
    demo_geometric_summation()?;
    demo_power_summation()?;
    demo_complex_expressions()?;
    demo_performance_comparison()?;

    Ok(())
}

/// Demo 1: Constant summation with perfect scoping
fn demo_constant_summation() -> Result<()> {
    println!("📊 Demo 1: Constant Summation");
    println!("=============================");
    println!("Computing: Σ(i=1 to 10) 5.0");
    println!("Expected: 5.0 * 10 = 50.0");
    println!();

    let mut processor = SummationProcessor::new()?;
    let range = IntRange::new(1, 10);

    // The index variable `i` is properly scoped within the closure
    let result = processor.sum(range, |_i| {
        let math = ExpressionBuilder::new();
        math.constant(5.0)
    })?;

    println!("🔍 Analysis Results:");
    println!("   Pattern: {:?}", result.pattern);
    println!("   Optimized: {}", result.is_optimized);
    println!("   Has closed form: {}", result.closed_form.is_some());

    let value = result.evaluate(&[])?;
    println!("   Result: {value}");
    println!("   ✅ Constant summation successful!");
    println!();

    Ok(())
}

/// Demo 2: Linear summation with automatic pattern recognition
fn demo_linear_summation() -> Result<()> {
    println!("📊 Demo 2: Linear Summation");
    println!("===========================");
    println!("Computing: Σ(i=1 to 10) i");
    println!("Expected: 1+2+...+10 = 55");
    println!();

    let mut processor = SummationProcessor::new()?;
    let range = IntRange::new(1, 10);

    // Natural syntax: just pass the index variable
    let result = processor.sum(range, |i| i)?;

    println!("🔍 Analysis Results:");
    match &result.pattern {
        SummationPattern::Linear {
            coefficient,
            constant,
        } => {
            println!("   Pattern: Linear (coefficient: {coefficient}, constant: {constant})");
        }
        other => println!("   Pattern: {other:?}"),
    }
    println!("   Optimized: {}", result.is_optimized);
    println!("   Has closed form: {}", result.closed_form.is_some());

    let value = result.evaluate(&[])?;
    println!("   Result: {value}");
    assert_eq!(value, 55.0);
    println!("   ✅ Linear summation successful!");
    println!();

    Ok(())
}

/// Demo 3: Factor extraction with closure safety
fn demo_factor_extraction() -> Result<()> {
    println!("📊 Demo 3: Factor Extraction");
    println!("============================");
    println!("Computing: Σ(i=1 to 10) 3.0 * i");
    println!("Expected: 3.0 * Σ(i) = 3.0 * 55 = 165");
    println!();

    let mut processor = SummationProcessor::new()?;
    let range = IntRange::new(1, 10);

    let result = processor.sum(range, |i| {
        let math = ExpressionBuilder::new();
        math.constant(3.0) * i
    })?;

    println!("🔍 Analysis Results:");
    println!("   Pattern: {:?}", result.pattern);
    println!("   Extracted factors: {:?}", result.extracted_factors);
    println!("   Factor speedup: {:.1}x", result.factor_speedup());
    println!("   Optimized: {}", result.is_optimized);

    let value = result.evaluate(&[])?;
    println!("   Result: {value}");
    assert_eq!(value, 165.0);
    println!("   ✅ Factor extraction successful!");
    println!();

    Ok(())
}

/// Demo 4: Geometric series recognition
fn demo_geometric_summation() -> Result<()> {
    println!("📊 Demo 4: Geometric Series");
    println!("===========================");
    println!("Computing: Σ(i=0 to 5) (0.5)^i");
    println!("Expected: geometric series sum ≈ 1.96875");
    println!();

    let mut processor = SummationProcessor::new()?;
    let range = IntRange::new(0, 5);

    let result = processor.sum(range, |i| {
        let math = ExpressionBuilder::new();
        math.constant(0.5).pow(i)
    })?;

    println!("🔍 Analysis Results:");
    match &result.pattern {
        SummationPattern::Geometric { coefficient, ratio } => {
            println!("   Pattern: Geometric (coefficient: {coefficient}, ratio: {ratio})");
        }
        other => println!("   Pattern: {other:?}"),
    }
    println!("   Optimized: {}", result.is_optimized);
    println!("   Has closed form: {}", result.closed_form.is_some());

    let value = result.evaluate(&[])?;
    println!("   Result: {value}");
    println!("   Expected: ~1.96875");
    assert!((value - 1.96875).abs() < 1e-5);
    println!("   ✅ Geometric series successful!");
    println!();

    Ok(())
}

/// Demo 5: Power summation with closed forms
fn demo_power_summation() -> Result<()> {
    println!("📊 Demo 5: Power Summation");
    println!("==========================");
    println!("Computing: Σ(i=1 to 5) i²");
    println!("Expected: 1² + 2² + 3² + 4² + 5² = 55");
    println!();

    let mut processor = SummationProcessor::new()?;
    let range = IntRange::new(1, 5);

    let result = processor.sum(range, |i| {
        let math = ExpressionBuilder::new();
        i.pow(math.constant(2.0))
    })?;

    println!("🔍 Analysis Results:");
    match &result.pattern {
        SummationPattern::Power { exponent } => {
            println!("   Pattern: Power (exponent: {exponent})");
        }
        other => println!("   Pattern: {other:?}"),
    }
    println!("   Optimized: {}", result.is_optimized);
    println!("   Has closed form: {}", result.closed_form.is_some());

    let value = result.evaluate(&[])?;
    println!("   Result: {value}");
    assert_eq!(value, 55.0);
    println!("   ✅ Power summation successful!");
    println!();

    Ok(())
}

/// Demo 6: Complex expressions with multiple optimizations
fn demo_complex_expressions() -> Result<()> {
    println!("📊 Demo 6: Complex Expressions");
    println!("==============================");
    println!("Computing: Σ(i=1 to 5) (2*i + 3)");
    println!("Expected: Linear pattern recognition");
    println!();

    let mut processor = SummationProcessor::new()?;
    let range = IntRange::new(1, 5);

    let result = processor.sum(range, |i| {
        let math = ExpressionBuilder::new();
        let two = math.constant(2.0);
        let three = math.constant(3.0);
        two * &i + three
    })?;

    println!("🔍 Analysis Results:");
    match &result.pattern {
        SummationPattern::Linear {
            coefficient,
            constant,
        } => {
            println!("   Pattern: Linear (coefficient: {coefficient}, constant: {constant})");
        }
        other => println!("   Pattern: {other:?}"),
    }
    println!("   Optimized: {}", result.is_optimized);

    let value = result.evaluate(&[])?;
    println!("   Result: {value}");

    // Manual verification: (2*1+3) + (2*2+3) + (2*3+3) + (2*4+3) + (2*5+3) = 5 + 7 + 9 + 11 + 13 = 45
    let expected = 5.0 + 7.0 + 9.0 + 11.0 + 13.0;
    assert_eq!(value, expected);
    println!("   Expected: {expected} (verified manually)");
    println!("   ✅ Complex expression successful!");
    println!();

    Ok(())
}

/// Demo 7: Performance comparison and safety demonstration
fn demo_performance_comparison() -> Result<()> {
    println!("📊 Demo 7: Performance & Safety");
    println!("===============================");
    println!("Demonstrating closure-based safety and performance");
    println!();

    let mut processor = SummationProcessor::new()?;
    let range = IntRange::new(1, 1000);
    let range_len = range.len(); // Get length before moving range
    let k = std::f64::consts::PI;

    println!("🔒 Safety Demonstration:");

    // This closure demonstrates that the index variable is properly scoped
    let result = processor.sum(range, |i| {
        // The variable 'i' is only accessible within this closure
        // After this closure ends, 'i' cannot be accessed anywhere else
        let math = ExpressionBuilder::new();
        math.constant(k) * i
    })?;

    // The variable 'i' is no longer accessible here - this is enforced by Rust's type system
    // This prevents the "index variable escape" problem entirely

    println!("   ✅ Index variable 'i' properly scoped within closure");
    println!("   ✅ No possibility of variable escape");
    println!("   ✅ Type-safe variable binding");

    println!("\n⚡ Performance Analysis:");
    println!("   Range: {range_len} terms");
    println!("   Pattern: {:?}", result.pattern);
    println!("   Extracted factors: {:?}", result.extracted_factors);
    println!("   Optimized: {}", result.is_optimized);

    let start = std::time::Instant::now();
    let value = result.evaluate(&[])?;
    let eval_time = start.elapsed();

    println!("   Result: {value:.6}");
    println!("   Evaluation time: {eval_time:?}");

    // Expected: k * (1+2+...+1000) = k * 500500
    let expected = k * 500500.0;
    assert!((value - expected).abs() < 1e-10);
    println!("   Expected: {expected:.6}");
    println!("   Accuracy: Perfect!");

    println!("\n✨ Summary of Improvements:");
    println!("   🔒 No string-based variable names");
    println!("   🛡️  Closure-based scoping prevents variable escape");
    println!("   📊 Same optimization capabilities as old system");
    println!("   🚀 Type-safe variable binding");
    println!("   ⚡ Direct AST manipulation");
    println!("   🧮 Natural mathematical syntax");

    Ok(())
}
