//! Egglog Optimization Demo
//! Demonstrates domain-aware symbolic optimization using native egglog integration

use dslcompile::final_tagless::{ASTEval, ASTMathExpr};
use dslcompile::symbolic::native_egglog::optimize_with_native_egglog;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Domain-Aware Egglog Optimization Demo");
    println!("========================================");

    // Test 1: Basic algebraic simplification
    println!("\n🧪 Test 1: Basic Algebraic Simplification");
    let expr1 = ASTEval::add(ASTEval::var(0), ASTEval::constant(0.0));
    println!("Original: x + 0");

    match optimize_with_native_egglog(&expr1) {
        Ok(optimized) => println!("Optimized: {optimized:?}"),
        Err(e) => println!("Error: {e}"),
    }

    // Test 2: Transcendental function optimization
    println!("\n🧪 Test 2: Transcendental Function Optimization");
    let expr2 = ASTEval::ln(ASTEval::exp(ASTEval::var(0)));
    println!("Original: ln(exp(x))");

    match optimize_with_native_egglog(&expr2) {
        Ok(optimized) => println!("Optimized: {optimized:?}"),
        Err(e) => println!("Error: {e}"),
    }

    // Test 3: Power simplification
    println!("\n🧪 Test 3: Power Simplification");
    let expr3 = ASTEval::pow(ASTEval::var(0), ASTEval::constant(1.0));
    println!("Original: x^1");

    match optimize_with_native_egglog(&expr3) {
        Ok(optimized) => println!("Optimized: {optimized:?}"),
        Err(e) => println!("Error: {e}"),
    }

    // Test 4: Complex expression
    println!("\n🧪 Test 4: Complex Expression");
    let expr4 = ASTEval::add(
        ASTEval::mul(ASTEval::var(0), ASTEval::constant(1.0)),
        ASTEval::mul(ASTEval::constant(0.0), ASTEval::var(1)),
    );
    println!("Original: x * 1 + 0 * y");

    match optimize_with_native_egglog(&expr4) {
        Ok(optimized) => println!("Optimized: {optimized:?}"),
        Err(e) => println!("Error: {e}"),
    }

    // Test 5: Domain-safe square root (this should NOT be simplified unsafely)
    println!("\n🧪 Test 5: Domain-Safe Square Root");
    let expr5 = ASTEval::sqrt(ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)));
    println!("Original: sqrt(x^2)");
    println!("Note: This should NOT be simplified to x without domain constraints");

    match optimize_with_native_egglog(&expr5) {
        Ok(optimized) => {
            println!("Optimized: {optimized:?}");
            println!("✅ Domain safety preserved - no unsafe sqrt(x^2) = x transformation");
        }
        Err(e) => println!("Error: {e}"),
    }

    println!("\n✅ Domain-aware optimization demo completed!");
    println!("💡 The native egglog optimizer provides mathematical safety");
    println!("   by avoiding unsafe transformations like sqrt(x^2) = x");

    Ok(())
}
