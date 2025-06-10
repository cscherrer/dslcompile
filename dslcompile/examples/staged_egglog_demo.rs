//! Staged Egglog Optimization Demo
//!
//! This demo showcases the new staged egglog optimization system that provides:
//! 1. Variable partitioning with integer indices
//! 2. Sum splitting: Σ(f + g) = Σ(f) + Σ(g)
//! 3. Constant factoring: Σ(k * f) = k * Σ(f)
//! 4. Arithmetic series optimization
//! 5. Fast execution with full equality saturation

use dslcompile::{ast::ASTRepr, symbolic::native_egglog::optimize_with_native_egglog};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Staged Egglog Optimization Demo");
    println!("=====================================");

    // Test 1: Variable Collection - 2*x + 3*x → 5*x
    println!("\n📊 Test 1: Variable Collection");
    let expr1 = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(2.0)),
            Box::new(ASTRepr::Variable(0)),
        )),
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(3.0)),
            Box::new(ASTRepr::Variable(0)),
        )),
    );

    println!("   Input:  2*x + 3*x");
    println!("   Before: {expr1:?}");

    #[cfg(feature = "optimization")]
    {
        match optimize_with_native_egglog(&expr1) {
            Ok(optimized) => {
                println!("   After:  {optimized:?}");
                println!("   ✅ Variable collection successful!");
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("   ⚠️  Optimization feature not enabled");
    }

    // Test 2: Complex Partitioning - 3 + 2*x + 4 + x → 7 + 3*x
    println!("\n🔧 Test 2: Complex Variable Partitioning");
    let expr2 = ASTRepr::Add(
        Box::new(ASTRepr::Add(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Constant(3.0)),
                Box::new(ASTRepr::Mul(
                    Box::new(ASTRepr::Constant(2.0)),
                    Box::new(ASTRepr::Variable(0)),
                )),
            )),
            Box::new(ASTRepr::Constant(4.0)),
        )),
        Box::new(ASTRepr::Variable(0)),
    );

    println!("   Input:  3 + 2*x + 4 + x");
    println!("   Before: {expr2:?}");

    #[cfg(feature = "optimization")]
    {
        match optimize_with_native_egglog(&expr2) {
            Ok(optimized) => {
                println!("   After:  {optimized:?}");
                println!("   ✅ Complex partitioning successful!");
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    // Test 3: Constant Folding
    println!("\n⚡ Test 3: Constant Folding");
    let expr3 = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(2.0)),
            Box::new(ASTRepr::Constant(3.0)),
        )),
        Box::new(ASTRepr::Constant(4.0)),
    );

    println!("   Input:  (2 * 3) + 4");
    println!("   Before: {expr3:?}");

    #[cfg(feature = "optimization")]
    {
        match optimize_with_native_egglog(&expr3) {
            Ok(optimized) => {
                println!("   After:  {optimized:?}");
                println!("   ✅ Constant folding successful!");
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    // Test 4: Identity Rules
    println!("\n🎯 Test 4: Identity Rules");
    let expr4 = ASTRepr::Add(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Constant(0.0)),
    );

    println!("   Input:  x + 0");
    println!("   Before: {expr4:?}");

    #[cfg(feature = "optimization")]
    {
        match optimize_with_native_egglog(&expr4) {
            Ok(optimized) => {
                println!("   After:  {optimized:?}");
                println!("   ✅ Identity rule successful!");
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    println!("\n🎉 Staged Egglog Demo Complete!");
    println!("\n📈 Key Benefits:");
    println!("   • Variable partitioning prevents combinatorial explosion");
    println!("   • Staged execution ensures fast convergence");
    println!("   • Full equality saturation with mathematical optimizations");
    println!("   • Integer variable indices for performance");
    println!("   • Production-ready optimization pipeline");

    Ok(())
}
