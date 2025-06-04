//! # Debug egglog Quadratic Expansion
//!
//! This example traces exactly what happens when egglog processes quadratic expressions.

use dslcompile::Result;
use dslcompile::final_tagless::{ASTRepr, ExpressionBuilder};
use dslcompile::symbolic::symbolic::SymbolicOptimizer;

fn main() -> Result<()> {
    println!("🔧 Debug: egglog Quadratic Expansion");
    println!("=====================================\n");

    // Create optimizer with egglog enabled
    let mut config = dslcompile::symbolic::symbolic::OptimizationConfig::default();
    config.egglog_optimization = true;
    let mut optimizer = SymbolicOptimizer::with_config(config)?;

    let math = ExpressionBuilder::new();

    // Test 1: Simple case (a + b)²
    println!("🔬 Test 1: Simple (a + b)²");
    let a = math.var(); // Variable(0)
    let b = math.var(); // Variable(1)
    let simple_square = (a + b).pow(math.constant(2.0));
    let simple_ast = simple_square.into_ast();

    println!("   Before optimization: {simple_ast:?}");
    println!("   Operations before: {}", simple_ast.count_operations());

    let optimized_simple = optimizer.optimize(&simple_ast)?;
    println!("   After optimization:  {optimized_simple:?}");
    println!(
        "   Operations after: {}",
        optimized_simple.count_operations()
    );
    println!();

    // Test 2: Check if Pow(Add(...), Num(2.0)) pattern is preserved
    println!("🔬 Test 2: Direct AST construction");
    let direct_pow = ASTRepr::Pow(
        Box::new(ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        )),
        Box::new(ASTRepr::Constant(2.0)),
    );

    println!("   Before optimization: {direct_pow:?}");
    println!("   Operations before: {}", direct_pow.count_operations());

    let optimized_direct = optimizer.optimize(&direct_pow)?;
    println!("   After optimization:  {optimized_direct:?}");
    println!(
        "   Operations after: {}",
        optimized_direct.count_operations()
    );
    println!();

    // Test 3: More complex case - our original problem
    println!("🔬 Test 3: Bayesian residual (y - β₀ - β₁*x)²");
    let y = math.var(); // Variable(0)
    let beta0 = math.var(); // Variable(1)
    let beta1 = math.var(); // Variable(2)
    let x = math.var(); // Variable(3)

    let residual = y - beta0 - beta1 * x;
    let residual_squared = residual.pow(math.constant(2.0));
    let complex_ast = residual_squared.into_ast();

    println!("   Before optimization: {complex_ast:?}");
    println!("   Operations before: {}", complex_ast.count_operations());

    let optimized_complex = optimizer.optimize(&complex_ast)?;
    println!("   After optimization:  {optimized_complex:?}");
    println!(
        "   Operations after: {}",
        optimized_complex.count_operations()
    );
    println!();

    // Test 4: Check if we need to apply optimization multiple times
    println!("🔬 Test 4: Multiple optimization passes");
    let mut multi_pass = complex_ast.clone();
    for i in 1..=5 {
        let before_ops = multi_pass.count_operations();
        multi_pass = optimizer.optimize(&multi_pass)?;
        let after_ops = multi_pass.count_operations();
        println!("   Pass {i}: {before_ops} → {after_ops} operations");
        if before_ops == after_ops {
            println!("   Converged at pass {i}");
            break;
        }
    }

    println!("   Final result: {multi_pass:?}");
    println!();

    println!("✅ Debug complete!");

    Ok(())
}
