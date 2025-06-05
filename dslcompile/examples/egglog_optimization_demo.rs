//! `EggLog` optimization demo - symbolic expression simplification

use dslcompile::ast::pretty::pretty_ast;
use dslcompile::prelude::*;

fn main() -> Result<()> {
    let math = DynamicContext::new();
    let mut optimizer = SymbolicOptimizer::new()?;

    // Example 1: Identity addition (x + 0 → x)
    let x = math.var();
    let expr1 = x.clone() + 0.0;
    let optimized1 = optimizer.optimize(&math.to_ast(&expr1))?;
    println!(
        "x + 0 → {}",
        pretty_ast(&optimized1, &math.registry().borrow())
    );

    // Example 2: Logarithm/exponential identity (ln(exp(x)) → x)
    let x = math.var();
    let expr2 = x.clone().exp().ln();
    let optimized2 = optimizer.optimize(&math.to_ast(&expr2))?;
    println!(
        "ln(exp(x)) → {}",
        pretty_ast(&optimized2, &math.registry().borrow())
    );

    // Example 3: Power identity (x^1 → x)
    let x = math.var();
    let expr3 = x.pow(math.constant(1.0));
    let optimized3 = optimizer.optimize(&math.to_ast(&expr3))?;
    println!(
        "x^1 → {}",
        pretty_ast(&optimized3, &math.registry().borrow())
    );

    // Verify correctness
    let test_val = 2.0;
    let original = math.eval(&expr1, &[test_val]);
    let optimized_result = optimized1.eval_with_vars(&[test_val]);
    assert_eq!(original, optimized_result);
    println!("Verification: {original} = {optimized_result}");

    Ok(())
}
