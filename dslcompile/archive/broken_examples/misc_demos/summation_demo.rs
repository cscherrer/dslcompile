//! Summation demo - basic mathematical summations

use dslcompile::prelude::*;

fn main() -> Result<()> {
    let math = DynamicContext::new();

    // Build summation expressions using basic math operations
    let x = math.var();
    let sum_expr = x.clone() * x.clone(); // x²

    println!("Expression: x²");
    println!("x² at x=5: {}", math.eval_old(&sum_expr, &[5.0]));

    // Linear expression
    let linear = x.clone() * 2.0 + 3.0; // 2x + 3
    println!("2x + 3 at x=4: {}", math.eval_old(&linear, &[4.0]));

    Ok(())
}
