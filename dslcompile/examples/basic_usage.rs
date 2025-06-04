//! Basic usage example showing runtime expression building

use dslcompile::prelude::*;

fn main() -> Result<()> {
    let math = ExpressionBuilder::new();

    // Create variables and build expression: 2*x + 3*y + 1
    let x = math.var();
    let y = math.var();
    let expr = &x * 2.0 + &y * 3.0 + 1.0;

    // Evaluate with test values
    let result = math.eval(&expr, &[1.0, 2.0]);
    println!("2*1 + 3*2 + 1 = {result}"); // Expected: 9

    // Pretty print the expression
    println!("Expression: {}", expr.pretty_print());

    // Transcendental functions
    let trig_expr = x.clone().sin() + y.clone().cos();
    let trig_result = math.eval(&trig_expr, &[std::f64::consts::PI / 2.0, 0.0]);
    println!("sin(π/2) + cos(0) = {trig_result:.1}"); // Expected: 2.0

    Ok(())
}
