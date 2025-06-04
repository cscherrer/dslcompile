//! Power operations demo - binary exponentiation optimization

use dslcompile::prelude::*;

fn main() -> Result<()> {
    let math = ExpressionBuilder::new();

    // Integer powers use binary exponentiation
    let x = math.var();
    let expr = x.pow(math.constant(8.0));
    println!("x^8: {}", expr.pretty_print());

    let result = math.eval(&expr, &[2.0]);
    println!("2^8 = {result}");

    // Fractional powers
    let x2 = math.var();
    let sqrt_expr = x2.pow(math.constant(0.5));
    let sqrt_result = math.eval(&sqrt_expr, &[4.0]);
    println!("4^0.5 = {sqrt_result}");

    // Variable powers
    let x3 = math.var();
    let y = math.var();
    let var_power = x3.pow(y);
    let var_result = math.eval(&var_power, &[3.0, 2.0]);
    println!("3^2 = {var_result}");

    Ok(())
}
