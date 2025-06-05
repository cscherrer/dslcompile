//! Power operations demo - binary exponentiation optimization

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("=== Power Operations Demo ===");
    let math = ExpressionBuilder::new();

    // Test 1: Integer powers use binary exponentiation
    println!("\n--- Test 1: Integer Power (2^8) ---");
    let x = math.var(); // Variable(0)
    let expr = x.pow(math.constant(8.0));
    println!("Expression: {}", expr.pretty_print());
    let result = math.eval(&expr, &[2.0]); // Variable(0) = 2.0
    println!("2^8 = {result}");

    // Test 2: Fractional powers
    println!("\n--- Test 2: Fractional Power (4^0.5) ---");
    let x2 = math.var(); // Variable(1)
    let sqrt_expr = x2.pow(math.constant(0.5));
    println!("Expression: {}", sqrt_expr.pretty_print());
    // Need to pass values for Variable(0) and Variable(1)
    let sqrt_result = math.eval(&sqrt_expr, &[0.0, 4.0]); // Variable(1) = 4.0
    println!("4^0.5 = {sqrt_result}");

    // Test 3: Variable powers
    println!("\n--- Test 3: Variable Power (3^2) ---");
    let x3 = math.var(); // Variable(2)
    let y = math.var(); // Variable(3)
    let var_power = x3.pow(y);
    println!("Expression: {}", var_power.pretty_print());
    // Need to pass values for Variable(0), Variable(1), Variable(2), Variable(3)
    let var_result = math.eval(&var_power, &[0.0, 0.0, 3.0, 2.0]); // Variable(2) = 3.0, Variable(3) = 2.0
    println!("3^2 = {var_result}");

    Ok(())
}
