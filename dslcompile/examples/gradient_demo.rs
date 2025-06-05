//! Gradient computation demo - automatic differentiation

use dslcompile::ast::pretty::pretty_ast;
use dslcompile::prelude::*;
use dslcompile::symbolic::symbolic_ad::convenience;

fn main() -> Result<()> {
    let math = DynamicContext::new();

    // Define function f(x, y) = x² + xy + y²
    let x = math.var();
    let y = math.var();
    let f = x.clone().pow(math.constant(2.0)) + x.clone() * y.clone() + y.pow(math.constant(2.0));

    println!("f(x, y) = {}", f.pretty_print());

    // Compute gradients using convenience function
    let gradient = convenience::gradient(&math.to_ast(&f), &["0", "1"])?;

    println!(
        "∂f/∂x = {}",
        pretty_ast(&gradient["0"], &math.registry().borrow())
    );
    println!(
        "∂f/∂y = {}",
        pretty_ast(&gradient["1"], &math.registry().borrow())
    );

    // Evaluate gradient at (2, 3)
    let grad_x = gradient["0"].eval_with_vars(&[2.0, 3.0]);
    let grad_y = gradient["1"].eval_with_vars(&[2.0, 3.0]);
    println!("∇f(2, 3) = [{grad_x}, {grad_y}]");

    Ok(())
}
