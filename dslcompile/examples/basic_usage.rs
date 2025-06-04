//! Basic Usage Examples
//!
//! This example demonstrates the two ways to build mathematical expressions
//! in `DSLCompile`: Runtime Expression Building (ergonomic) and Scoped Variables (composable).

use dslcompile::prelude::*;

fn main() {
    println!("=== DSLCompile Basic Usage Demo ===\n");

    // 1. Runtime Expression Building (Most Ergonomic)
    runtime_expression_demo();

    // 2. Scoped Variables (Compile-Time + Composability)
    scoped_variables_demo();
}

fn runtime_expression_demo() {
    println!("🚀 Runtime Expression Building (Most Ergonomic)");
    println!("===============================================");

    let math = MathBuilder::new();
    let x = math.var();
    let y = math.var();

    // Natural mathematical syntax
    let expr = &x * &x + 2.0 * &x * &y + &y * &y;
    println!("Expression: (x + y)²");

    let result = math.eval(&expr, &[3.0, 4.0]);
    println!("Result: (3 + 4)² = {result}");

    println!("✅ Perfect for: Interactive use, ergonomic syntax, debugging");
    println!();
}

fn scoped_variables_demo() {
    println!("⚡ Scoped Variables (Compile-Time + Composability)");
    println!("================================================");

    let mut builder = ScopedExpressionBuilder::new();

    // Define f(x) = x² in scope 0
    let f = builder.new_scope(|scope| {
        let (x, _scope) = scope.auto_var();
        x.clone().mul(x)
    });
    println!("f(x) = x² in scope 0");

    // Advance to next scope
    let mut builder = builder.next();

    // Define g(y) = 2y in scope 1 (no variable collision!)
    let g = builder.new_scope(|scope| {
        let (y, scope) = scope.auto_var();
        y.mul(scope.constant(2.0))
    });
    println!("g(y) = 2y in scope 1");

    // Perfect composition with automatic variable remapping
    let composed = compose(f, g);
    let combined = composed.add(); // h(x,y) = x² + 2y
    println!("h(x,y) = f(x) + g(y) = x² + 2y");

    let result = combined.eval(&[3.0, 4.0]);
    println!("Result: h(3,4) = 3² + 2*4 = 9 + 8 = {result}");

    println!("✅ Perfect for: Function composition, library development, zero overhead");
    println!();
}
