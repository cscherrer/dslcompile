//! Basic Usage Examples
//!
//! This example demonstrates all three ways to build mathematical expressions
//! in `DSLCompile`, from the most ergonomic to the most advanced.

use dslcompile::prelude::*;

fn main() {
    println!("=== DSLCompile Basic Usage Demo ===\n");

    // 1. Runtime Expression Building (Most Ergonomic)
    runtime_expression_demo();

    // 2. Scoped Variables (Compile-Time + Composability)
    scoped_variables_demo();

    // 3. Legacy Compile-Time (For Reference)
    legacy_compile_time_demo();
}

fn runtime_expression_demo() {
    println!("🚀 Runtime Expression Building (Recommended)");
    println!("============================================");

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

    // Define f(x) = x² in scope 0
    let x = scoped_var::<0, 0>();
    let f = x.clone().mul(x);
    println!("f(x) = x² in scope 0");

    // Define g(y) = 2y in scope 1 (no variable collision!)
    let y = scoped_var::<0, 1>();
    let g = y.mul(scoped_constant::<1>(2.0));
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

fn legacy_compile_time_demo() {
    println!("📚 Legacy Compile-Time (For Reference)");
    println!("======================================");

    let x = var::<0>();
    let expr = x.clone().mul(x).add(constant(1.0));
    println!("Expression: x² + 1");

    let result = expr.eval(&[3.0]);
    println!("Result: 3² + 1 = {result}");

    println!("⚠️  Note: Limited composability due to variable index collisions");
    println!("💡 Consider using Scoped Variables or Runtime Building instead");
    println!();
}
