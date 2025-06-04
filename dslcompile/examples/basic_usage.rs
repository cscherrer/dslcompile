//! Basic Usage Examples
//!
//! This example demonstrates the two ways to build mathematical expressions
//! in `DSLCompile`: Dynamic Context (ergonomic) and Static Context (composable).

use dslcompile::prelude::*;

fn main() {
    println!("=== DSLCompile Basic Usage Demo ===\n");

    // 1. Dynamic Context (Runtime Flexibility)
    dynamic_context_demo();

    // 2. Static Context (Compile-Time + Zero Overhead)
    static_context_demo();
}

fn dynamic_context_demo() {
    println!("🚀 Dynamic Context (Runtime Flexibility)");
    println!("========================================");

    let math = DynamicContext::new();
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

fn static_context_demo() {
    println!("⚡ Static Context (Compile-Time + Zero Overhead)");
    println!("==============================================");

    let mut builder = Context::new();

    // Define f(x) = x² in scope 0
    let f = builder.new_scope(|scope| {
        let (x, _scope) = scope.auto_var();
        x.clone() * x
    });
    println!("f(x) = x² in scope 0");

    // Advance to next scope
    let mut builder = Context::new();

    // Define g(y) = 2y in scope 1 (no variable collision!)
    let g = builder.new_scope(|scope| {
        let (y, scope) = scope.auto_var();
        y * scope.constant(2.0)
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
