use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("Testing Summation Evaluation Issues\n");

    // Test 1: Simple range summation
    println!("=== Test 1: Range Summation ===");
    let mut ctx = DynamicContext::<f64>::new();

    // Create a simple sum: sum(i for i in 1..=3) = 1 + 2 + 3 = 6
    let sum_expr = ctx.sum(1..=3, |i| i);
    let ast = ctx.to_ast(&sum_expr);

    println!("AST structure: {:#?}", ast);

    // Try to evaluate
    let result = ctx.eval(&sum_expr, hlist![]);
    println!("Evaluation result: {}", result);
    println!("Expected: 6.0");

    // Test 2: Simple math expression for comparison
    println!("\n=== Test 2: Simple Math (for comparison) ===");
    let x = ctx.var();
    let y = ctx.var();
    let simple_expr = &x + &y;
    let simple_result = ctx.eval(&simple_expr, hlist![3.0, 4.0]);
    println!("Simple math result: {} (expected: 7.0)", simple_result);
}
