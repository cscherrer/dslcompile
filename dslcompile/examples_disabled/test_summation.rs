use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("Testing Summation Evaluation Issues\n");

    // Test 1: Simple range summation
    println!("=== Test 1: Range Summation ===");
    let mut ctx = DynamicContext::new();

    // Create a simple sum: sum(i for i in 1..=3) = 1 + 2 + 3 = 6
    let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=3, |i: DynamicExpr<f64, 0>| i);
    let ast = ctx.to_ast(&sum_expr);

    println!("AST structure: {ast:#?}");

    // Try to evaluate
    let result = ctx.eval(&sum_expr, hlist![]);
    println!("Evaluation result: {result}");
    println!("Expected: 6.0");

    // Test 2: Simple math expression for comparison
    println!("\n=== Test 2: Simple Math (for comparison) ===");
    let x: DynamicExpr<f64, 0> = ctx.var();
    let y: DynamicExpr<f64, 0> = ctx.var();
    let simple_expr = &x + &y;
    let simple_result = ctx.eval(&simple_expr, hlist![3.0, 4.0]);
    println!("Simple math result: {simple_result} (expected: 7.0)");
}
