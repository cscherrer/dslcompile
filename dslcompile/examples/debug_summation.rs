use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("=== Debug Summation Issue ===");

    let mut ctx = DynamicContext::new();

    // Create a simple sum: sum(i for i in 1..=3) = 1 + 2 + 3 = 6
    let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=3, |i| i);

    // Look at the AST structure
    let ast = ctx.to_ast(&sum_expr);
    println!("AST: {ast:#?}");

    // Try to evaluate
    let result = ctx.eval(&sum_expr, hlist![]);
    println!("Result: {result} (expected: 6.0)");
}
