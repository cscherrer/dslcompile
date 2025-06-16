//! Evaluation Bug Test
//!
//! This test reproduces the bug where sum expressions evaluate to 0
//! instead of the expected value.

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🐛 Evaluation Bug Test");
    println!("======================\n");

    // Create a simple sum expression: Σ(x) for x in [1,2,3]
    let mut ctx = DynamicContext::new();
    let data = vec![1.0, 2.0, 3.0];

    println!("1️⃣ Simple Sum Test: Σ(x) for x in [1,2,3]");
    let sum_expr = ctx.sum(data.clone(), |x| x);

    println!("   AST: {:?}", sum_expr.as_ast());

    // Test evaluation
    use frunk::hlist;
    let params = hlist![];

    let result = ctx.eval(&sum_expr, params);
    println!("   Result: {:?}", result);
    println!("   Expected: 1 + 2 + 3 = 6");

    println!("\n2️⃣ Sum with Parameter Test: Σ(a * x) for x in [1,2,3]");
    let a = ctx.var::<f64>();
    let sum_expr2 = ctx.sum(data.clone(), |x| a.clone() * x);

    println!("   AST: {:?}", sum_expr2.as_ast());

    // Test evaluation with a=2
    let params2 = hlist![2.0];
    let result2 = ctx.eval(&sum_expr2, params2);
    println!("   Result: {:?}", result2);
    println!("   Expected: 2 * (1 + 2 + 3) = 2 * 6 = 12");

    println!("\n3️⃣ Debugging Variable Indices");
    println!("   Context created successfully");

    Ok(())
}
