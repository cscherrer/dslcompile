//! Debug Variable Indexing Issue
//!
//! This example isolates the variable indexing issue in summation evaluation.

use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("🔍 Debug Variable Indexing Issue");
    println!("=================================\n");

    // =======================================================================
    // 1. Simple Variable Creation
    // =======================================================================

    let mut ctx = DynamicContext::new();
    let a = ctx.var(); // Should be variable 0
    let b = ctx.var(); // Should be variable 1

    println!("1️⃣ Variable Creation");
    println!("   Variable a has index: {}", a.var_id());
    println!("   Variable b has index: {}", b.var_id());

    // Test simple evaluation
    let simple_expr = &a + &b;
    let result = ctx.eval(&simple_expr, hlist![10.0, 20.0]);
    println!("   Simple evaluation a + b = {} (expected 30.0)", result);

    // =======================================================================
    // 2. Simple Summation Without External Variables
    // =======================================================================

    println!("\n2️⃣ Simple Summation (No External Variables)");
    let data = vec![1.0, 2.0, 3.0];
    let simple_sum = ctx.sum(&data, |x| x * 2.0);

    println!("   Created summation: x * 2.0 for each x in [1, 2, 3]");

    // This should work fine - no external variables
    let result = ctx.eval(&simple_sum, hlist![]);
    println!("   Result: {} (expected 12.0)", result);

    // =======================================================================
    // 3. Summation With External Variable
    // =======================================================================

    println!("\n3️⃣ Summation With External Variable");
    let sum_with_external = ctx.sum(&data, |x| &x + &a); // x + a

    println!("   Created summation: x + a for each x in [1, 2, 3]");
    println!("   External variable a has index: {}", a.var_id());

    // This is where the issue likely occurs
    println!("   About to evaluate...");
    let result = ctx.eval(&sum_with_external, hlist![5.0]); // a = 5.0
    println!(
        "   Result: {} (expected 21.0: (1+5) + (2+5) + (3+5))",
        result
    );

    // =======================================================================
    // 4. More Complex Case - Two External Variables
    // =======================================================================

    println!("\n4️⃣ Two External Variables");
    let sum_two_external = ctx.sum(&data, |x| &x * &a + &b); // x * a + b

    println!("   Created summation: x * a + b for each x in [1, 2, 3]");
    println!("   External variables: a={}, b={}", a.var_id(), b.var_id());

    println!("   About to evaluate...");
    let result = ctx.eval(&sum_two_external, hlist![2.0, 10.0]); // a = 2.0, b = 10.0
    println!(
        "   Result: {} (expected 48.0: (1*2+10) + (2*2+10) + (3*2+10))",
        result
    );

    Ok(())
}
