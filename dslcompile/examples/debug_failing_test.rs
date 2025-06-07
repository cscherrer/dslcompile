//! Debug the failing test case to understand why complex expressions evaluate to 0.0

use dslcompile::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug: Analyzing failing test case");
    println!("=====================================");

    let math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    // The failing expression from the test
    let expr = &x * &x + 2.0 * &x * &y + &y * &y;

    println!("Variables created:");
    println!("  x: Variable(0)");
    println!("  y: Variable(1)");

    println!("\nExpression AST: {:?}", expr.as_ast());

    // Test simple expressions first
    println!("\nTesting simple expressions:");
    let simple_x = &x;
    let simple_y = &y;
    println!("x = {}", math.eval(simple_x, &[3.0, 4.0]));
    println!("y = {}", math.eval(simple_y, &[3.0, 4.0]));

    let add_expr = &x + &y;
    let mul_expr = &x * &y;
    println!("x + y = {}", math.eval(&add_expr, &[3.0, 4.0]));
    println!("x * y = {}", math.eval(&mul_expr, &[3.0, 4.0]));

    // Test the complex expression
    println!("\nTesting complex expression:");
    let result = math.eval(&expr, &[3.0, 4.0]);
    println!("x² + 2xy + y² = {result}");
    println!("Expected: 49.0 (since (3+4)² = 49)");

    if (result - 49.0).abs() < 1e-10 {
        println!("✅ Test PASSED!");
    } else {
        println!("❌ Test FAILED! Got {result} instead of 49.0");
    }

    println!("\n🔍 Debug: Testing interpretation-only mode");
    println!("==========================================");

    // Create context with interpretation-only strategy
    let math_interp = DynamicContext::new_interpreter();
    let x_interp = math_interp.var();
    let y_interp = math_interp.var();

    // Test the failing expression with interpretation only
    let expr_interp = &x_interp * &x_interp + 2.0 * &x_interp * &y_interp + &y_interp * &y_interp;

    println!("Expression AST: {:?}", expr_interp.as_ast());
    println!("JIT Strategy: {:?}", math_interp.jit_stats().strategy);

    let result_interp = math_interp.eval(&expr_interp, &[3.0, 4.0]);
    println!("Result with interpretation-only: {result_interp}");
    println!("Expected: 49.0 (since (3+4)² = 49)");

    // Test simpler expressions to isolate the issue
    let simple_add = &x_interp + &y_interp;
    let simple_mul = &x_interp * &y_interp;
    let x_squared = &x_interp * &x_interp;
    let y_squared = &y_interp * &y_interp;
    let cross_term = 2.0 * &x_interp * &y_interp;

    println!("\nBreaking down the expression:");
    println!("x + y = {}", math_interp.eval(&simple_add, &[3.0, 4.0]));
    println!("x * y = {}", math_interp.eval(&simple_mul, &[3.0, 4.0]));
    println!("x² = {}", math_interp.eval(&x_squared, &[3.0, 4.0]));
    println!("y² = {}", math_interp.eval(&y_squared, &[3.0, 4.0]));
    println!("2xy = {}", math_interp.eval(&cross_term, &[3.0, 4.0]));

    // Manual reconstruction
    let manual_expr = x_squared + cross_term + y_squared;
    println!(
        "Manual reconstruction (x² + 2xy + y²) = {}",
        math_interp.eval(&manual_expr, &[3.0, 4.0])
    );

    Ok(())
}
