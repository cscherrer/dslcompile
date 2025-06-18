//! Simple Mathematical Operations Demo
//!
//! This demo showcases basic mathematical operations with the current `DynamicContext` API
//! demonstrating expression building, evaluation, and pretty printing.

use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("🧮 Simple Mathematical Operations Demo");
    println!("=====================================\n");

    let mut ctx = DynamicContext::new();

    // Create variables
    let x = ctx.var();
    let y = ctx.var();
    let z = ctx.var();

    println!("1️⃣ Basic Arithmetic Operations");
    println!("------------------------------");

    // Addition
    let add_expr = &x + &y;
    println!("Addition: {}", ctx.pretty_print(&add_expr));
    let add_result = ctx.eval(&add_expr, hlist![3.0, 4.0]);
    println!("Evaluated at x=3, y=4: {add_result} (expected: 7)");
    assert_eq!(add_result, 7.0);

    // Multiplication
    let mul_expr = &x * &y;
    println!("Multiplication: {}", ctx.pretty_print(&mul_expr));
    let mul_result = ctx.eval(&mul_expr, hlist![3.0, 4.0]);
    println!("Evaluated at x=3, y=4: {mul_result} (expected: 12)");
    assert_eq!(mul_result, 12.0);

    // Subtraction
    let sub_expr = &x - &y;
    println!("Subtraction: {}", ctx.pretty_print(&sub_expr));
    let sub_result = ctx.eval(&sub_expr, hlist![3.0, 4.0]);
    println!("Evaluated at x=3, y=4: {sub_result} (expected: -1)");
    assert_eq!(sub_result, -1.0);

    // Division
    let div_expr = &x / &y;
    println!("Division: {}", ctx.pretty_print(&div_expr));
    let div_result = ctx.eval(&div_expr, hlist![8.0, 4.0]);
    println!("Evaluated at x=8, y=4: {div_result} (expected: 2)");
    assert_eq!(div_result, 2.0);

    println!("\n2️⃣ Complex Expressions");
    println!("----------------------");

    // Polynomial: 2x² + 3x + 1
    let poly = 2.0 * &x * &x + 3.0 * &x + 1.0;
    println!("Polynomial: {}", ctx.pretty_print(&poly));
    let poly_result = ctx.eval(&poly, hlist![2.0]);
    println!("Evaluated at x=2: {poly_result} (expected: 2*4 + 3*2 + 1 = 15)");
    assert_eq!(poly_result, 15.0);

    // Three variable expression: x*y + y*z + x*z
    let three_var = &x * &y + &y * &z + &x * &z;
    println!("Three variables: {}", ctx.pretty_print(&three_var));
    let three_result = ctx.eval(&three_var, hlist![2.0, 3.0, 4.0]);
    println!("Evaluated at x=2, y=3, z=4: {three_result} (expected: 6 + 12 + 8 = 26)");
    assert_eq!(three_result, 26.0);

    println!("\n3️⃣ Trigonometric Functions");
    println!("---------------------------");

    // Sin function
    let sin_expr = x.clone().sin();
    println!("Sine: {}", ctx.pretty_print(&sin_expr));
    let sin_result = ctx.eval(&sin_expr, hlist![0.0]);
    println!("Evaluated at x=0: {sin_result} (expected: ~0)");
    assert!((sin_result - 0.0).abs() < 1e-10);

    // Cos function
    let cos_expr = x.clone().cos();
    println!("Cosine: {}", ctx.pretty_print(&cos_expr));
    let cos_result = ctx.eval(&cos_expr, hlist![0.0]);
    println!("Evaluated at x=0: {cos_result} (expected: ~1)");
    assert!((cos_result - 1.0).abs() < 1e-10);

    // Combined: sin²(x) + cos²(x) = 1
    let identity = sin_expr.clone() * sin_expr + cos_expr.clone() * cos_expr;
    println!("Trig identity: {}", ctx.pretty_print(&identity));
    let identity_result = ctx.eval(&identity, hlist![0.5]);
    println!("Evaluated at x=0.5: {identity_result} (expected: ~1)");
    assert!((identity_result - 1.0).abs() < 1e-10);

    println!("\n4️⃣ Power and Exponential Functions");
    println!("-----------------------------------");

    // Power function
    let power_expr = x.clone().pow(y.clone());
    println!("Power: {}", ctx.pretty_print(&power_expr));
    let power_result = ctx.eval(&power_expr, hlist![2.0, 3.0]);
    println!("Evaluated at x=2, y=3: {power_result} (expected: 8)");
    assert_eq!(power_result, 8.0);

    // Exponential
    let exp_expr = x.clone().exp();
    println!("Exponential: {}", ctx.pretty_print(&exp_expr));
    let exp_result = ctx.eval(&exp_expr, hlist![0.0]);
    println!("Evaluated at x=0: {exp_result} (expected: ~1)");
    assert!((exp_result - 1.0).abs() < 1e-10);

    // Natural logarithm
    let ln_expr = x.clone().ln();
    println!("Natural log: {}", ctx.pretty_print(&ln_expr));
    let ln_result = ctx.eval(&ln_expr, hlist![1.0]);
    println!("Evaluated at x=1: {ln_result} (expected: ~0)");
    assert!((ln_result - 0.0).abs() < 1e-10);

    #[cfg(feature = "optimization")]
    {
        println!("\n5️⃣ Symbolic Optimization");
        println!("-------------------------");

        let mut optimizer = SymbolicOptimizer::new()?;

        // Simple optimization: x + 0 should become x
        let simple_opt = &x + 0.0;
        println!("Before optimization: {}", ctx.pretty_print(&simple_opt));

        let optimized = optimizer.optimize(simple_opt.as_ast())?;
        println!("After optimization: {optimized:?}");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n5️⃣ Symbolic Optimization");
        println!("-------------------------");
        println!("(Skipped - optimization feature not enabled)");
    }

    println!("\n✅ All mathematical operations completed successfully!");
    println!("The multiset-based AST handles all standard mathematical operations correctly.");

    Ok(())
}
