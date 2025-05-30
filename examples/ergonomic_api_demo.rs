//! API Demonstration
//!
//! This example showcases the unified typed variable system that provides
//! operator overloading syntax with compile-time type safety.

use mathcompile::MathBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MathCompile Unified Typed Variable System Demo");
    println!("==============================================\n");

    // ========================================================================
    // 1. Operator Overloading
    // ========================================================================

    println!("1. Operator Overloading");
    println!("-----------------------");

    let math = MathBuilder::new();
    let x = math.var("x");
    let y = math.var("y");

    // Mathematical syntax
    let expr1 = &x * &x + 2.0 * &x + &y;
    println!("  Expression: x² + 2x + y");

    let result1 = math.eval(&expr1, &[("x", 3.0), ("y", 1.0)]);
    println!("  Result at x=3, y=1: {result1}"); // 3² + 2*3 + 1 = 16
    assert_eq!(result1, 16.0);
    println!("  Correct\n");

    // ========================================================================
    // 2. Transcendental Functions
    // ========================================================================

    println!("2. Transcendental Functions");
    println!("---------------------------");

    let expr2 = x.clone().sin() * y.clone().cos() + x.clone().exp();
    println!("  Expression: sin(x) * cos(y) + exp(x)");

    let result2 = math.eval(&expr2, &[("x", 0.0), ("y", 0.0)]);
    println!("  Result at x=0, y=0: {result2}"); // sin(0) * cos(0) + exp(0) = 0 * 1 + 1 = 1
    assert_eq!(result2, 1.0);
    println!("  Correct\n");

    // ========================================================================
    // 3. High-Level Mathematical Functions
    // ========================================================================

    println!("3. High-Level Mathematical Functions");
    println!("------------------------------------");

    // Polynomial: 2x² + 3x + 1
    let poly = math.poly(&[1.0, 3.0, 2.0], &x);
    println!("  Polynomial: 2x² + 3x + 1");
    let poly_result = math.eval(&poly, &[("x", 2.0)]);
    println!("  Result at x=2: {poly_result}"); // 2*4 + 3*2 + 1 = 15
    assert_eq!(poly_result, 15.0);
    println!("  Correct");

    // Quadratic: x² - 4x + 3
    let quad = math.quadratic(1.0, -4.0, 3.0, &x);
    println!("  Quadratic: x² - 4x + 3");
    let quad_result = math.eval(&quad, &[("x", 1.0)]);
    println!("  Result at x=1: {quad_result}"); // 1 - 4 + 3 = 0
    assert_eq!(quad_result, 0.0);
    println!("  Correct");

    // Gaussian distribution (mean=0, std=1)
    let gaussian = math.gaussian(0.0, 1.0, &x);
    println!("  Gaussian: N(0,1)");
    let gauss_result = math.eval(&gaussian, &[("x", 0.0)]);
    println!("  Result at x=0: {gauss_result:.6}"); // Should be ~0.398942 (1/√(2π))
    assert!((gauss_result - 0.398942).abs() < 0.001);
    println!("  Correct");

    // Logistic function
    let logistic = math.logistic(&x);
    println!("  Logistic: 1/(1 + exp(-x))");
    let logistic_result = math.eval(&logistic, &[("x", 0.0)]);
    println!("  Result at x=0: {logistic_result}"); // Should be 0.5
    assert_eq!(logistic_result, 0.5);
    println!("  Correct\n");

    // ========================================================================
    // 4. Complex Expressions
    // ========================================================================

    println!("4. Complex Expressions");
    println!("----------------------");

    // Combine high-level functions with operators
    let complex = &poly + &logistic * math.constant(10.0);
    println!("  Expression: (2x² + 3x + 1) + 10 * logistic(x)");
    let complex_result = math.eval(&complex, &[("x", 1.0)]);
    println!("  Result at x=1: {complex_result:.6}");
    println!("  Complex expressions work\n");

    // ========================================================================
    // 5. Type Safety Demo
    // ========================================================================

    println!("5. Type Safety");
    println!("--------------");

    // Create typed variables
    let x_f64 = math.typed_var::<f64>("x_f64");
    let y_f32 = math.typed_var::<f32>("y_f32");

    let x_expr = math.expr_from(x_f64);
    let y_expr = math.expr_from(y_f32);

    // Cross-type operations work with automatic promotion
    let mixed = &x_expr + y_expr; // f32 automatically promotes to f64
    println!("  Mixed types: f64 + f32 → f64");
    let mixed_result = math.eval(&mixed, &[("x_f64", 2.5), ("y_f32", 1.5)]);
    println!("  Result: {mixed_result}");
    assert_eq!(mixed_result, 4.0);
    println!("  Type promotion works\n");

    // ========================================================================
    // 6. API Comparison
    // ========================================================================

    println!("6. API Comparison: Before vs After");
    println!("----------------------------------");

    println!("  Before (verbose):");
    println!("    ASTEval::add(");
    println!("        ASTEval::mul(ASTEval::var(0), ASTEval::constant(2.0)),");
    println!("        ASTEval::constant(1.0)");
    println!("    )");
    println!();

    println!("  After (concise):");
    println!("    let math = MathBuilder::new();");
    println!("    let x = math.var(\"x\");");
    println!("    let expr = &x * 2.0 + 1.0;");
    println!("    let result = math.eval(&expr, &[(\"x\", 3.0)]);");
    println!();

    // Demonstrate both work the same
    let concise_expr = &x * 2.0 + 1.0;
    let concise_result = math.eval(&concise_expr, &[("x", 3.0)]);
    println!("  Concise syntax result: {concise_result}");
    assert_eq!(concise_result, 7.0);
    println!("  Concise syntax works correctly\n");

    println!("Demo Complete");
    println!("\nFeatures:");
    println!("- Operator overloading syntax: x * x + 2.0 * x + y");
    println!("- Type safety with automatic promotion");
    println!("- High-level mathematical functions");
    println!("- Simple evaluation interface");
    println!("- Full backward compatibility");

    Ok(())
}
