//! Ergonomic API Demonstration
//!
//! This example showcases the new `MathBuilder` API that provides a unified,
//! user-friendly interface for building mathematical expressions. It demonstrates
//! how the ergonomic improvements make `MathJIT` much easier to use while maintaining
//! all the performance benefits.

use mathjit::prelude::*;

fn main() -> Result<()> {
    println!("🚀 MathJIT Ergonomic API Demonstration");
    println!("=====================================\n");

    // ========================================================================
    // 1. Basic Usage - Much Simpler Than Before
    // ========================================================================

    println!("1️⃣  Basic Expression Building");
    println!("-----------------------------");

    let mut math = MathBuilder::new();

    // Variables are automatically managed
    let x = math.var("x");
    let y = math.var("y");

    // Natural mathematical operations
    let expr = math.add(&math.mul(&x, &math.constant(2.0)), &y);

    // Easy evaluation with named variables
    let result = math.eval(&expr, &[("x", 3.0), ("y", 4.0)]);
    println!("  Expression: 2*x + y");
    println!("  At x=3, y=4: {result}");
    println!("  Expected: 2*3 + 4 = 10");
    assert_eq!(result, 10.0);
    println!("  ✓ Correct!\n");

    // ========================================================================
    // 2. High-Level Mathematical Functions
    // ========================================================================

    println!("2️⃣  High-Level Mathematical Functions");
    println!("------------------------------------");

    // Polynomial creation is now trivial
    let x = math.var("x");
    let quadratic = math.quadratic(2.0, -3.0, 1.0, &x); // 2x² - 3x + 1

    let result = math.eval(&quadratic, &[("x", 2.0)]);
    println!("  Quadratic: 2x² - 3x + 1");
    println!("  At x=2: {result}");
    println!("  Expected: 2(4) - 3(2) + 1 = 8 - 6 + 1 = 3");
    assert_eq!(result, 3.0);
    println!("  ✓ Correct!");

    // Polynomial with arbitrary coefficients
    let poly = math.poly(&[1.0, 0.0, 2.0, -1.0], &x); // 1 + 0x + 2x² - x³
    let result = math.eval(&poly, &[("x", 2.0)]);
    println!("  Polynomial: 1 + 2x² - x³");
    println!("  At x=2: {result}");
    println!("  Expected: 1 + 2(4) - 8 = 1 + 8 - 8 = 1");
    assert_eq!(result, 1.0);
    println!("  ✓ Correct!\n");

    // ========================================================================
    // 3. Mathematical Constants
    // ========================================================================

    println!("3️⃣  Mathematical Constants");
    println!("-------------------------");

    let pi = math.math_constant("pi")?;
    let e = math.math_constant("e")?;

    // Create e^(πi) + 1 (well, the real part since we don't have complex numbers yet)
    let x = math.var("x");
    let euler_identity_real = math.add(&math.exp(&math.mul(&pi, &x)), &math.constant(1.0));

    // At x=0, this should be e^0 + 1 = 2
    let result = math.eval(&euler_identity_real, &[("x", 0.0)]);
    println!("  Expression: exp(π*x) + 1");
    println!("  At x=0: {result}");
    println!("  Expected: exp(0) + 1 = 2");
    assert!((result - 2.0).abs() < 1e-10);
    println!("  ✓ Correct!");

    // Show available constants
    println!("  Available constants: pi, e, tau, sqrt2, ln2, ln10\n");

    // ========================================================================
    // 4. Statistical and ML Functions
    // ========================================================================

    println!("4️⃣  Statistical & Machine Learning Functions");
    println!("-------------------------------------------");

    // Gaussian (normal) distribution
    let x = math.var("x");
    let gaussian = math.gaussian(0.0, 1.0, &x); // Standard normal

    let result = math.eval(&gaussian, &[("x", 0.0)]);
    println!("  Standard Normal Distribution at x=0:");
    println!("  Result: {result:.6}");
    println!("  Expected: ~0.398942 (1/√(2π))");
    assert!((result - 0.398942).abs() < 0.001);
    println!("  ✓ Correct!");

    // Logistic (sigmoid) function
    let logistic = math.logistic(&x);
    let result = math.eval(&logistic, &[("x", 0.0)]);
    println!("  Logistic function at x=0:");
    println!("  Result: {result}");
    println!("  Expected: 0.5");
    assert!((result - 0.5).abs() < 1e-10);
    println!("  ✓ Correct!");

    // Hyperbolic tangent
    let tanh_expr = math.tanh(&x);
    let result = math.eval(&tanh_expr, &[("x", 0.0)]);
    println!("  Hyperbolic tangent at x=0:");
    println!("  Result: {result}");
    println!("  Expected: 0.0");
    assert!(result.abs() < 1e-10);
    println!("  ✓ Correct!\n");

    // ========================================================================
    // 5. Preset Functions for Common Use Cases
    // ========================================================================

    println!("5️⃣  Preset Functions");
    println!("-------------------");

    let x = math.var("x");

    // Standard normal distribution preset
    let std_normal = presets::standard_normal(&math, &x);
    let result = math.eval(&std_normal, &[("x", 0.0)]);
    println!("  Standard normal preset at x=0: {result:.6}");

    // Machine learning loss functions
    let y_pred = math.var("y_pred");
    let y_true = math.var("y_true");

    let mse = presets::mse_loss(&math, &y_pred, &y_true);
    let result = math.eval(&mse, &[("y_pred", 0.8), ("y_true", 1.0)]);
    println!("  MSE Loss (pred=0.8, true=1.0): {result}");
    println!("  Expected: (0.8-1.0)² = 0.04");
    assert!((result - 0.04).abs() < 1e-10);
    println!("  ✓ Correct!\n");

    // ========================================================================
    // 6. Expression Validation
    // ========================================================================

    println!("6️⃣  Expression Validation");
    println!("------------------------");

    // Valid expression
    let valid_expr = math.quadratic(1.0, 2.0, 3.0, &x); // x² + 2x + 3
    match math.validate(&valid_expr) {
        Ok(()) => println!("  ✓ Valid expression passed validation"),
        Err(e) => println!("  ✗ Unexpected validation error: {e}"),
    }

    // Invalid expression (division by zero)
    let invalid_expr = math.div(&math.constant(1.0), &math.constant(0.0));
    match math.validate(&invalid_expr) {
        Ok(()) => println!("  ✗ Invalid expression incorrectly passed validation"),
        Err(_) => println!("  ✓ Invalid expression correctly caught by validation"),
    }

    // Invalid constant (NaN)
    let nan_expr = math.constant(f64::NAN);
    match math.validate(&nan_expr) {
        Ok(()) => println!("  ✗ NaN expression incorrectly passed validation"),
        Err(_) => println!("  ✓ NaN expression correctly caught by validation"),
    }

    println!();

    // ========================================================================
    // 7. Integration with Optimization and Differentiation
    // ========================================================================

    println!("7️⃣  Integration with Advanced Features");
    println!("-------------------------------------");

    // Create a builder with optimization enabled
    let mut opt_math = MathBuilder::with_optimization()?;
    let x = opt_math.var("x");

    // Create a complex expression that can be optimized
    let complex_expr = opt_math.add(
        &opt_math.mul(&x, &opt_math.constant(0.0)), // This should optimize to 0
        &opt_math.pow(&x, &opt_math.constant(1.0)), // This should optimize to x
    );

    println!("  Original expression: x*0 + x^1");

    // Optimize the expression
    let optimized = opt_math.optimize(&complex_expr)?;
    println!("  After optimization: (should be simplified)");

    // Test that both give the same result
    let original_result = opt_math.eval(&complex_expr, &[("x", 5.0)]);
    let optimized_result = opt_math.eval(&optimized, &[("x", 5.0)]);

    println!("  Original result at x=5: {original_result}");
    println!("  Optimized result at x=5: {optimized_result}");
    assert_eq!(original_result, optimized_result);
    println!("  ✓ Optimization preserves correctness!");

    // Compute derivative
    let derivative = opt_math.derivative(&optimized, "x")?;
    let deriv_result = opt_math.eval(&derivative, &[("x", 5.0)]);
    println!("  Derivative at x=5: {deriv_result}");
    println!("  Expected: 1.0 (since optimized expression should be x)");

    println!();

    // ========================================================================
    // 8. Comparison with Traditional API
    // ========================================================================

    println!("8️⃣  Comparison: Traditional vs Ergonomic API");
    println!("-------------------------------------------");

    println!("  Traditional approach (verbose):");
    println!("    let expr = ASTEval::add(");
    println!("        ASTEval::mul(ASTEval::var(0), ASTEval::constant(2.0)),");
    println!("        ASTEval::var(1)");
    println!("    );");
    println!("    // Need to manage variable indices manually");
    println!("    // Need to create variable arrays for evaluation");
    println!();

    println!("  Ergonomic approach (intuitive):");
    println!("    let mut math = MathBuilder::new();");
    println!("    let x = math.var(\"x\");");
    println!("    let y = math.var(\"y\");");
    println!("    let expr = math.add(&math.mul(&x, &math.constant(2.0)), &y);");
    println!("    let result = math.eval(&expr, &[(\"x\", 3.0), (\"y\", 4.0)]);");
    println!();

    println!("  Benefits of the ergonomic API:");
    println!("  ✓ Automatic variable management");
    println!("  ✓ Named variable evaluation");
    println!("  ✓ Built-in validation");
    println!("  ✓ High-level mathematical functions");
    println!("  ✓ Integration with optimization and differentiation");
    println!("  ✓ Preset functions for common use cases");
    println!("  ✓ Better error messages");
    println!("  ✓ Type safety with helpful validation");

    println!("\n🎉 Ergonomic API Demo Complete!");
    println!("The new MathBuilder API makes MathJIT much more accessible while");
    println!("maintaining all the performance benefits of the underlying system.");

    Ok(())
}
