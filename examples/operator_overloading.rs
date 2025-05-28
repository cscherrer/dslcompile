//! Demonstration of ergonomic operator overloading for mathematical expressions
//!
//! This example shows how the new `MathBuilder` API enables natural mathematical syntax
//! with native operator overloading on `ASTRepr`<f64> while maintaining type safety.

use mathjit::prelude::*;
use mathjit::{DirectEval, PrettyPrint};

fn main() -> Result<()> {
    println!("=== MathJIT Operator Overloading Demo ===\n");

    // Example 1: Direct evaluation with natural syntax
    println!("1. Direct Evaluation with Operator Overloading:");

    let mut math = MathBuilder::new();
    let x = math.var("x");

    // Define a quadratic function: 2x² + 3x + 1 using natural syntax
    let quadratic = math.constant(2.0) * &x * &x + math.constant(3.0) * &x + math.constant(1.0);

    let x_val = 3.0;
    let result = math.eval(&quadratic, &[("x", x_val)]);

    println!("  f(x) = 2x² + 3x + 1");
    println!("  f({x_val}) = {result}");
    println!("  Expected: {}", 2.0 * x_val * x_val + 3.0 * x_val + 1.0);
    assert_eq!(result, 2.0 * x_val * x_val + 3.0 * x_val + 1.0);
    println!("  ✓ Calculation correct!");
    println!();

    // Example 2: Complex mathematical expressions
    println!("2. Complex Mathematical Functions:");

    // Gaussian function: exp(-x²/2) / sqrt(2π)
    let mut math = MathBuilder::new();
    let x = math.var("x");
    let pi = math.math_constant("pi")?;

    let gaussian =
        math.exp(&(-(&x * &x) / math.constant(2.0))) / math.sqrt(&(math.constant(2.0) * &pi));

    let x_vals = [0.0, 1.0, -1.0, 2.0];
    println!("  Gaussian function: exp(-x²/2) / sqrt(2π)");
    for &x_val in &x_vals {
        let result = math.eval(&gaussian, &[("x", x_val)]);
        println!("    f({x_val:4.1}) = {result:.6}");
    }
    println!();

    // Example 3: Trigonometric identities
    println!("3. Trigonometric Identity Verification:");

    // Verify sin²(x) + cos²(x) = 1
    let mut math = MathBuilder::new();
    let x = math.var("x");
    let trig_identity = x.sin_ref() * x.sin_ref() + x.cos_ref() * x.cos_ref();

    let test_angles = [
        0.0,
        std::f64::consts::PI / 4.0,
        std::f64::consts::PI / 2.0,
        std::f64::consts::PI,
    ];
    println!("  Verifying sin²(x) + cos²(x) = 1:");
    for &angle in &test_angles {
        let result = math.eval(&trig_identity, &[("x", angle)]);
        println!("    x = {angle:6.3}, sin²(x) + cos²(x) = {result:.10}");
        assert!((result - 1.0).abs() < 1e-10);
    }
    println!("  ✓ All trigonometric identities verified!");
    println!();

    // Example 4: Polynomial operations
    println!("4. Polynomial Operations:");

    // (x + 1)(x - 1) = x² - 1
    let mut math = MathBuilder::new();
    let x = math.var("x");
    let one = math.constant(1.0);
    let polynomial_product = (&x + &one) * (&x - &one);

    let x_val = 5.0;
    let result = math.eval(&polynomial_product, &[("x", x_val)]);

    println!("  (x + 1)(x - 1) = x² - 1");
    println!("  x = {x_val}, result = {result}");
    println!("  Expected: {}", x_val * x_val - 1.0);
    assert_eq!(result, x_val * x_val - 1.0);
    println!("  ✓ Polynomial expansion correct!");
    println!();

    // Example 5: Comparison with traditional approach
    println!("5. Comparison with Traditional Final Tagless:");

    // Traditional approach (verbose)
    fn traditional_quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        let a = E::constant(2.0);
        let b = E::constant(3.0);
        let c = E::constant(1.0);

        E::add(
            E::add(E::mul(a, E::mul(x.clone(), x.clone())), E::mul(b, x)),
            c,
        )
    }

    let x_val = 4.0;
    let traditional_result = traditional_quadratic::<DirectEval>(DirectEval::var("x", x_val));

    // New ergonomic approach with operator overloading
    let mut math = MathBuilder::new();
    let x = math.var("x");
    let ergonomic_quadratic =
        math.constant(2.0) * &x * &x + math.constant(3.0) * &x + math.constant(1.0);
    let ergonomic_result = math.eval(&ergonomic_quadratic, &[("x", x_val)]);

    println!("  Traditional result: {traditional_result}");
    println!("  Ergonomic result:   {ergonomic_result}");
    assert_eq!(traditional_result, ergonomic_result);
    println!("  ✓ Both approaches produce the same result!");
    println!();

    // Example 6: Advanced operator combinations
    println!("6. Advanced Operator Combinations:");

    let mut math = MathBuilder::new();
    let x = math.var("x");
    let y = math.var("y");

    // Complex expression: sin(x + y) * exp(x - y) + sqrt(x * y)
    let complex_expr = (&x + &y).sin_ref() * (&x - &y).exp_ref() + (&x * &y).sqrt_ref();

    let x_val = 1.0;
    let y_val = 2.0;
    let result = math.eval(&complex_expr, &[("x", x_val), ("y", y_val)]);

    println!("  Complex expression: sin(x + y) * exp(x - y) + sqrt(x * y)");
    println!("  x = {x_val}, y = {y_val}");
    println!("  Result: {result:.6}");

    // Verify manually
    let manual_result = (x_val + y_val).sin() * (x_val - y_val).exp() + (x_val * y_val).sqrt();
    println!("  Manual calculation: {manual_result:.6}");
    assert!((result - manual_result).abs() < 1e-10);
    println!("  ✓ Complex expression calculation correct!");
    println!();

    // Example 7: Pretty printing with operator overloading
    println!("7. Pretty Printing:");

    // Traditional pretty printing
    fn traditional_pretty<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        E::add(
            E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))),
            x,
        )
    }

    let pretty_traditional = traditional_pretty::<PrettyPrint>(PrettyPrint::var("x"));
    println!("  Traditional pretty print: {pretty_traditional}");

    // Note: The new operator overloading works directly on ASTRepr<f64>
    // For pretty printing, we can still use the traditional approach or
    // build expressions and then convert them to strings
    println!("  Modern approach uses direct ASTRepr<f64> with natural operators");
    println!("  This provides better performance and type safety!");

    println!("\n=== Key Benefits of Native Operator Overloading ===");
    println!("✓ Natural mathematical syntax: a * x + b instead of E::add(E::mul(a, x), b)");
    println!("✓ Direct operation on ASTRepr<f64> - no wrapper overhead");
    println!("✓ Type safety with compile-time checking");
    println!("✓ Seamless integration with MathBuilder API");
    println!("✓ Support for complex expressions with mixed operators");
    println!("✓ Reference-based operations to avoid unnecessary cloning");

    println!("\n=== Demo Complete ===");
    Ok(())
}
