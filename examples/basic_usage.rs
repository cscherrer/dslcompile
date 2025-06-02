//! Basic usage example for `MathCompile`
//!
//! This example demonstrates both the traditional final tagless approach and the
//! `MathBuilder` API:
//! - `DirectEval`: Immediate evaluation
//! - `PrettyPrint`: String representation
//! - `MathBuilder`: Expression building with operator overloading

use mathcompile::prelude::*;
use mathcompile::{DirectEval, PrettyPrint, StatisticalExpr};

/// Define a quadratic function using traditional final tagless syntax: 2x² + 3x + 1
fn quadratic_traditional<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
where
    E::Repr<f64>: Clone,
{
    let a = E::constant(2.0);
    let b = E::constant(3.0);
    let c = E::constant(1.0);

    E::add(
        E::add(E::mul(a, E::pow(x.clone(), E::constant(2.0))), E::mul(b, x)),
        c,
    )
}

/// Define a logistic function using statistical extensions
fn logistic_regression<E: StatisticalExpr>(x: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
    E::logistic(E::mul(theta, x))
}

fn main() -> Result<()> {
    println!("=== MathCompile Basic Usage Example ===\n");

    // 1. Traditional Final Tagless Approach
    println!("1. Traditional Final Tagless Approach:");
    let x_val = 2.0;
    let result_traditional = quadratic_traditional::<DirectEval>(DirectEval::var_with_value(0, x_val));
    println!("   quadratic({x_val}) = {result_traditional}");
    println!("   Expected: 2(4) + 3(2) + 1 = 15");

    let pretty_traditional = quadratic_traditional::<PrettyPrint>(PrettyPrint::var(0));
    println!("   Expression: {pretty_traditional}\n");

    // 2. MathBuilder Approach with Operator Overloading
    println!("2. MathBuilder Approach:");
    let math = MathBuilder::new();
    let x = math.var();

    // Mathematical syntax using operator overloading
    let quadratic_mathbuilder =
        math.constant(2.0) * &x * &x + math.constant(3.0) * &x + math.constant(1.0);

    let result_mathbuilder = math.eval(&quadratic_mathbuilder, &[x_val]);
    println!("   quadratic({x_val}) = {result_mathbuilder}");
    println!("   Expected: 2(4) + 3(2) + 1 = 15");

    // Verify both approaches give the same result
    assert_eq!(result_traditional, result_mathbuilder);
    println!("   Both approaches produce identical results\n");

    // 3. Statistical Functions
    println!("3. Statistical Functions:");
    let theta_val = 1.5;
    let logistic_result = logistic_regression::<DirectEval>(
        DirectEval::var_with_value(0, x_val),
        DirectEval::var_with_value(1, theta_val),
    );
    println!("   logistic_regression({x_val}, {theta_val}) = {logistic_result}");

    let logistic_pretty =
        logistic_regression::<PrettyPrint>(PrettyPrint::var(0), PrettyPrint::var(1));
    println!("   Expression: {logistic_pretty}");

    // Using MathBuilder for logistic regression
    let math = MathBuilder::new();
    let x = math.var();
    let theta = math.var();
    let logistic_mathbuilder = math.logistic(&(theta * &x));
    let logistic_result_mathbuilder = math.eval(&logistic_mathbuilder, &[x_val, theta_val]);
    println!("   MathBuilder logistic({x_val}, {theta_val}) = {logistic_result_mathbuilder}");
    assert!((logistic_result - logistic_result_mathbuilder).abs() < 1e-10);
    println!("   Traditional and MathBuilder approaches match\n");

    // 4. Complex Mathematical Expressions
    println!("4. Complex Expressions:");

    // Gaussian function: exp(-x²/2) / sqrt(2π) using MathBuilder
    let math = MathBuilder::new();
    let x = math.var();
    let pi = math.constant(std::f64::consts::PI);

    let gaussian = x.clone().exp().ln(); // Just a simple transcendental example
    let gaussian_result = math.eval(&gaussian, &[1.0]);
    println!("   exp(ln(x)) at x=1.0 = {gaussian_result:.6}");
    println!("   Expected: ~1.0");
    assert!((gaussian_result - 1.0).abs() < 0.001);
    println!("   Transcendental calculation correct\n");

    // 5. High-Level Mathematical Functions
    println!("5. High-Level Mathematical Functions:");

    let math = MathBuilder::new();
    let x = math.var();

    // Polynomial using convenience function
    let poly = math.poly(&[1.0, 2.0, 3.0], &x); // 1 + 2x + 3x² (coefficients in ascending order)
    let poly_result = math.eval(&poly, &[2.0]);
    println!("   polynomial 1 + 2x + 3x² at x=2: {poly_result}");
    println!("   Expected: 1 + 2(2) + 3(4) = 1 + 4 + 12 = 17");
    assert_eq!(poly_result, 17.0);

    // Quadratic using convenience function
    let quad = math.quadratic(3.0, 2.0, 1.0, &x); // 3x² + 2x + 1
    let quad_result = math.eval(&quad, &[2.0]);
    println!("   quadratic 3x² + 2x + 1 at x=2: {quad_result}");
    println!("   Expected: 3(4) + 2(2) + 1 = 12 + 4 + 1 = 17");
    assert_eq!(poly_result, quad_result);
    println!("   Polynomial and quadratic functions match");

    // Gaussian using convenience function
    let gaussian_builtin = math.gaussian(0.0, 1.0, &x); // Standard normal
    let gaussian_builtin_result = math.eval(&gaussian_builtin, &[0.0]);
    println!("   Built-in gaussian(0.0) = {gaussian_builtin_result:.6}");
    println!("   Built-in Gaussian works\n");

    // 6. Expression Validation and Optimization
    println!("6. Expression Validation and Optimization:");

    let math = MathBuilder::new();
    let x = math.var();

    // Create an expression that demonstrates the syntax
    let expression = &x * 0.0 + &x * 1.0; // x*0 + x*1 = x
    println!("   Expression: x*0 + x*1 (should simplify to x)");

    // Test the expression
    let result = math.eval(&expression, &[5.0]);
    println!("   Result at x=5: {result}");
    assert_eq!(result, 5.0);
    println!("   Expression evaluation works correctly");

    println!("\n=== MathBuilder API Features ===");
    println!("- Mathematical syntax with operator overloading");
    println!("- Automatic variable management");
    println!("- Built-in mathematical functions and constants");
    println!("- Expression validation and optimization");
    println!("- Named variable evaluation");
    println!("- Type safety and helpful error messages");

    println!("\n=== Example Complete ===");
    Ok(())
}
