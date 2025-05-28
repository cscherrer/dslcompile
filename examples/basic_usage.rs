//! Basic usage example for `MathJIT`
//!
//! This example demonstrates both the traditional final tagless approach and the new
//! ergonomic operator overloading syntax:
//! - `DirectEval`: Immediate evaluation
//! - `PrettyPrint`: String representation
//! - `ASTEval`: Native code compilation (with jit feature)

use mathjit::prelude::*;
use mathjit::{DirectEval, PrettyPrint, StatisticalExpr};

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

/// Define the same quadratic function using ergonomic operator overloading: 2x² + 3x + 1
fn quadratic_modern(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
    let a = Expr::constant(2.0);
    let b = Expr::constant(3.0);
    let c = Expr::constant(1.0);

    // Natural mathematical syntax!
    a * x.clone().pow(Expr::constant(2.0)) + b * x + c
}

/// Pretty printing version with operator overloading
fn quadratic_pretty(x: Expr<PrettyPrint, f64>) -> Expr<PrettyPrint, f64> {
    let a = Expr::constant(2.0);
    let b = Expr::constant(3.0);
    let c = Expr::constant(1.0);

    a * x.clone().pow(Expr::constant(2.0)) + b * x + c
}

/// Define a logistic function using statistical extensions
fn logistic_regression<E: StatisticalExpr>(x: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
    E::logistic(E::mul(theta, x))
}

fn main() {
    println!("=== MathJIT Basic Usage Example ===\n");

    // 1. Traditional Final Tagless Approach
    println!("1. Traditional Final Tagless Approach:");
    let x_val = 2.0;
    let result_traditional = quadratic_traditional::<DirectEval>(DirectEval::var("x", x_val));
    println!("   quadratic({x_val}) = {result_traditional}");
    println!("   Expected: 2(4) + 3(2) + 1 = 15");

    let pretty_traditional = quadratic_traditional::<PrettyPrint>(PrettyPrint::var("x"));
    println!("   Expression: {pretty_traditional}\n");

    // 2. Modern Operator Overloading Approach
    println!("2. Modern Operator Overloading Approach:");
    let result_modern = quadratic_modern(Expr::var_with_value("x", x_val));
    let result_modern_val = result_modern.eval();
    println!("   quadratic({x_val}) = {result_modern_val}");
    println!("   Expected: 2(4) + 3(2) + 1 = 15");

    let pretty_modern = quadratic_pretty(Expr::<PrettyPrint, f64>::var("x"));
    println!("   Expression: {}", pretty_modern.to_string());

    // Verify both approaches give the same result
    assert_eq!(result_traditional, result_modern_val);
    println!("   ✓ Both approaches produce identical results!\n");

    // 3. Statistical Functions
    println!("3. Statistical Functions:");
    let theta_val = 1.5;
    let logistic_result = logistic_regression::<DirectEval>(
        DirectEval::var("x", x_val),
        DirectEval::var("theta", theta_val),
    );
    println!("   logistic_regression({x_val}, {theta_val}) = {logistic_result}");

    let logistic_pretty =
        logistic_regression::<PrettyPrint>(PrettyPrint::var("x"), PrettyPrint::var("theta"));
    println!("   Expression: {logistic_pretty}\n");

    // 4. Complex Mathematical Expressions with Operator Overloading
    println!("4. Complex Expressions with Natural Syntax:");

    // Gaussian function: exp(-x²/2) / sqrt(2π)
    fn gaussian(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
        let two = Expr::constant(2.0);
        let pi = Expr::constant(std::f64::consts::PI);

        let x_squared = x.clone() * x;
        let neg_x_squared_over_two = -(x_squared / two.clone());
        let numerator = neg_x_squared_over_two.exp();
        let denominator = (two * pi).sqrt();

        numerator / denominator
    }

    let gaussian_result = gaussian(Expr::var_with_value("x", 0.0));
    println!("   gaussian(0.0) = {:.6}", gaussian_result.eval());
    println!("   Expected: ~0.398942 (1/sqrt(2π))\n");

    // 5. JIT Compilation (will be added in future versions)
    println!("5. JIT Compilation: (coming soon!)");

    println!("\n=== Example Complete ===");
}
