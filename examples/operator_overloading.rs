//! Demonstration of ergonomic operator overloading for final tagless expressions
//!
//! This example shows how the `Expr` wrapper enables natural mathematical syntax
//! while maintaining the final tagless approach's flexibility and type safety.

use mathjit::prelude::*;
use mathjit::{DirectEval, PrettyPrint};

fn main() {
    println!("=== MathJIT Operator Overloading Demo ===\n");

    // Example 1: Direct evaluation with natural syntax
    println!("1. Direct Evaluation:");

    // Define a quadratic function: 2x² + 3x + 1
    fn quadratic(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
        let a = Expr::constant(2.0);
        let b = Expr::constant(3.0);
        let c = Expr::constant(1.0);

        // Natural mathematical syntax!
        a * x.clone() * x.clone() + b * x + c
    }

    let x_val = 3.0;
    let x = Expr::var_with_value("x", x_val);
    let result = quadratic(x);

    println!("  f(x) = 2x² + 3x + 1");
    println!("  f({}) = {}", x_val, result.eval());
    println!("  Expected: {}", 2.0 * x_val * x_val + 3.0 * x_val + 1.0);
    println!();

    // Example 2: Pretty printing the same expression
    println!("2. Pretty Printing:");

    fn quadratic_pretty(x: Expr<PrettyPrint, f64>) -> Expr<PrettyPrint, f64> {
        let a = Expr::constant(2.0);
        let b = Expr::constant(3.0);
        let c = Expr::constant(1.0);

        a * x.clone() * x.clone() + b * x + c
    }

    let x_pretty = Expr::<PrettyPrint, f64>::var("x");
    let pretty_result = quadratic_pretty(x_pretty);

    println!("  Expression structure: {}", pretty_result.to_string());
    println!();

    // Example 3: Complex mathematical expressions
    println!("3. Complex Mathematical Functions:");

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

    let x_vals = [0.0, 1.0, -1.0, 2.0];
    println!("  Gaussian function: exp(-x²/2) / sqrt(2π)");
    for &x_val in &x_vals {
        let x = Expr::var_with_value("x", x_val);
        let result = gaussian(x);
        println!("    f({:4.1}) = {:.6}", x_val, result.eval());
    }
    println!();

    // Example 4: Trigonometric identities
    println!("4. Trigonometric Identity Verification:");

    // Verify sin²(x) + cos²(x) = 1
    fn trig_identity(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
        let sin_x = x.clone().sin();
        let cos_x = x.cos();
        sin_x.clone() * sin_x + cos_x.clone() * cos_x
    }

    let test_angles = [
        0.0,
        std::f64::consts::PI / 4.0,
        std::f64::consts::PI / 2.0,
        std::f64::consts::PI,
    ];
    println!("  Verifying sin²(x) + cos²(x) = 1:");
    for &angle in &test_angles {
        let x = Expr::var_with_value("x", angle);
        let result = trig_identity(x);
        println!(
            "    x = {:6.3}, sin²(x) + cos²(x) = {:.10}",
            angle,
            result.eval()
        );
    }
    println!();

    // Example 5: Polynomial operations
    println!("5. Polynomial Operations:");

    // (x + 1)(x - 1) = x² - 1
    fn polynomial_product(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
        let one = Expr::constant(1.0);
        (x.clone() + one.clone()) * (x.clone() - one)
    }

    let x_val = 5.0;
    let x = Expr::var_with_value("x", x_val);
    let result = polynomial_product(x);

    println!("  (x + 1)(x - 1) = x² - 1");
    println!("  x = {}, result = {}", x_val, result.eval());
    println!("  Expected: {}", x_val * x_val - 1.0);
    println!();

    // Example 6: Comparison with traditional approach
    println!("6. Comparison with Traditional Final Tagless:");

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

    // New ergonomic approach
    fn ergonomic_quadratic(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
        let a = Expr::constant(2.0);
        let b = Expr::constant(3.0);
        let c = Expr::constant(1.0);

        a * x.clone() * x.clone() + b * x + c
    }

    let x_val = 4.0;
    let traditional_result = traditional_quadratic::<DirectEval>(DirectEval::var("x", x_val));
    let ergonomic_result = ergonomic_quadratic(Expr::var_with_value("x", x_val));

    println!("  Traditional result: {traditional_result}");
    println!("  Ergonomic result:   {}", ergonomic_result.eval());
    println!("  Both approaches produce the same result!");

    println!("\n=== Demo Complete ===");
}
