//! Basic usage example for `MathJIT`
//!
//! This example demonstrates the final tagless approach with multiple interpreters:
//! - `DirectEval`: Immediate evaluation
//! - `PrettyPrint`: String representation
//! - `JITEval`: Native code compilation (with jit feature)

use mathjit::final_tagless::{DirectEval, MathExpr, PrettyPrint, StatisticalExpr};

// JIT support will be added in future versions

/// Define a quadratic function: 2x² + 3x + 1
fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathJIT Basic Usage Example ===\n");

    // 1. Direct Evaluation
    println!("1. Direct Evaluation:");
    let x_val = 2.0;
    let result = quadratic::<DirectEval>(DirectEval::var("x", x_val));
    println!("   quadratic({x_val}) = {result}");
    println!("   Expected: 2(4) + 3(2) + 1 = 15\n");

    // 2. Pretty Printing
    println!("2. Pretty Printing:");
    let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
    println!("   Expression: {pretty}\n");

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

    // 4. JIT Compilation (will be added in future versions)
    println!("4. JIT Compilation: (coming soon!)");

    println!("\n=== Example Complete ===");
    Ok(())
}
