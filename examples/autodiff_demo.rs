//! Automatic Differentiation Demo
//!
//! This example demonstrates how to use automatic differentiation with the `MathCompile` library.
//! It showcases both forward-mode and reverse-mode AD, higher-order derivatives, and
//! practical applications like optimization.

#[cfg(feature = "autodiff")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ad_trait::forward_ad::adfn::adfn;
    use mathcompile::autodiff::{convenience, ForwardAD, HigherOrderAD};

    println!("🚀 MathCompile Automatic Differentiation Demo");
    println!("==========================================\n");

    // 1. Basic Forward-Mode AD
    println!("1️⃣  Forward-Mode Automatic Differentiation");
    println!("-------------------------------------------");

    let forward_ad = ForwardAD::new();

    // Example: f(x) = x³ - 2x² + 3x - 1
    let polynomial = |x: adfn<1>| {
        let x2 = x * x;
        let x3 = x2 * x;
        let two = adfn::new(2.0, [0.0]);
        let three = adfn::new(3.0, [0.0]);
        let one = adfn::new(1.0, [0.0]);

        x3 - two * x2 + three * x - one
    };

    let x_val = 2.0;
    let (value, derivative) = forward_ad.differentiate(polynomial, x_val)?;

    println!("f(x) = x³ - 2x² + 3x - 1");
    println!("f({x_val}) = {value:.6}");
    println!("f'({x_val}) = {derivative:.6}");
    println!("Expected f'(x) = 3x² - 4x + 3");
    println!(
        "Expected f'({}) = {:.6}\n",
        x_val,
        3.0 * x_val * x_val - 4.0 * x_val + 3.0
    );

    // 2. Reverse-Mode AD for Multi-Variable Functions
    println!("2️⃣  Reverse-Mode AD for Gradients");
    println!("----------------------------------");

    // Example: f(x,y) = x²y + xy² + x + y
    let multi_var_func = |vars: &[f64]| {
        let x = vars[0];
        let y = vars[1];
        let x2 = x * x;
        let y2 = y * y;
        x2 * y + x * y2 + x + y
    };

    let inputs = [1.0, 2.0];
    let gradient = convenience::gradient(multi_var_func, &inputs)?;

    println!("f(x,y) = x²y + xy² + x + y");
    println!("At point ({:.1}, {:.1}):", inputs[0], inputs[1]);
    println!("∂f/∂x = {:.6}", gradient[0]);
    println!("∂f/∂y = {:.6}", gradient[1]);

    // Manual verification
    let x = inputs[0];
    let y = inputs[1];
    let expected_dx = 2.0 * x * y + y * y + 1.0;
    let expected_dy = x * x + 2.0 * x * y + 1.0;
    println!("Expected ∂f/∂x = {expected_dx:.6}");
    println!("Expected ∂f/∂y = {expected_dy:.6}\n");

    // 3. Higher-Order Derivatives
    println!("3️⃣  Higher-Order Derivatives");
    println!("-----------------------------");

    let higher_order = HigherOrderAD::new();

    // Example: f(x) = x⁴
    let quartic_func = |x: adfn<1>| {
        let x_squared = x * x;
        x_squared * x_squared
    };

    let x_val = 2.0;
    let (value, first_deriv, second_deriv) = higher_order.second_derivative(quartic_func, x_val)?;

    println!("f(x) = x⁴");
    println!("f({x_val}) = {value:.6}");
    println!("f'({x_val}) = {first_deriv:.6}");
    println!("f''({x_val}) = {second_deriv:.6}");

    // Manual verification: f'(x) = 4x³, f''(x) = 12x²
    let expected_first = 4.0 * x_val * x_val * x_val;
    let expected_second = 12.0 * x_val * x_val;
    println!("Expected f'({x_val}) = {expected_first:.6}");
    println!("Expected f''({x_val}) = {expected_second:.6}\n");

    // 4. Optimization Example: Newton's Method
    println!("4️⃣  Optimization with Newton's Method");
    println!("--------------------------------------");

    // Find root of f(x) = x³ - 2x - 5 using Newton's method
    let objective = |x: adfn<1>| {
        let x2 = x * x;
        let x3 = x2 * x;
        let two = adfn::new(2.0, [0.0]);
        let five = adfn::new(5.0, [0.0]);
        x3 - two * x - five
    };

    let mut x = 2.0; // Initial guess
    println!("Finding root of f(x) = x³ - 2x - 5");
    println!("Initial guess: x₀ = {x:.6}");

    for i in 0..5 {
        let (f_val, f_prime) = forward_ad.differentiate(objective, x)?;
        let x_new = x - f_val / f_prime;

        println!(
            "Iteration {}: x = {:.6}, f(x) = {:.6}, f'(x) = {:.6}",
            i + 1,
            x,
            f_val,
            f_prime
        );

        if (x_new - x).abs() < 1e-10 {
            println!("Converged to root: x = {x_new:.10}");
            break;
        }
        x = x_new;
    }
    println!();

    // 5. Jacobian Matrix for Vector Functions
    println!("5️⃣  Jacobian Matrix Computation");
    println!("-------------------------------");

    // Vector function: f(x,y) = [x² + y², xy, x + y]
    let vector_function = |vars: &[f64]| {
        let x = vars[0];
        let y = vars[1];
        vec![x * x + y * y, x * y, x + y]
    };

    let point = [1.0, 2.0];
    let jacobian = convenience::jacobian(vector_function, &point, 3)?;

    println!("f(x,y) = [x² + y², xy, x + y]");
    println!("Jacobian at ({:.1}, {:.1}):", point[0], point[1]);
    for (i, row) in jacobian.iter().enumerate() {
        println!("Row {}: [{:.6}, {:.6}]", i, row[0], row[1]);
    }
    println!();

    // 6. Performance Demonstration
    println!("6️⃣  Performance Characteristics");
    println!("-------------------------------");

    use std::time::Instant;

    // Function with many variables
    let many_var_func = |vars: &[f64]| {
        let mut result = 0.0;
        for (i, &var) in vars.iter().enumerate() {
            let coeff = (i + 1) as f64;
            result += coeff * var * var;
        }
        result
    };

    let many_inputs: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.1).collect();

    let start = Instant::now();
    for _ in 0..100 {
        let _ = convenience::gradient(many_var_func, &many_inputs)?;
    }
    let reverse_time = start.elapsed();

    println!("Gradient computation for 20 variables (100 iterations): {reverse_time:?}");
    println!("Using finite differences for demonstration");
    println!();

    // 7. Complex Function Composition
    println!("7️⃣  Complex Function Composition");
    println!("--------------------------------");

    // f(x) = (x² + 1)³
    let complex_func = |x: adfn<1>| {
        let x_squared = x * x;
        let one = adfn::new(1.0, [0.0]);
        let inner = x_squared + one;
        inner * inner * inner
    };

    let x_val = 1.0;
    let (value, derivative) = forward_ad.differentiate(complex_func, x_val)?;

    println!("f(x) = (x² + 1)³");
    println!("f({x_val:.1}) = {value:.6}");
    println!("f'({x_val:.1}) = {derivative:.6}");

    // Manual verification: f'(x) = 3(x² + 1)² * 2x = 6x(x² + 1)²
    let expected_deriv = 6.0 * x_val * (x_val * x_val + 1.0).powi(2);
    println!("Expected f'({x_val:.1}) = {expected_deriv:.6}");
    println!("This demonstrates automatic chain rule application!");
    println!();

    println!("✅ Demo completed successfully!");
    println!("The automatic differentiation system provides both forward and");
    println!("reverse mode AD with support for higher-order derivatives.");
    println!("While this demo uses simplified implementations, it demonstrates");
    println!("the core functionality and integration with the MathCompile library.");

    Ok(())
}

#[cfg(not(feature = "autodiff"))]
fn main() {
    println!("❌ Autodiff feature is not enabled!");
    println!("Run with: cargo run --example autodiff_demo --features autodiff");
    println!("Or add 'autodiff' to the default features in Cargo.toml");
}
