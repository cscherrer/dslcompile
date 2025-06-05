//! Macro-Based Expression System Showcase
//!
//! This example demonstrates the power and flexibility of the new macro-based
//! expression system, showing how it achieves zero-overhead abstraction while
//! supporting flexible arity and mixed types.

use dslcompile::compile_time::macro_expressions::{linear_combination, polynomial, PI};
use dslcompile::compile_time::{cos, exp, pow, sin, sqrt};
use dslcompile::{expr};

fn main() {
    println!("🚀 Macro-Based Expression System Showcase\n");

    // Basic arithmetic demonstrations
    basic_arithmetic_demo();

    // Mathematical functions
    mathematical_functions_demo();

    // Mixed types and array operations
    mixed_types_demo();

    // Neural network layer simulation
    neural_network_demo();

    // Financial calculations
    financial_calculations_demo();

    // Physics simulations
    physics_demo();


    // Builder pattern examples
    builder_pattern_demo();



    println!("\n🎯 All expressions compile to direct function calls!");
    println!("🎯 Zero runtime overhead - no type erasure!");
    println!("🎯 Flexible arity with natural syntax!");
}

fn basic_arithmetic_demo() {
    println!("📊 Basic Arithmetic Operations");
    println!("==============================");

    // Simple binary operations
    let add = expr!(|x: f64, y: f64| x + y);
    let multiply = expr!(|x: f64, y: f64| x * y);
    let divide = expr!(|x: f64, y: f64| x / y);

    println!("✅ Addition: 3 + 4 = {}", add(3.0, 4.0));
    println!("✅ Multiplication: 3 * 4 = {}", multiply(3.0, 4.0));
    println!("✅ Division: 8 / 2 = {}", divide(8.0, 2.0));

    // Ternary operations
    let sum3 = expr!(|x: f64, y: f64, z: f64| x + y + z);
    let product3 = expr!(|x: f64, y: f64, z: f64| x * y * z);

    println!("✅ Three-way sum: 1 + 2 + 3 = {}", sum3(1.0, 2.0, 3.0));
    println!(
        "✅ Three-way product: 2 * 3 * 4 = {}",
        product3(2.0, 3.0, 4.0)
    );

    // Complex arithmetic
    let complex = expr!(|a: f64, b: f64, c: f64, d: f64| (a + b) * (c - d));
    println!(
        "✅ Complex: (2 + 3) * (7 - 4) = {}",
        complex(2.0, 3.0, 7.0, 4.0)
    );

    println!();
}

fn mathematical_functions_demo() {
    println!("🧮 Mathematical Functions");
    println!("=========================");

    // Trigonometric functions
    let hypotenuse = expr!(|a: f64, b: f64| sqrt(a * a + b * b));
    println!("✅ Hypotenuse of 3,4 triangle: {:.2}", hypotenuse(3.0, 4.0));

    // Exponential and logarithmic
    let compound_interest =
        expr!(|principal: f64, rate: f64, time: f64| principal * exp(rate * time));
    println!(
        "✅ Compound interest ($1000, 5%, 2 years): ${:.2}",
        compound_interest(1000.0, 0.05, 2.0)
    );

    // Power functions
    let power_law = expr!(|x: f64, a: f64, b: f64| a * pow(x, b));
    println!(
        "✅ Power law (2 * x^3 at x=3): {}",
        power_law(3.0, 2.0, 3.0)
    );

    println!();
}

fn mixed_types_demo() {
    println!("🔀 Mixed Types and Array Operations");
    println!("===================================");

    // Array indexing with arithmetic
    let weights = [0.2, 0.5, 0.3, 0.8, 0.1];
    let weighted_sum = expr!(
        |arr: &[f64], idx1: usize, idx2: usize, factor: f64| (arr[idx1] + arr[idx2]) * factor
    );
    println!(
        "✅ Weighted sum: ({} + {}) * {} = {}",
        weights[1],
        weights[3],
        2.0,
        weighted_sum(&weights, 1, 3, 2.0)
    );

    // String length as numeric input (demonstrating type flexibility)
    let string_metric = expr!(|text_len: usize, multiplier: f64| text_len as f64 * multiplier);
    let text = "Hello, World!";
    println!(
        "✅ String metric: {} chars * 1.5 = {}",
        text.len(),
        string_metric(text.len(), 1.5)
    );

    // Boolean to numeric conversion
    let conditional_value = expr!(
        |condition: bool, true_val: f64, false_val: f64| if condition {
            true_val
        } else {
            false_val
        }
    );
    println!(
        "✅ Conditional: true ? 10.0 : 5.0 = {}",
        conditional_value(true, 10.0, 5.0)
    );

    println!();
}

fn neural_network_demo() {
    println!("🧠 Neural Network Layer Simulation");
    println!("==================================");

    // Single neuron with bias
    let neuron = expr!(
        |weights: &[f64], inputs: &[f64], bias: f64| weights[0] * inputs[0]
            + weights[1] * inputs[1]
            + bias
    );

    let weights = [0.5, 0.3];
    let inputs = [1.0, 2.0];
    let bias = 0.1;

    let output = neuron(&weights, &inputs, bias);
    println!("✅ Neuron output: 0.5*1.0 + 0.3*2.0 + 0.1 = {output}");

    // Activation function (ReLU)
    let relu = expr!(|x: f64| if x > 0.0 { x } else { 0.0 });
    let activated = relu(output);
    println!("✅ After ReLU activation: {activated}");

    // Sigmoid activation
    let sigmoid = expr!(|x: f64| 1.0 / (1.0 + exp(-x)));
    println!("✅ Sigmoid activation: {:.4}", sigmoid(output));

    println!();
}

fn financial_calculations_demo() {
    println!("💰 Financial Calculations");
    println!("=========================");

    // Present value calculation
    let present_value =
        expr!(|future_value: f64, rate: f64, periods: f64| future_value / pow(1.0 + rate, periods));
    println!(
        "✅ Present value of $1000 in 5 years at 6%: ${:.2}",
        present_value(1000.0, 0.06, 5.0)
    );

    // Black-Scholes option pricing (simplified)
    let black_scholes_call = expr!(
        |s: f64, k: f64, r: f64, t: f64, sigma: f64| s * 0.5 - k * exp(-r * t) * 0.5 // Simplified approximation
    );
    println!(
        "✅ Option value (simplified): ${:.2}",
        black_scholes_call(100.0, 95.0, 0.05, 0.25, 0.2)
    );

    // Portfolio return calculation
    let portfolio_return = expr!(
        |weights: &[f64], returns: &[f64], risk_free: f64| weights[0] * returns[0]
            + weights[1] * returns[1]
            + risk_free
    );
    let weights = [0.6, 0.4];
    let returns = [0.08, 0.12];
    println!(
        "✅ Portfolio return: {:.1}%",
        portfolio_return(&weights, &returns, 0.02) * 100.0
    );

    println!();
}

fn physics_demo() {
    println!("⚛️  Physics Simulations");
    println!("=======================");

    // Kinetic energy
    let kinetic_energy = expr!(|mass: f64, velocity: f64| 0.5 * mass * velocity * velocity);
    println!(
        "✅ Kinetic energy (10kg at 5m/s): {} J",
        kinetic_energy(10.0, 5.0)
    );

    // Gravitational force
    let gravity = expr!(|m1: f64, m2: f64, r: f64, g: f64| g * m1 * m2 / (r * r));
    let g_constant = 6.67430e-11;
    println!(
        "✅ Gravitational force: {:.2e} N",
        gravity(100.0, 200.0, 10.0, g_constant)
    );

    // Wave equation
    let wave = expr!(
        |amplitude: f64, frequency: f64, time: f64, phase: f64| amplitude
            * sin(2.0 * PI * frequency * time + phase)
    );
    println!("✅ Wave at t=0.5s: {:.3}", wave(2.0, 1.0, 0.5, 0.0));

    // Projectile motion (x-position)
    let projectile_x = expr!(|v0: f64, angle: f64, time: f64| v0 * cos(angle) * time);
    println!(
        "✅ Projectile x-position: {:.2} m",
        projectile_x(20.0, 0.785, 2.0)
    ); // 45 degrees in radians

    println!();
}


fn builder_pattern_demo() {
    println!("🏗️  Builder Pattern Examples");
    println!("============================");

    // Linear combination
    let linear_comb = linear_combination::<4>();
    let coefficients = [0.2, 0.3, 0.4, 0.1];
    let values = [10.0, 20.0, 30.0, 40.0];
    let result = linear_comb(&coefficients, &values);
    println!("✅ Linear combination: {result}");

    // Polynomial evaluation
    let poly = polynomial::<4>();
    let poly_coeffs = [1.0, 2.0, 1.0, 0.5]; // 1 + 2x + x² + 0.5x³
    let x = 2.0;
    let poly_result = poly(&poly_coeffs, x);
    println!("✅ Polynomial at x=2: {poly_result}");

    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_showcase_examples() {
        // Test a few key examples to ensure they work
        let add = expr!(|x: f64, y: f64| x + y);
        assert_eq!(add(3.0, 4.0), 7.0);

        let hypotenuse = expr!(|a: f64, b: f64| sqrt(a * a + b * b));
        assert_eq!(hypotenuse(3.0, 4.0), 5.0);

        let weights = [0.5, 0.3];
        let neuron = expr!(
            |weights: &[f64], inputs: &[f64], bias: f64| weights[0] * inputs[0]
                + weights[1] * inputs[1]
                + bias
        );
        let inputs = [1.0, 2.0];
        let result = neuron(&weights, &inputs, 0.1);
        assert!(
            (result - 1.2).abs() < 1e-10,
            "Expected ~1.2, got {}",
            result
        );
    }
}
