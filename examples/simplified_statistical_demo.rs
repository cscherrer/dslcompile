//! Simplified Statistical Computing Demo
//!
//! This example demonstrates how to do statistical computing using the general
//! mathematical expression system instead of specialized statistical methods.
//!
//! Key insight: Statistical functions are just mathematical expressions with more variables.
//! Instead of f(params, data), use f(param1, param2, ..., data1, data2, ...).

use mathcompile::prelude::*;

fn main() -> Result<()> {
    println!("=== Simplified Statistical Computing Demo ===\n");

    // Example: Bayesian Linear Regression
    // Instead of: f(params=[β₀, β₁], data=[x₁, y₁, x₂, y₂, ...])
    // Use:        f(β₀, β₁, x₁, y₁, x₂, y₂, ...)

    // Sample data points
    let data = vec![(1.0, 2.1), (2.0, 3.9), (3.0, 6.1), (4.0, 8.0)];

    println!("Data points: {data:?}");
    println!("Model: y = β₀ + β₁*x");
    println!("Goal: Minimize sum of squared residuals\n");

    // Create expression for sum of squared residuals using ASTRepr directly
    // Variables: β₀=var(0), β₁=var(1), x₁=var(2), y₁=var(3), x₂=var(4), y₂=var(5), ...

    let beta0 = ASTRepr::<f64>::Variable(0); // β₀ (intercept)
    let beta1 = ASTRepr::<f64>::Variable(1); // β₁ (slope)

    let mut sum_expr = ASTRepr::<f64>::Constant(0.0);

    // Add each data point to the sum: (yᵢ - β₀ - β₁*xᵢ)²
    for i in 0..data.len() {
        let x_var = ASTRepr::<f64>::Variable(2 + 2 * i); // x_i
        let y_var = ASTRepr::<f64>::Variable(2 + 2 * i + 1); // y_i

        // Prediction: β₀ + β₁*xᵢ
        let beta1_x = ASTRepr::Mul(Box::new(beta1.clone()), Box::new(x_var));
        let prediction = ASTRepr::Add(Box::new(beta0.clone()), Box::new(beta1_x));

        // Residual: yᵢ - prediction
        let residual = ASTRepr::Sub(Box::new(y_var), Box::new(prediction));

        // Squared residual: (yᵢ - β₀ - β₁*xᵢ)²
        let squared_residual = ASTRepr::Mul(Box::new(residual.clone()), Box::new(residual));

        // Add to sum
        sum_expr = ASTRepr::Add(Box::new(sum_expr), Box::new(squared_residual));
    }

    println!("Expression created with {} variables", 2 + 2 * data.len());

    // Compile the expression
    let compiler = RustCompiler::new();
    let generator = RustCodeGenerator::new();

    let rust_code = generator.generate_function(&sum_expr, "sum_squared_residuals")?;
    let compiled_fn = compiler.compile_and_load(&rust_code, "sum_squared_residuals")?;

    println!("Expression compiled successfully!\n");

    // Evaluate with different parameter values
    let test_cases = vec![
        (0.0, 1.0), // β₀=0, β₁=1
        (0.5, 1.5), // β₀=0.5, β₁=1.5
        (1.0, 2.0), // β₀=1, β₁=2
    ];

    for (beta0_val, beta1_val) in test_cases {
        // Create input vector: [β₀, β₁, x₁, y₁, x₂, y₂, ...]
        let mut vars = vec![beta0_val, beta1_val];
        for &(x, y) in &data {
            vars.push(x);
            vars.push(y);
        }

        // Use the general system - this actually works!
        let result = compiled_fn.call_multi_vars(&vars)?;

        println!("β₀={beta0_val}, β₁={beta1_val} → Sum of squared residuals = {result:.4}");
    }

    println!("\n=== Key Insights ===");
    println!("1. Statistical computing works perfectly with the general system");
    println!("2. No need for specialized 'call_with_data' methods");
    println!("3. Just use call_multi_vars() with all variables flattened");
    println!("4. Mathematically equivalent but architecturally simpler");

    Ok(())
}
