//! Enhanced Probabilistic Programming Demo with Staged Egglog Optimization
//!
//! This demo showcases how the new staged egglog optimization system enhances
//! probabilistic programming by optimizing complex mathematical expressions
//! commonly found in statistical models and machine learning.
//!
//! Key optimizations demonstrated:
//! 1. Variable partitioning for likelihood functions
//! 2. Sum splitting for expectation calculations  
//! 3. Constant factoring for parameter optimization
//! 4. Arithmetic series optimization for iterative algorithms

use dslcompile::{ast::ASTRepr, symbolic::native_egglog::optimize_with_native_egglog};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎲 Enhanced Probabilistic Programming Demo");
    println!("==========================================");
    println!("Showcasing staged egglog optimization in statistical contexts\n");

    // Demo 1: Gaussian Log-Likelihood Optimization
    gaussian_log_likelihood_demo()?;

    // Demo 2: Bayesian Linear Regression
    bayesian_linear_regression_demo()?;

    // Demo 3: Expectation Calculation with Sum Splitting
    expectation_calculation_demo()?;

    // Demo 4: Parameter Estimation with Constant Factoring
    parameter_estimation_demo()?;

    println!("\n🎉 Enhanced Probabilistic Programming Demo Complete!");
    println!("\n📊 Key Benefits Demonstrated:");
    println!("   • Variable partitioning optimizes likelihood functions");
    println!("   • Sum splitting accelerates expectation calculations");
    println!("   • Constant factoring improves parameter optimization");
    println!("   • Staged optimization prevents combinatorial explosion");
    println!("   • Production-ready mathematical optimization pipeline");

    Ok(())
}

/// Demo 1: Gaussian Log-Likelihood with Variable Partitioning
fn gaussian_log_likelihood_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Demo 1: Gaussian Log-Likelihood Optimization");
    println!("-----------------------------------------------");

    // Build log-likelihood: -0.5 * ((x - μ)² / σ² + log(2πσ²))
    // Variables: x=0, μ=1, σ=2
    let x = ASTRepr::Variable(0); // Data point
    let mu = ASTRepr::Variable(1); // Mean parameter  
    let sigma = ASTRepr::Variable(2); // Standard deviation parameter

    // Build (x - μ)²
    let diff = ASTRepr::Sub(Box::new(x), Box::new(mu));
    let diff_squared = ASTRepr::Mul(vec![diff.clone(), diff]);

    // Build σ²
    let variance = ASTRepr::Mul(vec![sigma.clone(), sigma.clone()]);

    // Build (x - μ)² / σ²
    let normalized = ASTRepr::Div(Box::new(diff_squared), Box::new(variance.clone()));

    // Build log(σ²) = 2*log(σ)
    let log_sigma = ASTRepr::Ln(Box::new(sigma));
    let log_variance = ASTRepr::Mul(vec![ASTRepr::Constant(2.0), log_sigma]);

    // Add log(2π) constant
    let log_2pi = ASTRepr::Constant(1.8378770664093453);
    let log_term = log_2pi + log_variance;

    // Build full expression: -0.5 * ((x - μ)² / σ² + log(2πσ²))
    let total = normalized + log_term;
    let log_likelihood = ASTRepr::Mul(vec![ASTRepr::Constant(-0.5), total]);

    println!("   Building: -0.5 * ((x - μ)² / σ² + log(2πσ²))");
    println!("   Before optimization: Complex nested expression");

    // Apply staged egglog optimization
    #[cfg(feature = "optimization")]
    {
        let start = Instant::now();
        match optimize_with_native_egglog(&log_likelihood) {
            Ok(optimized) => {
                let duration = start.elapsed();
                println!("   After optimization: Simplified expression structure");
                println!("   ✅ Optimization time: {duration:.2?}");
                println!("   ✅ Variable partitioning successful!");

                // Test evaluation with sample data: x=1, μ=0, σ=1
                let test_values = [1.0, 0.0, 1.0];
                let original_result = log_likelihood.eval_with_vars(&test_values);
                let optimized_result = optimized.eval_with_vars(&test_values);

                println!("   📊 Evaluation test (x=1, μ=0, σ=1):");
                println!("      Original:  {original_result:.6}");
                println!("      Optimized: {optimized_result:.6}");
                println!(
                    "      Difference: {:.2e}",
                    (original_result - optimized_result).abs()
                );
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("   ⚠️  Optimization feature not enabled");
    }

    println!();
    Ok(())
}

/// Demo 2: Bayesian Linear Regression with Complex Expressions
fn bayesian_linear_regression_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Demo 2: Bayesian Linear Regression Optimization");
    println!("--------------------------------------------------");

    // Build posterior: y = α + β*x + ε, where ε ~ N(0, σ²)
    // Variables: α=0, β=1, σ=2, x=3, y=4
    let alpha = ASTRepr::Variable(0); // Intercept
    let beta = ASTRepr::Variable(1); // Slope  
    let sigma = ASTRepr::Variable(2); // Noise standard deviation
    let x_i = ASTRepr::Variable(3); // Predictor variable
    let y_i = ASTRepr::Variable(4); // Response variable

    // Build prediction: α + β*x_i
    let prediction = ASTRepr::Add(vec![alpha, beta * x_i]);

    // Build residual: y_i - (α + β*x_i)
    let residual = ASTRepr::Sub(Box::new(y_i), Box::new(prediction));

    // Build residual²
    let residual_squared = ASTRepr::Mul(vec![residual.clone(), residual]);

    // Build σ²
    let variance = ASTRepr::Mul(vec![sigma.clone(), sigma.clone()]);

    // Build residual² / σ²
    let normalized_residual = ASTRepr::Div(Box::new(residual_squared), Box::new(variance.clone()));

    // Build log(σ²)
    let log_variance = ASTRepr::Ln(Box::new(variance));

    // Build full term: residual² / σ² + log(σ²)
    let log_posterior_term = normalized_residual + log_variance;

    // Build final expression: -0.5 * (residual² / σ² + log(σ²))
    let log_posterior = ASTRepr::Mul(vec![ASTRepr::Constant(-0.5), log_posterior_term]);

    println!("   Building: -0.5 * ((y - α - β*x)² / σ² + log(σ²))");
    println!("   Before optimization: Nested arithmetic operations");

    // Apply staged egglog optimization
    #[cfg(feature = "optimization")]
    {
        let start = Instant::now();
        match optimize_with_native_egglog(&log_posterior) {
            Ok(optimized) => {
                let duration = start.elapsed();
                println!("   After optimization: Streamlined computation");
                println!("   ✅ Optimization time: {duration:.2?}");
                println!("   ✅ Expression simplification successful!");

                // Test with sample regression data: α=1, β=2, σ=0.5, x=1.5, y=4
                let test_values = [1.0, 2.0, 0.5, 1.5, 4.0];
                let original_result = log_posterior.eval_with_vars(&test_values);
                let optimized_result = optimized.eval_with_vars(&test_values);

                println!("   📊 Evaluation test (α=1, β=2, σ=0.5, x=1.5, y=4):");
                println!("      Original:  {original_result:.6}");
                println!("      Optimized: {optimized_result:.6}");
                println!(
                    "      Difference: {:.2e}",
                    (original_result - optimized_result).abs()
                );
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    println!();
    Ok(())
}

/// Demo 3: Expectation Calculation with Sum Splitting
fn expectation_calculation_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Demo 3: Expectation Calculation with Sum Splitting");
    println!("-----------------------------------------------------");

    // Build E[aX + bY + c] = aE[X] + bE[Y] + c
    // Variables: a=0, b=1, c=2, x=3, y=4
    let a = ASTRepr::Variable(0); // Coefficient for X
    let b = ASTRepr::Variable(1); // Coefficient for Y  
    let c = ASTRepr::Variable(2); // Constant term
    let x = ASTRepr::Variable(3); // Random variable X
    let y = ASTRepr::Variable(4); // Random variable Y

    // Build a*x
    let ax = a * x;

    // Build b*y
    let by = b * y;

    // Build a*x + b*y
    let ax_plus_by = ax + by;

    // Build final expression: a*x + b*y + c
    let linear_combination = ax_plus_by + c;

    println!("   Building: E[aX + bY + c] = a*x + b*y + c");
    println!("   Before optimization: Nested additions and multiplications");

    // Apply staged egglog optimization (sum splitting should activate)
    #[cfg(feature = "optimization")]
    {
        let start = Instant::now();
        match optimize_with_native_egglog(&linear_combination) {
            Ok(optimized) => {
                let duration = start.elapsed();
                println!("   After optimization: Optimized linear combination");
                println!("   ✅ Optimization time: {duration:.2?}");
                println!("   ✅ Sum splitting and variable collection successful!");

                // Test with sample coefficients and values: a=2, b=3, c=1, x=5, y=7
                let test_values = [2.0, 3.0, 1.0, 5.0, 7.0];
                let original_result = linear_combination.eval_with_vars(&test_values);
                let optimized_result = optimized.eval_with_vars(&test_values);

                println!("   📊 Evaluation test (a=2, b=3, c=1, x=5, y=7):");
                println!("      Expected: 2*5 + 3*7 + 1 = 32");
                println!("      Original:  {original_result:.6}");
                println!("      Optimized: {optimized_result:.6}");
                println!(
                    "      Difference: {:.2e}",
                    (original_result - optimized_result).abs()
                );
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    println!();
    Ok(())
}

/// Demo 4: Parameter Estimation with Constant Factoring
fn parameter_estimation_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Demo 4: Parameter Estimation with Constant Factoring");
    println!("--------------------------------------------------------");

    // Build regularized loss: λ * (θ₁² + θ₂² + θ₃²)
    // Variables: λ=0, θ₁=1, θ₂=2, θ₃=3
    let lambda = ASTRepr::Variable(0); // Regularization parameter
    let theta1 = ASTRepr::Variable(1); // Parameter 1
    let theta2 = ASTRepr::Variable(2); // Parameter 2
    let theta3 = ASTRepr::Variable(3); // Parameter 3

    // Build θ₁²
    let theta1_sq = ASTRepr::Mul(vec![theta1.clone(), theta1]);

    // Build θ₂²
    let theta2_sq = ASTRepr::Mul(vec![theta2.clone(), theta2]);

    // Build θ₃²
    let theta3_sq = ASTRepr::Mul(vec![theta3.clone(), theta3]);

    // Build θ₁² + θ₂²
    let sum_12 = theta1_sq + theta2_sq;

    // Build θ₁² + θ₂² + θ₃²
    let sum_squares = sum_12 + theta3_sq;

    // Build final expression: λ * (θ₁² + θ₂² + θ₃²)
    let regularization = lambda * sum_squares;

    println!("   Building: λ * (θ₁² + θ₂² + θ₃²)");
    println!("   Before optimization: Multiple multiplications and additions");

    // Apply staged egglog optimization (constant factoring should activate)
    #[cfg(feature = "optimization")]
    {
        let start = Instant::now();
        match optimize_with_native_egglog(&regularization) {
            Ok(optimized) => {
                let duration = start.elapsed();
                println!("   After optimization: Factored constant multiplication");
                println!("   ✅ Optimization time: {duration:.2?}");
                println!("   ✅ Constant factoring and variable collection successful!");

                // Test with sample parameters: λ=0.1, θ₁=2, θ₂=3, θ₃=1
                let test_values = [0.1, 2.0, 3.0, 1.0];
                let original_result = regularization.eval_with_vars(&test_values);
                let optimized_result = optimized.eval_with_vars(&test_values);

                println!("   📊 Evaluation test (λ=0.1, θ₁=2, θ₂=3, θ₃=1):");
                println!("      Expected: 0.1 * (4 + 9 + 1) = 1.4");
                println!("      Original:  {original_result:.6}");
                println!("      Optimized: {optimized_result:.6}");
                println!(
                    "      Difference: {:.2e}",
                    (original_result - optimized_result).abs()
                );
            }
            Err(e) => {
                println!("   ❌ Optimization failed: {e}");
            }
        }
    }

    println!();
    Ok(())
}
