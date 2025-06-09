//! Proper Data-Driven Demo: Using the Actual Library APIs
//!
//! This demonstrates using the ACTUAL DynamicContext APIs we've built:
//! - ctx.sum(data, |x| expression)
//! - ctx.eval_with_data_arrays()
//! - Reusable gaussian_logdensity function
//! - Our new Div and Ln operations

use dslcompile::ast::{DynamicContext, TypedBuilderExpr};
use frunk::hlist;

/// Reusable Gaussian log-density function using the actual library
fn gaussian_logdensity(
    x: TypedBuilderExpr<f64>,     // Observation variable (from iterator)
    mu: TypedBuilderExpr<f64>,    // Mean parameter
    sigma: TypedBuilderExpr<f64>, // Std dev parameter
) -> TypedBuilderExpr<f64> {
    // log p(x|μ,σ) = -½((x-μ)/σ)² - log(σ√2π)
    let diff = x - mu;
    let standardized = diff / sigma.clone();
    let standardized_squared = &standardized * &standardized;

    // Create constants
    let neg_half = TypedBuilderExpr::from(-0.5);
    let log_density_term = neg_half * standardized_squared;

    // Normalization: log(σ√2π) = log(σ) + ½log(2π)
    let log_sigma = sigma.ln();
    let log_2pi = TypedBuilderExpr::from((2.0 * std::f64::consts::PI).ln());
    let half = TypedBuilderExpr::from(0.5);
    let half_log_2pi = half * log_2pi;
    let normalization = log_sigma + half_log_2pi;

    log_density_term - normalization
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Proper Data-Driven Demo: Using Actual Library APIs");
    println!("====================================================\n");

    let mut ctx = DynamicContext::new();

    // Create parameter variables
    let mu = ctx.var(); // Mean parameter (Variable 0)
    let sigma = ctx.var(); // Std dev parameter (Variable 1)

    println!("🧩 Demo: Using ctx.sum(data, |x| expression)");
    println!("Built with actual DynamicContext APIs");

    // Test datasets
    let datasets = [
        ("Dataset A", 2.0, 1.0, vec![1.8, 2.1, 1.9]),
        ("Dataset B", 1.5, 0.8, vec![1.4, 1.6, 1.3, 1.7]),
        ("Dataset C", 3.0, 0.5, vec![2.8, 3.2, 2.9, 3.1, 2.7]),
    ];

    for (name, mu_val, sigma_val, data) in &datasets {
        println!("\n   📊 {name}:");
        println!("      μ={mu_val:.1}, σ={sigma_val:.1}");
        println!("      Data: {:?} ({} observations)", data, data.len());

        // 🚀 USE THE ACTUAL LIBRARY API!
        // The closure parameter x is the iterator variable
        // We pass the parameter variables into the gaussian_logdensity function
        let iid_likelihood = ctx.sum(data.clone(), |x| {
            gaussian_logdensity(x, mu.clone(), sigma.clone())
        });

        // Evaluate with parameters using the actual API
        let result = ctx.eval_with_data_arrays(&iid_likelihood, hlist![*mu_val, *sigma_val]);

        println!("      Log-likelihood: {result:.6}");
        println!("      ✅ Using actual DynamicContext APIs!");
    }

    println!("\n🎉 Proper Demo Complete!");
    println!("\n📊 Key Benefits:");
    println!("   ✅ Uses actual ctx.sum(data, |x| expression) API");
    println!("   ✅ Uses actual ctx.eval_with_data_arrays() API");
    println!("   ✅ Reusable gaussian_logdensity function");
    println!("   ✅ True compositionality with library infrastructure");
    println!("   ✅ Uses Div and Ln operations we just added");
    println!("   ✅ No reinventing - pure library usage");

    Ok(())
}
