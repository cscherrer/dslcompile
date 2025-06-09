//! Simple IID Demo: Basic Compositionality with Div/Ln
//!
//! This demonstrates the core concept:
//! 1. Single Gaussian component (uses Div and Ln)
//! 2. IID composition (manual summation)
//! 3. Runtime evaluation on different datasets

use dslcompile::ast::ASTRepr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Simple IID Probabilistic Programming Demo");
    println!("===========================================\n");

    // Demo 1: Single Gaussian Component
    println!("🧩 Demo 1: Single Gaussian Log-Density");
    println!("Building: log p(x|μ,σ) = -½((x-μ)/σ)² - log(σ√2π)");

    let x = ASTRepr::<f64>::Variable(0); // Observation
    let mu = ASTRepr::Variable(1); // Mean
    let sigma = ASTRepr::Variable(2); // Std dev

    // Build single Gaussian: -½((x-μ)/σ)² - log(σ√2π)
    let diff = ASTRepr::Sub(Box::new(x), Box::new(mu));
    let standardized = ASTRepr::Div(Box::new(diff), Box::new(sigma.clone()));
    let standardized_squared = ASTRepr::Mul(Box::new(standardized.clone()), Box::new(standardized));
    let log_density_term = ASTRepr::Mul(
        Box::new(ASTRepr::Constant(-0.5)),
        Box::new(standardized_squared),
    );

    // Normalization: log(σ√2π) = log(σ) + ½log(2π)
    let log_sigma = ASTRepr::Ln(Box::new(sigma));
    let log_2pi = ASTRepr::Constant((2.0 * std::f64::consts::PI).ln());
    let half_log_2pi = ASTRepr::Mul(Box::new(ASTRepr::Constant(0.5)), Box::new(log_2pi));
    let normalization = ASTRepr::Add(Box::new(log_sigma), Box::new(half_log_2pi));

    let single_gaussian = ASTRepr::Sub(Box::new(log_density_term), Box::new(normalization));

    // Test single component
    let test_values = [1.5, 1.0, 0.5]; // x=1.5, μ=1.0, σ=0.5
    let result = single_gaussian.eval_with_vars(&test_values);
    println!("✅ Single Gaussian (x=1.5, μ=1.0, σ=0.5): {result:.6}");

    println!();

    // Demo 2: IID Composition
    println!("🔗 Demo 2: IID Composition");
    println!("Composing: Σ(log p(xᵢ|μ,σ) for i=1,2,3)");

    // Build IID likelihood for 3 observations
    let mu_shared = ASTRepr::Variable(0); // Shared mean
    let sigma_shared = ASTRepr::Variable(1); // Shared std dev
    let observations = [
        ASTRepr::Variable(2), // x₁
        ASTRepr::Variable(3), // x₂
        ASTRepr::Variable(4), // x₃
    ];

    let mut likelihood_terms = Vec::new();

    for (i, x_i) in observations.iter().enumerate() {
        println!("   Building term {} for observation x_{}", i + 1, i + 1);

        // Apply single Gaussian component to this observation
        let diff = ASTRepr::Sub(Box::new(x_i.clone()), Box::new(mu_shared.clone()));
        let standardized = ASTRepr::Div(Box::new(diff), Box::new(sigma_shared.clone()));
        let standardized_squared =
            ASTRepr::Mul(Box::new(standardized.clone()), Box::new(standardized));
        let log_density_term = ASTRepr::Mul(
            Box::new(ASTRepr::Constant(-0.5)),
            Box::new(standardized_squared),
        );

        let log_sigma = ASTRepr::Ln(Box::new(sigma_shared.clone()));
        let log_2pi = ASTRepr::Constant((2.0 * std::f64::consts::PI).ln());
        let half_log_2pi = ASTRepr::Mul(Box::new(ASTRepr::Constant(0.5)), Box::new(log_2pi));
        let normalization = ASTRepr::Add(Box::new(log_sigma), Box::new(half_log_2pi));

        let single_term = ASTRepr::Sub(Box::new(log_density_term), Box::new(normalization));
        likelihood_terms.push(single_term);
    }

    // Sum all terms manually
    let mut iid_likelihood = likelihood_terms[0].clone();
    for term in likelihood_terms.into_iter().skip(1) {
        iid_likelihood = ASTRepr::Add(Box::new(iid_likelihood), Box::new(term));
    }

    println!("✅ IID likelihood composed!");

    println!();

    // Demo 3: Runtime Data Evaluation
    println!("💾 Demo 3: Runtime Data Evaluation");
    println!("Testing with different datasets:");

    let datasets = [
        ("Dataset A", [2.0, 1.0, 1.8, 2.1, 1.9]),
        ("Dataset B", [1.5, 0.8, 1.4, 1.6, 1.3]),
        ("Dataset C", [3.0, 0.5, 2.8, 3.2, 2.9]),
    ];

    for (name, params) in &datasets {
        println!("\n   📊 {name}:");
        println!("      μ={:.1}, σ={:.1}", params[0], params[1]);
        println!(
            "      Data: [{:.1}, {:.1}, {:.1}]",
            params[2], params[3], params[4]
        );

        let result = iid_likelihood.eval_with_vars(params);
        println!("      Log-likelihood: {result:.6}");
        println!("      ✅ Evaluation successful!");
    }

    println!("\n🎉 Simple IID Demo Complete!");
    println!("\n📊 Key Compositionality Benefits:");
    println!("   ✅ Reusable single Gaussian component with Div/Ln");
    println!("   ✅ Manual IID composition (sum of components)");
    println!("   ✅ Runtime data binding for different datasets");
    println!("   ✅ Expression built once, evaluated many times");

    Ok(())
}
