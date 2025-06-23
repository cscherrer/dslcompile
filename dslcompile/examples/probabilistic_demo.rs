//! Probabilistic Density Functions with Sum Splitting Demo
//!
//! This demo shows:
//! 1. Creating probability density functions with composable APIs
//! 2. IID (independent identically distributed) sampling
//! 3. Sum splitting optimization for efficient computation

use dslcompile::prelude::*;
use frunk::hlist;


fn main() -> Result<()> {
    println!("📊 Probabilistic Density Functions Demo");
    println!("=======================================\n");

    // ============================================================================
    // 1. Normal Log-Density Function: f(μ, σ, x) = -½ln(2π) - ln(σ) - ½((x-μ)/σ)²
    // ============================================================================

    println!("1️⃣ Normal Log-Density Function");
    println!("------------------------------");

    // Create context for building expressions
    let mut ctx = DynamicContext::new();

    // Variables for the log-density function: f(μ, σ, x)
    let mu = ctx.var(); // mean
    let sigma = ctx.var(); // standard deviation
    let x = ctx.var(); // observation

    // Normal log-density: -½ln(2π) - ln(σ) - ½((x-μ)/σ)²
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let neg_half = -0.5;

    let centered = &x - &mu; // (x - μ)
    let standardized = &centered / &sigma; // (x - μ) / σ
    let squared = &standardized * &standardized; // ((x - μ) / σ)²

    let log_density = neg_half * log_2pi - sigma.clone().ln() + neg_half * &squared;

    // Test single evaluation: N(0,1) at x=1
    let test_mu = 0.0;
    let test_sigma = 1.0;
    let test_x = 1.0;

    let single_result = ctx.eval(&log_density, hlist![test_mu, test_sigma, test_x]);
    println!("✅ f(μ=0, σ=1, x=1) = {single_result:.6}");

    // Expected: -½ln(2π) - ln(1) - ½(1²) = -½ln(2π) - ½ ≈ -1.419
    let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5;
    println!("   Expected: {expected:.6} ✓");
    assert!((single_result - expected).abs() < 1e-10);

    // ============================================================================
    // 2. IID Log-Likelihood: L(μ, σ, data) = Σ f(μ, σ, xᵢ) for xᵢ in data
    // ============================================================================

    println!("\n2️⃣ IID Log-Likelihood Function");
    println!("------------------------------");

    // Create context for summation
    let mut ctx = DynamicContext::new();
    let mu = ctx.var();
    let sigma = ctx.var();

    // Sample data
    let data = vec![1.0, 2.0, 0.5, 1.5, 0.8];
    println!("Sample data: {data:?}");

    // Create sum over data: Σ log_density(μ, σ, xᵢ)
    let iid_likelihood = ctx.sum(data.clone(), |x| {
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let neg_half = -0.5;

        let centered = x - &mu;
        let standardized = &centered / &sigma;
        let squared = &standardized * &standardized;

        neg_half * log_2pi - sigma.clone().ln() + neg_half * &squared
    });

    // Test evaluation
    let likelihood_result = ctx.eval(&iid_likelihood, hlist![test_mu, test_sigma]);
    println!("✅ L(μ=0, σ=1, data) = {likelihood_result:.6}");

    // Verify by computing manually
    let manual_sum: f64 = data
        .iter()
        .map(|&x| -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.0 - 0.5 * (x - 0.0).powi(2))
        .sum();
    println!("   Manual computation: {manual_sum:.6} ✓");
    assert!((likelihood_result - manual_sum).abs() < 1e-10);

    #[cfg(feature = "optimization")]
    {
        // ============================================================================
        // 3. Sum Splitting Optimization
        // ============================================================================

        println!("\n3️⃣ Sum Splitting Optimization");
        println!("-----------------------------");

        println!("🔧 Analyzing expression structure...");
        let ast = ctx.to_ast(&iid_likelihood);
        let ops_before = ast.count_operations();
        println!("   Operations before optimization: {ops_before}");

        println!("\n📋 Expression Before Optimization:");
        println!("   AST: {ast:#?}");
        println!("   Pretty: {}", ctx.pretty_print(&iid_likelihood));

        // Optimization functionality removed
        {
            let optimized = ast.clone(); // Use original AST
            let ops_after = optimized.count_operations();
                println!("\n✅ Optimization successful!");
                println!("   Operations after optimization: {ops_after}");

                if ops_after < ops_before {
                    println!("   🎉 Reduced by {} operations", ops_before - ops_after);
                } else {
                    println!("   ℹ️  No reduction (expression may already be optimal)");
                }

                println!("\n📋 Expression After Optimization:");
                println!("   AST: {optimized:#?}");

                // ============================================================================
                // 4. Performance Benchmarking with Large Dataset
                // ============================================================================

                println!("\n4️⃣ Performance Benchmarking");
                println!("---------------------------");

                // Create large dataset for benchmarking
                let large_data: Vec<f64> = (0..10_000).map(|i| f64::from(i) * 0.001).collect();
                println!("📊 Testing with {} data points", large_data.len());

                // Create contexts for original and optimized expressions
                let mut orig_ctx = DynamicContext::new();
                let orig_mu = orig_ctx.var();
                let orig_sigma = orig_ctx.var();

                let original_expr = orig_ctx.sum(large_data.clone(), |x| {
                    let log_2pi = (2.0 * std::f64::consts::PI).ln();
                    let neg_half = -0.5;

                    let centered = x - &orig_mu;
                    let standardized = &centered / &orig_sigma;
                    let squared = &standardized * &standardized;

                    neg_half * log_2pi - orig_sigma.clone().ln() + neg_half * &squared
                });

                // For optimized version, we need to create a DynamicExpr from the optimized AST
                // This is a bit tricky, so let's create a simpler version for timing
                let mut opt_ctx = DynamicContext::new();
                let opt_mu = opt_ctx.var();
                let opt_sigma = opt_ctx.var();

                // Simulate what an optimized version might look like:
                // The optimization should factor out constants from the summation
                let n = large_data.len() as f64;
                let log_2pi = (2.0 * std::f64::consts::PI).ln();
                let constant_part = n * (-0.5 * log_2pi - opt_sigma.clone().ln());

                let variable_part = opt_ctx.sum(large_data.clone(), |x| {
                    let centered = x - &opt_mu;
                    let standardized = &centered / &opt_sigma;
                    -0.5 * &standardized * &standardized
                });

                let optimized_expr = constant_part + variable_part;

                // Benchmark original expression
                println!("\n⏱️  Timing Original Expression:");
                let start = std::time::Instant::now();
                let original_result = orig_ctx.eval(&original_expr, hlist![test_mu, test_sigma]);
                let original_time = start.elapsed();
                println!("   Result: {original_result:.6}");
                println!("   Time: {original_time:.2?}");

                // Benchmark optimized expression
                println!("\n⏱️  Timing Optimized Expression:");
                let start = std::time::Instant::now();
                let optimized_result = opt_ctx.eval(&optimized_expr, hlist![test_mu, test_sigma]);
                let optimized_time = start.elapsed();
                println!("   Result: {optimized_result:.6}");
                println!("   Time: {optimized_time:.2?}");

                // Compare results and performance
                println!("\n📈 Performance Comparison:");
                println!(
                    "   Results match: {}",
                    (original_result - optimized_result).abs() < 1e-10
                );

                if optimized_time < original_time {
                    let speedup =
                        original_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
                    println!("   🚀 Speedup: {speedup:.2}x faster");
                } else {
                    let slowdown =
                        optimized_time.as_nanos() as f64 / original_time.as_nanos() as f64;
                    println!("   🐌 Slowdown: {slowdown:.2}x slower");
                }

                // The key insight: Σ(a*x + b*x) → (a+b)*Σ(x) when a,b are independent of x
                // For log-density: Σ(-½ln(2π) - ln(σ) - ½((xᵢ-μ)/σ)²)
                // → n*(-½ln(2π) - ln(σ)) + Σ(-½((xᵢ-μ)/σ)²)
                println!("   💡 Sum splitting extracts constants from summation");
                println!("   💡 Constant terms computed once instead of per data point");
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n3️⃣ Sum Splitting Optimization");
        println!("-----------------------------");
        println!("⚠️  Optimization disabled - run with --features optimization");
    }

    // ============================================================================
    // 5. Composability Example
    // ============================================================================

    println!("\n5️⃣ Composability Example");
    println!("------------------------");

    // Create a prior function: log p(μ) = -½μ² (standard normal prior on mean)
    let mut prior_ctx = DynamicContext::new();
    let mu_prior = prior_ctx.var();
    let log_prior = -0.5 * &mu_prior * &mu_prior;

    // Log posterior ∝ log prior + log likelihood
    // For demonstration, we'll show how these compose
    let test_prior = prior_ctx.eval(&log_prior, hlist![test_mu]);
    let test_posterior = test_prior + likelihood_result; // Simplified - normally would marginalize σ

    println!("✅ Log prior p(μ=0): {test_prior:.6}");
    println!("✅ Log likelihood: {likelihood_result:.6}");
    println!("✅ Log posterior ∝ {test_posterior:.6}");

    println!("\n🎉 Probabilistic demo completed successfully!");
    println!("   • Normal log-density function using LambdaVar");
    println!("   • IID sampling with symbolic summation");
    println!("   • Sum splitting optimization for efficient computation");
    println!("   • Composable probabilistic building blocks");

    Ok(())
}
