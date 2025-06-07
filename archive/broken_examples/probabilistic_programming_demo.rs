//! Probabilistic Programming: Post-Cranelift Cleanup Demo
//!
//! This example demonstrates probabilistic programming using the Static Context System.
//! Key features:
//! - Gaussian log-density using Context (compile-time optimization)
//! - Static expression building with runtime data evaluation
//! - Zero-overhead mathematical expressions through ASTRepr
//! - Scales to large datasets with compile-time optimized expressions
//! - Clean codebase focused on core compile-time optimization goals
//!
//! This showcases the Static Context System after the Cranelift cleanup.

use dslcompile::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🎯 Probabilistic Programming: Static Context System Demo");
    println!("======================================================\n");

    // Test with increasingly large datasets to show scaling
    let test_sizes = [100, 1000, 10_000, 100_000];

    for &n in &test_sizes {
        println!("📊 Dataset size: {n}");
        test_gaussian_log_density_optimization(n)?;
        println!();
    }

    Ok(())
}

fn test_gaussian_log_density_optimization(n: usize) -> Result<()> {
    // Generate test data - normally distributed around mean=2.0, std=1.5
    let data: Vec<f64> = (0..n)
        .map(|i| 2.0 + 1.5 * (i as f64 / n as f64 - 0.5))
        .collect();
    let mu = 2.0;
    let sigma: f64 = 1.5;

    println!("  🔧 Building static Gaussian log-density expression...");

    // === STATIC CONTEXT SYSTEM: Compile-time optimization with runtime inputs ===
    let mut ctx = Context::new_f64();

    // 🏗️ STATIC EXPRESSION BUILDING - Build compile-time optimized expression
    let start_build = Instant::now();
    
    // Create static Gaussian log-density expression: log(p(x|μ,σ)) = -0.5*log(2π) - log(σ) - (x-μ)²/(2σ²)
    let log_density_expr = ctx.new_scope(|scope| {
        let (x, scope) = scope.auto_var();      // Input data point
        let (mu_param, scope) = scope.auto_var(); // Mean parameter
        let (sigma_param, scope) = scope.auto_var(); // Std deviation parameter
        
        // Build the mathematical expression with zero-overhead operations
        let two_pi = scope.clone().constant(2.0 * std::f64::consts::PI);
        let two = scope.clone().constant(2.0);
        let half = scope.constant(0.5);
        
        // Components of Gaussian log-density
        let diff = x.sub(mu_param); // (x - μ)
        let variance = sigma_param.clone().mul(sigma_param.clone()); // σ²
        let squared_error = diff.clone().mul(diff); // (x - μ)²
        let normalized_error = squared_error.div(two.mul(variance)); // (x-μ)²/(2σ²)
        let log_normalization = half.mul(two_pi.ln()).add(sigma_param.ln()); // 0.5*log(2π) + log(σ)

        // Full log-density: -0.5*log(2π) - log(σ) - (x-μ)²/(2σ²)
        // Note: We negate to get the negative log-likelihood
        log_normalization.add(normalized_error).neg()
    });
    
    let build_time = start_build.elapsed();

    // ⚡ STATIC EVALUATION WITH RUNTIME DATA
    let start_static = Instant::now();
    
    // Evaluate the static expression for each data point using array inputs
    let mut static_result = 0.0;
    for &x_val in &data {
        // Zero-overhead evaluation with array inputs: [x, μ, σ]
        let vars = ScopedVarArray::new(vec![x_val, mu, sigma]);
        let log_density = log_density_expr.eval(&vars);
        static_result += log_density;
    }
    
    let static_time = start_static.elapsed();

    println!("  ✅ Static result: {static_result:.6}");
    println!("  🏗️ Build time: {build_time:?}");
    println!("  ⚡ Static time: {static_time:?}");

    // === NAIVE APPROACH: Direct computation ===
    let start_naive = Instant::now();

    let mut naive_result = 0.0;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let log_sigma = sigma.ln();
    let two_sigma_sq = 2.0 * sigma * sigma;

    for &x in &data {
        let diff = x - mu;
        let log_density = -0.5 * log_2pi - log_sigma - (diff * diff) / two_sigma_sq;
        naive_result += log_density;
    }

    let naive_time = start_naive.elapsed();

    println!("  ✅ Naive result: {naive_result:.6}");
    println!("  🐌 Naive time: {naive_time:?}");

    // === PERFORMANCE ANALYSIS ===
    let static_vs_naive = naive_time.as_nanos() as f64 / static_time.as_nanos() as f64;
    let total_static_speedup = naive_time.as_nanos() as f64 / (build_time + static_time).as_nanos() as f64;
    let accuracy_static = (static_result - naive_result).abs();

    println!("  📈 Performance Analysis:");
    println!("    🚀 Static vs Naive: {static_vs_naive:.2}x faster");
    println!("    📊 Total static speedup (including build): {total_static_speedup:.2}x");
    println!("  🎯 Accuracy:");
    println!("    Static error: {accuracy_static:.2e}");

    // === PERFORMANCE CLASSIFICATION ===
    if static_vs_naive > 10.0 {
        println!(
            "  🚀 STATIC CONTEXT SYSTEM CRUSHING IT! ({}x faster than naive)",
            static_vs_naive as i32
        );
    } else if static_vs_naive > 2.0 {
        println!("  ⚡ Static Context System winning big!");
    } else if static_vs_naive > 1.2 {
        println!("  ✅ Static Context System performing well");
    } else {
        println!("  📝 Build cost may be dominating for this size");
    }

    // === STATIC CONTEXT SYSTEM INSIGHTS ===
    println!("  🔧 Static Context System Insights:");
    println!("    - Zero-overhead compile-time expression building through ASTRepr");
    println!("    - Type-safe variable scoping prevents composition errors");
    println!("    - Complete mathematical operations (sub, div, ln, neg)");
    println!("    - Native Rust performance with mathematical optimization");
    println!("    - Static expressions + runtime data = best of both worlds");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_correctness() -> Result<()> {
        let mut ctx = Context::new_f64();

        // Small test case for verification
        let data = vec![1.0, 2.0, 3.0];
        let mu = 2.0;
        let sigma: f64 = 1.0;

        // Build static expression
        let log_density_expr = ctx.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (mu_param, scope) = scope.auto_var();
            let (sigma_param, scope) = scope.auto_var();
            
            let two_pi = scope.clone().constant(2.0 * std::f64::consts::PI);
            let two = scope.clone().constant(2.0);
            let half = scope.constant(0.5);
            
            let diff = x.sub(mu_param);
            let variance = sigma_param.clone().mul(sigma_param.clone());
            let squared_error = diff.clone().mul(diff);
            let normalized_error = squared_error.div(two.mul(variance));
            let log_normalization = half.mul(two_pi.ln()).add(sigma_param.ln());

            log_normalization.add(normalized_error).neg()
        });

        // Evaluate with static system
        let mut static_result = 0.0;
        for &x_val in &data {
            let vars = ScopedVarArray::new(vec![x_val, mu, sigma]);
            let log_density = log_density_expr.eval(&vars);
            static_result += log_density;
        }

        // Manual calculation for verification
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_sigma = sigma.ln();
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut manual_result = 0.0;
        for &x in &data {
            let diff = x - mu;
            manual_result += -0.5 * log_2pi - log_sigma - (diff * diff) / two_sigma_sq;
        }

        let error = (static_result - manual_result).abs();

        assert!(
            error < 1e-10,
            "Static result differs from manual: {} vs {}",
            static_result,
            manual_result
        );

        println!("✅ Static Context System produces correct results");

        Ok(())
    }

    #[test]
    fn test_static_context_performance() -> Result<()> {
        let mut ctx = Context::new_f64();

        // Test that the Static Context System performs well
        let data = (0..1000).map(|i| i as f64).collect::<Vec<_>>();
        
        // Build static expression: f(x, k) = x * k + 1
        let expr = ctx.new_scope(|scope| {
            let (x, scope) = scope.auto_var();
            let (k, scope) = scope.auto_var();
            let one = scope.constant(1.0);
            
            x.mul(k).add(one)
        });

        let start = std::time::Instant::now();
        let mut result = 0.0;
        for &x_val in &data {
            let vars = ScopedVarArray::new(vec![x_val, 2.0]);
            result += expr.eval(&vars);
        }
        let duration = start.elapsed();

        // Should complete quickly
        assert!(duration.as_millis() < 100, "Evaluation took too long: {:?}", duration);
        
        // Verify correctness: Σ(2i + 1) for i=0..999 = 2*Σi + 1000 = 2*499500 + 1000 = 1000000
        let expected = 2.0 * (999.0 * 1000.0 / 2.0) + 1000.0;
        assert!((result - expected).abs() < 1e-10);

        println!("✅ Static Context System performs well: {:?}", duration);

        Ok(())
    }
}
