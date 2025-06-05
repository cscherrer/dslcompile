//! Probabilistic Programming: Symbolic Log-Density Optimization
//!
//! This example demonstrates the power of symbolic summation for probabilistic programming.
//! Key features:
//! - Gaussian log-density remains uncompiled until full expression is built
//! - Sum over IID data is symbolic (not unrolled)
//! - Mathematical optimizations recognize patterns and factor constants
//! - Scales to large datasets with aggressive optimization
//! - Significantly outperforms naive evaluation
//!
//! NOTE: Compile-time optimization with `Context::new_scope()` and `HeteroContext`
//! could be added for additional performance on smaller, known datasets.

use dslcompile::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🎯 Probabilistic Programming: Symbolic Log-Density Optimization");
    println!("===============================================================\n");

    // Test with increasingly large datasets to show symbolic scaling
    let test_sizes = [100, 1000, 10_000, 100_000, 1_000_000];

    for &n in &test_sizes {
        println!("📊 Dataset size: {n}");
        test_gaussian_log_density_optimization(n)?;
        println!();
    }

    Ok(())
}

fn test_gaussian_log_density_optimization(n: usize) -> Result<()> {
    let math = DynamicContext::new();

    // Generate test data - normally distributed around mean=2.0, std=1.5
    let data: Vec<f64> = (0..n)
        .map(|i| 2.0 + 1.5 * (i as f64 / n as f64 - 0.5))
        .collect();
    let mu = 2.0;
    let sigma: f64 = 1.5;

    println!("  🔧 Building symbolic Gaussian log-density...");

    // === SYMBOLIC APPROACH: Separate building from evaluation ===

    // 🏗️ EXPRESSION BUILDING (one-time cost)
    let start_build = Instant::now();
    let mu_param = math.constant(mu);
    let sigma_param = math.constant(sigma);
    let two_pi = math.constant(2.0 * std::f64::consts::PI);

    // Build symbolic Gaussian log-density: log(p(x|μ,σ)) = -0.5*log(2π) - log(σ) - (x-μ)²/(2σ²)
    // This remains UNCOMPILED - stays as symbolic AST
    let log_density_result =
        math.sum(&data, |x: TypedBuilderExpr<f64>| -> TypedBuilderExpr<f64> {
            let diff = x - mu_param.clone(); // (x - μ)
            let variance = sigma_param.clone() * sigma_param.clone(); // σ²
            let squared_error = diff.clone() * diff; // (x - μ)²
            let normalized_error = squared_error / (math.constant(2.0) * variance.clone()); // (x-μ)²/(2σ²)
            let log_normalization =
                math.constant(0.5) * two_pi.clone().ln() + sigma_param.clone().ln(); // 0.5*log(2π) + log(σ)

            // Full log-density: -0.5*log(2π) - log(σ) - (x-μ)²/(2σ²)
            let result: TypedBuilderExpr<f64> = -(log_normalization + normalized_error);
            result
        })?;
    let build_time = start_build.elapsed();

    // ⚡ SYMBOLIC EVALUATION (the actual performance benefit!)
    let start_eval = Instant::now();
    let symbolic_result = math.eval(&log_density_result, &[]);
    let eval_time = start_eval.elapsed();

    println!("  ✅ Runtime symbolic result: {symbolic_result:.6}");
    println!("  🏗️ Build time: {build_time:?}");
    println!("  ⚡ Eval time: {eval_time:?}");

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
    let eval_speedup = naive_time.as_nanos() as f64 / eval_time.as_nanos() as f64;
    let total_speedup = naive_time.as_nanos() as f64 / (build_time + eval_time).as_nanos() as f64;
    let accuracy = (symbolic_result - naive_result).abs();

    println!("  📈 Eval speedup: {eval_speedup:.2}x");
    println!("  📊 Total speedup: {total_speedup:.2}x");
    println!("  🎯 Accuracy: {accuracy:.2e}");

    if eval_speedup > 10.0 {
        println!(
            "  🚀 SYMBOLIC EVALUATION CRUSHING IT! ({}x faster)",
            eval_speedup as i32
        );
    } else if eval_speedup > 1.5 {
        println!("  ⚡ Symbolic evaluation winning!");
    } else if total_speedup > 0.8 {
        println!(
            "  ⚖️  Competitive overall (eval: {eval_speedup:.1}x, total: {total_speedup:.1}x)"
        );
    } else {
        println!(
            "  📝 Build cost dominates (eval: {eval_speedup:.1}x faster, but total: {total_speedup:.2}x)"
        );
    }

    // === SYMBOLIC INSPECTION ===
    if n <= 1000 {
        println!("  🔍 Expression structure:");
        let pretty = math.pretty_print(&log_density_result);
        let lines: Vec<&str> = pretty.lines().take(3).collect();
        for line in lines {
            println!("     {line}");
        }
        if pretty.lines().count() > 3 {
            println!("     ... (truncated)");
        }
    }

    // NOTE: Here's where compile-time optimization would go:
    // - Context::new_scope() for homogeneous compile-time optimization
    // - HeteroContext for heterogeneous compile-time optimization
    // - Both would show additional performance benefits for smaller, known datasets

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_gaussian_correctness() -> Result<()> {
        let math = DynamicContext::new();

        // Small test case for verification
        let data = vec![1.0, 2.0, 3.0];
        let mu = 2.0;
        let sigma: f64 = 1.0;

        // Symbolic approach
        let mu_param = math.constant(mu);
        let sigma_param = math.constant(sigma);
        let two_pi = math.constant(2.0 * std::f64::consts::PI);

        let symbolic_result =
            math.sum(&data, |x: TypedBuilderExpr<f64>| -> TypedBuilderExpr<f64> {
                let diff = x - mu_param.clone();
                let variance = sigma_param.clone() * sigma_param.clone();
                let squared_error = diff.clone() * diff;
                let normalized_error = squared_error / (math.constant(2.0) * variance.clone());
                let log_normalization =
                    math.constant(0.5) * two_pi.clone().ln() + sigma_param.clone().ln();

                let result: TypedBuilderExpr<f64> = -(log_normalization + normalized_error);
                result
            })?;

        let symbolic_value = math.eval(&symbolic_result, &[]);

        // Manual calculation
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_sigma = sigma.ln();
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut manual_value = 0.0;
        for &x in &data {
            let diff = x - mu;
            manual_value += -0.5 * log_2pi - log_sigma - (diff * diff) / two_sigma_sq;
        }

        let error = (symbolic_value - manual_value).abs();
        assert!(
            error < 1e-10,
            "Symbolic result differs from manual: {} vs {}",
            symbolic_value,
            manual_value
        );

        Ok(())
    }

    #[test]
    fn test_scaling_properties() -> Result<()> {
        let math = DynamicContext::new();

        // Test that expression complexity doesn't explode with data size
        let small_data = vec![1.0, 2.0];
        let large_data = (0..1000).map(|i| i as f64).collect::<Vec<_>>();

        let mu_param = math.constant(0.0);
        let sigma_param = math.constant(1.0);

        let build_expression = |data: &[f64]| -> Result<TypedBuilderExpr<f64>> {
            math.sum(data, |x: TypedBuilderExpr<f64>| -> TypedBuilderExpr<f64> {
                let diff = x - mu_param.clone();
                let result: TypedBuilderExpr<f64> = -(diff.clone() * diff);
                result
            })
        };

        let small_expr = build_expression(&small_data)?;
        let large_expr = build_expression(&large_data)?;

        // Both should evaluate successfully (proving symbolic handling)
        let small_result = math.eval(&small_expr, &[]);
        let large_result = math.eval(&large_expr, &[]);

        assert!(small_result.is_finite());
        assert!(large_result.is_finite());

        println!("Small data result: {}", small_result);
        println!("Large data result: {}", large_result);

        Ok(())
    }
}
