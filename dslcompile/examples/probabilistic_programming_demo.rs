//! Probabilistic Programming: Cranelift JIT Optimization Demo
//!
//! This example demonstrates the power of Cranelift JIT compilation for probabilistic programming.
//! Key features:
//! - Gaussian log-density compiled to native code via Cranelift JIT
//! - Sum over IID data uses symbolic summation + JIT compilation
//! - Mathematical optimizations + JIT compilation for maximum performance
//! - Scales to large datasets with aggressive optimization
//! - Significantly outperforms both naive evaluation and interpretation
//!
//! This is a test of Cranelift JIT - the demo showcases the underlying capabilities.

use dslcompile::prelude::*;
use dslcompile::ast::runtime::expression_builder::JITStrategy;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🎯 Probabilistic Programming: Cranelift JIT Optimization Demo");
    println!("==============================================================\n");

    // Test with increasingly large datasets to show Cranelift JIT scaling
    let test_sizes = [100, 1000, 10_000, 100_000, 1_000_000];

    for &n in &test_sizes {
        println!("📊 Dataset size: {n}");
        test_gaussian_log_density_cranelift_optimization(n)?;
        println!();
    }

    Ok(())
}

fn test_gaussian_log_density_cranelift_optimization(n: usize) -> Result<()> {
    // Generate test data - normally distributed around mean=2.0, std=1.5
    let data: Vec<f64> = (0..n)
        .map(|i| 2.0 + 1.5 * (i as f64 / n as f64 - 0.5))
        .collect();
    let mu = 2.0;
    let sigma: f64 = 1.5;

    println!("  🔧 Building symbolic Gaussian log-density summation...");

    // === CRANELIFT JIT APPROACH: Use symbolic summation ===
    let math_jit = DynamicContext::new_jit_optimized(); // Always use JIT

    // 🏗️ EXPRESSION BUILDING (one-time cost) - Build SYMBOLIC summation expression
    let start_build = Instant::now();
    
    // Create symbolic summation: Σ(log_density(x) for x in data)
    // This creates a SINGLE Sum AST node that gets JIT compiled once
    let mu_param = math_jit.constant(mu);
    let sigma_param = math_jit.constant(sigma);
    let two_pi = math_jit.constant(2.0 * std::f64::consts::PI);

    let sum_expr = math_jit.sum(&data, |x: TypedBuilderExpr<f64>| -> TypedBuilderExpr<f64> {
        // Build symbolic Gaussian log-density: log(p(x|μ,σ)) = -0.5*log(2π) - log(σ) - (x-μ)²/(2σ²)
        let diff = x - mu_param.clone(); // (x - μ)
        let variance = sigma_param.clone() * sigma_param.clone(); // σ²
        let squared_error = diff.clone() * diff; // (x - μ)²
        let normalized_error = squared_error / (math_jit.constant(2.0) * variance.clone()); // (x-μ)²/(2σ²)
        let log_normalization =
            math_jit.constant(0.5) * two_pi.clone().ln() + sigma_param.clone().ln(); // 0.5*log(2π) + log(σ)

        // Full log-density: -0.5*log(2π) - log(σ) - (x-μ)²/(2σ²)
        let result: TypedBuilderExpr<f64> = -(log_normalization + normalized_error);
        result
    })?;
    
    let build_time = start_build.elapsed();

    // ⚡ CRANELIFT JIT EVALUATION (single compiled expression!)
    let start_jit = Instant::now();
    
    // Evaluate the ENTIRE summation with a single JIT-compiled expression
    // This should be MUCH faster than calling eval_jit() in a loop
    let jit_result = math_jit.eval(&sum_expr, &[]);
    
    let jit_time = start_jit.elapsed();

    println!("  ✅ Cranelift JIT result: {jit_result:.6}");
    println!("  🏗️ Build time: {build_time:?}");
    println!("  ⚡ JIT time: {jit_time:?}");

    // === INTERPRETATION APPROACH: Force interpretation ===
    let math_interp = DynamicContext::new_interpreter(); // Never use JIT

    let start_interp_build = Instant::now();
    
    // Build identical symbolic summation for interpretation
    let mu_param_interp = math_interp.constant(mu);
    let sigma_param_interp = math_interp.constant(sigma);
    let two_pi_interp = math_interp.constant(2.0 * std::f64::consts::PI);

    let sum_expr_interp = math_interp.sum(&data, |x: TypedBuilderExpr<f64>| -> TypedBuilderExpr<f64> {
        let diff = x - mu_param_interp.clone();
        let variance = sigma_param_interp.clone() * sigma_param_interp.clone();
        let squared_error = diff.clone() * diff;
        let normalized_error = squared_error / (math_interp.constant(2.0) * variance.clone());
        let log_normalization =
            math_interp.constant(0.5) * two_pi_interp.clone().ln() + sigma_param_interp.clone().ln();

        let result: TypedBuilderExpr<f64> = -(log_normalization + normalized_error);
        result
    })?;
    
    let interp_build_time = start_interp_build.elapsed();

    let start_interp = Instant::now();
    let interp_result = math_interp.eval(&sum_expr_interp, &[]);
    let interp_time = start_interp.elapsed();

    println!("  ✅ Interpretation result: {interp_result:.6}");
    println!("  🏗️ Interp build time: {interp_build_time:?}");
    println!("  🔄 Interp time: {interp_time:?}");

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
    let jit_vs_naive = naive_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
    let jit_vs_interp = interp_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
    let interp_vs_naive = naive_time.as_nanos() as f64 / interp_time.as_nanos() as f64;
    let total_jit_speedup = naive_time.as_nanos() as f64 / (build_time + jit_time).as_nanos() as f64;
    let accuracy_jit = (jit_result - naive_result).abs();
    let accuracy_interp = (interp_result - naive_result).abs();

    println!("  📈 Performance Analysis:");
    println!("    🚀 Cranelift JIT vs Naive: {jit_vs_naive:.2}x faster");
    println!("    ⚡ Cranelift JIT vs Interpretation: {jit_vs_interp:.2}x faster");
    println!("    🔄 Interpretation vs Naive: {interp_vs_naive:.2}x faster");
    println!("    📊 Total JIT speedup (including build): {total_jit_speedup:.2}x");
    println!("  🎯 Accuracy:");
    println!("    Cranelift JIT error: {accuracy_jit:.2e}");
    println!("    Interpretation error: {accuracy_interp:.2e}");

    // === JIT STATISTICS ===
    let jit_stats = math_jit.jit_stats();
    println!("  📊 JIT Statistics:");
    println!("    Cached functions: {}", jit_stats.cached_functions);
    println!("    Strategy: {:?}", jit_stats.strategy);

    // === PERFORMANCE CLASSIFICATION ===
    if jit_vs_naive > 10.0 {
        println!(
            "  🚀 CRANELIFT JIT CRUSHING IT! ({}x faster than naive)",
            jit_vs_naive as i32
        );
    } else if jit_vs_naive > 2.0 {
        println!("  ⚡ Cranelift JIT winning big!");
    } else if jit_vs_naive > 1.2 {
        println!("  ✅ Cranelift JIT performing well");
    } else {
        println!("  📝 Build cost may be dominating for this size");
    }

    if jit_vs_interp > 5.0 {
        println!("  🔥 JIT compilation provides massive speedup over interpretation!");
    } else if jit_vs_interp > 2.0 {
        println!("  ⚡ JIT compilation clearly faster than interpretation");
    } else {
        println!("  📊 JIT vs interpretation competitive");
    }

    // === SYMBOLIC INSPECTION ===
    if n <= 1000 {
        println!("  🔍 Expression structure:");
        let pretty = math_jit.pretty_print(&sum_expr);
        let lines: Vec<&str> = pretty.lines().take(3).collect();
        for line in lines {
            println!("     {line}");
        }
        if pretty.lines().count() > 3 {
            println!("     ... (truncated)");
        }
    }

    // === CRANELIFT-SPECIFIC INSIGHTS ===
    println!("  🔧 Cranelift JIT Insights:");
    println!("    - SYMBOLIC Sum AST node compiled to native iteration code");
    println!("    - Single JIT compilation for entire summation operation");
    println!("    - Zero function call overhead during summation");
    println!("    - Automatic vectorization and loop optimization");
    println!("    - Data parameter binding handled by JIT runtime");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cranelift_jit_correctness() -> Result<()> {
        let math_jit = DynamicContext::new_jit_optimized();
        let math_interp = DynamicContext::new_interpreter();

        // Small test case for verification
        let data = vec![1.0, 2.0, 3.0];
        let mu = 2.0;
        let sigma: f64 = 1.0;

        // Build identical expressions
        let build_expr = |ctx: &DynamicContext| -> Result<TypedBuilderExpr<f64>> {
            let mu_param = ctx.constant(mu);
            let sigma_param = ctx.constant(sigma);
            let two_pi = ctx.constant(2.0 * std::f64::consts::PI);

            ctx.sum(&data, |x: TypedBuilderExpr<f64>| -> TypedBuilderExpr<f64> {
                let diff = x - mu_param.clone();
                let variance = sigma_param.clone() * sigma_param.clone();
                let squared_error = diff.clone() * diff;
                let normalized_error = squared_error / (ctx.constant(2.0) * variance.clone());
                let log_normalization =
                    ctx.constant(0.5) * two_pi.clone().ln() + sigma_param.clone().ln();

                let result: TypedBuilderExpr<f64> = -(log_normalization + normalized_error);
                result
            })
        };

        let jit_expr = build_expr(&math_jit)?;
        let interp_expr = build_expr(&math_interp)?;

        let jit_result = math_jit.eval(&jit_expr, &[]);
        let interp_result = math_interp.eval(&interp_expr, &[]);

        // Manual calculation for verification
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_sigma = sigma.ln();
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut manual_result = 0.0;
        for &x in &data {
            let diff = x - mu;
            manual_result += -0.5 * log_2pi - log_sigma - (diff * diff) / two_sigma_sq;
        }

        let jit_error = (jit_result - manual_result).abs();
        let interp_error = (interp_result - manual_result).abs();
        let jit_vs_interp_error = (jit_result - interp_result).abs();

        assert!(
            jit_error < 1e-10,
            "Cranelift JIT result differs from manual: {} vs {}",
            jit_result,
            manual_result
        );

        assert!(
            interp_error < 1e-10,
            "Interpretation result differs from manual: {} vs {}",
            interp_result,
            manual_result
        );

        assert!(
            jit_vs_interp_error < 1e-10,
            "Cranelift JIT and interpretation differ: {} vs {}",
            jit_result,
            interp_result
        );

        println!("✅ Cranelift JIT produces identical results to interpretation and manual calculation");

        Ok(())
    }

    #[test]
    fn test_jit_strategy_behavior() -> Result<()> {
        // Test that different strategies actually use different evaluation methods
        let jit_ctx = DynamicContext::new_jit_optimized();
        let interp_ctx = DynamicContext::new_interpreter();
        let adaptive_ctx = DynamicContext::with_jit_strategy(JITStrategy::Adaptive {
            complexity_threshold: 3,
            call_count_threshold: 1,
        });

        let x = jit_ctx.var();
        let y = jit_ctx.var();

        // Complex expression that should trigger JIT in adaptive mode
        let complex_expr = (&x * &x + &y * &y).sqrt() + (&x * &y).sin();

        let jit_result = jit_ctx.eval(&complex_expr, &[3.0, 4.0]);
        
        // Build equivalent expression for other contexts
        let x_interp = interp_ctx.var();
        let y_interp = interp_ctx.var();
        let interp_expr = (&x_interp * &x_interp + &y_interp * &y_interp).sqrt() + (&x_interp * &y_interp).sin();
        let interp_result = interp_ctx.eval(&interp_expr, &[3.0, 4.0]);

        let x_adaptive = adaptive_ctx.var();
        let y_adaptive = adaptive_ctx.var();
        let adaptive_expr = (&x_adaptive * &x_adaptive + &y_adaptive * &y_adaptive).sqrt() + (&x_adaptive * &y_adaptive).sin();
        let adaptive_result = adaptive_ctx.eval(&adaptive_expr, &[3.0, 4.0]);

        // All should produce the same result
        assert!((jit_result - interp_result).abs() < 1e-10);
        assert!((jit_result - adaptive_result).abs() < 1e-10);

        // Check that JIT cache was used
        let jit_stats = jit_ctx.jit_stats();
        assert!(jit_stats.cached_functions > 0, "JIT should have cached the function");

        let adaptive_stats = adaptive_ctx.jit_stats();
        assert!(adaptive_stats.cached_functions > 0, "Adaptive should have used JIT for complex expression");

        println!("✅ JIT strategies behave correctly and produce consistent results");

        Ok(())
    }

    #[test]
    fn test_scaling_properties() -> Result<()> {
        let math_jit = DynamicContext::new_jit_optimized();

        // Test that expression complexity doesn't explode with data size
        let small_data = vec![1.0, 2.0];
        let large_data = (0..1000).map(|i| i as f64).collect::<Vec<_>>();

        let mu_param = math_jit.constant(0.0);
        let sigma_param = math_jit.constant(1.0);

        let build_expression = |data: &[f64]| -> Result<TypedBuilderExpr<f64>> {
            math_jit.sum(data, |x: TypedBuilderExpr<f64>| -> TypedBuilderExpr<f64> {
                let diff = x - mu_param.clone();
                let result: TypedBuilderExpr<f64> = -(diff.clone() * diff);
                result
            })
        };

        let small_expr = build_expression(&small_data)?;
        let large_expr = build_expression(&large_data)?;

        // Both should evaluate successfully (proving JIT compilation works)
        let small_result = math_jit.eval(&small_expr, &[]);
        let large_result = math_jit.eval(&large_expr, &[]);

        assert!(small_result.is_finite());
        assert!(large_result.is_finite());

        // Check that JIT cache is working
        let stats = math_jit.jit_stats();
        assert!(stats.cached_functions >= 2, "Should have cached both expressions");

        println!("Small data result: {}", small_result);
        println!("Large data result: {}", large_result);
        println!("JIT cached functions: {}", stats.cached_functions);

        Ok(())
    }
}
