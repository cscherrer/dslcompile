//! Partial Evaluation Demo: Data Specialization
//!
//! This example demonstrates partial evaluation by specializing on data:
//! 1. **IMPLEMENTED**: Standard approach - compile function + pass data at runtime
//! 2. **IMPLEMENTED**: Partial evaluation - compile function with data baked in
//! 3. **IMPLEMENTED**: Performance comparison and use cases
//! 4. **FUTURE WORK**: Abstract interpretation opportunities with known data
//!
//! ## Current Functionality
//! - Data specialization vs. runtime data binding comparison
//! - Performance benchmarking of both approaches
//! - Trade-off analysis for different use cases
//!
//! ## Future Work (see ROADMAP.md)
//! - Data range analysis for optimization opportunities
//! - Sparsity pattern detection and elimination
//! - Statistical property analysis for numerical stability
//! - Advanced partial evaluation scenarios

use mathcompile::prelude::*;
use std::time::Instant;

/// Create a simple regression function: f(β₀, β₁) = Σ(yᵢ - β₀ - β₁*xᵢ)²
/// This represents the sum of squared residuals for linear regression
fn create_runtime_data_function(data: &[(f64, f64)]) -> ASTRepr<f64> {
    let beta0 = <ASTEval as ASTMathExpr>::var(0); // β₀ (intercept)
    let beta1 = <ASTEval as ASTMathExpr>::var(1); // β₁ (slope)

    let mut sum_expr = <ASTEval as ASTMathExpr>::constant(0.0);

    for (i, &(x_i, y_i)) in data.iter().enumerate() {
        // For each data point: (yᵢ - β₀ - β₁*xᵢ)²
        let x_const = <ASTEval as ASTMathExpr>::constant(x_i);
        let y_const = <ASTEval as ASTMathExpr>::constant(y_i);

        // β₁ * xᵢ
        let beta1_x = <ASTEval as ASTMathExpr>::mul(beta1.clone(), x_const);

        // β₀ + β₁*xᵢ
        let prediction = <ASTEval as ASTMathExpr>::add(beta0.clone(), beta1_x);

        // yᵢ - (β₀ + β₁*xᵢ)
        let residual = <ASTEval as ASTMathExpr>::sub(y_const, prediction);

        // (yᵢ - β₀ - β₁*xᵢ)²
        let squared_residual =
            <ASTEval as ASTMathExpr>::pow(residual, <ASTEval as ASTMathExpr>::constant(2.0));

        // Add to sum
        sum_expr = <ASTEval as ASTMathExpr>::add(sum_expr, squared_residual);

        // Limit expression size for demo
        if i >= 4 {
            println!("   (Truncated to first 5 data points for demo)");
            break;
        }
    }

    sum_expr
}

/// Create a function that takes data as runtime parameters
/// f(β₀, β₁, x₁, y₁, x₂, y₂, ...) = Σ(yᵢ - β₀ - β₁*xᵢ)²
fn create_runtime_binding_function(n_points: usize) -> ASTRepr<f64> {
    let beta0 = <ASTEval as ASTMathExpr>::var(0); // β₀
    let beta1 = <ASTEval as ASTMathExpr>::var(1); // β₁

    let mut sum_expr = <ASTEval as ASTMathExpr>::constant(0.0);

    for i in 0..n_points {
        // Variables: β₀, β₁, x₁, y₁, x₂, y₂, ...
        let x_var = <ASTEval as ASTMathExpr>::var(2 + i * 2); // x_i
        let y_var = <ASTEval as ASTMathExpr>::var(2 + i * 2 + 1); // y_i

        // β₁ * xᵢ
        let beta1_x = <ASTEval as ASTMathExpr>::mul(beta1.clone(), x_var);

        // β₀ + β₁*xᵢ
        let prediction = <ASTEval as ASTMathExpr>::add(beta0.clone(), beta1_x);

        // yᵢ - (β₀ + β₁*xᵢ)
        let residual = <ASTEval as ASTMathExpr>::sub(y_var, prediction);

        // (yᵢ - β₀ - β₁*xᵢ)²
        let squared_residual =
            <ASTEval as ASTMathExpr>::pow(residual, <ASTEval as ASTMathExpr>::constant(2.0));

        sum_expr = <ASTEval as ASTMathExpr>::add(sum_expr, squared_residual);
    }

    sum_expr
}

fn main() -> Result<()> {
    println!("🔬 MathCompile: Partial Evaluation on Data");
    println!("===========================================\n");

    // Check if Rust compiler is available
    if !RustCompiler::is_available() {
        println!("❌ Rust compiler not available - this demo requires rustc");
        println!("   Please install Rust toolchain to run this example");
        return Ok(());
    }

    // Generate test data
    let data = vec![
        (1.0, 2.1),  // (x₁, y₁)
        (2.0, 4.2),  // (x₂, y₂)
        (3.0, 5.9),  // (x₃, y₃)
        (4.0, 8.1),  // (x₄, y₄)
        (5.0, 10.0), // (x₅, y₅)
    ];

    println!("📊 Test Data:");
    for (i, &(x, y)) in data.iter().enumerate() {
        println!("   Point {}: x={:.1}, y={:.1}", i + 1, x, y);
    }
    println!();

    // Test parameters
    let beta0 = 0.1; // intercept
    let beta1 = 2.0; // slope
    let test_params = vec![beta0, beta1];

    // ========================================
    // Part 1: Partial Evaluation (Data Specialized)
    // ========================================
    println!("📊 PART 1: Partial Evaluation - Data Baked Into Function");
    println!("=========================================================");
    println!("   Approach: Compile f(β₀, β₁) with data constants embedded");

    let specialized_start = Instant::now();
    let specialized_expr = create_runtime_data_function(&data);
    let specialized_build_time = specialized_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "   Expression operations: {}",
        specialized_expr.count_operations()
    );
    println!("   Build time: {specialized_build_time:.2}ms");

    // Compile specialized function
    let compile_start = Instant::now();
    let rust_generator = RustCodeGenerator::new();
    let rust_compiler = RustCompiler::new();

    let specialized_code =
        rust_generator.generate_function(&specialized_expr, "specialized_func")?;
    let specialized_compiled =
        rust_compiler.compile_and_load(&specialized_code, "specialized_func")?;
    let specialized_compile_time = compile_start.elapsed().as_secs_f64() * 1000.0;

    println!("   Compilation time: {specialized_compile_time:.2}ms");

    // Test evaluation (only parameters needed)
    let specialized_result = specialized_compiled.call_multi_vars(&test_params)?;
    let specialized_direct = DirectEval::eval_with_vars(&specialized_expr, &test_params);

    println!("   Test evaluation (β₀={beta0}, β₁={beta1}):");
    println!("     Compiled: {specialized_result:.6}");
    println!("     DirectEval: {specialized_direct:.6}");
    println!(
        "     Match: {}",
        (specialized_result - specialized_direct).abs() < 1e-10
    );

    // ========================================
    // Part 2: Runtime Data Binding
    // ========================================
    println!("\n📊 PART 2: Runtime Data Binding - Data Passed at Runtime");
    println!("=========================================================");
    println!("   Approach: Compile f(β₀, β₁, x₁, y₁, x₂, y₂, ...) with data as parameters");

    let runtime_start = Instant::now();
    let runtime_expr = create_runtime_binding_function(data.len());
    let runtime_build_time = runtime_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "   Expression operations: {}",
        runtime_expr.count_operations()
    );
    println!("   Build time: {runtime_build_time:.2}ms");

    // Compile runtime function
    let runtime_compile_start = Instant::now();
    let runtime_code = rust_generator.generate_function(&runtime_expr, "runtime_func")?;
    let runtime_compiled = rust_compiler.compile_and_load(&runtime_code, "runtime_func")?;
    let runtime_compile_time = runtime_compile_start.elapsed().as_secs_f64() * 1000.0;

    println!("   Compilation time: {runtime_compile_time:.2}ms");

    // Prepare runtime parameters: [β₀, β₁, x₁, y₁, x₂, y₂, ...]
    let mut runtime_params = vec![beta0, beta1];
    for &(x, y) in &data {
        runtime_params.push(x);
        runtime_params.push(y);
    }

    // Test evaluation
    let runtime_result = runtime_compiled.call_multi_vars(&runtime_params)?;
    let runtime_direct = DirectEval::eval_with_vars(&runtime_expr, &runtime_params);

    println!("   Test evaluation (β₀={beta0}, β₁={beta1} + data):");
    println!("     Compiled: {runtime_result:.6}");
    println!("     DirectEval: {runtime_direct:.6}");
    println!(
        "     Match: {}",
        (runtime_result - runtime_direct).abs() < 1e-10
    );

    // ========================================
    // Part 3: Comparison & Analysis
    // ========================================
    println!("\n📈 PART 3: Comparison & Analysis");
    println!("================================");

    println!("\n⏱️  Build & Compilation Time:");
    println!("                           │ Build (ms) │ Compile (ms) │ Total (ms)");
    println!("───────────────────────────┼────────────┼──────────────┼───────────");
    println!(
        "Partial Evaluation         │ {:>8.2}   │ {:>10.2}   │ {:>8.2}",
        specialized_build_time,
        specialized_compile_time,
        specialized_build_time + specialized_compile_time
    );
    println!(
        "Runtime Data Binding       │ {:>8.2}   │ {:>10.2}   │ {:>8.2}",
        runtime_build_time,
        runtime_compile_time,
        runtime_build_time + runtime_compile_time
    );

    println!("\n🔍 Expression Complexity:");
    println!(
        "   Partial Evaluation:  {} operations",
        specialized_expr.count_operations()
    );
    println!(
        "   Runtime Binding:     {} operations",
        runtime_expr.count_operations()
    );

    println!("\n📊 Parameter Count:");
    println!(
        "   Partial Evaluation:  {} parameters (β₀, β₁)",
        test_params.len()
    );
    println!(
        "   Runtime Binding:     {} parameters (β₀, β₁ + data)",
        runtime_params.len()
    );

    // Performance comparison
    let n_evals = 50_000;
    println!("\n⚡ Runtime Performance ({n_evals} evaluations):");

    // Benchmark specialized function
    let specialized_perf_start = Instant::now();
    for _ in 0..n_evals {
        let _ = specialized_compiled.call_multi_vars(&test_params)?;
    }
    let specialized_time = specialized_perf_start.elapsed();

    // Benchmark runtime function
    let runtime_perf_start = Instant::now();
    for _ in 0..n_evals {
        let _ = runtime_compiled.call_multi_vars(&runtime_params)?;
    }
    let runtime_time = runtime_perf_start.elapsed();

    println!("                           │ Time (ms)  │ Rate (M evals/s) │ Speedup");
    println!("───────────────────────────┼────────────┼──────────────────┼─────────");
    println!(
        "Partial Evaluation         │ {:>8.2}   │ {:>14.2}   │ {:>5.1}x",
        specialized_time.as_secs_f64() * 1000.0,
        f64::from(n_evals) / specialized_time.as_secs_f64() / 1_000_000.0,
        runtime_time.as_secs_f64() / specialized_time.as_secs_f64()
    );
    println!(
        "Runtime Data Binding       │ {:>8.2}   │ {:>14.2}   │ {:>5.1}x",
        runtime_time.as_secs_f64() * 1000.0,
        f64::from(n_evals) / runtime_time.as_secs_f64() / 1_000_000.0,
        1.0
    );

    // ========================================
    // Part 4: Use Cases & Trade-offs
    // ========================================
    println!("\n🧠 PART 4: Use Cases & Trade-offs");
    println!("=================================");

    println!("\n✅ Partial Evaluation (Data Specialized) - Best When:");
    println!("   • Data is fixed and known at compile time");
    println!("   • Same data used for many parameter evaluations");
    println!("   • Memory usage is critical (fewer parameters)");
    println!("   • Maximum runtime performance needed");
    println!("   • Examples: Model fitting, hyperparameter optimization");

    println!("\n✅ Runtime Data Binding - Best When:");
    println!("   • Data changes frequently");
    println!("   • Same function used with different datasets");
    println!("   • Flexibility is more important than peak performance");
    println!("   • Examples: Online learning, streaming data, A/B testing");

    println!("\n🎯 Abstract Interpretation Opportunities (FUTURE WORK):");
    println!("   • Data range analysis: min/max values enable optimizations");
    println!("   • Sparsity patterns: zero values can eliminate terms");
    println!("   • Statistical properties: mean, variance for numerical stability");
    println!("   • Correlation structure: redundant computations identification");

    println!("\n🚀 Advanced Partial Evaluation Scenarios (FUTURE WORK):");
    println!("   • Partial data specialization: fix some data points, vary others");
    println!("   • Hierarchical models: specialize on group-level data");
    println!("   • Time series: specialize on historical data, predict future");
    println!("   • Ensemble methods: specialize each model on different data subsets");

    // Verify results match
    println!("\n🔍 Verification:");
    println!(
        "   Results match: {}",
        (specialized_result - runtime_result).abs() < 1e-10
    );
    println!("   Both approaches compute identical values with different trade-offs");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_evaluation_vs_runtime_binding() -> Result<()> {
        if !RustCompiler::is_available() {
            return Ok(());
        }

        let data = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
        let params = vec![0.0, 2.0]; // β₀=0, β₁=2 (perfect fit)

        // Create both approaches
        let specialized_expr = create_runtime_data_function(&data);
        let runtime_expr = create_runtime_binding_function(data.len());

        // Evaluate with DirectEval
        let specialized_result = DirectEval::eval_with_vars(&specialized_expr, &params);

        let mut runtime_params = params.clone();
        for &(x, y) in &data {
            runtime_params.push(x);
            runtime_params.push(y);
        }
        let runtime_result = DirectEval::eval_with_vars(&runtime_expr, &runtime_params);

        // Should produce identical results
        assert!((specialized_result - runtime_result).abs() < 1e-10);

        // Perfect fit should give near-zero sum of squared residuals
        assert!(specialized_result < 1e-10);

        Ok(())
    }

    #[test]
    fn test_expression_complexity() {
        let data = vec![(1.0, 2.0), (2.0, 4.0)];

        let specialized_expr = create_runtime_data_function(&data);
        let runtime_expr = create_runtime_binding_function(data.len());

        // Both should have similar complexity for same data size
        let specialized_ops = specialized_expr.count_operations();
        let runtime_ops = runtime_expr.count_operations();

        // Allow some variation due to different construction approaches
        assert!((specialized_ops as i32 - runtime_ops as i32).abs() <= 2);
    }
}
