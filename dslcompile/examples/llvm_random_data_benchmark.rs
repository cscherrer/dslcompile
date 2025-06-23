//! LLVM JIT Benchmark with Random Data
//!
//! This benchmark uses random data to eliminate any potential compile-time
//! optimizations, constant folding, or loop vectorization that could skew
//! the performance comparison between JIT and hand-written code.

#[cfg(feature = "llvm_jit")]
use dslcompile::{backends::LLVMJITCompiler, composition::MathFunction, prelude::*};

#[cfg(feature = "llvm_jit")]
use inkwell::{OptimizationLevel, context::Context};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

#[cfg(feature = "llvm_jit")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🎲 LLVM JIT Benchmark with Random Data");
    println!("======================================\n");

    // Create deterministic random data
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible results
    let iterations = 1_000_000;
    let test_data: Vec<f64> = (0..iterations)
        .map(|_| rng.random_range(-10.0..10.0))
        .collect();

    println!(
        "📊 Generated {} random test values in range [-10.0, 10.0]",
        iterations
    );
    println!("🔢 Using fixed seed for reproducible results\n");

    // Test simple expression: x² + 2x + 1
    println!("🧮 Testing Simple Expression: f(x) = x² + 2x + 1");
    println!("================================================");
    test_expression_performance(&test_data, "simple", |builder| {
        builder.lambda(|x| x.clone() * x.clone() + x.clone() * 2.0 + 1.0)
    })?;

    // Test complex expression with trigonometry
    println!("\n🌊 Testing Complex Expression: f(x) = sin(x²) + cos(2x) + ln(x² + 1) + exp(-x/5)");
    println!("==============================================================================");
    test_expression_performance(&test_data, "complex", |builder| {
        builder.lambda(|x| {
            let x_sq = x.clone() * x.clone();
            let sin_part = x_sq.sin();
            let cos_part = (x.clone() * 2.0).cos();
            // Use x² + 1 instead of |x| + 1 since abs is not available
            let ln_part = (x.clone() * x.clone() + 1.0).ln();
            let exp_part = (x.clone() * -0.2).exp();
            sin_part + cos_part + ln_part + exp_part
        })
    })?;

    // Test very complex expression with multiple operations
    println!("\n🔥 Testing Very Complex Expression:");
    println!("f(x) = ((x³ + sin(x))/(cos(x) + 2)) * ln(x² + 1) + sqrt(x² + 1) * exp(-x²/10)");
    println!("===========================================================================");
    test_expression_performance(&test_data, "very_complex", |builder| {
        builder.lambda(|x| {
            let x_cubed = x.clone() * x.clone() * x.clone();
            let numerator = x_cubed + x.clone().sin();
            let denominator = x.clone().cos() + 2.0;
            let fraction = numerator / denominator;

            let x_sq = x.clone() * x.clone();
            let ln_part = (x_sq + 1.0).ln();
            // Use x² + 1 instead of |x| since abs is not available
            let sqrt_part = (x.clone() * x.clone() + 1.0).sqrt();
            let exp_part = (x.clone() * x.clone() * -0.1).exp();

            fraction * ln_part + sqrt_part * exp_part
        })
    })?;

    Ok(())
}

#[cfg(feature = "llvm_jit")]
fn test_expression_performance<F>(
    test_data: &[f64],
    name: &str,
    lambda_builder: F,
) -> std::result::Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce(
        &mut dslcompile::composition::FunctionBuilder<f64>,
    ) -> dslcompile::ast::ast_repr::Lambda<f64>,
{
    // Create expression with LambdaVar
    let math_func = MathFunction::from_lambda(name, lambda_builder);
    let ast = math_func.to_ast();

    // Set up JIT compiler
    let context = Context::create();
    let mut jit_compiler = LLVMJITCompiler::new(&context);

    // Compile with aggressive optimization
    let start_compile = Instant::now();
    let compiled_fn =
        jit_compiler.compile_expression_with_opt(&ast, OptimizationLevel::Aggressive)?;
    let compile_time = start_compile.elapsed();

    println!("⚡ JIT Compilation time: {:?}", compile_time);

    // Benchmark JIT compiled function with random data
    let start = Instant::now();
    let mut jit_sum = 0.0;
    for &x_val in test_data {
        jit_sum += unsafe { compiled_fn.call(x_val) };
    }
    let jit_time = start.elapsed();
    let jit_ns_per_call = jit_time.as_nanos() as f64 / test_data.len() as f64;

    println!("🚀 JIT Performance:");
    println!("   Total time: {:?}", jit_time);
    println!("   Time per call: {:.2} ns", jit_ns_per_call);
    println!("   Sum: {:.6}", jit_sum);

    // Create equivalent hand-written function for comparison
    let hand_written_result = match name {
        "simple" => {
            #[inline(always)]
            fn simple_func(x: f64) -> f64 {
                x * x + 2.0 * x + 1.0
            }

            let start = Instant::now();
            let mut sum = 0.0;
            for &x_val in test_data {
                sum += simple_func(x_val);
            }
            let time = start.elapsed();
            (sum, time)
        }
        "complex" => {
            #[inline(always)]
            fn complex_func(x: f64) -> f64 {
                let x_sq = x * x;
                let sin_part = x_sq.sin();
                let cos_part = (x * 2.0).cos();
                let ln_part = (x * x + 1.0).ln();
                let exp_part = (x * -0.2).exp();
                sin_part + cos_part + ln_part + exp_part
            }

            let start = Instant::now();
            let mut sum = 0.0;
            for &x_val in test_data {
                sum += complex_func(x_val);
            }
            let time = start.elapsed();
            (sum, time)
        }
        "very_complex" => {
            #[inline(always)]
            fn very_complex_func(x: f64) -> f64 {
                let x_cubed = x * x * x;
                let numerator = x_cubed + x.sin();
                let denominator = x.cos() + 2.0;
                let fraction = numerator / denominator;

                let x_sq = x * x;
                let ln_part = (x_sq + 1.0).ln();
                let sqrt_part = (x * x + 1.0).sqrt();
                let exp_part = (x * x * -0.1).exp();

                fraction * ln_part + sqrt_part * exp_part
            }

            let start = Instant::now();
            let mut sum = 0.0;
            for &x_val in test_data {
                sum += very_complex_func(x_val);
            }
            let time = start.elapsed();
            (sum, time)
        }
        _ => panic!("Unknown expression type"),
    };

    let (hand_written_sum, hand_written_time) = hand_written_result;
    let hand_written_ns = hand_written_time.as_nanos() as f64 / test_data.len() as f64;

    println!("✍️  Hand-Written Performance:");
    println!("   Total time: {:?}", hand_written_time);
    println!("   Time per call: {:.2} ns", hand_written_ns);
    println!("   Sum: {:.6}", hand_written_sum);

    // Compare with interpreted evaluation (for context)
    let mut dynamic_ctx = DynamicContext::new();
    let x_var = dynamic_ctx.var();

    let interpreted_expr = match name {
        "simple" => &x_var * &x_var + 2.0 * &x_var + 1.0,
        "complex" => {
            let x_sq = &x_var * &x_var;
            let sin_part = x_sq.sin();
            let cos_part = (&x_var * 2.0).cos();
            let ln_part = (&x_var * &x_var + 1.0).ln();
            let exp_part = (&x_var * -0.2).exp();
            sin_part + cos_part + ln_part + exp_part
        }
        "very_complex" => {
            let x_cubed = &x_var * &x_var * &x_var;
            let numerator = x_cubed + x_var.clone().sin();
            let denominator = x_var.clone().cos() + 2.0;
            let fraction = numerator / denominator;

            let x_sq = &x_var * &x_var;
            let ln_part = (x_sq.clone() + 1.0).ln();
            let sqrt_part = (x_sq + 1.0).sqrt();
            let exp_part = (&x_var * &x_var * -0.1).exp();

            fraction * ln_part + sqrt_part * exp_part
        }
        _ => panic!("Unknown expression type"),
    };

    let start = Instant::now();
    let mut interpreted_sum = 0.0;
    for &x_val in test_data.iter().take(10000) {
        // Sample only 10k for interpreted (too slow)
        interpreted_sum += dynamic_ctx.eval(&interpreted_expr, frunk::hlist![x_val]);
    }
    let interpreted_time = start.elapsed();
    let interpreted_ns = interpreted_time.as_nanos() as f64 / 10000.0;

    println!("🔄 Interpreted Performance (10k sample):");
    println!("   Total time: {:?}", interpreted_time);
    println!("   Time per call: {:.2} ns", interpreted_ns);
    println!("   Sum (sample): {:.6}", interpreted_sum);

    // Analysis
    println!("\n📈 Performance Analysis:");
    let jit_vs_hand_written = jit_ns_per_call / hand_written_ns;
    let jit_vs_interpreted = interpreted_ns / jit_ns_per_call;

    println!(
        "   JIT vs Hand-Written: {:.2}x overhead",
        jit_vs_hand_written
    );
    println!("   JIT vs Interpreted: {:.1}x faster", jit_vs_interpreted);

    // Mathematical accuracy
    let accuracy_diff = (jit_sum - hand_written_sum).abs();
    let relative_error = accuracy_diff / hand_written_sum.abs();

    println!("   Accuracy difference: {:.2e}", accuracy_diff);
    println!("   Relative error: {:.2e}", relative_error);

    if relative_error < 1e-10 {
        println!("   ✅ Perfect mathematical accuracy!");
    } else if relative_error < 1e-6 {
        println!("   ✅ Excellent accuracy");
    } else {
        println!("   ⚠️  Accuracy concerns");
    }

    // Performance assessment
    if jit_vs_hand_written <= 1.5 {
        println!("   ✅ Excellent JIT performance - very close to hand-written!");
    } else if jit_vs_hand_written <= 3.0 {
        println!("   ⚠️  Moderate JIT overhead - acceptable for complex expressions");
    } else {
        println!("   ❌ Significant JIT overhead");
    }

    Ok(())
}

#[cfg(not(feature = "llvm_jit"))]
fn main() {
    println!("🚫 LLVM Random Data Benchmark - Feature Not Enabled");
    println!("===================================================");
    println!();
    println!("To run this benchmark, enable the LLVM JIT feature:");
    println!("   cargo run --example llvm_random_data_benchmark --features llvm_jit --release");
    println!();
    println!("This benchmark will:");
    println!("   🎲 Use random data to eliminate compile-time optimizations");
    println!("   📊 Test simple, complex, and very complex expressions");
    println!("   🔬 Compare JIT vs hand-written vs interpreted performance");
    println!("   ✅ Verify mathematical accuracy across all approaches");
}
