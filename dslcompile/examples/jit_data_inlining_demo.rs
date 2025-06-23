//! JIT Data Inlining vs Runtime Data Demo
//!
//! This demo demonstrates the critical difference between:
//! 1. Data embedded in expressions (compile-time optimization)
//! 2. Data passed as function parameters (runtime computation)
//!
//! Shows how LLVM optimization can achieve 1ns timing through constant folding
//! when data is embedded, vs realistic timing when data is passed at runtime.

#[cfg(feature = "llvm_jit")]
use dslcompile::{backends::LLVMJITCompiler, prelude::*};

#[cfg(feature = "llvm_jit")]
use inkwell::{OptimizationLevel, context::Context};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

#[cfg(feature = "llvm_jit")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🎯 JIT Data Inlining vs Runtime Data Demo");
    println!("=========================================\n");

    // =======================================================================
    // 1. Data Embedded in Expression (Current Issue)
    // =======================================================================
    
    println!("1️⃣ Data Embedded in Expression (Ultra-fast but misleading)");
    println!("----------------------------------------------------------");
    
    // Generate random data
    let mut rng = StdRng::seed_from_u64(42);
    let data_size = 1000;
    let embedded_data: Vec<f64> = (0..data_size)
        .map(|_| rng.random_range(-5.0..5.0))
        .collect();
    
    println!("   📊 Generated {} random data points", embedded_data.len());
    
    // Create expression with embedded data
    let mut ctx = DynamicContext::new();
    let data_expr = ctx.data_array(embedded_data.clone());
    let sum_expr = data_expr.map(|x| x.clone() * x.clone() + 2.0 * x + 1.0).sum(); // x² + 2x + 1 for each element
    let embedded_ast = ctx.to_ast(&sum_expr);
    
    println!("   🔧 Created expression with embedded data");
    println!("   📝 Formula: Σ(xi² + 2xi + 1) where data is baked into AST");
    
    // Compile with LLVM JIT
    let context = Context::create();
    let mut jit_compiler = LLVMJITCompiler::new(&context);
    
    let start_compile = Instant::now();
    let embedded_fn = jit_compiler.compile_multi_var(&embedded_ast)?;
    let compile_time = start_compile.elapsed();
    
    println!("   ⚡ Compilation time: {:?}", compile_time);
    
    // Benchmark the "computation" (actually just returning a constant)
    let num_runs = 100_000;
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..num_runs {
        result = unsafe { embedded_fn.call([].as_ptr()) }; // No parameters needed for embedded data
    }
    let embedded_time = start.elapsed();
    let avg_time = embedded_time / num_runs;
    
    println!("   🚀 JIT execution: {:.2?} per call ({} runs)", avg_time, num_runs);
    println!("   🎯 Result: {:.6}", result);
    
    // Verify with manual computation
    let expected: f64 = embedded_data.iter()
        .map(|&x| x * x + 2.0 * x + 1.0)
        .sum();
    println!("   ✅ Expected: {:.6}", expected);
    println!("   🔍 Difference: {:.2e}", (result - expected).abs());
    
    if avg_time.as_nanos() < 10 {
        println!("   ⚠️  ULTRA-FAST: LLVM constant-folded the entire computation!");
        println!("   💡 The 'function' just returns a pre-computed constant");
    }

    // =======================================================================
    // 2. Data Passed as Function Parameters (Realistic Benchmark)
    // =======================================================================
    
    println!("\n2️⃣ Data Passed as Function Parameters (Realistic timing)");
    println!("--------------------------------------------------------");
    
    // Create a simple mathematical expression without embedded data
    let mut param_ctx = DynamicContext::new();
    let x = param_ctx.var(); // Variable(0) - will be the input parameter
    let param_expr = &x * &x + 2.0 * &x + 1.0; // x² + 2x + 1
    let param_ast = param_ctx.to_ast(&param_expr);
    
    println!("   📝 Created expression: f(x) = x² + 2x + 1");
    println!("   🔧 No data embedded - function takes parameter at runtime");
    
    // Compile the parameterized function
    let start_compile = Instant::now();
    let param_fn = jit_compiler.compile_single_var(&param_ast)?;
    let param_compile_time = start_compile.elapsed();
    
    println!("   ⚡ Compilation time: {:?}", param_compile_time);
    
    // Generate fresh random data AFTER compilation
    let mut fresh_rng = StdRng::seed_from_u64(123); // Different seed
    let runtime_data: Vec<f64> = (0..data_size)
        .map(|_| fresh_rng.random_range(-5.0..5.0))
        .collect();
    
    println!("   📊 Generated {} fresh random data points AFTER compilation", runtime_data.len());
    
    // Benchmark realistic computation - iterate over data
    let start = Instant::now();
    let mut param_result = 0.0;
    for &x_val in &runtime_data {
        param_result += unsafe { param_fn.call(x_val) };
    }
    let param_time = start.elapsed();
    let param_avg_time = param_time / runtime_data.len() as u32;
    
    println!("   🚀 JIT execution: {:.2?} per call ({} data points)", param_avg_time, runtime_data.len());
    println!("   🎯 Result: {:.6}", param_result);
    
    // Verify with manual computation  
    let param_expected: f64 = runtime_data.iter()
        .map(|&x| x * x + 2.0 * x + 1.0)
        .sum();
    println!("   ✅ Expected: {:.6}", param_expected);
    println!("   🔍 Difference: {:.2e}", (param_result - param_expected).abs());
    
    if param_avg_time.as_nanos() > 10 {
        println!("   ✅ REALISTIC: JIT is doing actual computation per call!");
        println!("   💡 Each call processes a different input value");
    }

    // =======================================================================
    // 3. Hand-written Comparison
    // =======================================================================
    
    println!("\n3️⃣ Hand-written Rust Function Comparison");
    println!("----------------------------------------");
    
    #[inline(always)]
    fn hand_written(x: f64) -> f64 {
        x * x + 2.0 * x + 1.0
    }
    
    // Benchmark hand-written function
    let start = Instant::now();
    let mut hand_result = 0.0;
    for &x_val in &runtime_data {
        hand_result += hand_written(x_val);
    }
    let hand_time = start.elapsed();
    let hand_avg_time = hand_time / runtime_data.len() as u32;
    
    println!("   🚀 Hand-written execution: {:.2?} per call", hand_avg_time);
    println!("   🎯 Result: {:.6}", hand_result);
    println!("   🔍 Difference from JIT: {:.2e}", (hand_result - param_result).abs());
    
    // =======================================================================
    // 4. Performance Analysis
    // =======================================================================
    
    println!("\n4️⃣ Performance Analysis");
    println!("------------------------");
    
    let embedded_speedup = param_avg_time.as_nanos() as f64 / avg_time.as_nanos() as f64;
    let jit_vs_hand = param_avg_time.as_nanos() as f64 / hand_avg_time.as_nanos() as f64;
    
    println!("   📈 Embedded data timing: {:.2?} per operation", avg_time);
    println!("   📈 Parameterized timing: {:.2?} per operation", param_avg_time);
    println!("   📈 Hand-written timing: {:.2?} per operation", hand_avg_time);
    println!();
    println!("   🚀 Embedded is {:.0}x faster (constant folding)", embedded_speedup);
    println!("   ⚖️  JIT vs Hand-written: {:.2}x overhead", jit_vs_hand);
    
    if embedded_speedup > 100.0 {
        println!("\n🎯 Key Insights:");
        println!("   • Embedded data → LLVM constant folding → 1ns 'execution'");
        println!("   • Parameterized data → Real computation → Realistic timing");
        println!("   • JIT achieves near-native performance for actual computation");
        println!("   • Always generate data AFTER compilation for realistic benchmarks");
    }
    
    println!("\n✅ Demo completed - Both approaches are valid for different use cases!");
    
    Ok(())
}

#[cfg(not(feature = "llvm_jit"))]
fn main() {
    println!("🚫 JIT Data Inlining Demo - LLVM JIT Feature Not Enabled");
    println!("=========================================================");
    println!();
    println!("To run this demo, enable the LLVM JIT feature:");
    println!("   cargo run --example jit_data_inlining_demo --features llvm_jit --release");
    println!();
    println!("This demo shows the difference between:");
    println!("   🔥 Data embedded in expressions (ultra-fast constant folding)");
    println!("   ⚡ Data passed as parameters (realistic performance measurement)");
}