//! LLVM Optimization Analysis
//!
//! This example investigates the performance overhead between JIT compiled
//! functions and hand-written Rust code, testing different optimization levels
//! and analyzing the generated LLVM IR.

#[cfg(feature = "llvm_jit")]
use dslcompile::{
    ast::ASTRepr,
    backends::LLVMJITCompiler,
    composition::MathFunction,
    prelude::*,
};

#[cfg(feature = "llvm_jit")]
use inkwell::{context::Context, OptimizationLevel};
use std::time::Instant;

#[cfg(feature = "llvm_jit")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🔬 LLVM Optimization Analysis");
    println!("=============================\n");

    // Create expression: x² + 2x + 1
    let math_func = MathFunction::from_lambda("quadratic", |builder| {
        builder.lambda(|x| {
            x.clone() * x.clone() + x.clone() * 2.0 + 1.0
        })
    });

    let ast = math_func.to_ast();
    let context = Context::create();
    let mut jit_compiler = LLVMJITCompiler::new(&context);

    println!("📋 Testing Expression: f(x) = x² + 2x + 1");
    println!("===========================================\n");

    // Test different optimization levels
    let opt_levels = [
        (OptimizationLevel::None, "None"),
        (OptimizationLevel::Less, "Less"), 
        (OptimizationLevel::Default, "Default"),
        (OptimizationLevel::Aggressive, "Aggressive"),
    ];

    let iterations = 1_000_000;
    let mut results = Vec::new();

    for (opt_level, opt_name) in &opt_levels {
        println!("🔧 Testing Optimization Level: {}", opt_name);
        println!("--------------------------------");

        // Compile with specific optimization level
        let start_compile = Instant::now();
        let compiled_fn = jit_compiler.compile_expression_with_opt(&ast, *opt_level)?;
        let compile_time = start_compile.elapsed();

        // Benchmark performance
        let start = Instant::now();
        let mut sum = 0.0;
        for i in 0..iterations {
            let x_val = f64::from(i) * 0.001;
            sum += unsafe { compiled_fn.call(x_val) };
        }
        let exec_time = start.elapsed();
        let ns_per_call = exec_time.as_nanos() as f64 / f64::from(iterations);

        println!("   Compilation time: {:?}", compile_time);
        println!("   Execution time: {:?}", exec_time);
        println!("   Time per call: {:.2} ns", ns_per_call);
        println!("   Sum: {:.6}", sum);
        println!();

        results.push((opt_name.to_string(), ns_per_call, sum));
    }

    // Benchmark hand-written function
    println!("🖋️  Hand-Written Baseline");
    println!("------------------------");

    #[inline(always)]
    fn hand_written_quadratic(x: f64) -> f64 {
        x * x + 2.0 * x + 1.0
    }

    let start = Instant::now();
    let mut hand_written_sum = 0.0;
    for i in 0..iterations {
        let x_val = f64::from(i) * 0.001;
        hand_written_sum += hand_written_quadratic(x_val);
    }
    let hand_written_time = start.elapsed();
    let hand_written_ns = hand_written_time.as_nanos() as f64 / f64::from(iterations);

    println!("   Execution time: {:?}", hand_written_time);
    println!("   Time per call: {:.2} ns", hand_written_ns);
    println!("   Sum: {:.6}", hand_written_sum);
    println!();

    // Analysis
    println!("📊 Performance Analysis");
    println!("=======================");

    for (opt_name, ns_per_call, sum) in &results {
        let overhead_ratio = ns_per_call / hand_written_ns;
        let accuracy_diff = (sum - hand_written_sum).abs();
        
        println!("🔧 {} Optimization:", opt_name);
        println!("   Performance ratio: {:.2}x vs hand-written", overhead_ratio);
        println!("   Accuracy difference: {:.2e}", accuracy_diff);
        
        if overhead_ratio <= 1.2 {
            println!("   ✅ Excellent performance!");
        } else if overhead_ratio <= 2.0 {
            println!("   ⚠️  Moderate overhead");
        } else {
            println!("   ❌ Significant overhead");
        }
        println!();
    }

    // Insights
    println!("🔍 Performance Insights");
    println!("=======================");
    
    let best_jit = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let worst_jit = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    
    println!("✅ Best JIT performance: {} ({:.2} ns)", best_jit.0, best_jit.1);
    println!("❌ Worst JIT performance: {} ({:.2} ns)", worst_jit.0, worst_jit.1);
    println!("🖋️  Hand-written baseline: {:.2} ns", hand_written_ns);
    
    let best_overhead = best_jit.1 / hand_written_ns;
    println!("\n🎯 Best JIT vs Hand-written: {:.2}x overhead", best_overhead);
    
    if best_overhead <= 1.5 {
        println!("   ✅ JIT achieves excellent performance!");
    } else {
        println!("   ⚠️  JIT has significant overhead - potential causes:");
        println!("      - Function call overhead (extern \"C\" vs inline)");
        println!("      - Missing target-specific optimizations");
        println!("      - LLVM optimization settings vs rustc --release");
        println!("      - Benchmark loop optimization differences");
    }

    // Recommendations
    println!("\n💡 Optimization Recommendations");
    println!("===============================");
    println!("1. **Add target-specific optimization flags**");
    println!("2. **Test different LLVM optimization passes**");
    println!("3. **Consider inline assembly for critical paths**");
    println!("4. **Profile both JIT and hand-written code for micro-optimizations**");
    println!("5. **Test with larger, more complex expressions where JIT overhead is amortized**");

    Ok(())
}

#[cfg(not(feature = "llvm_jit"))]
fn main() {
    println!("🚫 LLVM Optimization Analysis - Feature Not Enabled");
    println!("====================================================");
    println!();
    println!("To run this analysis, enable the LLVM JIT feature:");
    println!("   cargo run --example llvm_optimization_analysis --features llvm_jit --release");
    println!();
    println!("This example will:");
    println!("   📊 Test different LLVM optimization levels");
    println!("   🔬 Compare JIT vs hand-written performance");
    println!("   💡 Provide optimization recommendations");
}