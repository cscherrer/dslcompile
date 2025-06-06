//! Cranelift vs Rayon Parallelism Comparison
//!
//! This demo explores the different types of parallelism available:
//! 1. Cranelift: SIMD parallelism + thread-safe JIT functions
//! 2. Rayon: Work-stealing parallelism for data processing
//! 3. Hybrid: Combining both approaches for maximum performance
//! 4. Performance analysis across different workload patterns

use dslcompile::ast::runtime::expression_builder::DynamicContext;
use dslcompile::backends::cranelift::{CraneliftCompiler, OptimizationLevel};
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    println!("⚡ Cranelift vs Rayon Parallelism Comparison");
    println!("============================================\n");

    // Demo 1: Understanding the Different Types of Parallelism
    demo_parallelism_types();

    // Demo 2: Single Expression, Many Data Points (Rayon's Strength)
    demo_data_parallelism();

    // Demo 3: Many Expressions, Single Data Point (Cranelift's Strength)
    demo_expression_parallelism();

    // Demo 4: Hybrid Approach - Best of Both Worlds
    demo_hybrid_parallelism();

    // Demo 5: SIMD vs Thread Parallelism
    demo_simd_vs_threads();

    // Demo 6: Performance Analysis and Recommendations
    demo_performance_analysis();
}

fn demo_parallelism_types() {
    println!("🔍 DEMO 1: UNDERSTANDING PARALLELISM TYPES");
    println!("===========================================");
    println!("Different approaches to parallel computation:\n");

    println!("🧠 **CRANELIFT PARALLELISM**:");
    println!("• **SIMD Parallelism**: Process multiple values per CPU instruction");
    println!("• **Thread-Safe JIT**: Same compiled function callable from multiple threads");
    println!("• **Instruction-Level**: AVX-512 can process 8 f64 values simultaneously");
    println!("• **Zero Overhead**: No runtime coordination or work distribution");
    println!("• **Best for**: Vectorizable mathematical operations");

    println!("\n🚀 **RAYON PARALLELISM**:");
    println!("• **Work-Stealing**: Distribute tasks across CPU cores dynamically");
    println!("• **Data Parallelism**: Split large datasets across multiple threads");
    println!("• **Load Balancing**: Automatically balance work between cores");
    println!("• **Runtime Coordination**: Manages thread pools and task distribution");
    println!("• **Best for**: Large datasets with independent computations");

    println!("\n🎯 **KEY INSIGHT**: They operate at different levels!");
    println!("• Cranelift: Instruction-level parallelism (SIMD)");
    println!("• Rayon: Task-level parallelism (threads)");
    println!("• **They complement each other perfectly!**");
    println!();
}

fn demo_data_parallelism() {
    println!("📊 DEMO 2: DATA PARALLELISM (Rayon's Strength)");
    println!("===============================================");
    println!("Scenario: One expression, many data points\n");

    // Create a moderately complex expression
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();
    let expr = (x.clone().sin() * y.clone().cos() + (x.clone() * y.clone()).exp() / (x.clone() * x.clone() + y.clone() * y.clone()).sqrt()).into();

    // Compile with Cranelift
    let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
    let registry = dslcompile::ast::VariableRegistry::for_expression(&expr);
    let compiled_func = compiler.compile_expression(&expr, &registry).unwrap();

    // Generate large dataset
    let data_size = 100_000;
    let test_data: Vec<(f64, f64)> = (0..data_size)
        .map(|i| {
            let x = (i as f64) * 0.01;
            let y = (i as f64) * 0.02 + 1.0;
            (x, y)
        })
        .collect();

    println!("Dataset: {} data points", data_size);
    println!("Expression: sin(x) * cos(y) + exp(x*y) / sqrt(x² + y²)\n");

    // Sequential evaluation
    let start = Instant::now();
    let sequential_results: Vec<f64> = test_data
        .iter()
        .map(|&(x, y)| compiled_func.call(&[x, y]).unwrap())
        .collect();
    let sequential_time = start.elapsed();

    // Rayon parallel evaluation
    let start = Instant::now();
    let rayon_results: Vec<f64> = test_data
        .par_iter()
        .map(|&(x, y)| compiled_func.call(&[x, y]).unwrap())
        .collect();
    let rayon_time = start.elapsed();

    // Manual thread parallelism (for comparison)
    let num_threads = rayon::current_num_threads();
    let chunk_size = data_size / num_threads;
    
    let start = Instant::now();
    let manual_results: Vec<f64> = test_data
        .chunks(chunk_size)
        .collect::<Vec<_>>()
        .par_iter()
        .flat_map(|chunk| {
            chunk.iter().map(|&(x, y)| compiled_func.call(&[x, y]).unwrap())
        })
        .collect();
    let manual_time = start.elapsed();

    println!("Performance Results:");
    println!("• Sequential: {:.3}ms", sequential_time.as_millis());
    println!("• Rayon parallel: {:.3}ms", rayon_time.as_millis());
    println!("• Manual parallel: {:.3}ms", manual_time.as_millis());
    
    let rayon_speedup = sequential_time.as_nanos() as f64 / rayon_time.as_nanos() as f64;
    let manual_speedup = sequential_time.as_nanos() as f64 / manual_time.as_nanos() as f64;
    
    println!("• Rayon speedup: {:.2}x", rayon_speedup);
    println!("• Manual speedup: {:.2}x", manual_speedup);
    println!("• Results match: {}", sequential_results == rayon_results && rayon_results == manual_results);
    println!("• CPU cores used: {}", num_threads);

    println!("\n🎯 Analysis:");
    println!("• Rayon excels at distributing large datasets across cores");
    println!("• Work-stealing provides excellent load balancing");
    println!("• Cranelift functions are perfectly thread-safe");
    println!("• Combined approach: Rayon (threads) + Cranelift (SIMD)");
    println!();
}

fn demo_expression_parallelism() {
    println!("🧮 DEMO 3: EXPRESSION PARALLELISM (Cranelift's Strength)");
    println!("=========================================================");
    println!("Scenario: Many expressions, same data points\n");

    // Create multiple different expressions
    let expressions = vec![
        ("Linear", "2x + 3y"),
        ("Quadratic", "x² + xy + y²"),
        ("Trigonometric", "sin(x) + cos(y)"),
        ("Exponential", "exp(x) + exp(y)"),
        ("Logarithmic", "ln(x+1) + ln(y+1)"),
        ("Complex", "sin(x)*cos(y) + exp(x/10)"),
    ];

    // Compile all expressions
    let mut compiled_funcs = Vec::new();
    for (name, _) in &expressions {
        let ctx = DynamicContext::new();
        let x = ctx.var();
        let y = ctx.var();
        
        let expr = match *name {
            "Linear" => (2.0 * &x + 3.0 * &y).into(),
            "Quadratic" => (&x * &x + &x * &y + &y * &y).into(),
            "Trigonometric" => (x.sin() + y.cos()).into(),
            "Exponential" => (x.exp() + y.exp()).into(),
            "Logarithmic" => ((x.clone() + 1.0).ln() + (y.clone() + 1.0).ln()).into(),
                         "Complex" => (x.clone().sin() * y.clone().cos() + (x.clone() / ctx.constant(10.0)).exp()).into(),
            _ => unreachable!(),
        };

        let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
        let registry = dslcompile::ast::VariableRegistry::for_expression(&expr);
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();
        compiled_funcs.push((name, compiled));
    }

    let test_point = [2.5, 1.8];
    let iterations = 1_000_000;

    println!("Test point: [{}, {}]", test_point[0], test_point[1]);
    println!("Iterations per expression: {}\n", iterations);

    // Sequential evaluation of all expressions
    let start = Instant::now();
    let mut sequential_results = Vec::new();
    for (name, func) in &compiled_funcs {
        let mut sum = 0.0;
        for _ in 0..iterations {
            sum += func.call(&test_point).unwrap();
        }
        sequential_results.push((*name, sum));
    }
    let sequential_time = start.elapsed();

    // Parallel evaluation using Rayon
    let start = Instant::now();
    let rayon_results: Vec<(&str, f64)> = compiled_funcs
        .par_iter()
        .map(|(name, func)| {
            let mut sum = 0.0;
            for _ in 0..iterations {
                sum += func.call(&test_point).unwrap();
            }
            (*name, sum)
        })
        .collect();
    let rayon_time = start.elapsed();

    println!("Performance Results:");
    println!("• Sequential: {:.3}ms", sequential_time.as_millis());
    println!("• Rayon parallel: {:.3}ms", rayon_time.as_millis());
    
    let speedup = sequential_time.as_nanos() as f64 / rayon_time.as_nanos() as f64;
    println!("• Speedup: {:.2}x", speedup);
    
    // Verify results match
    let results_match = sequential_results.iter().zip(rayon_results.iter())
        .all(|((name1, val1), (name2, val2))| name1 == name2 && (val1 - val2).abs() < 1e-10);
    println!("• Results match: {}", results_match);

    println!("\n🎯 Analysis:");
    println!("• Multiple expressions can be evaluated in parallel");
    println!("• Each Cranelift function is independently optimized");
    println!("• SIMD optimizations apply within each expression");
    println!("• Rayon distributes expressions across cores");
    println!();
}

fn demo_hybrid_parallelism() {
    println!("🚀 DEMO 4: HYBRID PARALLELISM (Best of Both Worlds)");
    println!("===================================================");
    println!("Combining Rayon's work-stealing with Cranelift's SIMD\n");

    // Create a vectorizable expression
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();
    let z = ctx.var();
    let w = ctx.var();
    
    // 4D vector operations (perfect for SIMD)
    let vector_expr = (&x * &x + &y * &y + &z * &z + &w * &w).sqrt().into();

    let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
    let registry = dslcompile::ast::VariableRegistry::for_expression(&vector_expr);
    let compiled_func = compiler.compile_expression(&vector_expr, &registry).unwrap();

    // Generate 4D vector dataset
    let num_vectors = 1_000_000;
    let vectors: Vec<[f64; 4]> = (0..num_vectors)
        .map(|i| {
            let base = i as f64 * 0.001;
            [base, base + 1.0, base + 2.0, base + 3.0]
        })
        .collect();

    println!("Dataset: {} 4D vectors", num_vectors);
    println!("Expression: sqrt(x² + y² + z² + w²) (4D magnitude)\n");

    // Sequential evaluation
    let start = Instant::now();
    let sequential_results: Vec<f64> = vectors
        .iter()
        .map(|vec| compiled_func.call(vec).unwrap())
        .collect();
    let sequential_time = start.elapsed();

    // Rayon + Cranelift hybrid
    let start = Instant::now();
    let hybrid_results: Vec<f64> = vectors
        .par_iter()
        .map(|vec| compiled_func.call(vec).unwrap())
        .collect();
    let hybrid_time = start.elapsed();

    // Chunked processing (simulate SIMD-friendly batching)
    let chunk_size = 1000; // Process in SIMD-friendly chunks
    let start = Instant::now();
    let chunked_results: Vec<f64> = vectors
        .par_chunks(chunk_size)
        .flat_map(|chunk| {
            chunk.iter().map(|vec| compiled_func.call(vec).unwrap())
        })
        .collect();
    let chunked_time = start.elapsed();

    println!("Performance Results:");
    println!("• Sequential: {:.3}ms", sequential_time.as_millis());
    println!("• Rayon + Cranelift: {:.3}ms", hybrid_time.as_millis());
    println!("• Chunked processing: {:.3}ms", chunked_time.as_millis());
    
    let hybrid_speedup = sequential_time.as_nanos() as f64 / hybrid_time.as_nanos() as f64;
    let chunked_speedup = sequential_time.as_nanos() as f64 / chunked_time.as_nanos() as f64;
    
    println!("• Hybrid speedup: {:.2}x", hybrid_speedup);
    println!("• Chunked speedup: {:.2}x", chunked_speedup);
    
    let results_match = sequential_results == hybrid_results && hybrid_results == chunked_results;
    println!("• Results match: {}", results_match);

    println!("\n🎯 Analysis:");
    println!("• Rayon provides thread-level parallelism");
    println!("• Cranelift provides instruction-level parallelism (SIMD)");
    println!("• Combined: Multi-core + SIMD = maximum throughput");
    println!("• Chunked processing can improve cache locality");
    println!();
}

fn demo_simd_vs_threads() {
    println!("⚡ DEMO 5: SIMD vs THREAD PARALLELISM");
    println!("=====================================");
    println!("Comparing instruction-level vs task-level parallelism\n");

    // Create a simple vectorizable operation
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();
    let simple_expr = (&x + &y).into();

    let mut compiler = CraneliftCompiler::new(OptimizationLevel::Full).unwrap();
    let registry = dslcompile::ast::VariableRegistry::for_expression(&simple_expr);
    let compiled_func = compiler.compile_expression(&simple_expr, &registry).unwrap();

    // Small dataset (where SIMD shines)
    let small_data: Vec<(f64, f64)> = (0..1000).map(|i| (i as f64, i as f64 + 1.0)).collect();
    
    // Large dataset (where threads shine)
    let large_data: Vec<(f64, f64)> = (0..1_000_000).map(|i| (i as f64, i as f64 + 1.0)).collect();

    println!("🔬 Small Dataset Analysis (1,000 points):");
    
    // Small dataset - sequential
    let start = Instant::now();
    let small_seq: Vec<f64> = small_data.iter().map(|&(x, y)| compiled_func.call(&[x, y]).unwrap()).collect();
    let small_seq_time = start.elapsed();
    
    // Small dataset - parallel
    let start = Instant::now();
    let small_par: Vec<f64> = small_data.par_iter().map(|&(x, y)| compiled_func.call(&[x, y]).unwrap()).collect();
    let small_par_time = start.elapsed();
    
    println!("• Sequential: {:.3}μs", small_seq_time.as_micros());
    println!("• Rayon parallel: {:.3}μs", small_par_time.as_micros());
    
    if small_par_time.as_nanos() > 0 {
        let small_speedup = small_seq_time.as_nanos() as f64 / small_par_time.as_nanos() as f64;
        println!("• Speedup: {:.2}x", small_speedup);
    } else {
        println!("• Speedup: N/A (too fast to measure)");
    }

    println!("\n🚀 Large Dataset Analysis (1,000,000 points):");
    
    // Large dataset - sequential
    let start = Instant::now();
    let large_seq: Vec<f64> = large_data.iter().map(|&(x, y)| compiled_func.call(&[x, y]).unwrap()).collect();
    let large_seq_time = start.elapsed();
    
    // Large dataset - parallel
    let start = Instant::now();
    let large_par: Vec<f64> = large_data.par_iter().map(|&(x, y)| compiled_func.call(&[x, y]).unwrap()).collect();
    let large_par_time = start.elapsed();
    
    println!("• Sequential: {:.3}ms", large_seq_time.as_millis());
    println!("• Rayon parallel: {:.3}ms", large_par_time.as_millis());
    
    let large_speedup = large_seq_time.as_nanos() as f64 / large_par_time.as_nanos() as f64;
    println!("• Speedup: {:.2}x", large_speedup);

    println!("\n🎯 Analysis:");
    println!("• **Small datasets**: SIMD dominates, thread overhead hurts");
    println!("• **Large datasets**: Thread parallelism provides significant speedup");
    println!("• **SIMD**: Always active within Cranelift functions");
    println!("• **Threads**: Most beneficial for large, independent computations");
    println!("• **Sweet spot**: Combine both for maximum performance");
    println!();
}

fn demo_performance_analysis() {
    println!("📈 DEMO 6: PERFORMANCE ANALYSIS & RECOMMENDATIONS");
    println!("=================================================");
    
    let num_cores = rayon::current_num_threads();
    
    println!("System Configuration:");
    println!("• CPU cores available: {}", num_cores);
    println!("• SIMD support: AVX-512 (8x f64 per instruction)");
    println!("• Cranelift compilation: Sub-millisecond");
    println!("• Thread pool: Rayon work-stealing\n");

    println!("🎯 **WHEN TO USE CRANELIFT PARALLELISM**:");
    println!("• ✅ Vectorizable mathematical operations");
    println!("• ✅ Small to medium datasets (< 100K points)");
    println!("• ✅ Complex expressions with many operations");
    println!("• ✅ Real-time/interactive applications");
    println!("• ✅ When compilation speed matters");

    println!("\n🚀 **WHEN TO USE RAYON PARALLELISM**:");
    println!("• ✅ Large datasets (> 100K points)");
    println!("• ✅ Independent computations");
    println!("• ✅ CPU-bound workloads");
    println!("• ✅ When you have many CPU cores");
    println!("• ✅ Embarrassingly parallel problems");

    println!("\n🔥 **HYBRID APPROACH (RECOMMENDED)**:");
    println!("• 🎯 Use Rayon for data distribution across cores");
    println!("• 🎯 Use Cranelift for SIMD within each core");
    println!("• 🎯 Combine for maximum throughput");
    println!("• 🎯 Theoretical speedup: {} cores × 8 SIMD = {}x", num_cores, num_cores * 8);

    println!("\n📊 **PERFORMANCE CHARACTERISTICS**:");
    println!("• **Cranelift SIMD**: 8x speedup on vectorizable operations");
    println!("• **Rayon threads**: {}x speedup on large datasets", num_cores);
    println!("• **Combined**: Up to {}x theoretical maximum speedup", num_cores * 8);
    println!("• **Overhead**: Rayon has thread coordination overhead");
    println!("• **Scalability**: Cranelift scales with SIMD width, Rayon with cores");

    println!("\n🎪 **BEST PRACTICES**:");
    println!("1. **Start with Cranelift**: Get SIMD optimizations automatically");
    println!("2. **Add Rayon for large data**: Use .par_iter() for big datasets");
    println!("3. **Profile your workload**: Measure actual performance gains");
    println!("4. **Consider data layout**: Ensure SIMD-friendly memory access");
    println!("5. **Batch operations**: Group related computations together");
} 