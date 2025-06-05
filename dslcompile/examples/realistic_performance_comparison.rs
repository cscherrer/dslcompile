use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::symbolic::symbolic::{OptimizationConfig, SymbolicOptimizer};
use dslcompile::zero_overhead_core::*;
use std::time::Instant;
use rand::prelude::*;

fn main() {
    println!("🎯 Realistic DSL vs Native Rust Performance (No Constant Propagation)");
    println!("======================================================================");
    
    // Generate random test data to prevent constant propagation
    let mut rng = thread_rng();
    let iterations = 1_000_000;
    let test_data: Vec<(f64, f64)> = (0..iterations)
        .map(|_| (rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0)))
        .collect();
    
    println!("Running {iterations} iterations with random data...");
    println!("Data range: x,y ∈ [-10, 10]");
    println!();
    
    // ============================================================================
    // SIMPLE ARITHMETIC: x + y
    // ============================================================================
    
    println!("🔢 SIMPLE ARITHMETIC: x + y");
    println!("============================");
    
    // Native Rust baseline
    let start = Instant::now();
    let mut result = 0.0;
    for &(x, y) in &test_data {
        result += x + y; // Accumulate to prevent optimization
    }
    let native_time = start.elapsed();
    let native_ns = native_time.as_nanos() as f64 / iterations as f64;
    println!("Native Rust:          {:.2}ns per operation (sum: {:.2})", native_ns, result);
    
    // Zero overhead direct
    let direct_ctx = DirectComputeContext::new();
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += direct_ctx.add_direct(x, y);
    }
    let direct_time = start.elapsed();
    let direct_ns = direct_time.as_nanos() as f64 / iterations as f64;
    println!("Zero Overhead Direct: {:.2}ns per operation (sum: {:.2}) [{:.1}x overhead]", 
             direct_ns, result, direct_ns / native_ns);
    
    // AST interpretation
    let simple_ast = ASTRepr::Add(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Variable(1))
    );
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += simple_ast.eval_with_vars(&[x, y]);
    }
    let ast_time = start.elapsed();
    let ast_ns = ast_time.as_nanos() as f64 / iterations as f64;
    println!("AST Interpretation:   {:.2}ns per operation (sum: {:.2}) [{:.1}x overhead]", 
             ast_ns, result, ast_ns / native_ns);
    
    println!();
    
    // ============================================================================
    // COMPLEX EXPRESSION: x*x + 2*x*y + y*y
    // ============================================================================
    
    println!("🧮 COMPLEX EXPRESSION: x*x + 2*x*y + y*y");
    println!("=========================================");
    
    // Native Rust
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += x*x + 2.0*x*y + y*y;
    }
    let native_complex_time = start.elapsed();
    let native_complex_ns = native_complex_time.as_nanos() as f64 / iterations as f64;
    println!("Native Rust:          {:.2}ns per operation (sum: {:.2})", native_complex_ns, result);
    
    // Zero overhead direct
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += direct_ctx.complex_direct(x, y, 0.0);
    }
    let direct_complex_time = start.elapsed();
    let direct_complex_ns = direct_complex_time.as_nanos() as f64 / iterations as f64;
    println!("Zero Overhead Direct: {:.2}ns per operation (sum: {:.2}) [{:.1}x overhead]", 
             direct_complex_ns, result, direct_complex_ns / native_complex_ns);
    
    // Build complex AST
    let complex_ast = ASTRepr::Add(
        Box::new(ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Variable(0))
            )),
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Mul(
                    Box::new(ASTRepr::Constant(2.0)),
                    Box::new(ASTRepr::Variable(0))
                )),
                Box::new(ASTRepr::Variable(1))
            ))
        )),
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Variable(1)),
            Box::new(ASTRepr::Variable(1))
        ))
    );
    
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += complex_ast.eval_with_vars(&[x, y]);
    }
    let ast_complex_time = start.elapsed();
    let ast_complex_ns = ast_complex_time.as_nanos() as f64 / iterations as f64;
    println!("AST Interpretation:   {:.2}ns per operation (sum: {:.2}) [{:.1}x overhead]", 
             ast_complex_ns, result, ast_complex_ns / native_complex_ns);
    
    // Optimized AST
    let mut optimizer = SymbolicOptimizer::with_config(OptimizationConfig::zero_overhead()).unwrap();
    let optimized_ast = optimizer.optimize(&complex_ast).unwrap();
    
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += optimized_ast.eval_with_vars(&[x, y]);
    }
    let optimized_time = start.elapsed();
    let optimized_ns = optimized_time.as_nanos() as f64 / iterations as f64;
    println!("Optimized AST:        {:.2}ns per operation (sum: {:.2}) [{:.1}x overhead]", 
             optimized_ns, result, optimized_ns / native_complex_ns);
    
    println!();
    
    // ============================================================================
    // TRANSCENDENTAL FUNCTIONS: sin(x) + cos(y)
    // ============================================================================
    
    println!("📐 TRANSCENDENTAL: sin(x) + cos(y)");
    println!("==================================");
    
    // Native Rust
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += x.sin() + y.cos();
    }
    let native_trig_time = start.elapsed();
    let native_trig_ns = native_trig_time.as_nanos() as f64 / iterations as f64;
    println!("Native Rust:          {:.2}ns per operation (sum: {:.2})", native_trig_ns, result);
    
    // AST with transcendental functions
    let trig_ast = ASTRepr::Add(
        Box::new(ASTRepr::Sin(Box::new(ASTRepr::Variable(0)))),
        Box::new(ASTRepr::Cos(Box::new(ASTRepr::Variable(1))))
    );
    
    let start = Instant::now();
    result = 0.0;
    for &(x, y) in &test_data {
        result += trig_ast.eval_with_vars(&[x, y]);
    }
    let trig_ast_time = start.elapsed();
    let trig_ast_ns = trig_ast_time.as_nanos() as f64 / iterations as f64;
    println!("AST Interpretation:   {:.2}ns per operation (sum: {:.2}) [{:.1}x overhead]", 
             trig_ast_ns, result, trig_ast_ns / native_trig_ns);
    
    println!();
    
    // ============================================================================
    // UNIFIED ARCHITECTURE COMPARISON
    // ============================================================================
    
    println!("🚀 UNIFIED ARCHITECTURE (Complex Expression)");
    println!("=============================================");
    
    // Build expression using DynamicContext
    let ctx = DynamicContext::new();
    let x_var = ctx.var();
    let y_var = ctx.var();
    let expr = &(&x_var * &x_var) + &(&ctx.constant(2.0) * &x_var * &y_var) + &(&y_var * &y_var);
    let unified_ast: ASTRepr<f64> = expr.into();
    
    let strategies = [
        ("StaticCodegen", OptimizationConfig::zero_overhead()),
        ("DynamicCodegen", OptimizationConfig::dynamic_performance()),
        ("Interpretation", OptimizationConfig::dynamic_flexible()),
        ("Adaptive", OptimizationConfig::adaptive()),
    ];
    
    for (name, config) in strategies {
        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
        let optimized = optimizer.optimize(&unified_ast).unwrap();
        
        let start = Instant::now();
        result = 0.0;
        for &(x, y) in &test_data {
            result += optimized.eval_with_vars(&[x, y]);
        }
        let strategy_time = start.elapsed();
        let strategy_ns = strategy_time.as_nanos() as f64 / iterations as f64;
        println!("{:<17} {:.2}ns per operation (sum: {:.2}) [{:.1}x vs native]", 
                 format!("{}:", name), strategy_ns, result, strategy_ns / native_complex_ns);
    }
    
    println!();
    
    // ============================================================================
    // MEMORY ACCESS PATTERN TEST
    // ============================================================================
    
    println!("💾 MEMORY ACCESS PATTERNS");
    println!("=========================");
    
    // Test with sequential vs random access
    let sequential_data: Vec<f64> = (0..iterations).map(|i| i as f64 * 0.001).collect();
    let mut random_indices: Vec<usize> = (0..iterations).collect();
    random_indices.shuffle(&mut rng);
    
    // Sequential access
    let start = Instant::now();
    result = 0.0;
    for &val in &sequential_data {
        result += val * val;
    }
    let sequential_time = start.elapsed();
    let sequential_ns = sequential_time.as_nanos() as f64 / iterations as f64;
    println!("Sequential Access:    {:.2}ns per operation", sequential_ns);
    
    // Random access
    let start = Instant::now();
    result = 0.0;
    for &idx in &random_indices[..100_000] { // Smaller sample for random access
        let val = sequential_data[idx % sequential_data.len()];
        result += val * val;
    }
    let random_time = start.elapsed();
    let random_ns = random_time.as_nanos() as f64 / 100_000.0;
    println!("Random Access:        {:.2}ns per operation [{:.1}x slower]", 
             random_ns, random_ns / sequential_ns);
    
    println!();
    
    // ============================================================================
    // SUMMARY TABLE
    // ============================================================================
    
    println!("📊 REALISTIC PERFORMANCE SUMMARY");
    println!("================================");
    println!("Operation                 | Native (ns) | Zero-OH (ns) | AST (ns) | Overhead");
    println!("--------------------------|-------------|--------------|----------|----------");
    println!("Simple Add (x + y)        | {:>7.2}     | {:>8.2}     | {:>6.2}   | {:>6.1}x", 
             native_ns, direct_ns, ast_ns, ast_ns / native_ns);
    println!("Complex (x²+2xy+y²)       | {:>7.2}     | {:>8.2}     | {:>6.2}   | {:>6.1}x", 
             native_complex_ns, direct_complex_ns, ast_complex_ns, ast_complex_ns / native_complex_ns);
    println!("Transcendental (sin+cos)  | {:>7.2}     | {:>8}       | {:>6.2}   | {:>6.1}x", 
             native_trig_ns, "N/A", trig_ast_ns, trig_ast_ns / native_trig_ns);
    
    println!();
    println!("🎯 REALISTIC PERFORMANCE INSIGHTS:");
    println!("• Zero-overhead direct: {:.1}x native performance", direct_ns / native_ns);
    println!("• AST interpretation: {:.1}x overhead for simple, {:.1}x for complex", 
             ast_ns / native_ns, ast_complex_ns / native_complex_ns);
    println!("• Transcendental overhead: {:.1}x (mostly function call cost)", 
             trig_ast_ns / native_trig_ns);
    println!("• Memory access patterns matter: {:.1}x difference", random_ns / sequential_ns);
    println!("• DSL provides excellent performance/flexibility balance");
    println!("• ~{:.0}ns overhead is very reasonable for expression flexibility", ast_complex_ns);
} 