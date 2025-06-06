use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::symbolic::symbolic::{OptimizationConfig, SymbolicOptimizer};
use dslcompile::zero_overhead_core::DirectComputeContext;
use std::time::Instant;

fn main() {
    println!("🎯 Focused DSL vs Native Rust Performance Analysis");
    println!("==================================================");
    
    // Use more iterations for better timing resolution
    let iterations = 10_000_000;
    let x = 1.5;
    let y = 2.5;
    
    println!("Running {iterations} iterations for accurate timing...");
    println!("Test values: x = {x}, y = {y}");
    println!();
    
    // ============================================================================
    // SIMPLE ARITHMETIC: x + y
    // ============================================================================
    
    println!("🔢 SIMPLE ARITHMETIC: x + y");
    println!("============================");
    
    // Native Rust baseline
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result = x + y;
    }
    let native_time = start.elapsed();
    let native_ns = native_time.as_nanos() as f64 / f64::from(iterations);
    println!("Native Rust:          {native_ns:.2}ns per operation → {result}");
    
    // Zero overhead direct
    let direct_ctx = DirectComputeContext::new();
    let start = Instant::now();
    for _ in 0..iterations {
        result = direct_ctx.add_direct(x, y);
    }
    let direct_time = start.elapsed();
    let direct_ns = direct_time.as_nanos() as f64 / f64::from(iterations);
    println!("Zero Overhead Direct: {:.2}ns per operation → {result} ({:.1}x overhead)", 
             direct_ns, direct_ns / native_ns);
    
    // AST interpretation
    let simple_ast = ASTRepr::Add(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Variable(1))
    );
    let values = [x, y];
    let start = Instant::now();
    for _ in 0..iterations {
        result = simple_ast.eval_with_vars(&values);
    }
    let ast_time = start.elapsed();
    let ast_ns = ast_time.as_nanos() as f64 / f64::from(iterations);
    println!("AST Interpretation:   {:.2}ns per operation → {result} ({:.1}x overhead)", 
             ast_ns, ast_ns / native_ns);
    
    println!();
    
    // ============================================================================
    // COMPLEX EXPRESSION: x*x + 2*x*y + y*y
    // ============================================================================
    
    println!("🧮 COMPLEX EXPRESSION: x*x + 2*x*y + y*y");
    println!("=========================================");
    
    // Native Rust
    let start = Instant::now();
    for _ in 0..iterations {
        result = x*x + 2.0*x*y + y*y;
    }
    let native_complex_time = start.elapsed();
    let native_complex_ns = native_complex_time.as_nanos() as f64 / f64::from(iterations);
    println!("Native Rust:          {native_complex_ns:.2}ns per operation → {result}");
    
    // Zero overhead direct
    let start = Instant::now();
    for _ in 0..iterations {
        result = direct_ctx.complex_direct(x, y, 0.0);
    }
    let direct_complex_time = start.elapsed();
    let direct_complex_ns = direct_complex_time.as_nanos() as f64 / f64::from(iterations);
    println!("Zero Overhead Direct: {:.2}ns per operation → {result} ({:.1}x overhead)", 
             direct_complex_ns, direct_complex_ns / native_complex_ns);
    
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
    for _ in 0..iterations {
        result = complex_ast.eval_with_vars(&values);
    }
    let ast_complex_time = start.elapsed();
    let ast_complex_ns = ast_complex_time.as_nanos() as f64 / f64::from(iterations);
    println!("AST Interpretation:   {:.2}ns per operation → {result} ({:.1}x overhead)", 
             ast_complex_ns, ast_complex_ns / native_complex_ns);
    
    // Optimized AST
    let mut optimizer = SymbolicOptimizer::with_config(OptimizationConfig::zero_overhead()).unwrap();
    let optimized_ast = optimizer.optimize(&complex_ast).unwrap();
    
    let start = Instant::now();
    for _ in 0..iterations {
        result = optimized_ast.eval_with_vars(&values);
    }
    let optimized_time = start.elapsed();
    let optimized_ns = optimized_time.as_nanos() as f64 / f64::from(iterations);
    println!("Optimized AST:        {:.2}ns per operation → {result} ({:.1}x overhead)", 
             optimized_ns, optimized_ns / native_complex_ns);
    
    println!();
    
    // ============================================================================
    // TRANSCENDENTAL FUNCTIONS: sin(x) + cos(y)
    // ============================================================================
    
    println!("📐 TRANSCENDENTAL: sin(x) + cos(y)");
    println!("==================================");
    
    // Native Rust
    let start = Instant::now();
    for _ in 0..iterations {
        result = x.sin() + y.cos();
    }
    let native_trig_time = start.elapsed();
    let native_trig_ns = native_trig_time.as_nanos() as f64 / f64::from(iterations);
    println!("Native Rust:          {native_trig_ns:.2}ns per operation → {result}");
    
    // AST with transcendental functions
    let trig_ast = ASTRepr::Add(
        Box::new(ASTRepr::Sin(Box::new(ASTRepr::Variable(0)))),
        Box::new(ASTRepr::Cos(Box::new(ASTRepr::Variable(1))))
    );
    
    let start = Instant::now();
    for _ in 0..iterations {
        result = trig_ast.eval_with_vars(&values);
    }
    let trig_ast_time = start.elapsed();
    let trig_ast_ns = trig_ast_time.as_nanos() as f64 / f64::from(iterations);
    println!("AST Interpretation:   {:.2}ns per operation → {result} ({:.1}x overhead)", 
             trig_ast_ns, trig_ast_ns / native_trig_ns);
    
    println!();
    
    // ============================================================================
    // CONSTANT FOLDING DEMONSTRATION
    // ============================================================================
    
    println!("🎯 CONSTANT FOLDING: 3*4 + 5*6");
    println!("===============================");
    
    // Native constant
    let start = Instant::now();
    for _ in 0..iterations {
        result = 3.0*4.0 + 5.0*6.0; // = 42
    }
    let native_const_time = start.elapsed();
    let native_const_ns = native_const_time.as_nanos() as f64 / f64::from(iterations);
    println!("Native Constant:      {native_const_ns:.2}ns per operation → {result}");
    
    // Unoptimized AST
    let const_ast = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(3.0)),
            Box::new(ASTRepr::Constant(4.0))
        )),
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(5.0)),
            Box::new(ASTRepr::Constant(6.0))
        ))
    );
    
    let start = Instant::now();
    for _ in 0..iterations {
        result = const_ast.eval_with_vars(&[]);
    }
    let const_ast_time = start.elapsed();
    let const_ast_ns = const_ast_time.as_nanos() as f64 / f64::from(iterations);
    println!("Unoptimized AST:      {:.2}ns per operation → {result} ({:.1}x overhead)", 
             const_ast_ns, const_ast_ns / native_const_ns);
    
    // Optimized (folded) AST
    let mut optimizer = SymbolicOptimizer::with_config(OptimizationConfig::zero_overhead()).unwrap();
    let folded_ast = optimizer.optimize(&const_ast).unwrap();
    
    println!("Folded AST: {folded_ast:?}");
    
    let start = Instant::now();
    for _ in 0..iterations {
        result = folded_ast.eval_with_vars(&[]);
    }
    let folded_time = start.elapsed();
    let folded_ns = folded_time.as_nanos() as f64 / f64::from(iterations);
    println!("Folded AST:           {:.2}ns per operation → {result} ({:.1}x overhead)", 
             folded_ns, folded_ns / native_const_ns);
    
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
        for _ in 0..iterations {
            result = optimized.eval_with_vars(&values);
        }
        let strategy_time = start.elapsed();
        let strategy_ns = strategy_time.as_nanos() as f64 / f64::from(iterations);
        println!("{:<17} {:.2}ns per operation → {result} ({:.1}x vs native)", 
                 format!("{}:", name), strategy_ns, strategy_ns / native_complex_ns);
    }
    
    println!();
    
    // ============================================================================
    // SUMMARY TABLE
    // ============================================================================
    
    println!("📊 PERFORMANCE SUMMARY TABLE");
    println!("============================");
    println!("Operation                 | Native (ns) | DSL (ns) | Overhead");
    println!("--------------------------|-------------|----------|----------");
    println!("Simple Add (x + y)        | {:>7.2}     | {:>6.2}   | {:>6.1}x", 
             native_ns, direct_ns, direct_ns / native_ns);
    println!("Complex (x²+2xy+y²)       | {:>7.2}     | {:>6.2}   | {:>6.1}x", 
             native_complex_ns, direct_complex_ns, direct_complex_ns / native_complex_ns);
    println!("Transcendental (sin+cos)  | {:>7.2}     | {:>6.2}   | {:>6.1}x", 
             native_trig_ns, trig_ast_ns, trig_ast_ns / native_trig_ns);
    println!("Constant Folding          | {:>7.2}     | {:>6.2}   | {:>6.1}x", 
             native_const_ns, folded_ns, folded_ns / native_const_ns);
    
    println!();
    println!("🎯 KEY TAKEAWAYS:");
    println!("• Zero-overhead direct operations: ~{:.1}x native performance", direct_ns / native_ns);
    println!("• AST interpretation overhead: ~{:.1}x for simple ops, ~{:.1}x for complex", 
             ast_ns / native_ns, ast_complex_ns / native_complex_ns);
    println!("• Constant folding achieves near-native performance: ~{:.1}x", folded_ns / native_const_ns);
    println!("• Transcendental functions have consistent overhead: ~{:.1}x", trig_ast_ns / native_trig_ns);
    println!("• DSL provides excellent performance/flexibility tradeoff");
} 