//! JIT Compilation Demo
//!
//! This example demonstrates the JIT compilation capabilities of MathJIT using the final tagless approach.
//! It shows how to define mathematical expressions and compile them to native code for high performance.

use mathjit::{JITCompiler, JITEval, JITMathExpr, Result};

fn main() -> Result<()> {
    println!("🚀 MathJIT - JIT Compilation Demo");
    println!("==================================\n");

    // Demo 1: Simple linear expression
    demo_linear_expression()?;
    
    // Demo 2: Quadratic polynomial
    demo_quadratic_polynomial()?;
    
    // Demo 3: Complex mathematical expression
    demo_complex_expression()?;
    
    // Demo 4: Performance comparison
    demo_performance_comparison()?;

    Ok(())
}

/// Demo 1: Simple linear expression (2x + 3)
fn demo_linear_expression() -> Result<()> {
    println!("📊 Demo 1: Linear Expression (2x + 3)");
    println!("--------------------------------------");

    // Define the expression using the final tagless approach
    let expr = JITEval::add(
        JITEval::mul(JITEval::constant(2.0), JITEval::var("x")),
        JITEval::constant(3.0)
    );

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&expr, "x")?;

    // Test the compiled function
    let test_values = [0.0, 1.0, 2.0, 5.0, -1.0];
    println!("Testing compiled function:");
    for x in test_values {
        let result = jit_func.call_single(x);
        let expected = 2.0 * x + 3.0;
        println!("  f({:.1}) = {:.1} (expected: {:.1})", x, result, expected);
        assert!((result - expected).abs() < 1e-10);
    }

    println!("✅ All tests passed!");
    println!("📈 Compilation stats: {:?}\n", jit_func.stats);

    Ok(())
}

/// Demo 2: Quadratic polynomial (x² + 2x + 1)
fn demo_quadratic_polynomial() -> Result<()> {
    println!("📊 Demo 2: Quadratic Polynomial (x² + 2x + 1)");
    println!("----------------------------------------------");

    // Define the quadratic expression
    let x = JITEval::var("x");
    let expr = JITEval::add(
        JITEval::add(
            JITEval::pow(x.clone(), JITEval::constant(2.0)),
            JITEval::mul(JITEval::constant(2.0), x)
        ),
        JITEval::constant(1.0)
    );

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&expr, "x")?;

    // Test the compiled function
    let test_values = [0.0, 1.0, 2.0, -1.0, 3.0];
    println!("Testing compiled quadratic function:");
    for x in test_values {
        let result = jit_func.call_single(x);
        let expected = x * x + 2.0 * x + 1.0;
        println!("  f({:.1}) = {:.1} (expected: {:.1})", x, result, expected);
        assert!((result - expected).abs() < 1e-10);
    }

    println!("✅ All tests passed!");
    println!("📈 Compilation stats: {:?}\n", jit_func.stats);

    Ok(())
}

/// Demo 3: Complex mathematical expression with transcendental functions
fn demo_complex_expression() -> Result<()> {
    println!("📊 Demo 3: Complex Expression (x² + sin(x) + sqrt(x))");
    println!("----------------------------------------------------");

    // Define a complex expression: x² + sin(x) + sqrt(x)
    let x = JITEval::var("x");
    let expr = JITEval::add(
        JITEval::add(
            JITEval::pow(x.clone(), JITEval::constant(2.0)),
            JITEval::sin(x.clone())
        ),
        JITEval::sqrt(x)
    );

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&expr, "x")?;

    // Test the compiled function
    let test_values = [1.0, 2.0, 4.0, 9.0];
    println!("Testing compiled complex function:");
    for x in test_values {
        let result = jit_func.call_single(x);
        // Note: Our placeholder implementations don't actually compute sin/sqrt correctly
        // In a real implementation, these would be properly implemented
        println!("  f({:.1}) = {:.6} (placeholder implementation)", x, result);
    }

    println!("⚠️  Note: Transcendental functions use placeholder implementations");
    println!("📈 Compilation stats: {:?}\n", jit_func.stats);

    Ok(())
}

/// Demo 4: Performance comparison between direct evaluation and JIT
fn demo_performance_comparison() -> Result<()> {
    println!("📊 Demo 4: Performance Comparison");
    println!("----------------------------------");

    // Define a moderately complex polynomial: 3x³ - 2x² + x - 5
    let x = JITEval::var("x");
    let expr = JITEval::sub(
        JITEval::add(
            JITEval::sub(
                JITEval::mul(JITEval::constant(3.0), JITEval::pow(x.clone(), JITEval::constant(3.0))),
                JITEval::mul(JITEval::constant(2.0), JITEval::pow(x.clone(), JITEval::constant(2.0)))
            ),
            x
        ),
        JITEval::constant(5.0)
    );

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&expr, "x")?;

    // Performance test parameters
    let test_value = 2.5;
    let iterations = 1_000_000;

    // Test JIT performance
    let start = std::time::Instant::now();
    let mut jit_result = 0.0;
    for _ in 0..iterations {
        jit_result = jit_func.call_single(test_value);
    }
    let jit_time = start.elapsed();

    // Test native Rust performance (for comparison)
    let start = std::time::Instant::now();
    let mut native_result = 0.0;
    for _ in 0..iterations {
        let x = test_value;
        native_result = 3.0 * x * x * x - 2.0 * x * x + x - 5.0;
    }
    let native_time = start.elapsed();

    println!("Performance comparison ({} iterations):", iterations);
    println!("  JIT compiled:  {:.2?} ({:.1} ns/call)", jit_time, jit_time.as_nanos() as f64 / iterations as f64);
    println!("  Native Rust:   {:.2?} ({:.1} ns/call)", native_time, native_time.as_nanos() as f64 / iterations as f64);
    
    let speedup = native_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
    if speedup > 1.0 {
        println!("  🚀 JIT is {:.1}x faster than native!", speedup);
    } else {
        println!("  📊 JIT is {:.1}x slower than native (expected for simple expressions)", 1.0 / speedup);
    }

    println!("  Results match: {}", (jit_result - native_result).abs() < 1e-10);
    println!("📈 JIT compilation stats: {:?}", jit_func.stats);

    Ok(())
} 