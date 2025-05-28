//! JIT Compilation Demo
//!
//! This example demonstrates the JIT compilation capabilities of `MathJIT` using the final tagless approach.
//! It shows how to define mathematical expressions and compile them to native code for high performance.

use mathjit::{ASTEval, ASTMathExpr, JITCompiler, Result};

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

    // Demo 5: Two-variable JIT compilation
    demo_two_variables()?;

    // Demo 6: Multi-variable JIT compilation
    demo_multi_variables()?;

    // Demo 7: Maximum variables (6 variables)
    demo_max_variables()?;

    Ok(())
}

/// Demo 1: Simple linear expression (2x + 3)
fn demo_linear_expression() -> Result<()> {
    println!("📊 Demo 1: Linear Expression (2x + 3)");
    println!("--------------------------------------");

    // Define the expression using the final tagless approach
    let expr = ASTEval::add(
        ASTEval::mul(ASTEval::constant(2.0), ASTEval::var_by_name("x")),
        ASTEval::constant(3.0),
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
        println!("  f({x:.1}) = {result:.1} (expected: {expected:.1})");
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
    let x = ASTEval::var_by_name("x");
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::pow(x.clone(), ASTEval::constant(2.0)),
            ASTEval::mul(ASTEval::constant(2.0), x),
        ),
        ASTEval::constant(1.0),
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
        println!("  f({x:.1}) = {result:.1} (expected: {expected:.1})");
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
    let x = ASTEval::var_by_name("x");
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::pow(x.clone(), ASTEval::constant(2.0)),
            ASTEval::sin(x.clone()),
        ),
        ASTEval::sqrt(x),
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
        println!("  f({x:.1}) = {result:.6} (placeholder implementation)");
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
    let x = ASTEval::var_by_name("x");
    let expr = ASTEval::sub(
        ASTEval::add(
            ASTEval::sub(
                ASTEval::mul(
                    ASTEval::constant(3.0),
                    ASTEval::pow(x.clone(), ASTEval::constant(3.0)),
                ),
                ASTEval::mul(
                    ASTEval::constant(2.0),
                    ASTEval::pow(x.clone(), ASTEval::constant(2.0)),
                ),
            ),
            x,
        ),
        ASTEval::constant(5.0),
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

    println!("Performance comparison ({iterations} iterations):");
    println!(
        "  JIT compiled:  {:.2?} ({:.1} ns/call)",
        jit_time,
        jit_time.as_nanos() as f64 / f64::from(iterations)
    );
    println!(
        "  Native Rust:   {:.2?} ({:.1} ns/call)",
        native_time,
        native_time.as_nanos() as f64 / f64::from(iterations)
    );

    let speedup = native_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
    if speedup > 1.0 {
        println!("  🚀 JIT is {speedup:.1}x faster than native!");
    } else {
        println!(
            "  📊 JIT is {:.1}x slower than native (expected for simple expressions)",
            1.0 / speedup
        );
    }

    println!(
        "  Results match: {}",
        (jit_result - native_result).abs() < 1e-10
    );
    println!("📈 JIT compilation stats: {:?}", jit_func.stats);

    Ok(())
}

/// Demo 5: Two-variable JIT compilation
fn demo_two_variables() -> Result<()> {
    println!("📊 Demo 5: Two Variables (x² + 2xy + y²)");
    println!("------------------------------------------");

    // Define a two-variable expression: x² + 2xy + y² = (x + y)²
    let x = ASTEval::var_by_name("x");
    let y = ASTEval::var_by_name("y");
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::pow(x.clone(), ASTEval::constant(2.0)),
            ASTEval::mul(ASTEval::mul(ASTEval::constant(2.0), x), y.clone()),
        ),
        ASTEval::pow(y, ASTEval::constant(2.0)),
    );

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_two_vars(&expr, "x", "y")?;

    // Test the compiled function
    let test_pairs = [(1.0, 2.0), (2.0, 3.0), (-1.0, 1.0), (0.5, 1.5)];
    println!("Testing compiled two-variable function:");
    for (x, y) in test_pairs {
        let result = jit_func.call_two_vars(x, y);
        let expected = (x + y).powi(2);
        println!("  f({x:.1}, {y:.1}) = {result:.1} (expected: {expected:.1})");
        assert!((result - expected).abs() < 1e-10);
    }

    println!("✅ All tests passed!");
    println!("📈 Compilation stats: {:?}\n", jit_func.stats);

    Ok(())
}

/// Demo 6: Multi-variable JIT compilation
fn demo_multi_variables() -> Result<()> {
    println!("📊 Demo 6: Multiple Variables (x*y + y*z + z*x)");
    println!("-----------------------------------------------");

    // Define a three-variable expression: x*y + y*z + z*x
    let x = ASTEval::var_by_name("x");
    let y = ASTEval::var_by_name("y");
    let z = ASTEval::var_by_name("z");
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::mul(x.clone(), y.clone()),
            ASTEval::mul(y, z.clone()),
        ),
        ASTEval::mul(z, x),
    );

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_multi_vars(&expr, &["x", "y", "z"])?;

    // Test the compiled function
    let test_triples = [
        (1.0, 2.0, 3.0),
        (2.0, 3.0, 4.0),
        (0.5, 1.0, 1.5),
        (-1.0, 2.0, -3.0),
    ];
    println!("Testing compiled multi-variable function:");
    for (x, y, z) in test_triples {
        let result = jit_func.call_multi_vars(&[x, y, z]);
        let expected = x * y + y * z + z * x;
        println!("  f({x:.1}, {y:.1}, {z:.1}) = {result:.1} (expected: {expected:.1})");
        assert!((result - expected).abs() < 1e-10);
    }

    println!("✅ All tests passed!");
    println!("📈 Compilation stats: {:?}\n", jit_func.stats);

    Ok(())
}

/// Demo 7: Maximum variables (6 variables)
fn demo_max_variables() -> Result<()> {
    println!("📊 Demo 7: Maximum Variables (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)");
    println!("----------------------------------------------------------");

    // Define a six-variable expression: sum of all variables
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::add(
                    ASTEval::add(ASTEval::var_by_name("x1"), ASTEval::var_by_name("x2")),
                    ASTEval::var_by_name("x3"),
                ),
                ASTEval::var_by_name("x4"),
            ),
            ASTEval::var_by_name("x5"),
        ),
        ASTEval::var_by_name("x6"),
    );

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_multi_vars(&expr, &["x1", "x2", "x3", "x4", "x5", "x6"])?;

    // Test the compiled function
    let test_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = jit_func.call_multi_vars(&test_values);
    let expected: f64 = test_values.iter().sum();

    println!("Testing compiled six-variable function:");
    println!(
        "  f({:.1}, {:.1}, {:.1}, {:.1}, {:.1}, {:.1}) = {:.1} (expected: {:.1})",
        test_values[0],
        test_values[1],
        test_values[2],
        test_values[3],
        test_values[4],
        test_values[5],
        result,
        expected
    );
    assert!((result - expected).abs() < 1e-10);

    println!("✅ Test passed!");
    println!("📈 Compilation stats: {:?}\n", jit_func.stats);

    Ok(())
}
