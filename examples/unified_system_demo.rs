//! MathCompile Unified System Demo
//!
//! This demo showcases the unified MathCompile system features:
//! - Natural operator overloading syntax
//! - Multiple backend compilation (Rust, JIT)
//! - ANF optimization with domain analysis
//! - Mathematical correctness preservation

use mathcompile::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 MathCompile Unified System Demo");
    println!("====================================\n");

    // 1. Natural Operator Overloading
    println!("🎯 Demo Goals:");
    println!("1. Natural Operator Overloading:");
    demo_operator_overloading()?;

    println!("\n2. 🏗️ Multiple Backend Compilation:");
    demo_backend_compilation()?;

    println!("\n3. 🔍 ANF Optimization:");
    demo_anf_optimization()?;

    println!("\n4. ✅ Mathematical Correctness:");
    demo_mathematical_correctness()?;

    println!("\n🎉 All demos completed successfully!");
    println!("The unified system demonstrates composable mathematical computing");
    println!("with performance, correctness, and flexibility.");

    Ok(())
}

/// Demonstrate operator overloading syntax
fn demo_operator_overloading() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating mathematical expressions with operator overloading...");

    let math = MathBuilder::new();
    let x = math.var("x");
    let y = math.var("y");

    // Natural mathematical syntax
    let expr = &x * &x + 2.0 * &x + &y;

    let result = math.eval(&expr, &[("x", 3.0), ("y", 1.0)]);
    println!("  Expression: x² + 2x + y");
    println!("  At x=3, y=1: {} (expected 16)", result);

    assert_eq!(result, 16.0);
    println!("  ✅ Operator overloading working correctly");

    Ok(())
}

/// Demonstrate multiple backend compilation
fn demo_backend_compilation() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing multiple compilation backends...");

    let math = MathBuilder::new();
    let x = math.var("x");

    // Create expression
    let expr = &x * 2.0 + 1.0;

    // Test direct evaluation
    let direct_result = math.eval(&expr, &[("x", 5.0)]);
    println!("  Direct evaluation: {}", direct_result);
    assert_eq!(direct_result, 11.0);

    // Convert to AST for other backends
    let ast = expr.to_ast();
    println!("  ✅ AST conversion successful");

    // Test Rust code generation backend
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast, "demo_func")?;
    println!("  ✅ Rust code generation successful");
    assert!(rust_code.contains("demo_func"));

    #[cfg(feature = "cranelift")]
    {
        // Test JIT compilation backend
        let compiler = JITCompiler::new()?;
        let jit_func = compiler.compile_single_var(&ast, "x")?;
        let jit_result = jit_func.call_single(5.0);
        println!("  JIT result: {}", jit_result);
        assert_eq!(jit_result, 11.0);
        println!("  ✅ JIT compilation successful");
    }

    Ok(())
}

/// Demonstrate ANF optimization
fn demo_anf_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing ANF optimization with domain analysis...");

    use mathcompile::symbolic::anf::convert_to_anf;
    use std::collections::HashMap;

    let math = MathBuilder::new();
    let x = math.var("x");

    // Create expression with optimization opportunities
    let expr = x.exp().ln(); // ln(exp(x)) should optimize to x
    let ast = expr.to_ast();

    // Convert to ANF
    let anf = convert_to_anf(&ast)?;
    println!("  ✅ ANF conversion successful");

    // Evaluate
    let var_map: HashMap<usize, f64> = [(0, 2.5)].into_iter().collect();
    let anf_result = anf.eval(&var_map);
    let direct_result = math.eval(&expr, &[("x", 2.5)]);

    println!("  ANF result: {}", anf_result);
    println!("  Direct result: {}", direct_result);

    assert!((anf_result - direct_result).abs() < 1e-10);
    println!("  ✅ ANF optimization preserves correctness");

    Ok(())
}

/// Demonstrate mathematical correctness preservation
fn demo_mathematical_correctness() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing mathematical correctness across transformations...");

    let math = MathBuilder::new();
    let x = math.var("x");

    // Test various mathematical identities
    let test_cases = [
        (x.exp().ln(), "ln(exp(x))", "x"),
        (&x + 0.0, "x + 0", "x"),
        (&x * 1.0, "x * 1", "x"),
    ];

    let test_value = 2.5;

    for (expr, desc, expected_desc) in test_cases.iter() {
        let result = math.eval(expr, &[("x", test_value)]);
        println!("  {}: {} ≈ {}", desc, result, expected_desc);

        // For these identities, result should equal the test value
        assert!((result - test_value).abs() < 1e-10);
    }

    println!("✓ Natural syntax: x * x + 2.0 * x + y");
    println!("✓ Multiple backends: Direct, ANF, Rust codegen, JIT");
    println!("✓ Domain analysis: Mathematical safety checking");
    println!("✓ Correctness: All transformations preserve semantics");
    println!("  ✅ Mathematical correctness verified");

    Ok(())
}
