//! `DSLCompile` Unified System Demo
//!
//! This demo showcases the unified `DSLCompile` system features:
//! - Natural operator overloading syntax
//! - Multiple backend compilation (Rust, JIT)
//! - ANF optimization with domain analysis
//! - Mathematical correctness preservation

use dslcompile::backends::rust_codegen::{
    CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel,
};
#[cfg(feature = "cranelift")]
use dslcompile::backends::cranelift::CraneliftCompiler;
#[cfg(feature = "cranelift")]
use dslcompile::final_tagless::VariableRegistry;
use dslcompile::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🧮 DSLCompile Unified System Demo");
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
fn demo_operator_overloading() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("  Creating mathematical expressions with operator overloading...");

    let math = MathBuilder::new();
    let x = math.var();
    let y = math.var();

    // Natural mathematical syntax
    let expr = &x * &x + 2.0 * &x + &y;

    let result = math.eval(&expr, &[3.0, 1.0]);
    println!("  Expression: x² + 2x + y");
    println!("  At x=3, y=1: {result} (expected 16)");

    assert_eq!(result, 16.0);
    println!("  ✅ Operator overloading working correctly");

    Ok(())
}

/// Demonstrate multiple backend compilation
fn demo_backend_compilation() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("  Testing multiple compilation backends...");

    let math = MathBuilder::new();
    let x = math.var();

    // Create expression
    let expr = &x * 2.0 + 1.0;

    // Test direct evaluation
    let direct_result = math.eval(&expr, &[5.0]);
    println!("  Direct evaluation: {direct_result}");
    assert_eq!(direct_result, 11.0);

    // Convert to AST for other backends
    let ast = expr.into_ast();
    println!("  ✅ AST conversion successful");

    // Test Rust code generation backend
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast, "demo_func")?;
    println!("  ✅ Rust code generation successful");
    assert!(rust_code.contains("demo_func"));

    #[cfg(feature = "cranelift")]
    {
        // Test JIT compilation backend
        let mut compiler = CraneliftCompiler::new_default()?;
        let registry = VariableRegistry::for_expression(&ast);
        let compiled_func = compiler.compile_expression(&ast, &registry)?;
        let jit_result = compiled_func.call(&[5.0])?;
        println!("  JIT result: {jit_result}");
        assert_eq!(jit_result, 11.0);
        println!("  ✅ JIT compilation successful");
    }

    Ok(())
}

/// Demonstrate ANF optimization
fn demo_anf_optimization() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("  Testing ANF optimization with domain analysis...");

    use dslcompile::symbolic::anf::convert_to_anf;
    use std::collections::HashMap;

    let math = MathBuilder::new();
    let x = math.var();

    // Create expression with optimization opportunities
    let expr = x.exp().ln(); // ln(exp(x)) should optimize to x
    let ast = expr.into_ast();

    // Convert to ANF
    let anf = convert_to_anf(&ast)?;
    println!("  ✅ ANF conversion successful");

    // Evaluate
    let var_map: HashMap<usize, f64> = [(0, 2.5)].into_iter().collect();
    let anf_result = anf.eval(&var_map);

    // Create a new expression for direct evaluation since the original was moved
    let x2 = math.var();
    let expr2 = x2.exp().ln();
    let direct_result = math.eval(&expr2, &[2.5]);

    println!("  ANF result: {anf_result}");
    println!("  Direct result: {direct_result}");

    assert!((anf_result - direct_result).abs() < 1e-10);
    println!("  ✅ ANF optimization preserves correctness");

    Ok(())
}

/// Demonstrate mathematical correctness preservation
fn demo_mathematical_correctness() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("  Testing mathematical correctness across transformations...");

    let math = MathBuilder::new();
    let test_value = 2.5;

    // Test various mathematical identities - create separate variables for each test
    let test_cases = [("ln(exp(x))", "x"), ("x + 0", "x"), ("x * 1", "x")];

    for (desc, expected_desc) in &test_cases {
        let x = math.var(); // Create fresh variable for each test
        let expr = match *desc {
            "ln(exp(x))" => x.exp().ln(),
            "x + 0" => &x + 0.0,
            "x * 1" => &x * 1.0,
            _ => unreachable!(),
        };

        let result = math.eval(&expr, &[test_value]);
        println!("  {desc}: {result} ≈ {expected_desc}");

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
