//! README Example
//!
//! This example demonstrates all the key functionality shown in the README.
//! We test this example to ensure the README code actually works, then copy
//! the exact working code into the README.

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("=== DSLCompile README Examples ===\n");

    // Example 1: Symbolic → Numeric Optimization
    println!("1. Symbolic → Numeric Optimization");
    symbolic_to_numeric_example()?;

    println!("\n2. Basic Usage");
    basic_usage_example()?;

    println!("\n3. Automatic Differentiation");
    automatic_differentiation_example()?;

    println!("\n4. Multiple Compilation Backends");
    multiple_backends_example()?;

    println!("\n=== All README examples completed successfully ===");
    Ok(())
}

fn symbolic_to_numeric_example() -> Result<()> {
    // Define symbolic expression
    let math = MathBuilder::new();
    let x = math.var();
    let expr = math.poly(&[1.0, 2.0, 3.0], &x); // 1 + 2x + 3x² (coefficients in ascending order)

    // Evaluate with named variables
    let result = math.eval(&expr, &[3.0]);
    println!("  Direct evaluation: f(3) = {result}");
    assert_eq!(result, 34.0); // 1 + 2*3 + 3*3² = 1 + 6 + 27 = 34

    // Convert to AST for code generation
    let ast_expr = expr.into_ast();

    // Generate Rust code
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "my_function")?;
    println!(
        "  Generated Rust code (first 100 chars): {}",
        &rust_code[..100.min(rust_code.len())]
    );

    // Compile and load the function (paths auto-generated from function name)
    let compiler = RustCompiler::new();
    if RustCompiler::is_available() {
        let compiled_func = compiler.compile_and_load(&rust_code, "my_function")?;
        let compiled_result = compiled_func.call(3.0)?;
        println!("  Compiled function: f(3) = {compiled_result}");
        assert_eq!(compiled_result, result); // Should match direct evaluation
    } else {
        println!("  Rust compiler not available - skipping compilation");
    }

    Ok(())
}

fn basic_usage_example() -> Result<()> {
    // Create mathematical expressions
    let math = MathBuilder::new();
    let x = math.var();
    let expr = &x * &x + 2.0 * &x + 1.0; // x² + 2x + 1

    // Evaluate using the API
    let result = math.eval(&expr, &[3.0]);
    println!("  Direct evaluation: f(3) = {result}");
    assert_eq!(result, 16.0); // 9 + 6 + 1

    // Convert to AST for code generation
    let ast_expr = expr.into_ast();

    // Generate and compile Rust code
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "quadratic")?;

    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "quadratic")?;
        let compiled_result = compiled_func.call(3.0)?;
        println!("  Compiled function: f(3) = {compiled_result}");
        assert_eq!(compiled_result, 16.0);
    } else {
        println!("  Rust compiler not available - skipping compilation");
    }

    // Or use JIT compilation (if available)
    #[cfg(feature = "cranelift")]
    {
        let mut compiler = CraneliftCompiler::new_default()?;
        let registry = VariableRegistry::for_expression(&ast_expr);
        let compiled_func = compiler.compile_expression(&ast_expr, &registry)?;
        let jit_result = compiled_func.call(&[3.0]).unwrap();
        println!("  JIT compiled: f(3) = {jit_result}");
        assert_eq!(jit_result, 16.0);
    }
    #[cfg(not(feature = "cranelift"))]
    {
        println!("  Cranelift JIT not available (feature not enabled)");
    }

    Ok(())
}

fn automatic_differentiation_example() -> Result<()> {
    // Define a function
    let math = MathBuilder::new();
    let x = math.var();
    let f = math.poly(&[1.0, 2.0, 1.0], &x); // 1 + 2x + x² (coefficients in ascending order)

    // Convert to AST for AD processing
    let ast_f = f.into_ast();

    // Compute function and derivatives with optimization
    let mut ad = SymbolicAD::new()?;
    let result = ad.compute_with_derivatives(&ast_f)?;

    println!("  f(x) = polynomial (1 + 2x + x²)");
    println!("  f'(x) computed (derivative of 1 + 2x + x² = 2 + 2x)");
    println!(
        "  Shared subexpressions: {}",
        result.stats.shared_subexpressions_count
    );

    Ok(())
}

fn multiple_backends_example() -> Result<()> {
    let math = MathBuilder::new();
    let x = math.var();
    let expr = 2.0 * &x + 1.0; // 2x + 1

    // Convert to AST for backend processing
    let ast_expr = expr.into_ast();

    // Cranelift JIT
    #[cfg(feature = "cranelift")]
    {
        let mut compiler = CraneliftCompiler::new_default()?;
        let registry = VariableRegistry::for_expression(&ast_expr);
        let compiled_func = compiler.compile_expression(&ast_expr, &registry)?;
        let jit_result = compiled_func.call(&[3.0]).unwrap();
        println!("  Cranelift JIT: f(3) = {jit_result}");
        assert_eq!(jit_result, 7.0); // 2*3 + 1 = 7
    }
    #[cfg(not(feature = "cranelift"))]
    {
        println!("  Cranelift JIT not available (feature not enabled)");
    }

    // Rust code generation
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "my_func")?;
    println!("  Rust code generated successfully");

    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "my_func")?;
        let compiled_result = compiled_func.call(3.0)?;
        println!("  Rust compiled: f(3) = {compiled_result}");
        assert_eq!(compiled_result, 7.0);
    } else {
        println!("  Rust compiler not available - skipping compilation");
    }

    Ok(())
}
