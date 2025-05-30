//! Integration tests for README examples
//!
//! This test file ensures that all the examples shown in the README continue
//! to work as the codebase evolves. It's based on the working examples/readme.rs.

use mathcompile::prelude::*;

#[test]
fn test_symbolic_to_numeric_optimization() -> Result<()> {
    // Define symbolic expression using beautiful syntax
    let math = MathBuilder::new();
    let x = math.var("x");
    let expr = math.poly(&[1.0, 2.0, 3.0], &x); // 1 + 2x + 3x² (coefficients in ascending order)

    // Evaluate efficiently with named variables
    let result = math.eval(&expr, &[("x", 3.0)]);
    assert_eq!(result, 34.0); // 1 + 2*3 + 3*3² = 1 + 6 + 27 = 34

    // Convert to AST for code generation
    let ast_expr = expr.into_ast();

    // Generate optimized Rust code for maximum performance
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_function")?;
    assert!(rust_code.contains("test_function"));

    // Test compilation if rustc is available
    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "test_function")?;
        let compiled_result = compiled_func.call(3.0)?;
        assert_eq!(compiled_result, result); // Should match direct evaluation
        // Cleanup handled automatically when compiled_func is dropped
    }

    Ok(())
}

#[test]
fn test_basic_usage_example() -> Result<()> {
    // Create mathematical expressions using beautiful syntax
    let math = MathBuilder::new();
    let x = math.var("x");
    let expr = &x * &x + 2.0 * &x + 1.0; // x² + 2x + 1

    // Evaluate efficiently using the new API
    let result = math.eval(&expr, &[("x", 3.0)]);
    assert_eq!(result, 16.0); // 9 + 6 + 1

    // Convert to AST for code generation
    let ast_expr = expr.into_ast();

    // Generate and compile Rust code for maximum performance
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_quadratic")?;

    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "test_quadratic")?;
        let compiled_result = compiled_func.call(3.0)?; // Native speed execution
        assert_eq!(compiled_result, 16.0);
        // Cleanup handled automatically when compiled_func is dropped
    }

    // Test JIT compilation if available
    #[cfg(feature = "cranelift")]
    {
        let compiler = JITCompiler::new()?;
        let compiled = compiler.compile_single_var(&ast_expr, "x")?;
        let fast_result = compiled.call_single(3.0);
        assert_eq!(fast_result, 16.0);
    }

    Ok(())
}

#[test]
fn test_automatic_differentiation_example() -> Result<()> {
    // Define a complex function using beautiful syntax
    let math = MathBuilder::new();
    let x = math.var("x");
    let f = math.poly(&[1.0, 2.0, 1.0], &x); // 1 + 2x + x² (coefficients in ascending order)

    // Convert to AST for AD processing
    let ast_f = f.into_ast();

    // Compute function and derivatives with optimization
    let mut ad = SymbolicAD::new()?;
    let result = ad.compute_with_derivatives(&ast_f)?;

    // Verify we got a result with the expected structure
    let _subexpr_count = result.stats.shared_subexpressions_count; // Should be accessible
    assert!(!result.first_derivatives.is_empty()); // Should have at least one derivative

    Ok(())
}

#[test]
fn test_multiple_backends_example() -> Result<()> {
    let math = MathBuilder::new();
    let x = math.var("x");
    let expr = 2.0 * &x + 1.0; // 2x + 1 using beautiful syntax

    // Convert to AST for backend processing
    let ast_expr = expr.into_ast();

    // Test Cranelift JIT if available
    #[cfg(feature = "cranelift")]
    {
        let compiler = JITCompiler::new()?;
        let jit_func = compiler.compile_single_var(&ast_expr, "x")?;
        let fast_result = jit_func.call_single(3.0);
        assert_eq!(fast_result, 7.0); // 2*3 + 1 = 7
    }

    // Test Rust code generation
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_backends")?;
    assert!(rust_code.contains("test_backends"));

    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "test_backends")?;
        let compiled_result = compiled_func.call(3.0)?;
        assert_eq!(compiled_result, 7.0);
        // Cleanup handled automatically when compiled_func is dropped
    }

    Ok(())
}

#[test]
fn test_compile_and_load_api() -> Result<()> {
    // Test the new compile_and_load API specifically
    let math = MathBuilder::new();
    let x = math.var("x");
    let expr = 3.0 * &x; // 3x using beautiful syntax

    // Convert to AST for code generation
    let ast_expr = expr.into_ast();

    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_api")?;

    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();

        // Test the convenience method
        let compiled_func = compiler.compile_and_load(&rust_code, "test_api")?;

        // Test different call methods
        let result1 = compiled_func.call(4.0)?;
        assert_eq!(result1, 12.0); // 3*4 = 12

        // Test function name access
        assert_eq!(compiled_func.name(), "test_api");
        // Cleanup handled automatically when compiled_func is dropped
    }

    Ok(())
}
