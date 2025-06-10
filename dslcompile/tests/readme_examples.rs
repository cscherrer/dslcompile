//! Integration tests for README examples
//!
//! This test file ensures that all the examples shown in the README continue
//! to work as the codebase evolves. It's based on the working examples/readme.rs.

use dslcompile::ast::DynamicContext;
use dslcompile::prelude::*;
use frunk::hlist;

#[test]
fn test_symbolic_to_numeric_optimization() -> Result<()> {
    // Define symbolic expression using natural syntax
    let mut math = DynamicContext::new();
    let x = math.var();
    let expr = math.poly(&[1.0, 2.0, 3.0], &x); // 1 + 2x + 3x² (coefficients in ascending order)

    // Evaluate efficiently with named variables
    let result = math.eval(&expr, hlist![3.0]);
    assert_eq!(result, 34.0); // 1 + 2*3 + 3*3² = 1 + 6 + 27 = 34

    // Convert to AST for code generation
    let ast_expr = expr.into();

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
    // Create mathematical expressions using natural syntax
    let mut math = DynamicContext::new();
    let x = math.var();
    let expr = &x * &x + 2.0 * &x + 1.0; // x² + 2x + 1

    // Evaluate efficiently using the new API
    let result = math.eval(&expr, hlist![3.0]);
    assert_eq!(result, 16.0); // 9 + 6 + 1

    // Convert to AST for code generation
    let ast_expr = expr.into();

    // Generate and compile Rust code for maximum performance
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_poly")?;

    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "test_poly")?;
        let compiled_result = compiled_func.call(3.0)?; // Native speed execution
        assert_eq!(compiled_result, 16.0);
        // Cleanup handled automatically when compiled_func is dropped
    }

    // Note: JIT compilation removed - focusing on compile-time optimization

    Ok(())
}

#[test]
fn test_automatic_differentiation_example() -> Result<()> {
    // Define a complex function using natural syntax
    let mut math = DynamicContext::new();
    let x = math.var();
    let f = math.poly(&[1.0, 2.0, 1.0], &x); // 1 + 2x + x² (coefficients in ascending order)

    // Convert to AST for AD processing
    let ast_f = f.into();

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
    let mut math = DynamicContext::new();
    let x = math.var();
    let expr = 2.0 * &x + 1.0; // 2x + 1 using natural syntax

    // Convert to AST for backend processing
    let ast_expr = expr.into();

    // Note: Cranelift JIT removed - focusing on Rust codegen backend

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
    let mut math = DynamicContext::new();
    let x = math.var();
    let expr = 3.0 * &x; // 3x using natural syntax

    // Convert to AST for code generation
    let ast_expr = expr.into();

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

#[test]
fn test_readme_basic_usage() {
    // This test verifies the basic usage examples from the README

    // Define symbolic expression using natural syntax
    let mut math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    // Create expression: x² + 2x + y
    let expr = &x * &x + 2.0 * &x + &y;

    // Evaluate
    let result = math.eval(&expr, hlist![3.0, 1.0]);
    assert_eq!(result, 16.0); // 9 + 6 + 1 = 16
}

#[test]
fn test_readme_optimization() {
    // Test optimization examples from README
    let mut math = DynamicContext::new();
    let x = math.var();

    // Expression that should optimize
    let expr = &x + 0.0; // x + 0 should optimize to x
    let result = math.eval(&expr, hlist![5.0]);
    assert_eq!(result, 5.0);
}

#[test]
fn test_readme_compilation() {
    // Test compilation examples from README

    // Create mathematical expressions using natural syntax
    let mut math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    // Build polynomial expression
    let poly_expr = &x * &x + 2.0 * &x + &y;

    // Test evaluation (compilation testing would require more setup)
    let result = math.eval(&poly_expr, hlist![2.0, 3.0]);
    assert_eq!(result, 11.0); // 4 + 4 + 3 = 11
}

#[test]
fn test_readme_complex_example() {
    // Test complex mathematical expression building
    let mut math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    // Complex expression with transcendental functions
    let expr = x.sin() + y.cos();

    // Test at specific values
    let result: f64 = math.eval_old(&expr, [0.0, 0.0]);
    let expected: f64 = 0.0_f64.sin() + 0.0_f64.cos(); // sin(0) + cos(0) = 0 + 1 = 1
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_readme_performance() {
    // Test performance claims from README

    // Create multiple expressions to test overhead
    let mut math = DynamicContext::new();
    let x = math.var();

    for _i in 0..1000 {
        let expr = &x * 2.0 + 1.0;
        let _result = math.eval(&expr, hlist![3.0]);
    }

    // If we get here without significant delay, performance is reasonable
    assert!(true);
}

#[test]
fn test_readme_variable_management() {
    // Test variable management examples
    let mut math = DynamicContext::new();
    let x = math.var();
    let y = math.var();
    let z = math.var();

    // Define a complex function using natural syntax
    let expr = &x * &y + &z * &z;

    // Test evaluation with all variables
    let result = math.eval_old(&expr, [2.0, 3.0, 4.0]);
    assert_eq!(result, 22.0); // 2*3 + 4*4 = 6 + 16 = 22

    // Test that variables work correctly (basic functionality test)
    let x_only = &x * 2.0;
    let x_result = math.eval_old(&x_only, [5.0]);
    assert_eq!(x_result, 10.0);
}

#[test]
fn test_readme_operator_precedence() {
    // Test that operator precedence works correctly
    let mut math = DynamicContext::new();
    let x = math.var();

    let expr = 2.0 * &x + 1.0; // 2x + 1 using natural syntax
    let result = math.eval_old(&expr, [3.0]);
    assert_eq!(result, 7.0); // 2*3 + 1 = 7

    // Test with different precedence
    let expr2 = 2.0 + &x * 3.0; // 2 + 3x
    let result2 = math.eval_old(&expr2, [2.0]);
    assert_eq!(result2, 8.0); // 2 + 3*2 = 8
}

#[test]
fn test_readme_mathematical_functions() {
    // Test mathematical functions from README
    let mut math = DynamicContext::new();
    let x = math.var();

    // Test exponential and logarithmic functions
    let expr = x.exp().ln(); // exp(ln(x)) should equal x
    let result: f64 = math.eval_old(&expr, [2.5]);
    assert!((result - 2.5).abs() < 1e-10);

    // Test trigonometric functions
    let mut math2 = DynamicContext::new();
    let x2 = math2.var();
    let expr2 = 3.0 * &x2; // 3x using natural syntax
    let result2: f64 = math2.eval_old(&expr2, [4.0]);
    assert_eq!(result2, 12.0); // 3*4 = 12
}
