//! Integration tests for README examples
//!
//! This test file ensures that all the examples shown in the README continue
//! to work as the codebase evolves. It's based on the working examples/readme.rs.

use dslcompile::{ast::DynamicContext, prelude::*};
use frunk::hlist;

#[test]
fn test_basic_usage_and_compilation() -> Result<()> {
    // Test basic usage: expression building, evaluation, and compilation
    let mut math = DynamicContext::new();
    let x = math.var();
    let expr: Expr<f64> = &x * &x + 2.0 * &x + 1.0; // x² + 2x + 1

    // Test evaluation
    let result = math.eval(&expr, hlist![3.0]);
    assert_eq!(result, 16.0); // 9 + 6 + 1

    // Test code generation
    let ast_expr = expr.into();
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "test_poly")?;
    assert!(rust_code.contains("test_poly"));

    // Test compilation if available
    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "test_poly")?;
        let compiled_result = compiled_func.call(3.0)?;
        assert_eq!(compiled_result, 16.0);
    }

    Ok(())
}

#[test]
fn test_polynomial_helper() -> Result<()> {
    // Test polynomial helper function
    let mut math = DynamicContext::new();
    let x = math.var();
    let expr = math.poly(&[1.0, 2.0, 3.0], &x); // 1 + 2x + 3x²

    let result = math.eval(&expr, hlist![3.0]);
    assert_eq!(result, 34.0); // 1 + 2*3 + 3*9 = 1 + 6 + 27 = 34

    Ok(())
}

#[test]
fn test_automatic_differentiation() -> Result<()> {
    // Test automatic differentiation integration
    let mut math = DynamicContext::new();
    let x = math.var();
    let f = math.poly(&[1.0, 2.0, 1.0], &x); // 1 + 2x + x²

    let ast_f = f.into();

    // Test that AD can process the expression
    let mut ad = SymbolicAD::new()?;
    let result = ad.compute_with_derivatives(&ast_f)?;

    // Verify we got derivatives
    assert!(!result.first_derivatives.is_empty());

    Ok(())
}

#[test]
fn test_multi_variable_expressions() {
    // Test multi-variable expressions and operator precedence
    let mut math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    // Test basic multi-variable: x² + 2x + y
    let expr = &x * &x + 2.0 * &x + &y;
    let result = math.eval(&expr, hlist![3.0, 1.0]);
    assert_eq!(result, 16.0); // 9 + 6 + 1 = 16

    // Test operator precedence: 2x + 1
    let expr2: Expr<f64> = 2.0 * &x + 1.0;
    let result2 = math.eval(&expr2, hlist![3.0]);
    assert_eq!(result2, 7.0); // 2*3 + 1 = 7

    // Test different precedence: 2 + 3x
    let expr3: Expr<f64> = 2.0 + &x * 3.0;
    let result3 = math.eval(&expr3, hlist![2.0]);
    assert_eq!(result3, 8.0); // 2 + 3*2 = 8
}

#[test]
fn test_transcendental_functions() {
    // Test transcendental functions
    let mut math = DynamicContext::new();
    let x = math.var();

    // Test exp(ln(x)) = x identity
    let expr: Expr<f64> = x.clone().exp().ln();
    let result: f64 = math.eval(&expr, hlist![2.5]);
    assert!((result - 2.5).abs() < 1e-10);

    // Test trigonometric functions
    let expr2: Expr<f64> = x.clone().sin() + x.clone().cos();
    let result2: f64 = math.eval(&expr2, hlist![0.0]);
    let expected: f64 = 0.0_f64.sin() + 0.0_f64.cos(); // sin(0) + cos(0) = 0 + 1 = 1
    assert!((result2 - expected).abs() < 1e-10);
}

#[test]
fn test_optimization_examples() {
    // Test optimization examples that should work
    let mut math = DynamicContext::new();
    let x = math.var();

    // Test x + 0 optimization
    let expr: Expr<f64> = &x + 0.0;
    let result = math.eval(&expr, hlist![5.0]);
    assert_eq!(result, 5.0);

    // Test x * 1 optimization
    let expr2: Expr<f64> = &x * 1.0;
    let result2 = math.eval(&expr2, hlist![7.0]);
    assert_eq!(result2, 7.0);
}
