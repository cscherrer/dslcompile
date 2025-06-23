//! Integration tests for LLVM JIT compilation backend
//!
//! This test suite ensures that the LLVM JIT backend correctly compiles
//! and evaluates mathematical expressions with performance comparable to
//! native Rust code.

#![cfg(feature = "llvm_jit")]

use dslcompile::{ast::ASTRepr, backends::LLVMJITCompiler, error::Result, prelude::*};
use inkwell::context::Context;
use proptest::prelude::*;

/// Helper to create a context and compiler
fn setup_compiler<'ctx>(context: &'ctx Context) -> LLVMJITCompiler<'ctx> {
    LLVMJITCompiler::new(context)
}

#[test]
fn test_constant_expressions() -> Result<()> {
    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Test various constants
    let test_cases = [
        (42.0, 42.0),
        (0.0, 0.0),
        (-17.5, -17.5),
        (std::f64::consts::PI, std::f64::consts::PI),
    ];

    for (constant, expected) in test_cases {
        let expr = ASTRepr::Constant(constant);
        let compiled_fn = compiler.compile_single_var(&expr)?;
        let result = unsafe { compiled_fn.call(0.0) }; // Parameter ignored for constant
        assert!(
            (result - expected).abs() < f64::EPSILON,
            "Constant {} failed: expected {}, got {}",
            constant,
            expected,
            result
        );
    }

    Ok(())
}

#[test]
fn test_arithmetic_operations() -> Result<()> {
    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Test addition: x + 5
    let add_expr = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Constant(5.0)]);
    let add_fn = compiler.compile_single_var(&add_expr)?;
    assert!((unsafe { add_fn.call(3.0) } - 8.0).abs() < f64::EPSILON);

    // Test subtraction: x - 3
    let sub_expr: ASTRepr<f64> = ASTRepr::Sub(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Constant(3.0)),
    );
    let sub_fn = compiler.compile_single_var(&sub_expr)?;
    assert!((unsafe { sub_fn.call(10.0) } - 7.0).abs() < f64::EPSILON);

    // Test multiplication: x * 2
    let mul_expr = ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Constant(2.0)]);
    let mul_fn = compiler.compile_single_var(&mul_expr)?;
    assert!((unsafe { mul_fn.call(5.0) } - 10.0).abs() < f64::EPSILON);

    // Test division: x / 2
    let div_expr: ASTRepr<f64> = ASTRepr::Div(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Constant(2.0)),
    );
    let div_fn = compiler.compile_single_var(&div_expr)?;
    assert!((unsafe { div_fn.call(10.0) } - 5.0).abs() < f64::EPSILON);

    Ok(())
}

#[test]
fn test_complex_polynomial() -> Result<()> {
    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Test polynomial: 3x³ - 2x² + x - 5
    let expr: ASTRepr<f64> = ASTRepr::add_from_array([
        ASTRepr::mul_from_array([
            ASTRepr::Constant(3.0),
            ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(3.0)),
            ),
        ]),
        ASTRepr::Neg(Box::new(ASTRepr::mul_from_array([
            ASTRepr::Constant(2.0),
            ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(2.0)),
            ),
        ]))),
        ASTRepr::Variable(0),
        ASTRepr::Neg(Box::new(ASTRepr::Constant(5.0))),
    ]);

    let compiled_fn = compiler.compile_single_var(&expr)?;

    // Test at x = 2: 3(8) - 2(4) + 2 - 5 = 24 - 8 + 2 - 5 = 13
    let result = unsafe { compiled_fn.call(2.0) };
    assert!((result - 13.0).abs() < f64::EPSILON);

    // Test at x = 0: -5
    let result = unsafe { compiled_fn.call(0.0) };
    assert!((result - (-5.0)).abs() < f64::EPSILON);

    Ok(())
}

#[test]
fn test_transcendental_functions() -> Result<()> {
    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Test sin(x)
    let sin_expr: ASTRepr<f64> = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
    let sin_fn = compiler.compile_single_var(&sin_expr)?;
    assert!((unsafe { sin_fn.call(std::f64::consts::PI / 2.0) } - 1.0).abs() < 1e-10);

    // Test cos(x)
    let cos_expr: ASTRepr<f64> = ASTRepr::Cos(Box::new(ASTRepr::Variable(0)));
    let cos_fn = compiler.compile_single_var(&cos_expr)?;
    assert!((unsafe { cos_fn.call(std::f64::consts::PI) } - (-1.0)).abs() < 1e-10);

    // Test ln(x)
    let ln_expr: ASTRepr<f64> = ASTRepr::Ln(Box::new(ASTRepr::Variable(0)));
    let ln_fn = compiler.compile_single_var(&ln_expr)?;
    assert!((unsafe { ln_fn.call(std::f64::consts::E) } - 1.0).abs() < 1e-10);

    // Test exp(x)
    let exp_expr: ASTRepr<f64> = ASTRepr::Exp(Box::new(ASTRepr::Variable(0)));
    let exp_fn = compiler.compile_single_var(&exp_expr)?;
    assert!((unsafe { exp_fn.call(1.0) } - std::f64::consts::E).abs() < 1e-10);

    // Test sqrt(x)
    let sqrt_expr: ASTRepr<f64> = ASTRepr::Sqrt(Box::new(ASTRepr::Variable(0)));
    let sqrt_fn = compiler.compile_single_var(&sqrt_expr)?;
    assert!((unsafe { sqrt_fn.call(16.0) } - 4.0).abs() < f64::EPSILON);

    Ok(())
}

#[test]
fn test_multi_variable_basic() -> Result<()> {
    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Test: x + y
    let expr: ASTRepr<f64> = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Variable(1)]);
    let compiled_fn = compiler.compile_multi_var(&expr)?;

    let vars = [3.0, 5.0];
    let result = unsafe { compiled_fn.call(vars.as_ptr()) };
    assert!((result - 8.0).abs() < f64::EPSILON);

    Ok(())
}

#[test]
fn test_multi_variable_complex() -> Result<()> {
    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Test: x²y + xyz - z³
    let expr: ASTRepr<f64> = ASTRepr::add_from_array([
        ASTRepr::mul_from_array([
            ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(2.0)),
            ),
            ASTRepr::Variable(1),
        ]),
        ASTRepr::mul_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Variable(1),
            ASTRepr::Variable(2),
        ]),
        ASTRepr::Neg(Box::new(ASTRepr::Pow(
            Box::new(ASTRepr::Variable(2)),
            Box::new(ASTRepr::Constant(3.0)),
        ))),
    ]);

    let compiled_fn = compiler.compile_multi_var(&expr)?;

    // Test with x=2, y=3, z=1: 4*3 + 2*3*1 - 1³ = 12 + 6 - 1 = 17
    let vars = [2.0, 3.0, 1.0];
    let result = unsafe { compiled_fn.call(vars.as_ptr()) };
    assert!((result - 17.0).abs() < f64::EPSILON);

    Ok(())
}

#[test]
fn test_optimization_levels() -> Result<()> {
    use inkwell::OptimizationLevel;

    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Complex expression to show optimization effects
    let expr: ASTRepr<f64> = ASTRepr::add_from_array([
        ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Constant(2.0)]),
        ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Constant(3.0)]),
    ]); // Should optimize to 5*x

    let opt_levels = [
        OptimizationLevel::None,
        OptimizationLevel::Less,
        OptimizationLevel::Default,
        OptimizationLevel::Aggressive,
    ];

    for opt_level in opt_levels {
        let compiled_fn = compiler.compile_single_var_with_opt(&expr, opt_level)?;
        let result = unsafe { compiled_fn.call(10.0) };
        assert!(
            (result - 50.0).abs() < f64::EPSILON,
            "Optimization level {:?} failed",
            opt_level
        );
    }

    Ok(())
}

#[test]
fn test_edge_cases() -> Result<()> {
    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Test division by zero handling
    let div_expr: ASTRepr<f64> = ASTRepr::Div(
        Box::new(ASTRepr::Constant(1.0)),
        Box::new(ASTRepr::Variable(0)),
    );
    let div_fn = compiler.compile_single_var(&div_expr)?;
    let result = unsafe { div_fn.call(0.0) };
    assert!(result.is_infinite());

    // Test very large exponents
    let pow_expr: ASTRepr<f64> = ASTRepr::Pow(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Constant(100.0)),
    );
    let pow_fn = compiler.compile_single_var(&pow_expr)?;
    let result = unsafe { pow_fn.call(1.1) };
    assert!(result.is_finite());

    Ok(())
}

// Property-based tests
proptest! {
    #[test]
    fn prop_arithmetic_correctness(x in -100.0..100.0f64, y in -100.0..100.0f64) {
        let context = Context::create();
        let mut compiler = setup_compiler(&context);

        // Test that LLVM JIT produces same results as direct evaluation
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::mul_from_array([
                ASTRepr::Variable(0),
                ASTRepr::Variable(1),
            ]),
            ASTRepr::Constant(1.0),
        ]);

        if let Ok(compiled_fn) = compiler.compile_multi_var(&expr) {
            let vars = [x, y];
            let jit_result = unsafe { compiled_fn.call(vars.as_ptr()) };
            let expected = x * y + 1.0;

            prop_assert!((jit_result - expected).abs() < 1e-10,
                        "JIT result {} != expected {} for x={}, y={}",
                        jit_result, expected, x, y);
        }
    }

    #[test]
    fn prop_transcendental_domain(x in 0.1..10.0f64) {
        let context = Context::create();
        let mut compiler = setup_compiler(&context);

        // Test ln(x) is defined for positive x
        let ln_expr: ASTRepr<f64> = ASTRepr::Ln(Box::new(ASTRepr::Variable(0)));
        if let Ok(ln_fn) = compiler.compile_single_var(&ln_expr) {
            let result = unsafe { ln_fn.call(x) };
            prop_assert!(result.is_finite(), "ln({}) should be finite", x);
        }
    }
}

#[test]
fn test_performance_comparison() -> Result<()> {
    use std::time::Instant;

    let context = Context::create();
    let mut compiler = setup_compiler(&context);

    // Complex expression for benchmarking
    let expr: ASTRepr<f64> = ASTRepr::add_from_array([
        ASTRepr::Sin(Box::new(ASTRepr::mul_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(2.0),
        ]))),
        ASTRepr::Cos(Box::new(ASTRepr::mul_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(3.0),
        ]))),
    ]);

    let compiled_fn = compiler.compile_single_var(&expr)?;

    // Warm up
    for _ in 0..1000 {
        unsafe { compiled_fn.call(1.0) };
    }

    // Benchmark JIT
    let iterations = 100_000;
    let start = Instant::now();
    for i in 0..iterations {
        let x = (i as f64) / 1000.0;
        unsafe { compiled_fn.call(x) };
    }
    let jit_time = start.elapsed();

    // Benchmark native Rust
    let start = Instant::now();
    for i in 0..iterations {
        let x = (i as f64) / 1000.0;
        let _result = (x * 2.0).sin() + (x * 3.0).cos();
    }
    let rust_time = start.elapsed();

    println!("JIT time: {:?}", jit_time);
    println!("Rust time: {:?}", rust_time);
    println!(
        "JIT/Rust ratio: {:.2}x",
        jit_time.as_nanos() as f64 / rust_time.as_nanos() as f64
    );

    // JIT should be within 2x of native Rust performance
    assert!(
        jit_time.as_nanos() < rust_time.as_nanos() * 2,
        "JIT performance should be comparable to native Rust"
    );

    Ok(())
}
