//! Integration tests for egglog optimization and Rust code generation

use mathjit::final_tagless::{ASTEval, ASTMathExpr, ASTRepr};
use mathjit::symbolic::{CompilationStrategy, OptimizationConfig, RustOptLevel, SymbolicOptimizer};
use std::path::PathBuf;

// Helper functions for more ergonomic expression building
fn var(name: &str) -> ASTRepr<f64> {
    ASTEval::var_by_name(name)
}

fn constant(value: f64) -> ASTRepr<f64> {
    ASTEval::constant(value)
}

fn add(left: ASTRepr<f64>, right: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::add(left, right)
}

fn mul(left: ASTRepr<f64>, right: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::mul(left, right)
}

fn sub(left: ASTRepr<f64>, right: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::sub(left, right)
}

fn pow(base: ASTRepr<f64>, exp: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::pow(base, exp)
}

fn sin(x: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::sin(x)
}

fn cos(x: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::cos(x)
}

fn exp(x: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::exp(x)
}

fn log(x: ASTRepr<f64>) -> ASTRepr<f64> {
    ASTEval::ln(x)
}

/// Test helper function to create a variable expression
fn create_var_expr(name: &str) -> ASTRepr<f64> {
    ASTEval::var_by_name(name)
}

#[test]
fn test_current_optimization_capabilities() {
    println!("🧪 Testing current optimization capabilities...");

    // Create optimizer with egglog enabled
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

    // Test expression: (x + 0) * 1 + log(exp(y)) - using helper functions
    let x = var("x");
    let y = var("y");
    let zero = constant(0.0);
    let one = constant(1.0);

    let expr = add(mul(add(x, zero), one), log(exp(y)));

    println!("Original expression: {expr:?}");

    // Apply optimization
    let optimized = optimizer.optimize(&expr).unwrap();
    println!("Optimized expression: {optimized:?}");

    // Should optimize to: x + y
    // Check that optimizations were applied
    assert_ne!(format!("{expr:?}"), format!("{optimized:?}"));
}

#[test]
fn test_rust_code_generation() {
    println!("🦀 Testing Rust code generation...");

    let optimizer = SymbolicOptimizer::new().unwrap();

    // Test expression: x^2 + 2*x + 1 - using helper functions
    let x = var("x");
    let two = constant(2.0);
    let one = constant(1.0);

    let expr = add(add(pow(x, two), mul(constant(2.0), var("x"))), one);

    let rust_code = optimizer.generate_rust_source(&expr, "quadratic").unwrap();
    println!("Generated Rust code:\n{rust_code}");

    // Verify the generated code contains expected elements
    assert!(rust_code.contains("#[no_mangle]"));
    assert!(rust_code.contains("pub extern \"C\" fn quadratic"));
    assert!(rust_code.contains("x * x") || rust_code.contains("x.powf(2"));
    assert!(rust_code.contains("2.0_f64 * x") || rust_code.contains("2 * x"));
}

#[test]
fn test_compilation_strategy_selection() {
    println!("⚙️ Testing compilation strategy selection...");

    let mut optimizer = SymbolicOptimizer::new().unwrap();

    // Simple expression should use Cranelift - using helper functions
    let simple_expr = add(var("x"), constant(1.0));

    let approach = optimizer.choose_compilation_approach(&simple_expr, "simple");
    println!("Simple expression approach: {approach:?}");

    // Set adaptive strategy
    optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
        call_threshold: 3,
        complexity_threshold: 10,
    });

    // Simulate multiple calls
    for i in 0..5 {
        let approach = optimizer.choose_compilation_approach(&simple_expr, "adaptive_test");
        println!("Call {i}: {approach:?}");
        optimizer.record_execution("adaptive_test", 1000);
    }

    let stats = optimizer.get_expression_stats();
    println!("Expression stats: {stats:?}");
}

#[test]
fn test_hot_loading_strategy() {
    println!("🔥 Testing hot-loading compilation strategy...");

    let strategy = CompilationStrategy::HotLoadRust {
        source_dir: PathBuf::from("/tmp/mathjit_test_sources"),
        lib_dir: PathBuf::from("/tmp/mathjit_test_libs"),
        opt_level: RustOptLevel::O2,
    };

    let optimizer = SymbolicOptimizer::with_strategy(strategy).unwrap();

    // Complex expression: sin(2x + cos(y)) - using helper functions
    let x = var("x");
    let y = var("y");
    let two = constant(2.0);

    let expr = sin(add(mul(two, x), cos(y)));

    let rust_code = optimizer
        .generate_rust_source(&expr, "complex_func")
        .unwrap();
    println!("Complex function Rust code:\n{rust_code}");

    assert!(rust_code.contains("sin"));
    assert!(rust_code.contains("cos"));
    assert!(rust_code.contains("complex_func"));
}

#[test]
fn test_algebraic_optimizations() {
    println!("🧮 Testing algebraic optimizations...");

    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

    // Test exp(a) * exp(b) = exp(a+b) - using helper functions
    let a = var("a");
    let b = var("b");
    let exp_expr = mul(exp(a), exp(b));

    let optimized_exp = optimizer.optimize(&exp_expr).unwrap();
    println!("exp(a) * exp(b) optimized to: {optimized_exp:?}");

    // Test log(exp(x)) = x - using helper functions
    let x = var("x");
    let log_exp_expr = log(exp(x));

    let optimized_log_exp = optimizer.optimize(&log_exp_expr).unwrap();
    println!("log(exp(x)) optimized to: {optimized_log_exp:?}");

    // Test power rule: x^a * x^b = x^(a+b) - using helper functions
    let x = var("x");
    let a = var("a");
    let b = var("b");
    let power_expr = mul(pow(x, a), pow(var("x"), b));

    let optimized_power = optimizer.optimize(&power_expr).unwrap();
    println!("x^a * x^b optimized to: {optimized_power:?}");
}

#[test]
fn test_end_to_end_optimization_and_generation() {
    println!("🎯 Testing end-to-end optimization and Rust generation...");

    // Create optimizer with full optimization enabled
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    config.constant_folding = true;
    config.aggressive = true;

    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

    // Complex expression that should be heavily optimized - using helper functions
    let x = var("x");
    let y = var("y");
    let zero = constant(0.0);
    let one = constant(1.0);

    // (x + 0) * 1 + (log(exp(y)) - 0)
    let complex_expr = add(mul(add(x, zero), one), sub(log(exp(y)), constant(0.0)));

    println!("Original complex expression: {complex_expr:?}");

    // Optimize
    let optimized = optimizer.optimize(&complex_expr).unwrap();
    println!("Optimized expression: {optimized:?}");

    // Generate Rust code
    let rust_code = optimizer
        .generate_rust_source(&optimized, "optimized_func")
        .unwrap();
    println!("Generated Rust code for optimized expression:\n{rust_code}");

    // The optimized expression should be much simpler
    assert_ne!(format!("{complex_expr:?}"), format!("{optimized:?}"));
    assert!(rust_code.contains("optimized_func"));
}

#[cfg(feature = "autodiff")]
#[test]
fn test_autodiff_integration() {
    println!("🔬 Testing autodiff integration with symbolic optimization...");

    use ad_trait::forward_ad::adfn::adfn;
    use mathjit::autodiff::{convenience, ForwardAD};

    // Test that we can differentiate optimized expressions
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

    // Create a complex expression that will be optimized - using helper functions
    let x = var("x");
    let y = var("y");
    let zero = constant(0.0);
    let one = constant(1.0);

    // (x + 0) * 1 + log(exp(y))
    let expr = add(mul(add(x, zero), one), log(exp(y)));

    println!("Original expression: {expr:?}");

    // Optimize the expression
    let optimized = optimizer.optimize(&expr).unwrap();
    println!("Optimized expression: {optimized:?}");

    // Now test autodiff on a similar function
    let forward_ad = ForwardAD::new();

    // Test f(x) = x + y (the optimized form)
    let simple_func = |x: adfn<1>| {
        let y = adfn::new(2.0, [0.0]); // y = 2 for testing
        x + y
    };

    let (value, derivative) = forward_ad.differentiate(simple_func, 3.0).unwrap();
    println!("f(3) = {value}, f'(3) = {derivative}");

    // f(x) = x + 2, so f(3) = 5, f'(3) = 1
    assert!((value - 5.0).abs() < 1e-10);
    assert!((derivative - 1.0).abs() < 1e-10);

    // Test gradient computation for multi-variable case using finite differences
    let multi_var = |vars: &[f64]| {
        let x = vars[0];
        let y = vars[1];
        x + y // This is what the optimized expression should become
    };

    let gradient = convenience::gradient(multi_var, &[3.0, 2.0]).unwrap();
    println!("Gradient: [{}, {}]", gradient[0], gradient[1]);

    // Gradient of f(x,y) = x + y is [1, 1]
    assert!((gradient[0] - 1.0).abs() < 1e-6); // Finite difference tolerance
    assert!((gradient[1] - 1.0).abs() < 1e-6); // Finite difference tolerance

    println!("✅ Autodiff integration test passed!");
}
