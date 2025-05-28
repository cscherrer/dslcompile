//! Integration tests for egglog optimization and Rust code generation

use mathjit::final_tagless::{JITEval, JITMathExpr};
use mathjit::symbolic::{CompilationStrategy, OptimizationConfig, RustOptLevel, SymbolicOptimizer};
use std::path::PathBuf;

#[test]
fn test_current_optimization_capabilities() {
    println!("🧪 Testing current optimization capabilities...");

    // Create optimizer with egglog enabled
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

    // Test expression: (x + 0) * 1 + ln(exp(y))
    let expr = JITEval::add(
        JITEval::mul(
            JITEval::add(JITEval::var("x"), JITEval::constant(0.0)),
            JITEval::constant(1.0),
        ),
        JITEval::ln(JITEval::exp(JITEval::var("y"))),
    );

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

    // Test expression: x^2 + 2*x + 1
    let expr = JITEval::add(
        JITEval::add(
            JITEval::pow(JITEval::var("x"), JITEval::constant(2.0)),
            JITEval::mul(JITEval::constant(2.0), JITEval::var("x")),
        ),
        JITEval::constant(1.0),
    );

    let rust_code = optimizer.generate_rust_source(&expr, "quadratic").unwrap();
    println!("Generated Rust code:\n{rust_code}");

    // Verify the generated code contains expected elements
    assert!(rust_code.contains("#[no_mangle]"));
    assert!(rust_code.contains("pub extern \"C\" fn quadratic"));
    assert!(rust_code.contains("x.powf(2"));
    assert!(rust_code.contains("2.0 * x"));
}

#[test]
fn test_compilation_strategy_selection() {
    println!("⚙️ Testing compilation strategy selection...");

    let mut optimizer = SymbolicOptimizer::new().unwrap();

    // Simple expression should use Cranelift
    let simple_expr = JITEval::add(JITEval::var("x"), JITEval::constant(1.0));
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

    // Complex expression
    let expr = JITEval::sin(JITEval::add(
        JITEval::mul(JITEval::var("x"), JITEval::constant(2.0)),
        JITEval::cos(JITEval::var("y")),
    ));

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

    // Test exp(a) * exp(b) = exp(a+b)
    let exp_expr = JITEval::mul(
        JITEval::exp(JITEval::var("a")),
        JITEval::exp(JITEval::var("b")),
    );

    let optimized_exp = optimizer.optimize(&exp_expr).unwrap();
    println!("exp(a) * exp(b) optimized to: {optimized_exp:?}");

    // Test ln(exp(x)) = x
    let ln_exp_expr = JITEval::ln(JITEval::exp(JITEval::var("x")));
    let optimized_ln_exp = optimizer.optimize(&ln_exp_expr).unwrap();
    println!("ln(exp(x)) optimized to: {optimized_ln_exp:?}");

    // Test power rule: x^a * x^b = x^(a+b)
    let power_expr = JITEval::mul(
        JITEval::pow(JITEval::var("x"), JITEval::var("a")),
        JITEval::pow(JITEval::var("x"), JITEval::var("b")),
    );

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

    // Complex expression that should be heavily optimized
    let complex_expr = JITEval::add(
        JITEval::mul(
            JITEval::add(JITEval::var("x"), JITEval::constant(0.0)), // x + 0 = x
            JITEval::constant(1.0),                                  // * 1 = identity
        ),
        JITEval::sub(
            JITEval::ln(JITEval::exp(JITEval::var("y"))), // ln(exp(y)) = y
            JITEval::constant(0.0),                       // - 0 = identity
        ),
    );

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
