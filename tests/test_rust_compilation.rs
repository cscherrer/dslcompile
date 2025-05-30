//! Test actual Rust compilation and execution

use mathcompile::backends::{RustCodeGenerator, RustCompiler};
use mathcompile::final_tagless::{ASTEval, ASTMathExpr};
use mathcompile::{CompilationStrategy, RustOptLevel, SymbolicOptimizer};

#[test]
fn test_rust_code_generation() {
    let codegen = RustCodeGenerator::new();
    let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
    let code = codegen.generate_function(&expr, "test_fn").unwrap();

    assert!(code.contains("#[no_mangle]"));
    assert!(code.contains("pub extern \"C\" fn test_fn"));
    assert!(code.contains("(var_0 + 1_f64)"));
}

#[test]
fn test_rust_compilation_and_execution() {
    // Only run this test if rustc is available
    if !RustCompiler::is_available() {
        println!("Rust compiler not available - skipping compilation test");
        return;
    }

    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::new();

    // Create a simple expression: f(x) = x + 1
    let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));

    // Generate Rust code
    let rust_code = codegen.generate_function(&expr, "test_func").unwrap();

    // Compile and load the function using dlopen2
    let compiled_func = compiler.compile_and_load(&rust_code, "test_func").unwrap();

    // Test the compiled function
    let result = compiled_func.call(5.0).unwrap();
    assert_eq!(result, 6.0);

    println!("Rust compilation test passed: f(5) = {result}");
}

#[test]
fn test_complex_expression_compilation() {
    // Only run this test if rustc is available
    if !RustCompiler::is_available() {
        println!("Rust compiler not available - skipping complex compilation test");
        return;
    }

    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::new();

    // Create a more complex expression: f(x, y) = x^2 + 2*x*y + y^2
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)),
            ASTEval::mul(
                ASTEval::mul(ASTEval::constant(2.0), ASTEval::var(0)),
                ASTEval::var(1),
            ),
        ),
        ASTEval::pow(ASTEval::var(1), ASTEval::constant(2.0)),
    );

    // Generate and compile
    let rust_code = codegen.generate_function(&expr, "complex_func").unwrap();
    let compiled_func = compiler.compile_and_load(&rust_code, "complex_func").unwrap();

    // Test with two variables: f(3, 4) = 9 + 24 + 16 = 49
    let result = compiled_func.call_two_vars(3.0, 4.0).unwrap();
    assert_eq!(result, 49.0);

    println!("Complex expression compilation test passed: f(3, 4) = {result}");
}

#[test]
fn test_optimization_with_compilation_strategy() {
    use mathcompile::OptimizationConfig;
    println!("⚡ Testing optimization with compilation strategy selection...");

    let mut optimizer = SymbolicOptimizer::new().unwrap();

    // Test adaptive strategy
    optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
        call_threshold: 2,
        complexity_threshold: 5,
    });

    // Simple expression
    let expr = ASTEval::add(
        ASTEval::mul(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
        ASTEval::constant(1.0),
    );

    // Optimize the expression
    let mut config = OptimizationConfig::default();
    config.egglog_optimization = true;
    let mut opt = SymbolicOptimizer::with_config(config).unwrap();
    let optimized = opt.optimize(&expr).unwrap();

    println!("Original: {expr:?}");
    println!("Optimized: {optimized:?}");

    // Test compilation approach selection
    for i in 0..5 {
        let approach = optimizer.choose_compilation_approach(&optimized, "test_expr");
        println!("Iteration {i}: Compilation approach: {approach:?}");
        optimizer.record_execution("test_expr", 1000 + i * 100);
    }

    // Check statistics
    let stats = optimizer.get_expression_stats();
    if let Some(expr_stats) = stats.get("test_expr") {
        println!("Expression statistics: {expr_stats:?}");
        assert!(expr_stats.call_count > 0);
    }
}
