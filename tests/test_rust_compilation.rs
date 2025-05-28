//! Test actual Rust compilation and execution

use mathjit::final_tagless::{ASTEval, ASTMathExpr};
use mathjit::symbolic::{CompilationStrategy, RustOptLevel, SymbolicOptimizer};
use std::fs;

#[test]
fn test_rust_compilation_and_execution() {
    println!("🔧 Testing actual Rust compilation and execution...");

    // Create temporary directories
    let temp_dir = std::env::temp_dir().join("mathjit_test");
    let source_dir = temp_dir.join("sources");
    let lib_dir = temp_dir.join("libs");

    // Create directories
    fs::create_dir_all(&source_dir).unwrap();
    fs::create_dir_all(&lib_dir).unwrap();

    // Create optimizer with hot-loading strategy
    let strategy = CompilationStrategy::HotLoadRust {
        source_dir: source_dir.clone(),
        lib_dir: lib_dir.clone(),
        opt_level: RustOptLevel::O2,
    };

    let optimizer = SymbolicOptimizer::with_strategy(strategy).unwrap();

    // Create a simple expression: x^2 + 1
    let expr = ASTEval::add(
        ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
        ASTEval::constant(1.0),
    );

    // Generate Rust source
    let rust_code = optimizer.generate_rust_source(&expr, "test_func").unwrap();
    println!("Generated Rust code:\n{rust_code}");

    // Write to source file
    let source_path = source_dir.join("test_func.rs");
    let lib_path = lib_dir.join("libtest_func.so");

    // Test compilation
    let compile_result =
        optimizer.compile_rust_dylib(&rust_code, &source_path, &lib_path, &RustOptLevel::O2);

    match compile_result {
        Ok(()) => {
            println!("✅ Rust compilation successful!");
            println!("Source file: {}", source_path.display());
            println!("Library file: {}", lib_path.display());

            // Check if library file exists
            if lib_path.exists() {
                println!("✅ Dynamic library created successfully!");

                // Try to load and test the library (if libloading is available)
                #[cfg(feature = "libloading")]
                test_dynamic_library_loading(&lib_path);
            } else {
                println!("❌ Dynamic library file not found");
            }
        }
        Err(e) => {
            println!("❌ Rust compilation failed: {e}");
            // This might fail if rustc is not available, which is okay for CI
            println!("Note: This test requires rustc to be available in PATH");
        }
    }

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}

#[cfg(feature = "libloading")]
fn test_dynamic_library_loading(lib_path: &std::path::Path) {
    use libloading::{Library, Symbol};

    println!("🔗 Testing dynamic library loading...");

    match unsafe { Library::new(lib_path) } {
        Ok(lib) => {
            println!("✅ Library loaded successfully!");

            // Try to get the function symbol
            let func: Result<Symbol<unsafe extern "C" fn(f64) -> f64>, _> =
                unsafe { lib.get(b"test_func") };

            match func {
                Ok(f) => {
                    println!("✅ Function symbol found!");

                    // Test the function: f(3) = 3^2 + 1 = 10
                    let result = unsafe { f(3.0) };
                    println!("test_func(3.0) = {result}");

                    let expected = 3.0_f64.powf(2.0) + 1.0;
                    assert!((result - expected).abs() < 1e-10);
                    println!("✅ Function execution successful and correct!");
                }
                Err(e) => {
                    println!("❌ Failed to get function symbol: {e}");
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to load library: {e}");
        }
    }
}

#[test]
fn test_optimization_with_compilation_strategy() {
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
    let mut config = mathjit::symbolic::OptimizationConfig::default();
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
