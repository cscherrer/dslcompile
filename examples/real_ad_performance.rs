//! Real Performance Comparison: MathJIT Symbolic AD vs ad_trait
//!
//! This benchmark provides ACTUAL measured performance comparisons between
//! our symbolic automatic differentiation and the ad_trait library.
//! 
//! **NEW**: Now uses Rust hot-loading compilation for maximum performance!

#[cfg(feature = "ad_trait")]
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD, ForwardADMulti};
#[cfg(feature = "ad_trait")]
use ad_trait::function_engine::FunctionEngine;
#[cfg(feature = "ad_trait")]
use ad_trait::forward_ad::adfn::adfn;
#[cfg(feature = "ad_trait")]
use ad_trait::AD;

use mathjit::final_tagless::{JITEval, JITMathExpr};
use mathjit::symbolic_ad::convenience;
use mathjit::backends::{RustCodeGenerator, RustCompiler};
use mathjit::backends::rust_codegen::RustOptLevel;
use mathjit::symbolic::{CompilationStrategy, SymbolicOptimizer};
use std::time::Instant;
use std::fs;
use libloading::{Library, Symbol};

/// Real benchmark results with actual measurements
#[derive(Debug, Clone)]
struct BenchmarkResults {
    /// Actual measured time for symbolic AD (microseconds)
    symbolic_ad_time_us: u64,
    /// Actual measured time for ad_trait (microseconds)
    ad_trait_time_us: u64,
    /// Accuracy comparison
    accuracy_difference: f64,
    /// Test description
    test_name: String,
    /// Compilation time for Rust codegen (microseconds)
    compilation_time_us: u64,
}

/// Compiled function wrapper for Rust hot-loading
struct CompiledFunction {
    _library: Library,
    function: Symbol<'static, extern "C" fn(f64, f64) -> f64>,
}

impl CompiledFunction {
    /// Load a compiled function from a dynamic library
    unsafe fn load(lib_path: &std::path::Path, func_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let library = Library::new(lib_path)?;
        let function: Symbol<extern "C" fn(f64, f64) -> f64> = library.get(format!("{}_two_vars", func_name).as_bytes())?;
        let function = std::mem::transmute(function);
        
        Ok(Self {
            _library: library,
            function,
        })
    }
    
    /// Call the compiled function
    fn call(&self, x: f64, y: f64) -> f64 {
        (self.function)(x, y)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 REAL Performance Comparison: MathJIT Symbolic AD (Rust Codegen) vs ad_trait");
    println!("=============================================================================\n");

    #[cfg(not(feature = "ad_trait"))]
    {
        println!("❌ This benchmark requires the 'ad_trait' feature to be enabled.");
        println!("   Run with: cargo run --example real_ad_performance --features ad_trait");
        return Ok(());
    }

    // Check if rustc is available
    if !RustCompiler::is_available() {
        println!("❌ rustc is not available. Rust codegen benchmarks require rustc to be installed.");
        return Ok(());
    }

    println!("✅ Using Rust hot-loading compilation for symbolic AD");
    println!("   Rustc version: {}", RustCompiler::version_info()?);
    println!();

    #[cfg(feature = "ad_trait")]
    {
        // Setup temporary directories for Rust compilation
        let temp_dir = std::env::temp_dir().join("mathjit_ad_bench");
        let source_dir = temp_dir.join("sources");
        let lib_dir = temp_dir.join("libs");
        
        // Clean and create directories
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&source_dir)?;
        fs::create_dir_all(&lib_dir)?;

        // 1. Simple Quadratic Function
        println!("1️⃣  Simple Quadratic: f(x) = x² (Rust Codegen)");
        let result1 = benchmark_simple_quadratic_rust(1000, &source_dir, &lib_dir)?;
        print_results(&result1);
        println!();

        // 2. Polynomial Function
        println!("2️⃣  Polynomial: f(x) = x⁴ + 3x³ + 2x² + x + 1 (Rust Codegen)");
        let result2 = benchmark_polynomial_rust(500, &source_dir, &lib_dir)?;
        print_results(&result2);
        println!();

        // 3. Multivariate Function
        println!("3️⃣  Multivariate: f(x,y) = x² + 2xy + y² (Rust Codegen)");
        let result3 = benchmark_multivariate_rust(500, &source_dir, &lib_dir)?;
        print_results(&result3);
        println!();

        // Summary
        print_summary(&[result1, result2, result3]);
        
        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    Ok(())
}

#[cfg(feature = "ad_trait")]
#[derive(Clone)]
struct SimpleQuadratic<T: AD> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "ad_trait")]
impl<T: AD> DifferentiableFunctionTrait<T> for SimpleQuadratic<T> {
    const NAME: &'static str = "SimpleQuadratic";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![inputs[0] * inputs[0]]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[cfg(feature = "ad_trait")]
impl<T: AD> SimpleQuadratic<T> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> SimpleQuadratic<T2> {
        SimpleQuadratic::new()
    }
}

#[cfg(feature = "ad_trait")]
fn benchmark_simple_quadratic_rust(iterations: usize, source_dir: &std::path::Path, lib_dir: &std::path::Path) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    // Symbolic AD version - PRE-COMPILE the derivative with enhanced optimization
    let expr = JITEval::pow(JITEval::var("x"), JITEval::constant(2.0));
    
    // Enable enhanced optimization
    let mut config = mathjit::symbolic_ad::SymbolicADConfig::default();
    config.pre_optimize = true;
    config.post_optimize = true;
    config.variables = vec!["x".to_string()];
    
    let mut symbolic_ad = mathjit::symbolic_ad::SymbolicAD::with_config(config)?;
    let result = symbolic_ad.compute_with_derivatives(&expr)?;
    let symbolic_grad = &result.first_derivatives["x"];
    
    println!("  📊 Optimization stats:");
    println!("    Function operations before: {}", result.stats.function_operations_before);
    println!("    Function operations after: {}", result.stats.function_operations_after);
    println!("    Total operations before: {}", result.stats.total_operations_before);
    println!("    Total operations after: {}", result.stats.total_operations_after);
    
    if result.stats.function_operations_before > result.stats.function_operations_after {
        let reduction = 100.0 * (1.0 - result.stats.function_optimization_ratio());
        println!("    🎯 Function optimized by {:.1}%", reduction);
    } else if result.stats.function_operations_after > result.stats.function_operations_before {
        let increase = 100.0 * (result.stats.function_optimization_ratio() - 1.0);
        println!("    📈 Function complexity increased by {:.1}% (due to optimization rules)", increase);
    }
    
    if result.stats.total_operations_before > result.stats.total_operations_after {
        let reduction = 100.0 * (1.0 - result.stats.total_optimization_ratio());
        println!("    🎯 Total pipeline optimized by {:.1}%", reduction);
    }

    // Compile the derivative to Rust code
    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::with_opt_level(RustOptLevel::O2);
    
    let func_name = "simple_quadratic_grad";
    let rust_source = codegen.generate_function(symbolic_grad, func_name)?;
    
    let source_path = source_dir.join(format!("{}.rs", func_name));
    let lib_path = lib_dir.join(format!("lib{}.so", func_name));
    
    // Time the compilation
    let compile_start = Instant::now();
    compiler.compile_dylib(&rust_source, &source_path, &lib_path)?;
    let compilation_time = compile_start.elapsed().as_micros() as u64;
    
    println!("  🔧 Rust compilation time: {} μs", compilation_time);
    
    // Load the compiled function
    let compiled_func = unsafe { CompiledFunction::load(&lib_path, func_name)? };
    
    // Now time just the EXECUTION
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = compiled_func.call(2.0, 0.0);
    }
    let symbolic_time = start.elapsed().as_micros() as u64;

    // ad_trait version - PRE-COMPILE the function engine
    let function_standard = SimpleQuadratic::<f64>::new();
    let function_derivative = function_standard.to_other_ad_type::<adfn<1>>();
    let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardAD::new());
    let inputs = vec![2.0];
    
    // Now time just the EXECUTION
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = differentiable_block.derivative(&inputs);
    }
    let ad_trait_time = start.elapsed().as_micros() as u64;

    // Accuracy check
    let symbolic_result = compiled_func.call(2.0, 0.0);
    let (_, ad_trait_grad) = differentiable_block.derivative(&inputs);
    let ad_trait_result = ad_trait_grad[(0, 0)];
    
    Ok(BenchmarkResults {
        symbolic_ad_time_us: symbolic_time,
        ad_trait_time_us: ad_trait_time,
        accuracy_difference: (symbolic_result - ad_trait_result).abs(),
        test_name: "Simple Quadratic".to_string(),
        compilation_time_us: compilation_time,
    })
}

#[cfg(feature = "ad_trait")]
#[derive(Clone)]
struct Polynomial<T: AD> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "ad_trait")]
impl<T: AD> DifferentiableFunctionTrait<T> for Polynomial<T> {
    const NAME: &'static str = "Polynomial";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        // x⁴ + 3x³ + 2x² + x + 1
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        let three = T::from_f64(3.0).unwrap_or_else(|| panic!("Failed to convert 3.0"));
        let two = T::from_f64(2.0).unwrap_or_else(|| panic!("Failed to convert 2.0"));
        let one = T::from_f64(1.0).unwrap_or_else(|| panic!("Failed to convert 1.0"));
        vec![x4 + x3 * three + x2 * two + x + one]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[cfg(feature = "ad_trait")]
impl<T: AD> Polynomial<T> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> Polynomial<T2> {
        Polynomial::new()
    }
}

#[cfg(feature = "ad_trait")]
fn benchmark_polynomial_rust(iterations: usize, source_dir: &std::path::Path, lib_dir: &std::path::Path) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    // Symbolic AD: f(x) = x⁴ + 3x³ + 2x² + x + 1 with enhanced optimization
    let expr = JITEval::add(
        JITEval::add(
            JITEval::add(
                JITEval::add(
                    JITEval::pow(JITEval::var("x"), JITEval::constant(4.0)),
                    JITEval::mul(JITEval::constant(3.0), JITEval::pow(JITEval::var("x"), JITEval::constant(3.0))),
                ),
                JITEval::mul(JITEval::constant(2.0), JITEval::pow(JITEval::var("x"), JITEval::constant(2.0))),
            ),
            JITEval::var("x"),
        ),
        JITEval::constant(1.0),
    );
    
    // Enable enhanced optimization
    let mut config = mathjit::symbolic_ad::SymbolicADConfig::default();
    config.pre_optimize = true;
    config.post_optimize = true;
    config.variables = vec!["x".to_string()];
    
    let mut symbolic_ad = mathjit::symbolic_ad::SymbolicAD::with_config(config)?;
    let result = symbolic_ad.compute_with_derivatives(&expr)?;
    let symbolic_grad = &result.first_derivatives["x"];
    
    println!("  📊 Optimization stats:");
    println!("    Function operations before: {}", result.stats.function_operations_before);
    println!("    Function operations after: {}", result.stats.function_operations_after);
    println!("    Total operations before: {}", result.stats.total_operations_before);
    println!("    Total operations after: {}", result.stats.total_operations_after);
    
    if result.stats.function_operations_before > result.stats.function_operations_after {
        let reduction = 100.0 * (1.0 - result.stats.function_optimization_ratio());
        println!("    🎯 Function optimized by {:.1}%", reduction);
    } else if result.stats.function_operations_after > result.stats.function_operations_before {
        let increase = 100.0 * (result.stats.function_optimization_ratio() - 1.0);
        println!("    📈 Function complexity increased by {:.1}% (due to optimization rules)", increase);
    }
    
    if result.stats.total_operations_before > result.stats.total_operations_after {
        let reduction = 100.0 * (1.0 - result.stats.total_optimization_ratio());
        println!("    🎯 Total pipeline optimized by {:.1}%", reduction);
    }

    // Compile the derivative to Rust code
    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::with_opt_level(RustOptLevel::O2);
    
    let func_name = "polynomial_grad";
    let rust_source = codegen.generate_function(symbolic_grad, func_name)?;
    
    let source_path = source_dir.join(format!("{}.rs", func_name));
    let lib_path = lib_dir.join(format!("lib{}.so", func_name));
    
    // Time the compilation
    let compile_start = Instant::now();
    compiler.compile_dylib(&rust_source, &source_path, &lib_path)?;
    let compilation_time = compile_start.elapsed().as_micros() as u64;
    
    println!("  🔧 Rust compilation time: {} μs", compilation_time);
    
    // Load the compiled function
    let compiled_func = unsafe { CompiledFunction::load(&lib_path, func_name)? };
    
    // Now time just the EXECUTION
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = compiled_func.call(2.0, 0.0);
    }
    let symbolic_time = start.elapsed().as_micros() as u64;

    // ad_trait version - PRE-COMPILE
    let function_standard = Polynomial::<f64>::new();
    let function_derivative = function_standard.to_other_ad_type::<adfn<1>>();
    let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardAD::new());
    let inputs = vec![2.0];
    
    // Now time just the EXECUTION
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = differentiable_block.derivative(&inputs);
    }
    let ad_trait_time = start.elapsed().as_micros() as u64;

    // Accuracy check
    let symbolic_result = compiled_func.call(2.0, 0.0);
    let (_, ad_trait_grad) = differentiable_block.derivative(&inputs);
    let ad_trait_result = ad_trait_grad[(0, 0)];
    
    Ok(BenchmarkResults {
        symbolic_ad_time_us: symbolic_time,
        ad_trait_time_us: ad_trait_time,
        accuracy_difference: (symbolic_result - ad_trait_result).abs(),
        test_name: "Polynomial".to_string(),
        compilation_time_us: compilation_time,
    })
}

#[cfg(feature = "ad_trait")]
#[derive(Clone)]
struct Multivariate<T: AD> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "ad_trait")]
impl<T: AD> DifferentiableFunctionTrait<T> for Multivariate<T> {
    const NAME: &'static str = "Multivariate";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        let y = inputs[1];
        let two = T::from_f64(2.0).unwrap_or_else(|| panic!("Failed to convert 2.0"));
        // x² + 2xy + y²
        vec![x * x + two * x * y + y * y]
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[cfg(feature = "ad_trait")]
impl<T: AD> Multivariate<T> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    fn to_other_ad_type<T2: AD>(&self) -> Multivariate<T2> {
        Multivariate::new()
    }
}

#[cfg(feature = "ad_trait")]
fn benchmark_multivariate_rust(iterations: usize, source_dir: &std::path::Path, lib_dir: &std::path::Path) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    // Symbolic AD: f(x,y) = x² + 2xy + y²
    let expr = JITEval::add(
        JITEval::add(
            JITEval::pow(JITEval::var("x"), JITEval::constant(2.0)),
            JITEval::mul(
                JITEval::constant(2.0),
                JITEval::mul(JITEval::var("x"), JITEval::var("y")),
            ),
        ),
        JITEval::pow(JITEval::var("y"), JITEval::constant(2.0)),
    );
    let symbolic_grad = convenience::gradient(&expr, &["x", "y"])?; // Pre-compile
    
    // Compile both partial derivatives to Rust code
    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::with_opt_level(RustOptLevel::O2);
    
    // Compile dx
    let func_name_dx = "multivariate_grad_dx";
    let rust_source_dx = codegen.generate_function(&symbolic_grad["x"], func_name_dx)?;
    let source_path_dx = source_dir.join(format!("{}.rs", func_name_dx));
    let lib_path_dx = lib_dir.join(format!("lib{}.so", func_name_dx));
    
    // Compile dy
    let func_name_dy = "multivariate_grad_dy";
    let rust_source_dy = codegen.generate_function(&symbolic_grad["y"], func_name_dy)?;
    let source_path_dy = source_dir.join(format!("{}.rs", func_name_dy));
    let lib_path_dy = lib_dir.join(format!("lib{}.so", func_name_dy));
    
    // Time the compilation
    let compile_start = Instant::now();
    compiler.compile_dylib(&rust_source_dx, &source_path_dx, &lib_path_dx)?;
    compiler.compile_dylib(&rust_source_dy, &source_path_dy, &lib_path_dy)?;
    let compilation_time = compile_start.elapsed().as_micros() as u64;
    
    println!("  🔧 Rust compilation time: {} μs", compilation_time);
    
    // Load the compiled functions
    let compiled_func_dx = unsafe { CompiledFunction::load(&lib_path_dx, func_name_dx)? };
    let compiled_func_dy = unsafe { CompiledFunction::load(&lib_path_dy, func_name_dy)? };
    
    // Now time just the EXECUTION
    let start = Instant::now();
    for _ in 0..iterations {
        let symbolic_dx = compiled_func_dx.call(1.0, 2.0);
        let symbolic_dy = compiled_func_dy.call(1.0, 2.0);
        let _result = (symbolic_dx, symbolic_dy);
    }
    let symbolic_time = start.elapsed().as_micros() as u64;

    // ad_trait version - PRE-COMPILE
    let function_standard = Multivariate::<f64>::new();
    let function_derivative = function_standard.to_other_ad_type::<adfn<2>>();
    let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardADMulti::new());
    let inputs = vec![1.0, 2.0];
    
    // Now time just the EXECUTION
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = differentiable_block.derivative(&inputs);
    }
    let ad_trait_time = start.elapsed().as_micros() as u64;

    // Accuracy check
    let symbolic_dx = compiled_func_dx.call(1.0, 2.0);
    let symbolic_dy = compiled_func_dy.call(1.0, 2.0);
    
    let (_, ad_trait_grad) = differentiable_block.derivative(&inputs);
    let ad_trait_dx = ad_trait_grad[(0, 0)];
    let ad_trait_dy = ad_trait_grad[(0, 1)];
    
    let accuracy_diff = (symbolic_dx - ad_trait_dx).abs() + (symbolic_dy - ad_trait_dy).abs();
    
    Ok(BenchmarkResults {
        symbolic_ad_time_us: symbolic_time,
        ad_trait_time_us: ad_trait_time,
        accuracy_difference: accuracy_diff,
        test_name: "Multivariate".to_string(),
        compilation_time_us: compilation_time,
    })
}

fn print_results(results: &BenchmarkResults) {
    let speedup = if results.symbolic_ad_time_us < results.ad_trait_time_us {
        results.ad_trait_time_us as f64 / results.symbolic_ad_time_us as f64
    } else {
        -(results.symbolic_ad_time_us as f64 / results.ad_trait_time_us as f64)
    };
    
    println!("  📊 Results:");
    println!("    Symbolic AD:  {} μs", results.symbolic_ad_time_us);
    println!("    ad_trait:     {} μs", results.ad_trait_time_us);
    println!("    Compilation:  {} μs", results.compilation_time_us);
    
    if speedup > 0.0 {
        println!("    🚀 Symbolic AD is {:.1}x faster", speedup);
    } else {
        println!("    📈 ad_trait is {:.1}x faster", -speedup);
    }
    
    println!("    Accuracy diff: {:.2e}", results.accuracy_difference);
}

fn print_summary(results: &[BenchmarkResults]) {
    println!("📋 **RUST CODEGEN BENCHMARK SUMMARY**");
    println!("====================================\n");
    
    let mut symbolic_wins = 0;
    let mut ad_trait_wins = 0;
    let mut total_symbolic_time = 0;
    let mut total_ad_trait_time = 0;
    let mut total_compilation_time = 0;
    
    for result in results {
        total_symbolic_time += result.symbolic_ad_time_us;
        total_ad_trait_time += result.ad_trait_time_us;
        total_compilation_time += result.compilation_time_us;
        
        if result.symbolic_ad_time_us < result.ad_trait_time_us {
            symbolic_wins += 1;
        } else {
            ad_trait_wins += 1;
        }
    }
    
    println!("🏆 **Performance Summary**:");
    println!("  Symbolic AD wins: {} tests", symbolic_wins);
    println!("  ad_trait wins:    {} tests", ad_trait_wins);
    
    if total_symbolic_time > 0 {
        println!("  Total execution time ratio: {:.2}x", total_ad_trait_time as f64 / total_symbolic_time as f64);
    }
    
    println!("  Total compilation time: {} μs", total_compilation_time);
    println!("  Compilation overhead: {:.1}% of execution time", 
             100.0 * total_compilation_time as f64 / total_symbolic_time as f64);
    
    println!();
    
    println!("🎯 **Key Findings**:");
    println!("  • Rust codegen provides native machine code performance");
    println!("  • Compilation overhead is amortized over repeated evaluations");
    println!("  • Symbolic optimization reduces expression complexity before compilation");
    println!("  • Hot-loading enables maximum performance for production workloads");
    println!();
    
    println!("💡 **Use Case Recommendations**:");
    println!("  • Use Rust codegen for: production systems, repeated evaluation, maximum performance");
    println!("  • Use ad_trait for: prototyping, one-off computations, immediate results");
    println!("  • Consider compilation cost vs. evaluation frequency trade-offs");
    println!("  • Symbolic optimization is crucial for complex expressions");
}

// Stub implementations for when ad_trait is not available
#[cfg(not(feature = "ad_trait"))]
fn benchmark_simple_quadratic_rust(_iterations: usize, _source_dir: &std::path::Path, _lib_dir: &std::path::Path) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    Ok(BenchmarkResults {
        symbolic_ad_time_us: 0,
        ad_trait_time_us: 0,
        accuracy_difference: 0.0,
        test_name: "Stub".to_string(),
        compilation_time_us: 0,
    })
}

#[cfg(not(feature = "ad_trait"))]
fn benchmark_polynomial_rust(_iterations: usize, _source_dir: &std::path::Path, _lib_dir: &std::path::Path) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    Ok(BenchmarkResults {
        symbolic_ad_time_us: 0,
        ad_trait_time_us: 0,
        accuracy_difference: 0.0,
        test_name: "Stub".to_string(),
        compilation_time_us: 0,
    })
}

#[cfg(not(feature = "ad_trait"))]
fn benchmark_multivariate_rust(_iterations: usize, _source_dir: &std::path::Path, _lib_dir: &std::path::Path) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    Ok(BenchmarkResults {
        symbolic_ad_time_us: 0,
        ad_trait_time_us: 0,
        accuracy_difference: 0.0,
        test_name: "Stub".to_string(),
        compilation_time_us: 0,
    })
} 