//! Heterogeneous System vs Cranelift vs Rust Codegen Benchmark
//!
//! Direct comparison of our zero-overhead heterogeneous system against
//! traditional compilation backends for identical mathematical operations.

use divan::Bencher;
use dlopen2::raw::Library;
use dslcompile::SymbolicOptimizer;
use dslcompile::ast::{ASTRepr, ExpressionBuilder};
#[cfg(feature = "cranelift")]
use dslcompile::backends::cranelift::{CompiledFunction, CraneliftCompiler};
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use dslcompile::compile_time::heterogeneous::{
    HeteroContext, HeteroExpr, HeteroInputs, hetero_add,
};
use dslcompile::compile_time::scoped::{Context, ScopedMathExpr, ScopedVarArray};
use std::fs;

/// Compiled Rust function wrapper
struct CompiledRustFunction {
    _library: Library,
    function: extern "C" fn(f64, f64) -> f64,
}

impl CompiledRustFunction {
    fn load(
        lib_path: &std::path::Path,
        func_name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let library = Library::open(lib_path)?;
        let function: extern "C" fn(f64, f64) -> f64 =
            unsafe { library.symbol::<extern "C" fn(f64, f64) -> f64>(func_name)? };
        Ok(Self {
            _library: library,
            function,
        })
    }

    fn call(&self, x: f64, y: f64) -> f64 {
        (self.function)(x, y)
    }
}

/// Create the test expression: x + y
fn create_test_expr() -> ASTRepr<f64> {
    let math = ExpressionBuilder::new();
    let x = math.var();
    let y = math.var();
    (&x + &y).into()
}

/// Setup all systems for comparison
fn setup_all_systems() -> Result<
    (
        // Heterogeneous system - function of two variables
        impl Fn(f64, f64) -> f64,
        // Optimized heterogeneous system - stack allocated
        impl Fn(f64, f64) -> f64,
        // Old Context system - function of two variables
        impl Fn(f64, f64) -> f64,
        // Cranelift system
        CompiledFunction,
        // Rust codegen system
        CompiledRustFunction,
    ),
    Box<dyn std::error::Error>,
> {
    // 1. HETEROGENEOUS SYSTEM - FUNCTION OF TWO VARIABLES
    let mut hetero_ctx = HeteroContext::<0, 8>::new();
    let x_hetero = hetero_ctx.var::<f64>();
    let y_hetero = hetero_ctx.var::<f64>();
    let hetero_expr = hetero_add::<f64, _, _, 0>(x_hetero, y_hetero);

    let hetero_fn = move |x: f64, y: f64| -> f64 {
        let mut inputs = HeteroInputs::<8>::new();
        inputs.add_f64(0, x);
        inputs.add_f64(1, y);
        hetero_expr.eval(&inputs)
    };

    // 1b. OPTIMIZED HETEROGENEOUS SYSTEM - STACK ALLOCATED
    let mut hetero_ctx2 = HeteroContext::<0, 8>::new();
    let x_hetero2 = hetero_ctx2.var::<f64>();
    let y_hetero2 = hetero_ctx2.var::<f64>();
    let hetero_expr2 = hetero_add::<f64, _, _, 0>(x_hetero2, y_hetero2);

    let hetero_optimized_fn = move |x: f64, y: f64| -> f64 {
        // Stack-allocated, zero-cost initialization
        let mut inputs = HeteroInputs::<8>::default();
        // Direct array access instead of method calls
        inputs.f64_values[0] = Some(x);
        inputs.f64_values[1] = Some(y);
        hetero_expr2.eval(&inputs)
    };

    // 2. OLD CONTEXT SYSTEM - FUNCTION OF TWO VARIABLES
    let mut old_ctx = Context::new();
    let old_expr = old_ctx.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var();
        x + y
    });

    let old_fn = move |x: f64, y: f64| -> f64 {
        let vars = ScopedVarArray::new(vec![x, y]);
        old_expr.eval(&vars)
    };

    // 3. CRANELIFT SYSTEM
    let expr = create_test_expr();
    let mut optimizer = SymbolicOptimizer::new()?;
    let optimized = optimizer.optimize(&expr)?;

    let mut cranelift_compiler = CraneliftCompiler::new_default()?;
    let registry = dslcompile::ast::VariableRegistry::for_expression(&optimized);
    let cranelift_func = cranelift_compiler.compile_expression(&optimized, &registry)?;

    // 4. RUST CODEGEN SYSTEM
    let temp_dir = std::env::temp_dir().join("dslcompile_hetero_vs_cranelift_bench");
    let source_dir = temp_dir.join("sources");
    let lib_dir = temp_dir.join("libs");

    fs::create_dir_all(&source_dir)?;
    fs::create_dir_all(&lib_dir)?;

    let codegen = RustCodeGenerator::new();
    let compiler = RustCompiler::with_opt_level(RustOptLevel::O3);

    let rust_source = codegen.generate_function(&optimized, "add_func")?;
    let source_path = source_dir.join("add_func.rs");
    let lib_path = lib_dir.join("libadd_func.so");

    compiler.compile_dylib(&rust_source, &source_path, &lib_path)?;
    let rust_func = CompiledRustFunction::load(&lib_path, "add_func")?;

    Ok((
        hetero_fn,
        hetero_optimized_fn,
        old_fn,
        cranelift_func,
        rust_func,
    ))
}

#[divan::bench]
fn heterogeneous_system(bencher: Bencher) {
    let (hetero_fn, _, _, _, _) = setup_all_systems().unwrap();
    bencher.bench_local(|| hetero_fn(3.0, 4.0));
}

#[divan::bench]
fn heterogeneous_optimized(bencher: Bencher) {
    let (_, hetero_optimized_fn, _, _, _) = setup_all_systems().unwrap();
    bencher.bench_local(|| hetero_optimized_fn(3.0, 4.0));
}

#[divan::bench]
fn old_context_system(bencher: Bencher) {
    let (_, _, old_fn, _, _) = setup_all_systems().unwrap();
    bencher.bench_local(|| old_fn(3.0, 4.0));
}

#[divan::bench]
fn cranelift_system(bencher: Bencher) {
    let (_, _, _, cranelift_func, _) = setup_all_systems().unwrap();
    bencher.bench_local(|| cranelift_func.call(&[3.0, 4.0]).unwrap());
}

#[divan::bench]
fn rust_codegen_system(bencher: Bencher) {
    let (_, _, _, _, rust_func) = setup_all_systems().unwrap();
    bencher.bench_local(|| rust_func.call(3.0, 4.0));
}

#[divan::bench]
fn direct_rust_baseline(bencher: Bencher) {
    bencher.bench_local(|| 3.0_f64 + 4.0_f64);
}

fn main() {
    println!("🚀 Heterogeneous System vs All Backends Comparison");
    println!("==================================================");
    println!();

    println!("🎯 Testing identical operation: x + y");
    println!("📊 All systems pre-compiled, measuring pure execution speed");
    println!();

    // Verify all systems produce the same result
    let (hetero_fn, hetero_optimized_fn, old_fn, cranelift_func, rust_func) =
        setup_all_systems().unwrap();

    let hetero_result = hetero_fn(3.0, 4.0);
    let hetero_optimized_result = hetero_optimized_fn(3.0, 4.0);
    let old_result = old_fn(3.0, 4.0);
    let cranelift_result = cranelift_func.call(&[3.0, 4.0]).unwrap();
    let rust_result = rust_func.call(3.0, 4.0);
    let direct_result = 3.0_f64 + 4.0_f64;

    println!("✅ Result verification:");
    println!("  Heterogeneous: {hetero_result}");
    println!("  Optimized Heterogeneous: {hetero_optimized_result}");
    println!("  Old Context:   {old_result}");
    println!("  Cranelift:     {cranelift_result}");
    println!("  Rust Codegen:  {rust_result}");
    println!("  Direct Rust:   {direct_result}");

    assert_eq!(hetero_result, 7.0);
    assert_eq!(hetero_optimized_result, 7.0);
    assert_eq!(old_result, 7.0);
    assert_eq!(cranelift_result, 7.0);
    assert_eq!(rust_result, 7.0);
    assert_eq!(direct_result, 7.0);

    println!("  🎯 All systems produce identical results!\n");

    divan::main();
}
