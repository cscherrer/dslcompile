use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use rand::prelude::*;
use std::process::Command;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 WHY SAME-PROCESS DYNAMIC COMPILATION HAS OVERHEAD");
    println!("====================================================");
    println!("Investigating the fundamental bottlenecks in runtime codegen:");
    println!("1. Same-process vs separate-process execution");
    println!("2. Dynamic library loading overhead");
    println!("3. Function pointer indirection costs");
    println!("4. JIT compilation patterns");
    println!("5. How normal external calls avoid these issues");
    println!();

    // Generate random test data to prevent constant propagation
    let mut rng = thread_rng();
    let iterations = 1_000_000;
    let test_data: Vec<(f64, f64)> = (0..iterations)
        .map(|_| (rng.gen_range(-100.0..100.0), rng.gen_range(-100.0..100.0)))
        .collect();

    println!("Generated {iterations} random test pairs");
    println!("Data range: x,y ∈ [-100, 100]");
    println!();

    // ============================================================================
    // BASELINE: NATIVE RUST PERFORMANCE
    // ============================================================================

    println!("📊 BASELINE: NATIVE RUST PERFORMANCE");
    println!("====================================");

    let start = Instant::now();
    let mut native_sum = 0.0;
    for &(x, y) in &test_data {
        native_sum += x + y;
    }
    let native_time = start.elapsed();
    let native_ns_per_op = native_time.as_nanos() as f64 / iterations as f64;

    println!("Native Rust (x + y): {native_ns_per_op:.3}ns per operation");
    println!();

    // ============================================================================
    // ANALYSIS 1: SAME-PROCESS DYNAMIC LIBRARY LOADING
    // ============================================================================

    println!("🔬 ANALYSIS 1: SAME-PROCESS DYNAMIC LIBRARY OVERHEAD");
    println!("====================================================");
    println!("Breaking down the overhead sources in runtime compilation");
    println!();

    // Build expression for codegen
    let ctx = DynamicContext::new();
    let x_var = ctx.var();
    let y_var = ctx.var();
    let add_expr = &x_var + &y_var;
    let ast_expr: ASTRepr<f64> = add_expr.into();

    let codegen = RustCodeGenerator::new();
    let temp_dir = std::env::temp_dir();

    // Test 1a: Measure compilation time
    let compile_start = Instant::now();
    let rust_code = codegen.generate_function(&ast_expr, "add_func_timing")?;
    let source_path = temp_dir.join("timing_add.rs");
    let lib_path = temp_dir.join("libtiming_add.so");
    std::fs::write(&source_path, &rust_code)?;

    let compiler = RustCompiler::with_opt_level(RustOptLevel::O3);
    compiler.compile_dylib(&rust_code, &source_path, &lib_path)?;
    let compile_time = compile_start.elapsed();

    println!("Compilation time: {:.3}ms", compile_time.as_millis());

    // Test 1b: Measure library loading time
    let load_start = Instant::now();
    use dlopen2::raw::Library;
    let library = Library::open(&lib_path)?;
    let load_time = load_start.elapsed();

    println!("Library loading time: {:.3}μs", load_time.as_micros());

    // Test 1c: Measure symbol resolution time
    let symbol_start = Instant::now();
    let add_func: extern "C" fn(f64, f64) -> f64 = unsafe { library.symbol("add_func_timing")? };
    let symbol_time = symbol_start.elapsed();

    println!("Symbol resolution time: {:.3}μs", symbol_time.as_micros());

    // Test 1d: Measure actual function call overhead
    let call_start = Instant::now();
    let mut dynamic_sum = 0.0;
    for &(x, y) in &test_data {
        dynamic_sum += add_func(x, y);
    }
    let call_time = call_start.elapsed();
    let call_ns_per_op = call_time.as_nanos() as f64 / iterations as f64;

    println!("Function call time: {call_ns_per_op:.3}ns per operation");

    let call_overhead = call_ns_per_op / native_ns_per_op;
    println!("Call overhead: {call_overhead:.2}x native");

    println!();

    // ============================================================================
    // ANALYSIS 2: SEPARATE PROCESS EXECUTION (HOW NORMAL CALLS WORK)
    // ============================================================================

    println!("🚀 ANALYSIS 2: SEPARATE PROCESS EXECUTION");
    println!("==========================================");
    println!("How normal external calls avoid same-process overhead");
    println!();

    // Create a simple external program that does the same computation
    let external_program = r#"
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: program <x> <y>");
        std::process::exit(1);
    }
    
    let x: f64 = args[1].parse().unwrap();
    let y: f64 = args[2].parse().unwrap();
    let result = x + y;
    println!("{}", result);
}
"#
    .to_string();

    let external_source = temp_dir.join("external_add.rs");
    let external_binary = temp_dir.join("external_add");
    std::fs::write(&external_source, &external_program)?;

    // Compile external program
    let compile_external_start = Instant::now();
    let compile_status = Command::new("rustc")
        .args([
            "-O",
            &external_source.to_string_lossy(),
            "-o",
            &external_binary.to_string_lossy(),
        ])
        .status()?;
    let compile_external_time = compile_external_start.elapsed();

    if !compile_status.success() {
        println!("Failed to compile external program");
        return Ok(());
    }

    println!(
        "External program compilation: {:.3}ms",
        compile_external_time.as_millis()
    );

    // Test external program call overhead (sample a few calls)
    let sample_size = 1000; // Much smaller sample for process calls
    let external_start = Instant::now();
    let mut external_results = Vec::new();

    for &(x, y) in &test_data[..sample_size] {
        let output = Command::new(&external_binary)
            .args(&[x.to_string(), y.to_string()])
            .output()?;

        if output.status.success() {
            let result: f64 = String::from_utf8(output.stdout)?.trim().parse()?;
            external_results.push(result);
        }
    }

    let external_time = external_start.elapsed();
    let external_ns_per_op = external_time.as_nanos() as f64 / sample_size as f64;
    let external_overhead = external_ns_per_op / native_ns_per_op;

    println!("External process call: {external_ns_per_op:.3}ns per operation");
    println!("External process overhead: {external_overhead:.2}x native");
    println!("(Note: This includes process spawn overhead)");

    println!();

    // ============================================================================
    // ANALYSIS 3: FUNCTION POINTER INDIRECTION BREAKDOWN
    // ============================================================================

    println!("🎯 ANALYSIS 3: FUNCTION POINTER INDIRECTION BREAKDOWN");
    println!("======================================================");
    println!("Isolating the specific overhead sources");
    println!();

    // Test 3a: Direct function call (baseline)
    #[inline(never)] // Prevent inlining to make it comparable
    fn direct_add(x: f64, y: f64) -> f64 {
        x + y
    }

    let direct_start = Instant::now();
    let mut direct_sum = 0.0;
    for &(x, y) in &test_data {
        direct_sum += direct_add(x, y);
    }
    let direct_time = direct_start.elapsed();
    let direct_ns_per_op = direct_time.as_nanos() as f64 / iterations as f64;

    // Test 3b: Function pointer (no dynamic loading)
    let fn_ptr: fn(f64, f64) -> f64 = direct_add;

    let fn_ptr_start = Instant::now();
    let mut fn_ptr_sum = 0.0;
    for &(x, y) in &test_data {
        fn_ptr_sum += fn_ptr(x, y);
    }
    let fn_ptr_time = fn_ptr_start.elapsed();
    let fn_ptr_ns_per_op = fn_ptr_time.as_nanos() as f64 / iterations as f64;

    // Test 3c: Option-wrapped function pointer (simulating dynamic loading)
    let optional_fn: Option<fn(f64, f64) -> f64> = Some(direct_add);

    let optional_start = Instant::now();
    let mut optional_sum = 0.0;
    for &(x, y) in &test_data {
        if let Some(func) = optional_fn {
            optional_sum += func(x, y);
        }
    }
    let optional_time = optional_start.elapsed();
    let optional_ns_per_op = optional_time.as_nanos() as f64 / iterations as f64;

    println!("Direct function call:     {direct_ns_per_op:.3}ns per operation");
    println!("Function pointer:         {fn_ptr_ns_per_op:.3}ns per operation");
    println!("Option-wrapped pointer:   {optional_ns_per_op:.3}ns per operation");
    println!("Dynamic library call:     {call_ns_per_op:.3}ns per operation");

    let direct_overhead = direct_ns_per_op / native_ns_per_op;
    let fn_ptr_overhead = fn_ptr_ns_per_op / native_ns_per_op;
    let optional_overhead = optional_ns_per_op / native_ns_per_op;

    println!("Direct call overhead:     {direct_overhead:.2}x");
    println!("Function pointer overhead: {fn_ptr_overhead:.2}x");
    println!("Option wrapper overhead:  {optional_overhead:.2}x");
    println!("Dynamic library overhead: {call_overhead:.2}x");

    println!();

    // ============================================================================
    // ANALYSIS 4: WHY SAME-PROCESS IS HARD
    // ============================================================================

    println!("💡 ANALYSIS 4: WHY SAME-PROCESS DYNAMIC COMPILATION IS HARD");
    println!("============================================================");
    println!();

    println!("🔍 OVERHEAD BREAKDOWN:");
    println!(
        "1. Compilation: {:.3}ms (one-time cost)",
        compile_time.as_millis()
    );
    println!(
        "2. Library loading: {:.3}μs (one-time cost)",
        load_time.as_micros()
    );
    println!(
        "3. Symbol resolution: {:.3}μs (one-time cost)",
        symbol_time.as_micros()
    );
    println!("4. Function call: {call_ns_per_op:.3}ns per call (repeated cost)");
    println!();

    println!("📊 OVERHEAD SOURCES ANALYSIS:");
    let total_one_time_overhead =
        compile_time.as_nanos() + load_time.as_nanos() + symbol_time.as_nanos();
    let per_call_overhead = call_ns_per_op - native_ns_per_op;

    println!(
        "• One-time overhead: {:.3}ms",
        total_one_time_overhead as f64 / 1_000_000.0
    );
    println!("• Per-call overhead: {per_call_overhead:.3}ns");
    println!(
        "• Break-even point: {:.0} calls",
        total_one_time_overhead as f64 / per_call_overhead
    );

    println!();

    println!("🚀 HOW NORMAL EXTERNAL CALLS AVOID THIS:");
    println!("1. **Pre-compilation**: Libraries compiled ahead of time");
    println!("2. **Static linking**: No dynamic loading overhead");
    println!("3. **Direct calls**: No function pointer indirection");
    println!("4. **Compiler optimizations**: Full optimization across call boundaries");
    println!();

    println!("⚠️  WHY SAME-PROCESS DYNAMIC COMPILATION IS HARD:");
    println!("1. **Security isolation**: Dynamic libraries can't be fully trusted");
    println!("2. **Memory management**: Shared address space complications");
    println!("3. **Symbol resolution**: Runtime symbol lookup overhead");
    println!("4. **Optimization barriers**: Compiler can't optimize across dynamic boundaries");
    println!("5. **ABI overhead**: extern \"C\" calling convention costs");
    println!();

    println!("🎯 POTENTIAL SOLUTIONS:");
    println!("1. **JIT compilation**: Generate machine code directly in memory");
    println!("2. **Bytecode interpretation**: Avoid compilation altogether");
    println!("3. **Template specialization**: Pre-compile common patterns");
    println!("4. **Hybrid approach**: Cache compiled functions, interpret complex ones");
    println!("5. **LLVM JIT**: Use LLVM's JIT infrastructure for zero-overhead calls");

    // Cleanup
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&lib_path);
    let _ = std::fs::remove_file(&external_source);
    let _ = std::fs::remove_file(&external_binary);

    Ok(())
}
