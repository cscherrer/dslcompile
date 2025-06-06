use dslcompile::ast::{ASTRepr, DynamicContext, VariableRegistry};
use dslcompile::backends::cranelift::CraneliftCompiler;
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};
use dlopen2::raw::Library;
use std::time::Instant;

/// Compiled Rust function wrapper
struct CompiledRustFunction {
    _library: Library,
    function: extern "C" fn(f64) -> f64,
}

impl CompiledRustFunction {
    fn load(
        lib_path: &std::path::Path,
        func_name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let library = Library::open(lib_path)?;
        let function: extern "C" fn(f64) -> f64 =
            unsafe { library.symbol::<extern "C" fn(f64) -> f64>(func_name)? };
        Ok(Self {
            _library: library,
            function,
        })
    }

    fn call(&self, x: f64) -> f64 {
        (self.function)(x)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 TRANSCENDENTAL FUNCTION CALL OVERHEAD ANALYSIS");
    println!("=================================================");
    println!("Investigating the performance impact of external function calls");
    println!("in Cranelift JIT vs Rust -O3 for transcendental functions");
    println!();

    let iterations = 1_000_000;
    let test_value = 0.5; // Safe value for all transcendental functions

    println!("Test setup:");
    println!("  Iterations: {iterations}");
    println!("  Test value: {test_value}");
    println!();

    // ============================================================================
    // BASELINE: NATIVE RUST TRANSCENDENTAL FUNCTIONS
    // ============================================================================

    println!("📊 BASELINE: NATIVE RUST TRANSCENDENTAL FUNCTIONS");
    println!("=================================================");

    // Test each transcendental function individually
    let transcendental_tests: [(&str, fn(f64) -> f64); 5] = [
        ("sin", |x: f64| x.sin()),
        ("cos", |x: f64| x.cos()),
        ("exp", |x: f64| x.exp()),
        ("ln", |x: f64| x.ln()),
        ("sqrt", |x: f64| x.sqrt()),
    ];

    let mut native_times = Vec::new();

    for (name, func) in &transcendental_tests {
        let start = Instant::now();
        let mut result = 0.0;
        for _ in 0..iterations {
            result += func(test_value);
        }
        let time = start.elapsed();
        let ns_per_op = time.as_nanos() as f64 / iterations as f64;
        
        println!("Native {name:>4}: {ns_per_op:>6.2}ns per call (result: {result:.6})", );
        native_times.push((name, ns_per_op, result));
    }

    println!();

    // ============================================================================
    // CRANELIFT JIT: TRANSCENDENTAL FUNCTION CALLS
    // ============================================================================

    {
        println!("🚀 CRANELIFT JIT: TRANSCENDENTAL FUNCTION CALLS");
        println!("===============================================");

        let mut cranelift_times = Vec::new();

        for (name, _) in &transcendental_tests {
            // Create expression for this transcendental function
            let math = DynamicContext::new();
            let x = math.var();
            
            let expr: ASTRepr<f64> = match *name {
                "sin" => x.sin().into(),
                "cos" => x.cos().into(),
                "exp" => x.exp().into(),
                "ln" => x.ln().into(),
                "sqrt" => x.sqrt().into(),
                _ => panic!("Unknown function: {}", name),
            };

            // Compile with Cranelift
            let mut compiler = CraneliftCompiler::new_default()?;
            let registry = VariableRegistry::for_expression(&expr);
            let compiled_func = compiler.compile_expression(&expr, &registry)?;

            // Benchmark the compiled function
            let start = Instant::now();
            let mut result = 0.0;
            for _ in 0..iterations {
                result += compiled_func.call(&[test_value])?;
            }
            let time = start.elapsed();
            let ns_per_op = time.as_nanos() as f64 / iterations as f64;

            println!("Cranelift {name:>4}: {ns_per_op:>6.2}ns per call (result: {result:.6})");
            cranelift_times.push((name, ns_per_op, result));
        }

        println!();

        // ============================================================================
        // RUST -O3: TRANSCENDENTAL FUNCTION CALLS
        // ============================================================================

        println!("⚡ RUST -O3: TRANSCENDENTAL FUNCTION CALLS");
        println!("=========================================");

        let temp_dir = std::env::temp_dir().join("transcendental_overhead_test");
        let source_dir = temp_dir.join("sources");
        let lib_dir = temp_dir.join("libs");

        std::fs::create_dir_all(&source_dir)?;
        std::fs::create_dir_all(&lib_dir)?;

        let mut rust_times = Vec::new();

        for (name, _) in &transcendental_tests {
            // Create expression for this transcendental function
            let math = DynamicContext::new();
            let x = math.var();
            
            let expr: ASTRepr<f64> = match *name {
                "sin" => x.sin().into(),
                "cos" => x.cos().into(),
                "exp" => x.exp().into(),
                "ln" => x.ln().into(),
                "sqrt" => x.sqrt().into(),
                _ => panic!("Unknown function: {}", name),
            };

            // Compile with Rust -O3
            let codegen = RustCodeGenerator::new();
            let compiler = RustCompiler::with_opt_level(RustOptLevel::O3);

            let func_name = format!("{}_func", name);
            let rust_source = codegen.generate_function(&expr, &func_name)?;
            let source_path = source_dir.join(format!("{}.rs", func_name));
            let lib_path = lib_dir.join(format!("lib{}.so", func_name));

            compiler.compile_dylib(&rust_source, &source_path, &lib_path)?;
            let rust_func = CompiledRustFunction::load(&lib_path, &func_name)?;

            // Benchmark the compiled function
            let start = Instant::now();
            let mut result = 0.0;
            for _ in 0..iterations {
                result += rust_func.call(test_value);
            }
            let time = start.elapsed();
            let ns_per_op = time.as_nanos() as f64 / iterations as f64;

            println!("Rust -O3 {name:>4}: {ns_per_op:>6.2}ns per call (result: {result:.6})");
            rust_times.push((name, ns_per_op, result));
        }

        println!();

        // ============================================================================
        // ANALYSIS: CALL OVERHEAD BREAKDOWN
        // ============================================================================

        println!("🔍 ANALYSIS: CALL OVERHEAD BREAKDOWN");
        println!("===================================");

        println!("Function │ Native    │ Cranelift │ Rust -O3  │ CL Overhead │ Rust Overhead │ CL vs Rust");
        println!("─────────┼───────────┼───────────┼───────────┼─────────────┼───────────────┼───────────");

        for i in 0..transcendental_tests.len() {
            let (name, native_ns, _) = &native_times[i];
            let (_, cranelift_ns, _) = &cranelift_times[i];
            let (_, rust_ns, _) = &rust_times[i];

            let cl_overhead = cranelift_ns / native_ns;
            let rust_overhead = rust_ns / native_ns;
            let cl_vs_rust = cranelift_ns / rust_ns;

            println!(
                "{name:>8} │ {native_ns:>7.2}ns │ {cranelift_ns:>7.2}ns │ {rust_ns:>7.2}ns │ {cl_overhead:>9.2}x │ {rust_overhead:>11.2}x │ {cl_vs_rust:>8.2}x",
            );
        }

        println!();

        // ============================================================================
        // DETAILED ANALYSIS: WHY THE OVERHEAD EXISTS
        // ============================================================================

        println!("💡 DETAILED ANALYSIS: WHY THE OVERHEAD EXISTS");
        println!("=============================================");
        println!();

        println!("🔬 CRANELIFT CALL MECHANISM:");
        println!("1. Cranelift generates: call sin_wrapper");
        println!("2. sin_wrapper is extern \"C\" fn(f64) -> f64");
        println!("3. sin_wrapper calls x.sin() (Rust std)");
        println!("4. x.sin() calls libm or intrinsic");
        println!();

        println!("⚡ RUST -O3 CALL MECHANISM:");
        println!("1. Rust generates: call to x.sin()");
        println!("2. LLVM may inline or optimize the call");
        println!("3. Direct call to libm or intrinsic");
        println!("4. No wrapper function overhead");
        println!();

        println!("🎯 KEY INSIGHTS:");
        
        // Calculate average overheads
        let avg_cl_overhead: f64 = cranelift_times.iter()
            .zip(native_times.iter())
            .map(|((_, cl_ns, _), (_, native_ns, _))| cl_ns / native_ns)
            .sum::<f64>() / cranelift_times.len() as f64;

        let avg_rust_overhead: f64 = rust_times.iter()
            .zip(native_times.iter())
            .map(|((_, rust_ns, _), (_, native_ns, _))| rust_ns / native_ns)
            .sum::<f64>() / rust_times.len() as f64;

        let avg_cl_vs_rust: f64 = cranelift_times.iter()
            .zip(rust_times.iter())
            .map(|((_, cl_ns, _), (_, rust_ns, _))| cl_ns / rust_ns)
            .sum::<f64>() / cranelift_times.len() as f64;

        println!("• Average Cranelift overhead: {avg_cl_overhead:.2}x native");
        println!("• Average Rust -O3 overhead: {avg_rust_overhead:.2}x native");
        println!("• Cranelift vs Rust -O3: {avg_cl_vs_rust:.2}x slower");
        println!();

        if avg_cl_vs_rust > 2.0 {
            println!("⚠️  SIGNIFICANT OVERHEAD DETECTED!");
            println!("   Cranelift's extern \"C\" wrapper functions add substantial overhead");
            println!("   for transcendental functions compared to Rust's direct calls.");
        } else if avg_cl_vs_rust > 1.5 {
            println!("⚠️  MODERATE OVERHEAD DETECTED");
            println!("   Cranelift has noticeable but manageable overhead for transcendental functions.");
        } else {
            println!("✅ MINIMAL OVERHEAD");
            println!("   Cranelift's transcendental function overhead is acceptable.");
        }

        println!();

        // ============================================================================
        // RECOMMENDATIONS
        // ============================================================================

        println!("🎯 RECOMMENDATIONS");
        println!("==================");
        println!();

        if avg_cl_vs_rust > 2.0 {
            println!("For transcendental-heavy expressions:");
            println!("1. 🚀 Use Rust -O3 compilation for production");
            println!("2. 🔧 Consider Cranelift for development/prototyping only");
            println!("3. 🎛️  Implement adaptive compilation based on function types");
            println!("4. 🔬 Profile actual workloads to measure real-world impact");
        } else {
            println!("Cranelift overhead is acceptable for most use cases:");
            println!("1. ✅ Use Cranelift as default for fast compilation");
            println!("2. 🎯 Upgrade to Rust -O3 only for performance-critical paths");
            println!("3. 📊 The compilation speed benefit often outweighs execution overhead");
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }



    Ok(())
} 