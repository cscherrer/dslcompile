use dslcompile::ast::DynamicContext;
use dslcompile::backends::{RustCodeGenerator, RustCompiler};
// Note: zero_overhead_core removed - using Static Scoped System instead
use std::time::Instant;

fn main() -> Result<()> {
    println!("🔥 Three Distinct Approaches: Static, Dynamic Interpretation, Dynamic Codegen");
    println!("==============================================================================");

    let x = 3.0;
    let y = 4.0;
    let z = 5.0;

    // ============================================================================
    // APPROACH 1: STATIC ZERO-OVERHEAD (Compile-time optimization)
    // ============================================================================
    println!("\n⚡ APPROACH 1: STATIC ZERO-OVERHEAD");
    println!("   - Direct computation, no runtime overhead");
    println!("   - Values known at compile time");

    // Note: DirectComputeContext removed - using Static Scoped System instead
    // TODO: Migrate to Static Scoped System
    println!("DirectComputeContext removed - demo needs migration to Static Scoped System");
    return Ok(());

    // These compile down to direct operations - literally x + y in assembly
    let static_add = static_ctx.add_direct(x, y);
    let static_mul = static_ctx.mul_direct(x, y);
    let static_complex = static_ctx.complex_direct(x, y, z);

    println!("  Static Add:     {static_add} (direct: x + y)");
    println!("  Static Mul:     {static_mul} (direct: x * y)");
    println!("  Static Complex: {static_complex} (direct: x*x + 2*x*y + y*y + z)");

    // ============================================================================
    // APPROACH 2: DYNAMIC INTERPRETATION (Runtime flexibility)
    // ============================================================================
    println!("\n🌊 APPROACH 2: DYNAMIC INTERPRETATION");
    println!("   - Expression tree interpretation");
    println!("   - Runtime flexibility, can change expressions");

    let dynamic_ctx = DynamicContext::new();

    // Build expressions at runtime - creates AST nodes
    let var_x = dynamic_ctx.var();
    let var_y = dynamic_ctx.var();
    let var_z = dynamic_ctx.var();

    // Create complex expression: x*x + 2*x*y + y*y + z
    let dynamic_expr = var_x.clone() * var_x.clone()
        + (var_x.clone() * var_y.clone()) * 2.0
        + var_y.clone() * var_y.clone()
        + var_z;

    // Evaluate by walking the AST tree at runtime
    let interp_result1 = dynamic_expr.eval(&[3.0, 4.0, 5.0]);
    let interp_result2 = dynamic_expr.eval(&[1.0, 2.0, 3.0]);
    let interp_result3 = dynamic_expr.eval(&[5.0, 6.0, 7.0]);

    println!("  Interpreted expr(3,4,5): {interp_result1} (AST traversal)");
    println!("  Interpreted expr(1,2,3): {interp_result2} (AST traversal)");
    println!("  Interpreted expr(5,6,7): {interp_result3} (AST traversal)");

    // ============================================================================
    // APPROACH 3: DYNAMIC CODEGEN (Runtime compilation)
    // ============================================================================
    println!("\n🔧 APPROACH 3: DYNAMIC CODEGEN");
    println!("   - Runtime code generation and compilation");
    println!("   - Native performance after compilation");

    // Convert expression to AST for codegen
    let ast_expr = dynamic_expr.clone().into();

    // Generate Rust code from the expression
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen
        .generate_function(&ast_expr, "compiled_expr")
        .unwrap();

    println!("  Generated Rust code:");
    println!(
        "  {}",
        rust_code.lines().take(5).collect::<Vec<_>>().join("\\n  ")
    );
    println!("  ... (truncated)");

    // Compile and load if compiler is available
    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        match compiler.compile_and_load(&rust_code, "compiled_expr") {
            Ok(compiled_func) => {
                let codegen_result1 = compiled_func.call(3.0).unwrap();
                let codegen_result2 = compiled_func.call(1.0).unwrap();
                let codegen_result3 = compiled_func.call(5.0).unwrap();

                println!("  Compiled expr(3): {codegen_result1} (native code)");
                println!("  Compiled expr(1): {codegen_result2} (native code)");
                println!("  Compiled expr(5): {codegen_result3} (native code)");
            }
            Err(e) => {
                println!("  Compilation failed: {e}");
                println!("  (This is expected in some environments)");
            }
        }
    } else {
        println!("  Rust compiler not available for dynamic compilation");
    }

    // ============================================================================
    // PERFORMANCE COMPARISON
    // ============================================================================
    println!("\n📊 PERFORMANCE COMPARISON:");

    let iterations = 100_000;

    // Static performance (zero overhead)
    let start = Instant::now();
    let mut static_sum = 0.0;
    for _ in 0..iterations {
        static_sum += static_ctx.add_direct(x, y);
    }
    let static_time = start.elapsed();

    // Dynamic interpretation performance
    let start = Instant::now();
    let mut interp_sum = 0.0;
    for _ in 0..iterations {
        interp_sum += dynamic_expr.eval(&[x, y, z]);
    }
    let interp_time = start.elapsed();

    println!("  Static time:        {static_time:?} ({iterations} iterations)");
    println!("  Interpretation time: {interp_time:?} ({iterations} iterations)");

    let interp_slowdown = interp_time.as_nanos() as f64 / static_time.as_nanos() as f64;
    println!("  Interpretation slowdown: {interp_slowdown:.1}x slower than static");

    // Note: Codegen performance would be similar to static after compilation,
    // but has compilation overhead upfront

    // ============================================================================
    // SUMMARY: WHEN TO USE EACH APPROACH
    // ============================================================================
    println!("\n✅ WHEN TO USE EACH APPROACH:");
    println!("  🚀 Static Zero-Overhead:");
    println!("     - Values known at compile time");
    println!("     - Maximum performance critical");
    println!("     - No runtime expression changes needed");

    println!("  🌊 Dynamic Interpretation:");
    println!("     - Expressions change frequently at runtime");
    println!("     - Quick prototyping and experimentation");
    println!("     - Don't want compilation overhead");

    println!("  🔧 Dynamic Codegen:");
    println!("     - Expressions known at runtime but stable");
    println!("     - Need native performance for repeated evaluation");
    println!("     - Can amortize compilation cost over many evaluations");

    println!("\n🎯 UNIFIED API: All three use similar syntax!");
    println!("   - static_ctx.add_direct(x, y)");
    println!("   - dynamic_expr.eval(&[x, y, z])");
    println!("   - compiled_func.call(x)");
}
