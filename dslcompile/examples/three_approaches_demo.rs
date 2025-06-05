use dslcompile::zero_overhead_core::*;
use dslcompile::ast::DynamicContext;
use dslcompile::backends::{RustCodeGenerator, RustCompiler};
use std::time::Instant;

fn main() {
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
    
    let static_ctx: DirectComputeContext<f64> = DirectComputeContext::new();
    
    // These compile down to direct operations - literally x + y in assembly
    let static_add = static_ctx.add_direct(x, y);
    let static_mul = static_ctx.mul_direct(x, y);
    let static_complex = static_ctx.complex_direct(x, y, z);
    
    println!("  Static Add:     {} (direct: x + y)", static_add);
    println!("  Static Mul:     {} (direct: x * y)", static_mul);
    println!("  Static Complex: {} (direct: x*x + 2*x*y + y*y + z)", static_complex);
    
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
    let dynamic_expr = var_x.clone() * var_x.clone() + 
                      (var_x.clone() * var_y.clone()) * 2.0 + 
                      var_y.clone() * var_y.clone() + 
                      var_z;
    
    // Evaluate by walking the AST tree at runtime
    let interp_result1 = dynamic_expr.eval(&[3.0, 4.0, 5.0]);
    let interp_result2 = dynamic_expr.eval(&[1.0, 2.0, 3.0]);
    let interp_result3 = dynamic_expr.eval(&[5.0, 6.0, 7.0]);
    
    println!("  Interpreted expr(3,4,5): {} (AST traversal)", interp_result1);
    println!("  Interpreted expr(1,2,3): {} (AST traversal)", interp_result2);
    println!("  Interpreted expr(5,6,7): {} (AST traversal)", interp_result3);
    
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
    let rust_code = codegen.generate_function(&ast_expr, "compiled_expr").unwrap();
    
    println!("  Generated Rust code:");
    println!("  {}", rust_code.lines().take(5).collect::<Vec<_>>().join("\\n  "));
    println!("  ... (truncated)");
    
    // Compile and load if compiler is available
    if RustCompiler::is_available() {
        let compiler = RustCompiler::new();
        match compiler.compile_and_load(&rust_code, "compiled_expr") {
            Ok(compiled_func) => {
                let codegen_result1 = compiled_func.call(3.0).unwrap();
                let codegen_result2 = compiled_func.call(1.0).unwrap(); 
                let codegen_result3 = compiled_func.call(5.0).unwrap();
                
                println!("  Compiled expr(3): {} (native code)", codegen_result1);
                println!("  Compiled expr(1): {} (native code)", codegen_result2);
                println!("  Compiled expr(5): {} (native code)", codegen_result3);
            }
            Err(e) => {
                println!("  Compilation failed: {}", e);
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
    
    println!("  Static time:        {:?} ({} iterations)", static_time, iterations);
    println!("  Interpretation time: {:?} ({} iterations)", interp_time, iterations);
    
    let interp_slowdown = interp_time.as_nanos() as f64 / static_time.as_nanos() as f64;
    println!("  Interpretation slowdown: {:.1}x slower than static", interp_slowdown);
    
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