//! StaticContext Codegen Integration Demo
//!
//! This example demonstrates how to use RustCodeGenerator with StaticContext expressions:
//! 1. Creating expressions with StaticContext
//! 2. Converting to AST via to_ast()
//! 3. Generating Rust code with RustCodeGenerator
//! 4. Comparing with DynamicContext approach

use dslcompile::prelude::*;
use dslcompile::backends::RustCompiler;
use dslcompile::contexts::Expr;  // Needed for to_ast() method
use frunk::hlist;

fn main() -> Result<()> {
    println!("🚀 StaticContext + RustCodeGenerator Demo");
    println!("=========================================\n");

    // ========================================================================
    // 1. DynamicContext approach (easier syntax for complex expressions)
    // ========================================================================
    
    println!("1️⃣ Creating expressions with DynamicContext for codegen");
    let mut dyn_ctx = DynamicContext::new();

    // Simple polynomial: x² + 2x + 1
    let x_dyn = dyn_ctx.var();
    let poly_expr_dyn = &x_dyn * &x_dyn + 2.0 * &x_dyn + 1.0;

    // More complex expression with multiple variables: x² + y² + xy
    let y_dyn = dyn_ctx.var();
    let multi_var_expr_dyn = &x_dyn * &x_dyn + &y_dyn * &y_dyn + &x_dyn * &y_dyn;

    // Sum expression: Σ(x[i] - μ)² for i in data
    let mu_dyn = dyn_ctx.var();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sum_expr_dyn = dyn_ctx.sum(data, |x| {
        let diff = &x - &mu_dyn;
        &diff * &diff
    });

    println!("✅ Created three DynamicContext expressions\n");

    // ========================================================================
    // 2. StaticContext approach (simpler expressions due to limited operators) 
    // ========================================================================
    
    println!("2️⃣ Creating simple StaticContext expressions");
    let mut static_ctx = StaticContext::new();

    // Simple addition: x + y (StaticContext has limited operator overloading)
    let simple_static_expr = static_ctx.new_scope(|scope| {
        let (x, scope) = scope.auto_var::<f64>();
        let (y, _scope) = scope.auto_var::<f64>();
        static_add(x, y)
    });

    // Simple sum with StaticContext
    let static_sum_expr = static_ctx.new_scope(|scope| {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (sum_result, _scope) = scope.sum(data, |x| x);
        sum_result
    });

    println!("✅ Created StaticContext expressions\n");

    // ========================================================================
    // 3. Convert expressions to AST for codegen
    // ========================================================================
    
    println!("3️⃣ Converting expressions to AST for codegen");
    
    // KEY STEP: Use to_ast() method to convert expressions to ASTRepr
    let poly_ast = dyn_ctx.to_ast(&poly_expr_dyn);
    let multi_var_ast = dyn_ctx.to_ast(&multi_var_expr_dyn);
    let sum_ast = dyn_ctx.to_ast(&sum_expr_dyn);
    
    // Convert StaticContext expressions to AST as well
    let simple_static_ast = simple_static_expr.to_ast();
    let static_sum_ast = static_sum_expr.to_ast();
    
    println!("✅ Converted to AST representations\n");
    
    // ========================================================================
    // 4. Generate Rust code using RustCodeGenerator
    // ========================================================================
    
    println!("4️⃣ Generating Rust code with RustCodeGenerator");
    
    let codegen = RustCodeGenerator::new();
    
    // Generate individual functions from DynamicContext expressions
    let poly_code = codegen.generate_function(&poly_ast, "polynomial_func")?;
    let multi_var_code = codegen.generate_function(&multi_var_ast, "multi_var_func")?;
    let sum_code = codegen.generate_function(&sum_ast, "sum_squares_func")?;
    
    println!("Generated polynomial function (from DynamicContext):");
    println!("{}\n", poly_code);
    
    println!("Generated multi-variable function (from DynamicContext):");
    println!("{}\n", multi_var_code);
    
    // Generate functions from StaticContext expressions
    let simple_static_code = codegen.generate_function(&simple_static_ast, "simple_static_func")?;
    let static_sum_code = codegen.generate_function(&static_sum_ast, "static_sum_func")?;
    
    println!("Generated simple function (from StaticContext):");
    println!("{}\n", simple_static_code);
    
    println!("Generated sum function (from StaticContext):");
    println!("{}\n", static_sum_code);
    
    // ========================================================================
    // 5. Generate a complete module with multiple functions
    // ========================================================================
    
    println!("5️⃣ Generating complete Rust module");
    
    let expressions = vec![
        ("polynomial_func".to_string(), poly_ast.clone()),
        ("multi_var_func".to_string(), multi_var_ast.clone()),
        ("simple_static_func".to_string(), simple_static_ast.clone()),
        ("static_sum_func".to_string(), static_sum_ast.clone()),
    ];
    
    let module_code = codegen.generate_module(&expressions, "mixed_expressions")?;
    println!("Generated module:");
    println!("{}\n", &module_code[..1000]); // Show first 1000 chars
    println!("... (truncated)\n");
    
    // ========================================================================
    // 6. Test compilation and execution (if compilation succeeds)
    // ========================================================================
    
    println!("6️⃣ Testing compilation and execution");
    
    // Try to compile and execute the polynomial function
    let compiler = RustCompiler::new();
    match compiler.compile_and_load(&poly_code, "polynomial_func") {
        Ok(compiled_func) => {
            println!("✅ Successfully compiled polynomial function!");
            
            // Test the compiled function
            let test_input = 3.0;
            let compiled_result = compiled_func.call(hlist![test_input])?;
            let direct_result = dyn_ctx.eval(&poly_expr_dyn, hlist![test_input]);
            
            println!("Test input: {}", test_input);
            println!("Compiled result: {}", compiled_result);
            println!("Direct eval result: {}", direct_result);
            println!("Results match: {}", (compiled_result - direct_result).abs() < 1e-10);
        }
        Err(e) => {
            println!("⚠️ Compilation failed (this is expected in some environments): {}", e);
            println!("The code generation worked, but compilation requires a Rust toolchain.");
        }
    }
    
    // ========================================================================
    // 7. Alternative: Use StaticCompiler for inline code generation
    // ========================================================================
    
    println!("\n7️⃣ Using StaticCompiler for inline code generation");
    
    use dslcompile::backends::StaticCompiler;
    
    let mut static_compiler = StaticCompiler::new();
    
    // Generate inline function (no FFI overhead)
    let inline_poly = static_compiler.generate_inline_function(&poly_ast, "inline_poly")?;
    println!("Generated inline function:");
    println!("{}\n", inline_poly);
    
    // Generate macro for compile-time expansion
    let poly_macro = static_compiler.generate_inline_macro(&simple_static_ast, "simple_macro")?;
    println!("Generated macro:");
    println!("{}\n", poly_macro);
    
    // ========================================================================
    // 8. Demonstrate the complete workflow
    // ========================================================================
    
    println!("8️⃣ Complete StaticContext → Codegen workflow summary");
    println!("=====================================================");
    println!("✅ DynamicContext expressions → AST → Codegen (easier for complex math)");
    println!("✅ StaticContext expressions → AST → Codegen (zero-overhead evaluation)");
    println!("\nKey API points:");
    println!("- DynamicContext: ctx.to_ast(&expr) converts expressions to ASTRepr");
    println!("- StaticContext: expr.to_ast() converts expressions to ASTRepr");  
    println!("- RustCodeGenerator works with ASTRepr<T>");
    println!("- StaticCompiler provides inline/macro generation for zero FFI overhead");
    println!("- RustCompiler.compile_and_load() creates executable functions");
    
    println!("\n📝 Usage recommendations:");
    println!("- Use DynamicContext for complex mathematical expressions with rich operators");
    println!("- Use StaticContext for compile-time optimized expressions with simpler operations");
    println!("- Both can generate the same high-performance Rust code via RustCodeGenerator");
    println!("- StaticCompiler generates inline code that can be embedded directly in your program");
    
    Ok(())
}