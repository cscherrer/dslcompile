// Debug codegen issue
use dslcompile::prelude::*;
use dslcompile::backends::{RustCodeGenerator, RustCompiler};
use dslcompile::contexts::Expr;
use frunk::hlist;

fn main() -> Result<()> {
    println!("Debugging codegen issue");
    
    // Create a simple StaticContext expression that should work with codegen
    let mut ctx = StaticContext::new();
    let simple_expr = ctx.new_scope(|scope| {
        let (x, scope) = scope.auto_var::<f64>();
        let (y, scope) = scope.auto_var::<f64>();
        // Simple arithmetic: x + y * 2.0 - use the helper function
        let two = scope.constant(2.0);
        let mul_result = y * two;
        static_add(x, mul_result)
    });
    
    println!("Simple expression created");
    
    // Convert to AST
    let ast = simple_expr.to_ast();
    println!("AST: {:#?}", ast);
    
    // Try codegen
    let codegen = RustCodeGenerator::new();
    match codegen.generate_function(&ast, "simple_func") {
        Ok(rust_code) => {
            println!("Generated Rust code:");
            println!("{}", rust_code);
            
            let compiler = RustCompiler::new();
            match compiler.compile_and_load(&rust_code, "simple_func") {
                Ok(compiled_func) => {
                    println!("Compilation successful!");
                    let result = compiled_func.call(hlist![3.0, 4.0])?;
                    println!("Result: {} (expected: 11.0)", result);
                }
                Err(e) => {
                    println!("Compilation failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Code generation failed: {}", e);
        }
    }
    
    // Now try with the problematic sum expression
    println!("\n--- Testing sum expression ---");
    let mut sum_ctx = StaticContext::new();
    let sum_expr = sum_ctx.new_scope(|scope| {
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| x.clone() * x.clone());
        sum_expr
    });
    
    let sum_ast = sum_expr.to_ast();
    println!("Sum AST: {:#?}", sum_ast);
    
    match codegen.generate_function(&sum_ast, "sum_func") {
        Ok(rust_code) => {
            println!("Sum generated Rust code:");
            println!("{}", rust_code);
        }
        Err(e) => {
            println!("Sum code generation failed: {}", e);
        }
    }
    
    Ok(())
}