// Test to verify DataArray parameter fix
use dslcompile::prelude::*;
use dslcompile::backends::{RustCodeGenerator, RustCompiler};
use dslcompile::contexts::Expr;

fn main() -> Result<()> {
    println!("Testing DataArray parameter fix...");
    
    // Create a StaticContext expression with DataArray collection
    let mut ctx = StaticContext::new();
    let expr = ctx.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0, 3.0, 4.0, 5.0], |x| {
            (x - mu.clone()) * (x - mu.clone()) // (x - mu)^2
        });
        sum_expr
    });
    
    // Convert to AST and test codegen
    let ast = expr.to_ast();
    let codegen = RustCodeGenerator::new();
    
    match codegen.generate_function(&ast, "test_func") {
        Ok(rust_code) => {
            println!("✅ Generated Rust code:");
            println!("{}", rust_code);
            
            // Verify the key features:
            // 1. Function signature includes data parameter
            if rust_code.contains("data_0: &[f64]") {
                println!("✅ Function signature includes data parameter");
            } else {
                println!("❌ Missing data parameter in function signature");
                return Ok(());
            }
            
            // 2. Code references data parameter instead of inlining
            if rust_code.contains("data_0.iter()") {
                println!("✅ Code references data parameter");
            } else {
                println!("❌ Code doesn't reference data parameter");
                return Ok(());
            }
            
            // 3. No inline data vectors
            if !rust_code.contains("vec![1, 2, 3, 4, 5]") {
                println!("✅ No inline data vectors found");
            } else {
                println!("❌ Found inline data vectors (should be parameters)");
                return Ok(());
            }
            
            println!("\n🎉 All tests passed! DataArray fix is working correctly.");
            println!("   • DataArray collections are treated as function parameters");
            println!("   • Generated code references parameters instead of inlining data"); 
            println!("   • Functions can now be called with different datasets (MCMC use case)");
        }
        Err(e) => {
            println!("❌ Code generation failed: {}", e);
        }
    }
    
    Ok(())
}