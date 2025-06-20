// Test all StaticContext evaluation paths to identify where variable collision occurs
use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("Testing all StaticContext evaluation paths for variable namespace collision");
    println!("Expected result: 2.0 (sum of (1.0-0.5) + (2.0-0.5))");
    println!();
    
    // Create the problematic expression
    let mut ctx = StaticContext::new();
    let test_expr = ctx.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();  // StaticVar<f64, 0, 0>
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| {  // x is StaticBoundVar<f64, 0, 0>
            x - mu.clone()  // This is where the collision happens in direct eval
        });
        sum_expr
    });
    
    // 1. Direct evaluation (currently broken)
    println!("1. DIRECT EVALUATION (StaticContext native eval):");
    let direct_result = test_expr.eval(hlist![0.5]);
    println!("   Result: {} {}", direct_result, if (direct_result - 2.0).abs() < 1e-10 { "✅" } else { "❌" });
    
    // 2. StaticContext -> ASTRepr -> AST evaluation 
    println!("\n2. STATICCONTEXT -> ASTREPR -> AST EVALUATION:");
    use dslcompile::contexts::Expr;
    let ast = test_expr.to_ast();
    let ast_result = ast.eval_with_vars(&[0.5]);
    println!("   Result: {} {}", ast_result, if (ast_result - 2.0).abs() < 1e-10 { "✅" } else { "❌" });
    
    // 3. Codegen path (if available)
    println!("\n3. CODEGEN EVALUATION:");
    // Check if we can generate and compile code
    match test_codegen_evaluation(&test_expr) {
        Ok(codegen_result) => {
            println!("   Result: {} {}", codegen_result, if (codegen_result - 2.0).abs() < 1e-10 { "✅" } else { "❌" });
        }
        Err(e) => {
            println!("   Not available or failed: {}", e);
        }
    }
    
    // 4. Test without optimization (if optimization is affecting anything)
    println!("\n4. WITHOUT EGG OPTIMIZATION:");
    let ast_no_opt = test_expr.to_ast();
    let no_opt_result = ast_no_opt.eval_with_vars(&[0.5]);
    println!("   Result: {} {}", no_opt_result, if (no_opt_result - 2.0).abs() < 1e-10 { "✅" } else { "❌" });
    println!("   (Note: No optimization applied in this test)");
    
    // Summary
    println!("\n=== SUMMARY ===");
    println!("Direct eval:     {} {}", direct_result, if (direct_result - 2.0).abs() < 1e-10 { "✅ WORKS" } else { "❌ BROKEN" });
    println!("AST eval:        {} {}", ast_result, if (ast_result - 2.0).abs() < 1e-10 { "✅ WORKS" } else { "❌ BROKEN" });
    println!("No optimization: {} {}", no_opt_result, if (no_opt_result - 2.0).abs() < 1e-10 { "✅ WORKS" } else { "❌ BROKEN" });
    
    Ok(())
}

fn test_codegen_evaluation<E>(expr: &E) -> Result<f64> 
where 
    E: Clone + std::fmt::Debug,
{
    // Try to use codegen if available
    // This is a placeholder - we'd need to check what codegen methods are available
    Err("Codegen test not implemented yet".into())
}