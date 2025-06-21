// Debug test to understand the namespace collision
use dslcompile::prelude::*;
use dslcompile::contexts::static_context::static_scoped::{StaticVar, StaticBoundVar};
use frunk::hlist;

fn main() {
    println!("=== DEBUGGING NAMESPACE COLLISION ===");
    
    let mut ctx = StaticContext::new();
    let test_expr = ctx.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();  // StaticVar<f64, 0, 0>
        println!("mu variable ID: {}", StaticVar::<f64, 0, 0>::var_id());
        
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| {  // x is StaticBoundVar<f64, 0, 0>
            println!("x bound variable ID: {}", StaticBoundVar::<f64, 0, 0>::bound_id());
            x - mu.clone()  // This is where the collision happens
        });
        sum_expr
    });
    
    println!("\n=== AST STRUCTURE ===");
    // Removed Expr trait - using StaticExpr trait methods directly
    let ast = test_expr.to_ast();
    println!("AST: {:#?}", ast);
    
    println!("\n=== EVALUATION COMPARISON ===");
    println!("Input: mu = 0.5");
    println!("Expected: (1.0-0.5) + (2.0-0.5) = 0.5 + 1.5 = 2.0");
    
    // Direct eval (broken)
    let direct_result = test_expr.eval(hlist![0.5]);
    println!("Direct eval result: {}", direct_result);
    
    // AST eval (works)
    let ast_result = ast.eval_with_vars(&[0.5]);
    println!("AST eval result: {}", ast_result);
}