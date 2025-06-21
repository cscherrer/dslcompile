// Test codegen approach vs direct evaluation for variable namespace collision
use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("Testing codegen approach vs direct evaluation");
    
    // Create the problematic expression that shows variable namespace collision
    let mut ctx = StaticContext::new();
    let test_expr = ctx.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();  // StaticVar<f64, 0, 0>
        let (sum_expr, _) = scope.sum(vec![1.0, 2.0], |x| {  // x is StaticBoundVar<f64, 0, 0>
            x - mu.clone()  // This is where the collision happens in direct eval
        });
        sum_expr
    });
    
    println!("Expression created successfully");
    
    // Test 1: Direct evaluation (currently broken - should give wrong result)
    println!("\n1. Direct evaluation (current approach):");
    let direct_result = test_expr.eval(hlist![0.5]);
    println!("   Result: {} (should be 2.0: (1.0-0.5) + (2.0-0.5))", direct_result);
    
    // Test 2: Convert to AST and evaluate through AST path
    println!("\n2. AST representation and evaluation:");
    
    // Convert to AST and evaluate using AST evaluation
    // Removed Expr trait - using StaticExpr trait methods directly
    let ast = test_expr.to_ast();
    println!("   AST conversion successful");
    println!("   AST structure: {:#?}", ast);
    
    // Evaluate using AST evaluation (should work correctly)
    let ast_result = ast.eval_with_vars(&[0.5]);
    println!("   AST evaluation result: {} (should be 2.0)", ast_result);
    
    // Test 3: If we have codegen capabilities, test those
    println!("\n3. Codegen approach:");
    println!("   Checking codegen capabilities...");
    
    // For now, let's see what methods are available on the expression
    println!("   Expression type: StaticSumExpr");
    
    // Test 4: Compare with DynamicContext approach (known working)
    println!("\n4. DynamicContext comparison (known working):");
    let mut dynamic_ctx = DynamicContext::new();
    let dynamic_mu = dynamic_ctx.var();
    let dynamic_sum = dynamic_ctx.sum(vec![1.0, 2.0], |x| &x - &dynamic_mu);
    let dynamic_result = dynamic_ctx.eval(&dynamic_sum, hlist![0.5]);
    println!("   DynamicContext result: {} (reference)", dynamic_result);
    
    Ok(())
}