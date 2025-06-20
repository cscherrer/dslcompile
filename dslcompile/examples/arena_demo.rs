//! Arena allocation demonstration
//!
//! This example shows the memory efficiency benefits of the new arena-based
//! AST allocation compared to the traditional Box-based allocation.

use dslcompile::{
    prelude::*,
    ast::{ASTRepr, ExprArena, ast_to_arena, arena_to_ast},
};

fn main() {
    println!("=== Arena Allocation Demonstration ===\n");

    // Create a shared subexpression using traditional Box-based AST
    let x = ASTRepr::Variable(0);
    let one = ASTRepr::Constant(1.0);
    let x_plus_one = ASTRepr::add_binary(x.clone(), one.clone());
    
    // Use the shared subexpression in multiple places: (x + 1) * (x + 1) + (x + 1)
    let expr1 = ASTRepr::mul_binary(x_plus_one.clone(), x_plus_one.clone());
    let final_expr = ASTRepr::add_binary(expr1, x_plus_one.clone());
    
    println!("Box-based AST:");
    println!("Expression: (x + 1) * (x + 1) + (x + 1)");
    println!("Memory usage: Multiple Box allocations for each occurrence of (x + 1)");
    println!("Evaluation: {}", final_expr.eval_with_vars(&[3.0])); // (3+1)*(3+1) + (3+1) = 16 + 4 = 20
    
    // Convert to arena-based representation
    let mut arena = ExprArena::new();
    let arena_expr = ast_to_arena(&final_expr, &mut arena);
    
    println!("\nArena-based AST:");
    println!("Expression: Same (x + 1) * (x + 1) + (x + 1)");
    println!("Arena nodes: {} (optimized through structural sharing)", arena.len());
    println!("Memory usage: Single arena allocation with ExprId references");
    
    // Convert back to Box-based AST to verify correctness
    let converted_back = arena_to_ast(arena_expr, &arena).unwrap();
    println!("Round-trip evaluation: {}", converted_back.eval_with_vars(&[3.0])); // Should be the same: 20
    
    // Demonstrate arena efficiency with a more complex example
    println!("\n=== Complex Expression Comparison ===");
    
    // Create a complex expression with many shared subexpressions
    let y = ASTRepr::Variable(1);
    let shared_sub1 = ASTRepr::add_binary(x.clone(), y.clone()); // x + y
    let shared_sub2 = ASTRepr::mul_binary(shared_sub1.clone(), shared_sub1.clone()); // (x + y) * (x + y)
    
    // Build: ((x + y) * (x + y)) + ((x + y) * (x + y)) + (x + y)
    let complex_expr = ASTRepr::add_binary(
        ASTRepr::add_binary(shared_sub2.clone(), shared_sub2.clone()),
        shared_sub1.clone()
    );
    
    println!("Complex expression: ((x + y)²)² + (x + y)");
    
    // Convert to arena
    let mut complex_arena = ExprArena::new();
    let _complex_arena_expr = ast_to_arena(&complex_expr, &mut complex_arena);
    
    println!("Arena nodes for complex expression: {}", complex_arena.len());
    println!("Traditional approach would require many more allocations due to duplication");
    
    // Evaluate both to verify correctness
    let test_values = [2.0, 3.0]; // x = 2, y = 3, so x + y = 5
    let box_result = complex_expr.eval_with_vars(&test_values);
    let arena_result = arena_to_ast(_complex_arena_expr, &complex_arena).unwrap().eval_with_vars(&test_values);
    
    println!("Box-based result: {}", box_result);     // ((2+3)²)² + (2+3) = 5² * 5² + 5 = 25 * 25 + 5 = 625 + 5 = 630
    println!("Arena-based result: {}", arena_result); // Should be the same
    
    assert_eq!(box_result, arena_result, "Results should be identical");
    
    println!("\n=== Memory Efficiency Summary ===");
    println!("✓ Arena allocation eliminates Box<> overhead");
    println!("✓ Structural sharing reduces memory usage for repeated subexpressions");
    println!("✓ Better cache locality for tree traversal");
    println!("✓ Single arena allocation instead of many small allocations");
    println!("✓ Preserved API compatibility through conversion utilities");
}