//! Debug script to investigate the map operation issue

use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    println!("🔍 Debug: Testing map operation with embedded data");
    
    let mut ctx = DynamicContext::new();
    let data = vec![1.0, 2.0, 3.0];
    
    // Create embedded data expression
    let data_expr = ctx.data_array(data.clone());
    println!("Data expr AST: {:?}", ctx.to_ast(&data_expr));
    
    // Create the map expression
    let mapped_expr = data_expr.map(|x| &x * &x);
    println!("Mapped expr AST: {:?}", ctx.to_ast(&mapped_expr));
    
    // Try to evaluate with no parameters (should work for embedded data)
    println!("Attempting to evaluate with hlist![]...");
    let result = ctx.eval(&mapped_expr, hlist![]);
    println!("Result: {:?}", result);
}