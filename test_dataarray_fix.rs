use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    let mut ctx = DynamicContext::new();
    let data = vec![1.0, 2.0, 3.0];
    
    // Simple test: sum over data without external variables
    let sum_expr = ctx.sum(data, |x| x * 2.0);
    
    // Print the AST to see if it uses DataArray now
    let ast = ctx.to_ast(&sum_expr);
    println!("AST: {:#?}", ast);
    
    // Try to evaluate with empty HList
    let result = ctx.eval(&sum_expr, hlist![]);
    println!("Result: {}", result);
}