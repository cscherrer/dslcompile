use dslcompile::prelude::*;
use frunk::hlist;

fn main() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var();
    
    // Test two separate sums: 
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];
    
    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    let total = &sum1 + &sum2;
    
    let total_ast = ctx.to_ast(&total);
    
    println!("Original AST structure:");
    println!("{:#?}", total_ast);
    
    let test_x = 2.0;
    let result = ctx.eval(&total, hlist![test_x]);
    println!("Original result: {}", result);
}