//! Test Normalization
//!
//! Simple test to verify that Sub -> Add+Neg normalization works

use dslcompile::{
    ast::{ASTRepr, normalization::normalize},
    prelude::*,
};

fn main() -> Result<()> {
    println!("🔧 Testing Expression Normalization");
    println!("===================================\n");

    // Test simple subtraction: x - y
    let x = ASTRepr::<f64>::Variable(0);
    let y = ASTRepr::<f64>::Variable(1);
    let sub_expr = ASTRepr::Sub(Box::new(x), Box::new(y));
    
    println!("1️⃣ Before normalization:");
    println!("   Expression: {:?}", sub_expr);
    
    let normalized = normalize(&sub_expr);
    println!("\n2️⃣ After normalization:");
    println!("   Expression: {:?}", normalized);
    
    // Test nested: (x - y) / z
    let z = ASTRepr::<f64>::Variable(2);
    let div_expr = ASTRepr::Div(Box::new(sub_expr.clone()), Box::new(z));
    
    println!("\n3️⃣ Nested expression before normalization:");
    println!("   Expression: {:?}", div_expr);
    
    let normalized_nested = normalize(&div_expr);
    println!("\n4️⃣ Nested expression after normalization:");
    println!("   Expression: {:?}", normalized_nested);
    
    // Test with DynamicContext to see actual structure
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    let b = ctx.var::<f64>();
    let c = ctx.var::<f64>();
    
    let expr = (&a - &b) / &c;
    println!("\n5️⃣ DynamicContext expression:");
    println!("   Pretty: {}", ctx.pretty_print(&expr));
    
    let ast = ctx.to_ast(&expr);
    println!("   AST: {:?}", ast);
    
    let normalized_ctx = normalize(&ast);
    println!("   Normalized AST: {:?}", normalized_ctx);

    Ok(())
}