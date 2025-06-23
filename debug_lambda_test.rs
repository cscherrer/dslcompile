use dslcompile::prelude::*;

fn main() {
    println!("🔍 Debugging lambda body variable issue");
    
    let mut ctx = DynamicContext::new();
    let data_param = ctx.constant_vector(vec![1.0, 2.0, 3.0]);
    
    // Create the map operation that causes the issue
    let mapped = data_param.map(|x| {
        println!("🔍 Input parameter x has AST: {:?}", x.as_ast());
        let x_ref1 = &x;
        let x_ref2 = &x;
        println!("🔍 x_ref1 has AST: {:?}", x_ref1.as_ast());
        println!("🔍 x_ref2 has AST: {:?}", x_ref2.as_ast());
        
        let result = x_ref1 * x_ref2;  // This is where the issue occurs
        println!("🔍 Multiplication result has AST: {:?}", result.as_ast());
        result
    });
    
    // Look at the lambda structure
    println!("🔍 Mapped collection structure: {:?}", mapped.collection);
}