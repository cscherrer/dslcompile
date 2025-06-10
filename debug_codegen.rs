use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::backends::RustCodeGenerator;

fn main() {
    // Create a simple Gaussian likelihood example
    let mut ctx = DynamicContext::new();
    
    let mu = ctx.var();      // var_0
    let sigma = ctx.var();   // var_1
    
    // Create simple data summation like the demo
    let data = vec![1.0, 2.0, 3.0]; // placeholder
    let sum_expr = ctx.sum(data, |x| {
        let standardized = (x - &mu) / &sigma;
        -0.5 * (&standardized * &standardized)
    });
    
    // Generate code
    let codegen = RustCodeGenerator::new();
    let ast = dslcompile::ast::advanced::ast_from_expr(&sum_expr);
    
    match codegen.generate_function(ast, "test_gaussian") {
        Ok(code) => {
            println!("Generated Code:");
            println!("{}", code);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
} 