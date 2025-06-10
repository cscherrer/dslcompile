use dslcompile::prelude::*;
use dslcompile::backends::RustCodeGenerator;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Debug: Data Array Detection");
    println!("==================================");
    
    // Create the same expression structure as the demo
    let mut ctx = DynamicContext::<f64>::new();
    
    let _x = ctx.var();  // x variable (data values)
    let mu_var = ctx.var();  // mu parameter  
    let sigma_var = ctx.var();  // sigma parameter
    
    // Build the same expression with actual data
    let const_neg_half = ctx.constant(-0.5);
    let const_log_sqrt_2pi = ctx.constant((2.0 * std::f64::consts::PI).sqrt().ln());
    
    let test_data = vec![1.0, 2.0, 3.0];
    let iid_expr = ctx.sum(test_data, |x| {
        let diff = &x - &mu_var;
        let standardized = diff / &sigma_var;
        let squared = standardized.clone() * standardized;
        let neg_half_squared = &const_neg_half * squared;
        
        let log_sigma = sigma_var.clone().ln();
        let normalization = -(log_sigma + &const_log_sqrt_2pi);
        
        neg_half_squared + normalization
    });
    
    // Get the AST to debug
    let ast = dslcompile::ast::advanced::ast_from_expr(&iid_expr);
    
    println!("📋 Expression AST: {:#?}", ast);
    
    // Test the detection methods
    let codegen = RustCodeGenerator::new();
    let uses_data_arrays = codegen.expression_uses_data_arrays(ast);
    let data_array_count = codegen.count_data_arrays(ast);
    
    println!("\n🔍 Detection Results:");
    println!("   Uses data arrays: {}", uses_data_arrays);
    println!("   Data array count: {}", data_array_count);
    
    // Test code generation to see what happens
    println!("\n🔨 Generated Code:");
    match codegen.generate_function(ast, "debug_function") {
        Ok(code) => println!("{}", code),
        Err(e) => println!("Error: {:#?}", e),
    }
    
    Ok(())
} 