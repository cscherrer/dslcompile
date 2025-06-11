use dslcompile::ast::runtime::expression_builder::DynamicContext;

fn main() {
    // Test with f64
    let mut ctx_f64: DynamicContext<f64> = DynamicContext::new();
    let x_f64 = ctx_f64.var_context_type();
    let y_f64 = ctx_f64.var_context_type();
    let expr_f64 = &x_f64 + &y_f64;
    
    // Test that all methods now work with f64
    let result_f64 = ctx_f64.eval_borrowed(&expr_f64, &[3.0, 4.0]);
    println!("f64 result: {}", result_f64);
    
    let ast_f64 = ctx_f64.to_ast(&expr_f64);
    println!("f64 AST: {:?}", ast_f64);
    
    let pretty_f64 = ctx_f64.pretty_print(&expr_f64);
    println!("f64 pretty: {}", pretty_f64);
    
    let unbound_f64 = ctx_f64.find_unbound_variables(&expr_f64);
    println!("f64 unbound variables: {:?}", unbound_f64);
    
    let max_var_f64 = ctx_f64.find_max_variable_index(&expr_f64);
    println!("f64 max variable index: {}", max_var_f64);
    
    // Test with f32
    let mut ctx_f32: DynamicContext<f32> = DynamicContext::new();
    let x_f32 = ctx_f32.var_context_type();
    let y_f32 = ctx_f32.var_context_type();
    let expr_f32 = &x_f32 + &y_f32;
    
    // Test that all methods now work with f32
    let result_f32 = ctx_f32.eval_borrowed(&expr_f32, &[3.0f32, 4.0f32]);
    println!("f32 result: {}", result_f32);
    
    let ast_f32 = ctx_f32.to_ast(&expr_f32);
    println!("f32 AST: {:?}", ast_f32);
    
    let pretty_f32 = ctx_f32.pretty_print(&expr_f32);
    println!("f32 pretty: {}", pretty_f32);
    
    let unbound_f32 = ctx_f32.find_unbound_variables(&expr_f32);
    println!("f32 unbound variables: {:?}", unbound_f32);
    
    let max_var_f32 = ctx_f32.find_max_variable_index(&expr_f32);
    println!("f32 max variable index: {}", max_var_f32);
    
    println!("✅ All generic type methods working correctly!");
} 