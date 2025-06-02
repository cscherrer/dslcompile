use mathcompile::prelude::*;

fn main() -> mathcompile::Result<()> {
    println!("=== A-Normal Form (ANF) Demonstration ===\n");

    // Create a variable registry
    let mut registry = VariableRegistry::new();
    let x_idx = registry.register_variable();
    let y_idx = registry.register_variable();

    // Create a complex expression with repeated subexpressions:
    // sin(x + y) + cos(x + y) + exp(x + y)
    // Notice how (x + y) appears three times!
    let x = <ASTEval as ASTMathExpr>::var(x_idx);
    let y = <ASTEval as ASTMathExpr>::var(y_idx);
    let x_plus_y = <ASTEval as ASTMathExpr>::add(x, y);

    let sin_term = <ASTEval as ASTMathExpr>::sin(x_plus_y.clone());
    let cos_term = <ASTEval as ASTMathExpr>::cos(x_plus_y.clone());
    let exp_term = <ASTEval as ASTMathExpr>::exp(x_plus_y);

    let sum1 = <ASTEval as ASTMathExpr>::add(sin_term, cos_term);
    let final_expr = <ASTEval as ASTMathExpr>::add(sum1, exp_term);

    println!("Original expression: sin(var_0 + var_1) + cos(var_0 + var_1) + exp(var_0 + var_1)");
    println!("Notice how (var_0 + var_1) is computed three times!\n");

    // Convert to ANF - this automatically performs CSE!
    let anf = convert_to_anf(&final_expr)?;

    println!("ANF automatically introduces temporary variables:");
    println!("Let count: {}", anf.let_count());
    println!("Variables used: {:?}\n", anf.used_variables());

    // Generate clean Rust code
    let codegen = ANFCodeGen::new(&registry);
    let function_code = codegen.generate_function("optimized_function", &anf);

    println!("Generated optimized Rust code:");
    println!("{function_code}");

    println!("\n✨ ANF Benefits:");
    println!("• Automatic common subexpression elimination");
    println!("• Clean, readable generated code");
    println!("• Efficient variable management (index-based for performance)");
    println!("• Ready for further optimization passes");

    Ok(())
}
