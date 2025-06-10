//! ANF demo - Administrative Normal Form conversion

use dslcompile::prelude::*;

fn main() -> Result<()> {
    let math = DynamicContext::new();

    // Create expression with common subexpressions
    let x = math.var();
    let y = math.var();
    let shared = x.clone() + y.clone();
    let expr = shared.clone() * shared + x * 2.0;

    println!("Original: {}", expr.pretty_print());

    // Convert to ANF
    let mut anf_converter = ANFConverter::new();
    let anf_expr = anf_converter.convert(&math.to_ast(&expr))?;

    println!("ANF form with CSE applied");

    // Evaluate both forms - need to provide user variables as HashMap
    let original_result = math.eval_old(&expr, &[3.0, 4.0]);
    let mut user_vars = std::collections::HashMap::new();
    user_vars.insert(0, 3.0); // x
    user_vars.insert(1, 4.0); // y
    let anf_result = anf_expr.eval(&user_vars);

    println!("Original result: {original_result}");
    println!("ANF result: {anf_result}");
    assert_eq!(original_result, anf_result);

    Ok(())
}
