use dslcompile::ast::ast_utils::{combine_expressions_with_remapping, remap_variables};
use dslcompile::final_tagless::DirectEval;
use dslcompile::prelude::*;
use std::collections::HashMap;

#[test]
fn test_manual_variable_remapping() {
    // Define f(x) = x² + 2x + 1 independently
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var(); // index 0 in f's context
    let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0;
    let f_ast = f_expr.as_ast();

    // Define g(y) = 3y + 5 independently
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var(); // index 0 in g's context (collision!)
    let g_expr = 3.0 * &y_g + 5.0;
    let g_ast = g_expr.as_ast();

    // Manually remap g's variables to avoid collision
    let mut var_map = HashMap::new();
    var_map.insert(0, 1); // Map g's variable 0 to index 1
    let g_remapped = remap_variables(g_ast, &var_map);

    // Now create h(x,y) = f(x) + g(y) with proper variable mapping
    let h_ast = dslcompile::ast::ASTRepr::Add(Box::new(f_ast.clone()), Box::new(g_remapped));

    // Test evaluation: h(2,3) = f(2) + g(3) = (4+4+1) + (9+5) = 9 + 14 = 23
    let result = DirectEval::eval_with_vars(&h_ast, &[2.0, 3.0]);
    assert_eq!(result, 23.0);

    println!("Manual remapping: h(2,3) = {result}");
}

#[test]
fn test_automatic_variable_remapping() {
    // Define f(x) = x² + 2x + 1 independently
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var();
    let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0;
    let f_ast = f_expr.as_ast();

    // Define g(y) = 3y + 5 independently
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var();
    let g_expr = 3.0 * &y_g + 5.0;
    let g_ast = g_expr.as_ast();

    // Use automatic remapping
    let (remapped_expressions, total_vars) =
        combine_expressions_with_remapping(&[f_ast.clone(), g_ast.clone()]);

    assert_eq!(remapped_expressions.len(), 2);
    assert_eq!(total_vars, 2); // f uses var 0, g uses var 1

    // Create h(x,y) = f(x) + g(y)
    let h_ast = dslcompile::ast::ASTRepr::Add(
        Box::new(remapped_expressions[0].clone()),
        Box::new(remapped_expressions[1].clone()),
    );

    // Test evaluation: h(2,3) = f(2) + g(3) = 9 + 14 = 23
    let result = DirectEval::eval_with_vars(&h_ast, &[2.0, 3.0]);
    assert_eq!(result, 23.0);

    println!("Automatic remapping: h(2,3) = {result}");
}

#[test]
fn test_simple_composition_api() {
    // Define f(x) = x² + 2x + 1 independently
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var();
    let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0;

    // Define g(y) = 3y + 5 independently
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var();
    let g_expr = 3.0 * &y_g + 5.0;

    // Use automatic remapping directly
    let (remapped_expressions, _) =
        combine_expressions_with_remapping(&[f_expr.as_ast().clone(), g_expr.as_ast().clone()]);

    let h_ast = dslcompile::ast::ASTRepr::Add(
        Box::new(remapped_expressions[0].clone()),
        Box::new(remapped_expressions[1].clone()),
    );

    // Test evaluation: h(2,3) = f(2) + g(3) = 9 + 14 = 23
    let result = DirectEval::eval_with_vars(&h_ast, &[2.0, 3.0]);
    assert_eq!(result, 23.0);

    println!("Simple composition: h(2,3) = {result}");
}

#[test]
fn test_complex_composition() {
    // Define f(x) = sin(x) + cos(x)
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var();
    let f_expr = x_f.clone().sin() + x_f.clone().cos();

    // Define g(y) = exp(y) - ln(y + 1)
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var();
    let g_expr = y_g.clone().exp() - (y_g.clone() + 1.0).ln();

    // Define h(z) = z²
    let math_h = ExpressionBuilder::new();
    let z_h = math_h.var();
    let h_expr = &z_h * &z_h;

    // Compose k(x,y,z) = f(x) * g(y) + h(z)
    let (remapped_expressions, _) = combine_expressions_with_remapping(&[
        f_expr.as_ast().clone(),
        g_expr.as_ast().clone(),
        h_expr.as_ast().clone(),
    ]);

    let k_ast = dslcompile::ast::ASTRepr::Add(
        Box::new(dslcompile::ast::ASTRepr::Mul(
            Box::new(remapped_expressions[0].clone()),
            Box::new(remapped_expressions[1].clone()),
        )),
        Box::new(remapped_expressions[2].clone()),
    );

    // Test evaluation with x=1.0, y=1.0, z=3.0
    let x_val = 1.0_f64;
    let y_val = 1.0_f64;
    let z_val = 3.0_f64;

    let result = DirectEval::eval_with_vars(&k_ast, &[x_val, y_val, z_val]);

    // Expected: f(1) * g(1) + h(3)
    // f(1) = sin(1) + cos(1) ≈ 0.8415 + 0.5403 ≈ 1.3818
    // g(1) = exp(1) - ln(2) ≈ 2.7183 - 0.6931 ≈ 2.0252
    // h(3) = 9
    // k(1,1,3) ≈ 1.3818 * 2.0252 + 9 ≈ 2.798 + 9 ≈ 11.798

    let expected_f = x_val.sin() + x_val.cos();
    let expected_g = y_val.exp() - (y_val + 1.0).ln();
    let expected_h = z_val * z_val;
    let expected = expected_f * expected_g + expected_h;

    assert!((result - expected).abs() < 1e-10);

    println!("Complex composition: k(1,1,3) = {result}");
}

#[test]
fn test_compile_time_variable_collision() {
    // This test demonstrates that the compile-time system has the same issue
    // but it's handled differently due to the type system

    use dslcompile::compile_time::*;

    // Define f(x) = 2x using compile-time variable 0
    let f = var::<0>().mul(constant(2.0));

    // Define g(y) = 3y using compile-time variable 0 (collision!)
    let g = var::<0>().mul(constant(3.0));

    // If we naively combine: h = f + g = 2*var[0] + 3*var[0] = 5*var[0]
    let h_wrong = f.add(g);

    // This gives us h(4) = 5*4 = 20, not f(4) + g(7) = 8 + 21 = 29
    let result_wrong = h_wrong.eval(&[4.0]);
    assert_eq!(result_wrong, 20.0);

    // The correct way: use different variable IDs
    let f_correct = var::<0>().mul(constant(2.0)); // f(x) uses var 0
    let g_correct = var::<1>().mul(constant(3.0)); // g(y) uses var 1
    let h_correct = f_correct.add(g_correct);

    // Now h(4,7) = f(4) + g(7) = 2*4 + 3*7 = 8 + 21 = 29
    let result_correct = h_correct.eval(&[4.0, 7.0]);
    assert_eq!(result_correct, 29.0);

    println!("Compile-time collision: {result_wrong} vs correct: {result_correct}");
}

#[test]
fn test_compilation_with_remapped_variables() {
    // Test that the remapped expressions work with code generation

    // Define independent functions
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var();
    let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0; // f(x) = x² + 2x + 1

    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var();
    let g_expr = 3.0 * &y_g + 5.0; // g(y) = 3y + 5

    // Compose with automatic remapping
    let (remapped_expressions, _) =
        combine_expressions_with_remapping(&[f_expr.as_ast().clone(), g_expr.as_ast().clone()]);

    let h_ast = dslcompile::ast::ASTRepr::Add(
        Box::new(remapped_expressions[0].clone()),
        Box::new(remapped_expressions[1].clone()),
    );

    // Generate Rust code
    let codegen = dslcompile::backends::rust_codegen::RustCodeGenerator::new();
    let mut registry =
        dslcompile::final_tagless::variables::typed_registry::VariableRegistry::new();
    let _var0 = registry.register_variable(); // x
    let _var1 = registry.register_variable(); // y

    let rust_code =
        codegen.generate_function_with_registry(&h_ast, "composed_func", "f64", &registry);
    assert!(rust_code.is_ok());

    let code = rust_code.unwrap();
    println!("Generated code:\n{code}");

    // The generated code should reference both var_0 and var_1
    assert!(code.contains("var_0"));
    assert!(code.contains("var_1"));
}
