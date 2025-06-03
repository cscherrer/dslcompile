use dslcompile::final_tagless::DirectEval;
use dslcompile::prelude::*;
use std::collections::HashMap;

#[test]
fn test_shared_variable_composition_naive() {
    // Define f(x,y) = x² + xy + y² independently
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var(); // index 0 in f's context
    let y_f = math_f.var(); // index 1 in f's context
    let f_expr = &x_f * &x_f + &x_f * &y_f + &y_f * &y_f;

    // Define g(y,z) = 2y + 3z independently
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var(); // index 0 in g's context (should be y!)
    let z_g = math_g.var(); // index 1 in g's context (should be z!)
    let g_expr = 2.0 * &y_g + 3.0 * &z_g;

    // Naive combination - this is WRONG!
    // f uses variables [0,1] for [x,y]
    // g uses variables [0,1] for [y,z]
    // When combined, g's y (index 0) collides with f's x (index 0)
    let h_wrong = f_expr.as_ast().clone() + g_expr.as_ast().clone();

    // This evaluates as f(x,y) + g(x,y) instead of f(x,y) + g(y,z)
    let result_wrong = DirectEval::eval_with_vars(&h_wrong, &[1.0, 2.0]);
    // f(1,2) = 1 + 2 + 4 = 7
    // g(1,2) = 2*1 + 3*2 = 8  (but this should be g(2,z) for some z!)
    // Total: 15 (but this is wrong!)

    println!("Naive (wrong) result: h(1,2) = {result_wrong}");
    assert_eq!(result_wrong, 15.0); // This is the wrong answer!
}
