use dslcompile::prelude::*;

#[test]
fn test_independent_function_composition() {
    // Define function f(x) = x² + 2x + 1 completely independently
    fn define_f() -> (MathBuilder, TypedBuilderExpr<f64>) {
        let math_f = MathBuilder::new();
        let x = math_f.var(); // This will be variable index 0 in f's registry
        let _f_expr = &x * &x + 2.0 * &x + 1.0; // x² + 2x + 1
        (math_f, _f_expr)
    }

    // Define function g(y) = 3y + 5 completely independently
    fn define_g() -> (MathBuilder, TypedBuilderExpr<f64>) {
        let math_g = MathBuilder::new();
        let y = math_g.var(); // This will be variable index 0 in g's registry
        let _g_expr = 3.0 * &y + 5.0; // 3y + 5
        (math_g, _g_expr)
    }

    // Get the independent functions
    let (math_f, _f_expr) = define_f();
    let (math_g, _g_expr) = define_g();

    // Test that f and g work independently
    let f_result = math_f.eval(&_f_expr, &[2.0]); // f(2) = 4 + 4 + 1 = 9
    let g_result = math_g.eval(&_g_expr, &[3.0]); // g(3) = 9 + 5 = 14

    assert_eq!(f_result, 9.0);
    assert_eq!(g_result, 14.0);

    // Now try to define h(x,y) = f(x) + g(y)
    // This is where the problem occurs!

    // Both _f_expr and _g_expr use variable index 0, but they represent different variables
    // _f_expr uses index 0 for "x", _g_expr uses index 0 for "y"
    let h_expr = &_f_expr + &_g_expr; // This combines two expressions that both use variable 0!

    // When we evaluate h_expr, both parts will use the same variable value
    // because they both reference variable index 0
    let h_result_wrong = math_f.eval(&h_expr, &[2.0]); // Both f and g get x=2

    // This gives us f(2) + g(2) = 9 + 11 = 20, NOT what we want for h(2,3)
    assert_eq!(h_result_wrong, 20.0); // f(2) + g(2) = 9 + 11 = 20

    println!("Problem: h_expr treats both parts as using the same variable!");
    println!("We got h(2,2) = {h_result_wrong} instead of h(2,3)");
}

#[test]
fn test_correct_function_composition() {
    // The correct way: define everything in the same MathBuilder context
    let math = MathBuilder::new();

    // Define f in terms of the first variable
    let x = math.var(); // index 0
    let _f_expr = &x * &x + 2.0 * &x + 1.0; // f(x) = x² + 2x + 1

    // Define g in terms of the second variable
    let y = math.var(); // index 1
    let _g_expr = 3.0 * &y + 5.0; // g(y) = 3y + 5

    // Now h(x,y) = f(x) + g(y) works correctly
    let h_expr = &_f_expr + &_g_expr;

    // Evaluate h(2,3) = f(2) + g(3) = 9 + 14 = 23
    let h_result = math.eval(&h_expr, &[2.0, 3.0]);
    assert_eq!(h_result, 23.0);

    println!("Correct: h(2,3) = f(2) + g(3) = {h_result}");
}

#[test]
fn test_variable_remapping_solution() {
    // Another solution: manually remap variables when combining independent expressions

    // Define f(x) = x² + 2x + 1 independently
    let math_f = MathBuilder::new();
    let x_f = math_f.var(); // index 0 in f's context
    let _f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0;

    // Define g(y) = 3y + 5 independently
    let math_g = MathBuilder::new();
    let y_g = math_g.var(); // index 0 in g's context  
    let _g_expr = 3.0 * &y_g + 5.0;

    // Create a new context for the combined function h(x,y)
    let math_h = MathBuilder::new();
    let x_h = math_h.var(); // index 0 in h's context
    let y_h = math_h.var(); // index 1 in h's context

    // We need to manually recreate f and g in the new context
    let f_in_h = &x_h * &x_h + 2.0 * &x_h + 1.0; // f(x) using x_h (index 0)
    let g_in_h = 3.0 * &y_h + 5.0; // g(y) using y_h (index 1)

    let h_expr = &f_in_h + &g_in_h;

    // Now h(2,3) works correctly
    let h_result = math_h.eval(&h_expr, &[2.0, 3.0]);
    assert_eq!(h_result, 23.0); // f(2) + g(3) = 9 + 14 = 23

    println!("Manual remapping: h(2,3) = {h_result}");
}

#[test]
fn test_variable_collision_demonstration() {
    // This test clearly shows the variable collision problem

    // Function f(x) = 2x (uses variable index 0)
    let math_f = MathBuilder::new();
    let x_f = math_f.var(); // index 0
    let _f_expr = 2.0 * &x_f;

    // Function g(y) = 3y (uses variable index 0 in its own context)
    let math_g = MathBuilder::new();
    let y_g = math_g.var(); // index 0 (different context!)
    let _g_expr = 3.0 * &y_g;

    // Verify they work independently
    assert_eq!(math_f.eval(&_f_expr, &[5.0]), 10.0); // f(5) = 2*5 = 10
    assert_eq!(math_g.eval(&_g_expr, &[7.0]), 21.0); // g(7) = 3*7 = 21

    // Try to combine: h(x,y) = f(x) + g(y)
    let h_expr = &_f_expr + &_g_expr;

    // Problem: both expressions use variable index 0!
    // So h_expr is effectively 2*var[0] + 3*var[0] = 5*var[0]

    // When we evaluate with one value, both parts get the same input
    let result = math_f.eval(&h_expr, &[4.0]); // Both f and g get 4
    assert_eq!(result, 20.0); // 2*4 + 3*4 = 8 + 12 = 20

    // This is NOT h(4,7) = f(4) + g(7) = 8 + 21 = 29
    // It's h(4,4) = f(4) + g(4) = 8 + 12 = 20

    println!("Variable collision: got h(4,4) = {result} instead of h(4,7) = 29");
}
