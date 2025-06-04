//! Scoped Variables Demo: Solving Compile-Time Composability
//!
//! This example demonstrates the problem with legacy compile-time variables
//! and shows how the scoped variables system solves it elegantly.

use dslcompile::prelude::*;

fn main() {
    println!("=== Scoped Variables Demo: Solving Compile-Time Composability ===\n");

    // Demonstrate the problem with legacy variables
    demonstrate_legacy_problem();

    // Demonstrate the scoped variables solution
    demonstrate_scoped_solution();

    // Show advanced composition patterns
    demonstrate_advanced_composition();
}

fn demonstrate_legacy_problem() {
    println!("🚨 THE PROBLEM: Legacy Variable Collision");
    println!("==========================================");

    // Define f(x) = 2x using legacy compile-time variable 0
    let f = var::<0>().mul(constant(2.0));
    println!("f(x) = 2x using var<0>");

    // Define g(y) = 3y using legacy compile-time variable 0 (COLLISION!)
    let g = var::<0>().mul(constant(3.0));
    println!("g(y) = 3y using var<0> (COLLISION!)");

    // Naive composition: h = f + g = 2*var[0] + 3*var[0] = 5*var[0]
    let h_wrong = f.add(g);
    println!("h = f + g becomes: 5*var[0] (WRONG!)");

    // This gives h(4) = 5*4 = 20, NOT f(4) + g(7) = 8 + 21 = 29
    let result_wrong = h_wrong.eval(&[4.0]);
    println!("h(4) = {result_wrong} (should be f(4) + g(7) = 8 + 21 = 29)");
    println!();
}

fn demonstrate_scoped_solution() {
    println!("✅ THE SOLUTION: Scoped Variables");
    println!("=================================");

    // Define f(x) = 2x using scoped variables in scope 0
    let x_f = scoped_var::<0, 0>();
    let f = x_f.mul(scoped_constant::<0>(2.0));
    println!("f(x) = 2x using scoped_var<0, 0>");

    // Define g(y) = 3y using scoped variables in scope 1 (NO COLLISION!)
    let y_g = scoped_var::<0, 1>();
    let g = y_g.mul(scoped_constant::<1>(3.0));
    println!("g(y) = 3y using scoped_var<0, 1> (different scope)");

    // Evaluate independently with proper scoped arrays
    let f_vars = ScopedVarArray::<0>::new(vec![4.0]);
    let g_vars = ScopedVarArray::<1>::new(vec![7.0]);

    let f_result = f.eval(&f_vars);
    let g_result = g.eval(&g_vars);

    println!("f(4) = {f_result} (2 * 4 = 8)");
    println!("g(7) = {g_result} (3 * 7 = 21)");

    // Composition with automatic variable remapping
    let composed = compose(f, g);
    let h_correct = composed.add();
    println!("h = compose(f, g).add() with automatic variable remapping");

    // This correctly evaluates: f(4) + g(7) = 8 + 21 = 29
    let result_correct = h_correct.eval(&[4.0, 7.0]);
    println!("h(4, 7) = {result_correct} (correctly: f(4) + g(7) = 8 + 21 = 29)");

    println!("✅ Perfect composability achieved!");
    println!();
}

fn demonstrate_advanced_composition() {
    println!("🚀 ADVANCED: Complex Composition Patterns");
    println!("=========================================");

    // Define quadratic(x,y) = x² + xy + y² in scope 0
    let x = scoped_var::<0, 0>();
    let y = scoped_var::<1, 0>();
    let quadratic = x
        .clone()
        .mul(x.clone())
        .add(x.mul(y.clone()))
        .add(y.clone().mul(y));
    println!("quadratic(x,y) = x² + xy + y² in scope 0");

    // Define linear(a,b) = 2a + 3b in scope 1
    let a = scoped_var::<0, 1>();
    let b = scoped_var::<1, 1>();
    let linear = a
        .mul(scoped_constant::<1>(2.0))
        .add(b.mul(scoped_constant::<1>(3.0)));
    println!("linear(a,b) = 2a + 3b in scope 1");

    // Test individual evaluations
    let quad_vars = ScopedVarArray::<0>::new(vec![1.0, 2.0]);
    let quad_result = quadratic.eval(&quad_vars);
    println!("quadratic(1,2) = {quad_result} (1² + 1*2 + 2² = 7)");

    let lin_vars = ScopedVarArray::<1>::new(vec![3.0, 4.0]);
    let lin_result = linear.eval(&lin_vars);
    println!("linear(3,4) = {lin_result} (2*3 + 3*4 = 18)");

    // Compose two different scoped expressions
    let composed = compose(quadratic, linear);
    let combined = composed.add();
    println!("combined = compose(quadratic, linear).add()");

    // Test evaluation: quadratic(1,2) + linear(3,4) = 7 + 18 = 25
    let test_values = [1.0, 2.0, 3.0, 4.0];
    let result = combined.eval(&test_values);

    let expected = quad_result + lin_result;
    println!("combined(1,2,3,4) = {result}");
    println!("Expected: {quad_result} + {lin_result} = {expected}");
    println!("✅ Match: {}", (result - expected).abs() < 1e-10);

    // Show exponential function separately since triple composition is complex
    println!("\nSeparate exponential function:");
    let z = scoped_var::<0, 2>();
    let exponential = z.exp();
    let exp_vars = ScopedVarArray::<2>::new(vec![0.0]);
    let exp_result = exponential.eval(&exp_vars);
    println!("exponential(0) = {exp_result} (e^0 = 1)");

    println!("\n🎯 Key Benefits:");
    println!("• Type-safe composition at compile time");
    println!("• Automatic variable remapping prevents collisions");
    println!("• Zero runtime overhead");
    println!("• Perfect composability for mathematical functions");
    println!("• Each function maintains its own variable scope");
    println!("• Complex multi-function composition possible with step-by-step building");
}
