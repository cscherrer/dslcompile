//! Scoped Variables Demo: Compile-Time Composability
//!
//! This example demonstrates the power of scoped variables for building
//! mathematical functions that can be composed without variable collisions.

use dslcompile::prelude::*;

fn main() {
    println!("=== Scoped Variables Demo: Perfect Compile-Time Composability ===\n");

    // Demonstrate basic scoped variables
    demonstrate_basic_scoped_usage();

    // Show the composition solution
    demonstrate_composition_solution();

    // Show advanced composition patterns
    demonstrate_advanced_composition();
}

fn demonstrate_basic_scoped_usage() {
    println!("🎯 BASIC USAGE: Scoped Variables");
    println!("===============================");

    // Define f(x) = 2x in scope 0
    let x = scoped_var::<0, 0>();
    let f = x.mul(scoped_constant::<0>(2.0));
    println!("f(x) = 2x in scope 0");

    // Define g(y) = 3y in scope 1
    let y = scoped_var::<0, 1>();
    let g = y.mul(scoped_constant::<1>(3.0));
    println!("g(y) = 3y in scope 1");

    // Evaluate independently  
    let f_vars = ScopedVarArray::<0>::new(vec![4.0]);
    let g_vars = ScopedVarArray::<1>::new(vec![5.0]);

    let f_result = f.eval(&f_vars);
    let g_result = g.eval(&g_vars);

    println!("f(4) = {f_result}"); // 2 * 4 = 8
    println!("g(5) = {g_result}"); // 3 * 5 = 15
    println!();
}

fn demonstrate_composition_solution() {
    println!("✅ PERFECT COMPOSITION: Automatic Variable Remapping");
    println!("===================================================");

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

    // Perfect composition with automatic variable remapping!
    let composed = compose(quadratic, linear);
    let combined = composed.add(); // h(x,y,a,b) = quadratic(x,y) + linear(a,b)
    println!("combined = quadratic + linear (automatic remapping!)");

    // Test the composition: quadratic(1,2) + linear(3,4) = 7 + 18 = 25
    let result = combined.eval(&[1.0, 2.0, 3.0, 4.0]);
    println!("combined(1,2,3,4) = quadratic(1,2) + linear(3,4) = 7 + 18 = {result}");
    println!("✅ Perfect: Variables automatically remapped to [x,y,a,b] = [0,1,2,3]");
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
