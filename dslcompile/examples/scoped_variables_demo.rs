//! Scoped Variables Demo: Automatic Compile-Time Composability
//!
//! This example demonstrates the power of automatic scoped variables for building
//! mathematical functions that can be composed without variable collisions.

use dslcompile::prelude::*;

fn main() {
    println!("=== Scoped Variables Demo: Automatic Compile-Time Composability ===\n");

    // Demonstrate basic automatic scoped usage
    demonstrate_automatic_scoped_usage();

    // Show the composition solution
    demonstrate_composition_solution();

    // Show advanced composition patterns
    demonstrate_advanced_composition();
}

fn demonstrate_automatic_scoped_usage() {
    println!("🎯 AUTOMATIC USAGE: Scoped Variables with Builder Pattern");
    println!("========================================================");

    let mut builder = ScopedExpressionBuilder::new_f64();

    // Define f(x) = 2x in scope 0
    let f = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        x.mul(scope.constant(2.0))
    });
    println!("f(x) = 2x in scope 0 (automatic variable assignment)");

    // Advance to next scope
    let mut builder = builder.next();

    // Define g(y) = 3y in scope 1
    let g = builder.new_scope(|scope| {
        let (y, scope) = scope.auto_var();
        y.mul(scope.constant(3.0))
    });
    println!("g(y) = 3y in scope 1 (automatic variable assignment)");

    // Evaluate independently
    let f_vars = ScopedVarArray::<f64, 0>::new(vec![4.0]);
    let g_vars = ScopedVarArray::<f64, 1>::new(vec![5.0]);

    let f_result = f.eval(&f_vars);
    let g_result = g.eval(&g_vars);

    println!("f(4) = {f_result}"); // 2 * 4 = 8
    println!("g(5) = {g_result}"); // 3 * 5 = 15
    println!();
}

fn demonstrate_composition_solution() {
    println!("✅ PERFECT COMPOSITION: Automatic Variable Remapping");
    println!("===================================================");

    let mut builder = ScopedExpressionBuilder::new_f64();

    // Define quadratic(x,y) = x² + xy + y² in scope 0
    let quadratic = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var();
        x.clone()
            .mul(x.clone())
            .add(x.mul(y.clone()))
            .add(y.clone().mul(y))
    });
    println!("quadratic(x,y) = x² + xy + y² in scope 0");

    // Advance to next scope
    let mut builder = builder.next();

    // Define linear(a,b) = 2a + 3b in scope 1
    let linear = builder.new_scope(|scope| {
        let (a, scope) = scope.auto_var();
        let (b, scope) = scope.auto_var();
        a.mul(scope.clone().constant(2.0))
            .add(b.mul(scope.constant(3.0)))
    });
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

    let mut builder = ScopedExpressionBuilder::new_f64();

    // Define quadratic(x,y) = x² + xy + y² in scope 0
    let quadratic = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var();
        x.clone()
            .mul(x.clone())
            .add(x.mul(y.clone()))
            .add(y.clone().mul(y))
    });
    println!("quadratic(x,y) = x² + xy + y² in scope 0");

    // Advance to next scope
    let mut builder = builder.next();

    // Define linear(a,b) = 2a + 3b in scope 1
    let linear = builder.new_scope(|scope| {
        let (a, scope) = scope.auto_var();
        let (b, scope) = scope.auto_var();
        a.mul(scope.clone().constant(2.0))
            .add(b.mul(scope.constant(3.0)))
    });
    println!("linear(a,b) = 2a + 3b in scope 1");

    // Test individual evaluations
    let quad_vars = ScopedVarArray::<f64, 0>::new(vec![1.0, 2.0]);
    let quad_result = quadratic.eval(&quad_vars);
    println!("quadratic(1,2) = {quad_result} (1² + 1*2 + 2² = 7)");

    let lin_vars = ScopedVarArray::<f64, 1>::new(vec![3.0, 4.0]);
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

    // Show trigonometric function separately
    println!("\nSeparate trigonometric function:");
    let mut builder = builder.next();
    let trig = builder.new_scope(|scope| {
        let (z, _scope) = scope.auto_var();
        z.clone().sin().add(z.cos())
    });
    let trig_vars = ScopedVarArray::<f64, 2>::new(vec![0.0]);
    let trig_result = trig.eval(&trig_vars);
    println!("sin(0) + cos(0) = {trig_result} (0 + 1 = 1)");

    println!("\n🎯 Key Benefits:");
    println!("• Type-safe composition at compile time");
    println!("• Automatic variable ID assignment prevents errors");
    println!("• Automatic variable remapping prevents collisions");
    println!("• Zero runtime overhead");
    println!("• Perfect composability for mathematical functions");
    println!("• Each function maintains its own variable scope");
    println!("• Clean, ergonomic builder pattern API");
}
