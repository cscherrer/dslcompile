//! Scoped Variables Demo
//!
//! This example demonstrates the scoped variable system that prevents
//! variable collisions at compile time while maintaining zero runtime overhead.

use dslcompile::prelude::*;

fn main() {
    println!("=== Scoped Variables Demo ===\n");

    // Demo 1: Basic scoped variables
    basic_scoped_variables();

    // Demo 2: Variable collision prevention
    variable_collision_prevention();

    // Demo 3: Complex composition
    complex_composition();
}

fn basic_scoped_variables() {
    println!("🔧 Basic Scoped Variables");
    println!("=========================");

    // Create a scoped expression builder
    let mut builder = Context::new_f64();

    // Define f(x) = x² in scope 0
    let f = builder.new_scope(|scope| {
        let (x, _scope) = scope.auto_var();
        x.clone().mul(x)
    });

    println!("f(x) = x² defined in scope 0");

    // Evaluate f(3) = 9
    let vars = ScopedVarArray::new(vec![3.0]);
    let result = f.eval(&vars);
    println!("f(3) = {result}");

    assert_eq!(result, 9.0);
    println!("✅ Basic scoped variables working!\n");
}

fn variable_collision_prevention() {
    println!("🛡️ Variable Collision Prevention");
    println!("=================================");

    // Create a scoped expression builder
    let mut builder = Context::new_f64();

    // Define f(x) = 2x in scope 0
    let f = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        x.mul(scope.constant(2.0))
    });

    // Advance to next scope
    let mut builder = builder.next();

    // Define g(x) = 3x in scope 1 - NO COLLISION!
    let g = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var(); // This 'x' is different from f's 'x'
        x.mul(scope.constant(3.0))
    });

    println!("f(x) = 2x defined in scope 0");
    println!("g(x) = 3x defined in scope 1 (no collision!)");

    // Evaluate independently
    let f_vars = ScopedVarArray::new(vec![4.0]);
    let g_vars = ScopedVarArray::new(vec![5.0]);

    let f_result = f.eval(&f_vars);
    let g_result = g.eval(&g_vars);

    println!("f(4) = {f_result}"); // 2 * 4 = 8
    println!("g(5) = {g_result}"); // 3 * 5 = 15

    assert_eq!(f_result, 8.0);
    assert_eq!(g_result, 15.0);
    println!("✅ Variable collision prevention working!\n");
}

fn complex_composition() {
    println!("🔗 Complex Composition");
    println!("======================");

    // Create a scoped expression builder
    let mut builder = Context::new_f64();

    // Define quadratic(x,y) = x² + xy + y² in scope 0
    let quadratic = builder.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var();
        x.clone()
            .mul(x.clone())
            .add(x.mul(y.clone()))
            .add(y.clone().mul(y))
    });

    // Advance to next scope
    let mut builder = builder.next();

    // Define linear(a,b) = 2a + 3b in scope 1
    let linear = builder.new_scope(|scope| {
        let (a, scope) = scope.auto_var();
        let (b, scope) = scope.auto_var();
        a.mul(scope.clone().constant(2.0))
            .add(b.mul(scope.constant(3.0)))
    });

    println!("quadratic(x,y) = x² + xy + y² in scope 0");
    println!("linear(a,b) = 2a + 3b in scope 1");

    // Test individual evaluations
    let quad_vars = ScopedVarArray::new(vec![1.0, 2.0]);
    let quad_result = quadratic.eval(&quad_vars); // 1² + 1*2 + 2² = 7
    println!("quadratic(1,2) = {quad_result}");

    let lin_vars = ScopedVarArray::new(vec![3.0, 4.0]);
    let lin_result = linear.eval(&lin_vars); // 2*3 + 3*4 = 18
    println!("linear(3,4) = {lin_result}");

    // Compose them: h(x,y,a,b) = quadratic(x,y) + linear(a,b)
    let composed = compose(quadratic, linear);
    let combined = composed.add();

    println!("h(x,y,a,b) = quadratic(x,y) + linear(a,b)");

    // Test with combined variable array [x, y, a, b] = [1, 2, 3, 4]
    // Should evaluate to quadratic(1,2) + linear(3,4) = 7 + 18 = 25
    let test_values = [1.0, 2.0, 3.0, 4.0];
    let result = combined.eval(&test_values);

    println!("h(1,2,3,4) = {result}");

    assert_eq!(result, 25.0);
    println!("✅ Complex composition working!");
    println!("🎯 Variable remapping automatically handled!");
}
