use dslcompile::compile_time::scoped::{
    ScopedMathExpr, ScopedVarArray, compose, scoped_constant, scoped_var,
};

fn main() {
    println!("=== Type-Level Scoped Variables Demo ===\n");

    // Define f(x) = x² + 2x in scope 0
    println!("1. Define f(x) = x² + 2x in scope 0");
    let x_f = scoped_var::<0, 0>();
    let f = x_f
        .clone()
        .mul(x_f)
        .add(scoped_var::<0, 0>().mul(scoped_constant::<0>(2.0)));

    // Define g(x) = 3x + 1 in scope 1 (note: same variable name 'x' but different scope!)
    println!("2. Define g(x) = 3x + 1 in scope 1 (same variable name, different scope!)");
    let x_g = scoped_var::<0, 1>();
    let g = x_g
        .mul(scoped_constant::<1>(3.0))
        .add(scoped_constant::<1>(1.0));

    // Evaluate functions independently - no collision!
    println!("\n3. Evaluate functions independently:");
    let f_vars = ScopedVarArray::<0>::new(vec![2.0]);
    let g_vars = ScopedVarArray::<1>::new(vec![4.0]);

    let f_result = f.eval(&f_vars); // f(2) = 4 + 4 = 8
    let g_result = g.eval(&g_vars); // g(4) = 12 + 1 = 13

    println!("   f(2) = 2² + 2*2 = {f_result}");
    println!("   g(4) = 3*4 + 1 = {g_result}");

    // Compose functions: h = f + g
    println!("\n4. Compose functions: h = f + g");
    let composed = compose(f, g);
    let h = composed.add();

    // Evaluate composed function h(2, 4) = f(2) + g(4) = 8 + 13 = 21
    let combined_vars = vec![2.0, 4.0]; // Variables from both scopes
    let h_result = h.eval(&combined_vars);

    println!("   h(2, 4) = f(2) + g(4) = {f_result} + {g_result} = {h_result}");

    // Demonstrate type safety - this would be a compile error:
    // let invalid = x_f.add(x_g);  // ERROR: Cannot mix variables from different scopes!

    println!("\n5. Type Safety Demonstration:");
    println!("   ✅ Variables in same scope can be combined");
    println!("   ❌ Variables in different scopes require explicit composition");
    println!("   ✅ Compiler prevents accidental variable collisions");

    // Show more complex scoped expressions
    println!("\n6. Complex scoped expressions:");

    // Scope 2: trigonometric function
    let theta = scoped_var::<0, 2>();
    let trig_expr = theta.clone().sin().add(theta.cos());
    let trig_vars = ScopedVarArray::<2>::new(vec![std::f64::consts::PI / 4.0]);
    let trig_result = trig_expr.eval(&trig_vars);

    println!("   sin(π/4) + cos(π/4) = {trig_result:.6}");

    // Scope 3: exponential function
    let y = scoped_var::<0, 3>();
    let exp_expr = y.clone().exp().ln(); // Should simplify to y
    let exp_vars = ScopedVarArray::<3>::new(vec![2.5]);
    let exp_result = exp_expr.eval(&exp_vars);

    println!("   ln(exp(2.5)) = {exp_result:.6}");

    println!("\n=== Benefits Achieved ===");
    println!("✅ Zero runtime overhead - all scope checking at compile time");
    println!("✅ Impossible variable collisions - type system prevents them");
    println!("✅ Clear intent - scope information explicit in types");
    println!("✅ Automatic variable remapping during composition");
    println!("✅ Maintains mathematical correctness");
}
