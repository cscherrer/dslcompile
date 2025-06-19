use dslcompile::ast::DynamicContext;

fn main() {
    use frunk::hlist;

    let mut math_f = DynamicContext::new();
    let x_f = math_f.var(); // Index 0 in math_f's registry
    let f = &x_f + 1.0; // f(x) = x + 1

    let mut math_g = DynamicContext::new();
    let x_g = math_g.var(); // Index 0 in math_g's registry
    let g = 2.0 * &x_g; // g(x) = 2x

    println!("f expression: {f:?}");
    println!("g expression: {g:?}");

    // Test independent evaluation first
    let f_result = math_f.eval(&f, hlist![3.0]);
    let g_result = math_g.eval(&g, hlist![4.0]);
    println!("f(3) = {f_result}"); // Should be 4
    println!("g(4) = {g_result}"); // Should be 8

    // This creates an expression that combines f and g
    let composed = &f + &g;
    println!("composed expression: {composed:?}");

    // With scope merging, we need TWO values: one for f's variable, one for g's variable
    let temp_ctx = DynamicContext::new();
    let result = temp_ctx.eval(&composed, hlist![3.0, 4.0]);
    println!("composed(3.0, 4.0) = {result}");

    // Let's also try the reverse order
    let result_rev = temp_ctx.eval(&composed, hlist![4.0, 3.0]);
    println!("composed(4.0, 3.0) = {result_rev}");
}
