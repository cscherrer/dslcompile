// Demonstration of the new function composition API
// Shows how to build and compose mathematical functions using lambda calculus

use dslcompile::{
    ast::ast_repr::Lambda,
    composition::{FunctionBuilder, MathFunction},
    prelude::*,
};
use frunk::hlist;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Function Composition API Demo ===\n");

    // Create mathematical functions using the builder pattern

    // f(x) = x² + 1
    let square_plus_one = MathFunction::from_lambda("square_plus_one", |builder| {
        builder.lambda(|x| x.clone() * x + 1.0)
    });

    // g(x) = 2x + 3
    let linear = MathFunction::from_lambda("linear", |builder| builder.lambda(|x| x * 2.0 + 3.0));

    // h(x) = sin(x)
    let sine = MathFunction::from_lambda("sin", |builder| builder.lambda(|x| x.sin()));

    println!("Created functions:");
    println!("  f(x) = {}", square_plus_one.name);
    println!("  g(x) = {}", linear.name);
    println!("  h(x) = {}", sine.name);
    println!();

    // Function composition using the new Lambda struct infrastructure

    // Compose f ∘ g: f(g(x)) = (2x + 3)² + 1
    let f_compose_g = square_plus_one.compose(&linear);
    println!("Function composition f ∘ g:");
    println!("  {} = (2x + 3)² + 1", f_compose_g.name);

    // Test evaluation at x = 2
    let x = 2.0_f64;
    let expected: f64 = {
        let g_result = 2.0 * x + 3.0; // g(2) = 7
        g_result * g_result + 1.0 // f(7) = 50
    };
    let actual = f_compose_g.eval(hlist![x]);
    println!(
        "  At x = {}: expected = {}, actual = {}",
        x, expected, actual
    );
    println!();

    // Chain multiple compositions: h ∘ f ∘ g
    let complex_composition = sine.compose(&f_compose_g);
    println!("Complex composition h ∘ f ∘ g:");
    println!("  {} = sin((2x + 3)² + 1)", complex_composition.name);

    let complex_result = complex_composition.eval(hlist![x]);
    let expected_complex: f64 = expected.sin();
    println!(
        "  At x = {}: expected = {}, actual = {}",
        x, expected_complex, complex_result
    );
    println!();

    // Demonstrate lambda infrastructure is being used
    println!("=== Lambda Infrastructure Verification ===");

    let lambda_ref = complex_composition.lambda();
    println!("✓ Uses Lambda struct with composition");
    println!("  Lambda arity: {}", lambda_ref.arity());
    println!("  Variable indices: {:?}", lambda_ref.var_indices);
    println!("  Body contains: Nested function substitutions");

    // Show the internal structure
    println!("  Composition creates: Single lambda with substituted expressions");
    println!("  This enables: Unified optimization and code generation");
    println!();

    // Show that functions can be built with different complexity levels

    // Simple identity function
    let identity = MathFunction::from_lambda_direct("identity", Lambda::identity(), 1);

    // Constant function
    let constant_5 = MathFunction::from_lambda_direct("constant_5", Lambda::constant(5.0), 1);

    println!("=== Simple Functions ===");
    println!("Identity function: {}", identity.name);
    println!("  At x = 10: {}", identity.eval(hlist![10.0]));

    println!("Constant function: {}", constant_5.name);
    println!("  At x = 99: {}", constant_5.eval(hlist![99.0]));
    println!();

    // Compose identity and constant to show basic composition
    let id_compose_const = identity.compose(&constant_5);
    println!("Identity ∘ Constant composition: {}", id_compose_const.name);
    println!(
        "  At x = 7: {} (should be identity of constant = 5)",
        id_compose_const.eval(hlist![7.0])
    );

    println!("\n=== Composition Benefits ===");
    println!("✅ Natural mathematical notation: f ∘ g");
    println!("✅ Automatic lambda substitution and optimization");
    println!("✅ Single unified expression for complex compositions");
    println!("✅ Type-safe function building with ergonomic syntax");
    println!("✅ HList evaluation for zero-cost parameter passing");

    Ok(())
}
