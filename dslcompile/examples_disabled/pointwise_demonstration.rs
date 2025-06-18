// Demonstration: Pointwise operations - convenience wrappers vs new primitives
// Shows different approaches to combining functions pointwise: (f + g)(x) = f(x) + g(x)

use dslcompile::{
    composition::MathFunction,
    prelude::*,
};
use frunk::hlist;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Pointwise Operations Demonstration ===\n");

    // Create two simple functions
    // f(x) = x²
    let f = MathFunction::from_lambda("f", |builder| builder.lambda(|x| x.clone() * x));

    // g(x) = 2x + 1
    let g = MathFunction::from_lambda("g", |builder| builder.lambda(|x| x * 2.0 + 1.0));

    let x = 3.0;
    let f_result = f.eval(hlist![x]);
    let g_result = g.eval(hlist![x]);

    println!("Individual function evaluation at x = {x}:");
    println!("  f(x) = x² = {f_result}"); // 9
    println!("  g(x) = 2x + 1 = {g_result}"); // 7
    println!();

    // APPROACH 1: Manual - using existing Lambda infrastructure directly
    println!("=== APPROACH 1: Manual Lambda Construction ===");

    // Want: (f + g)(x) = f(x) + g(x) = x² + (2x + 1)
    // This requires creating a new lambda that applies both f and g to the same input
    // and adds the results - this is NOT just composition!

    let manual_pointwise_add = MathFunction::from_lambda("manual_f_plus_g", |builder| {
        builder.lambda(|x| {
            // f(x) = x²
            let f_applied = x.clone() * x.clone();

            // g(x) = 2x + 1
            let g_applied = x * 2.0 + 1.0;

            // (f + g)(x) = f(x) + g(x)
            f_applied + g_applied
        })
    });

    let manual_result = manual_pointwise_add.eval(hlist![x]);
    let expected = f_result + g_result;

    println!("Manual construction:");
    println!("  (f + g)(x) = x² + (2x + 1) = {manual_result}");
    println!("  Expected: {f_result} + {g_result} = {expected}");
    println!("  ✓ Matches: {}", manual_result == expected);
    println!();

    // APPROACH 2: What a convenience wrapper might look like
    println!("=== APPROACH 2: Hypothetical Convenience Wrapper ===");

    // This would be a convenience method that generates the same Lambda as above
    // let convenient_add = f.pointwise_add(&g);  // <-- This doesn't exist yet

    println!("A convenience wrapper like f.pointwise_add(&g) would:");
    println!("  1. Take the lambda from f: λx. x²");
    println!("  2. Take the lambda from g: λx. 2x + 1");
    println!("  3. Generate NEW lambda: λx. (x²) + (2x + 1)");
    println!("  4. Return: MathFunction with the combined lambda");
    println!();
    println!("This is NOT a new primitive - it's code generation!");
    println!("It builds the same AST as the manual approach above.");
    println!();

    // APPROACH 3: Using existing Collection::Map infrastructure (if applicable)
    println!("=== APPROACH 3: Existing Collection::Map Approach ===");

    // Question: Can we use existing Collection::Map for this?
    // Collection::Map applies a lambda to each element of a collection
    // But pointwise addition is different - it's applying two lambdas to the same input

    println!("Collection::Map is for: λf. map(f, [a, b, c]) = [f(a), f(b), f(c)]");
    println!("Pointwise addition is: (f + g)(x) = f(x) + g(x)");
    println!("These are different patterns!");
    println!();

    // APPROACH 4: Show the Lambda structure difference
    println!("=== APPROACH 4: Lambda Structure Analysis ===");

    println!("Function composition f ∘ g creates:");
    println!("  Lambda::Compose {{ f: λx.x², g: λx.2x+1 }}");
    println!("  Meaning: f(g(x)) = (2x+1)²");

    println!();
    println!("Pointwise addition f + g would create:");
    println!("  Lambda::Lambda {{ var_index: 0, body: Add(f_body, g_body) }}");
    println!("  Where f_body and g_body both reference the same variable");
    println!("  Meaning: (f + g)(x) = f(x) + g(x)");

    println!();
    println!("Key insight: Pointwise operations require PARALLEL application,");
    println!("not sequential application like composition!");

    // DEMONSTRATION: Show this is fundamentally different from composition
    println!();
    println!("=== COMPARISON: Composition vs Pointwise ===");

    let composed = f.compose(&g); // f(g(x)) = (2x + 1)²
    let composed_result = composed.eval(hlist![x]);

    println!("At x = {x}:");
    println!("  f ∘ g = f(g(x)) = (2x + 1)² = {composed_result}");
    println!("  f + g = f(x) + g(x) = x² + (2x + 1) = {manual_result}");
    println!(
        "  These are completely different: {composed_result} ≠ {manual_result}"
    );

    Ok(())
}
