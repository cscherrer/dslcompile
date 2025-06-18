// Demonstration of natural function call syntax: f.call(g.call(x))
// Shows how to write mathematical expressions naturally instead of using .compose()

use dslcompile::{
    composition::MathFunction,
    contexts::dynamic::DynamicContext,
    prelude::*,
};
use frunk::hlist;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Natural Function Call Syntax Demo ===\n");

    // Create mathematical functions using ergonomic syntax
    let square_plus_one = MathFunction::<f64>::from_lambda("square_plus_one", |builder| {
        builder.lambda(|x| x.clone() * x + 1.0)
    });

    let linear =
        MathFunction::<f64>::from_lambda("linear", |builder| builder.lambda(|x| x * 2.0 + 3.0));

    let sine = MathFunction::<f64>::from_lambda("sine", |builder| builder.lambda(|x| x.sin()));

    println!("Created functions:");
    println!("  square_plus_one(x) = x² + 1");
    println!("  linear(x) = 2x + 3");
    println!("  sine(x) = sin(x)");
    println!();

    // Convert to callable functions for natural syntax
    let f = square_plus_one.as_callable();
    let g = linear.as_callable();
    let h = sine.as_callable();

    // APPROACH 1: Traditional .compose() method
    println!("=== Traditional .compose() Method ===");
    let traditional_composed = square_plus_one.compose(&linear);
    println!("square_plus_one.compose(&linear)");

    let x_test = 2.0;
    let traditional_result = traditional_composed.eval(hlist![x_test]);
    println!("At x = {x_test}: result = {traditional_result}");
    println!();

    // APPROACH 2: Natural function call syntax
    println!("=== Natural Function Call Syntax ===");
    let natural_composed = MathFunction::<f64>::from_lambda("f_g_natural", |builder| {
        builder.lambda(|x| f.call(g.call(x))) // f(g(x)) - reads like math!
    });
    println!("builder.lambda(|x| f.call(g.call(x)))");

    let natural_result = natural_composed.eval(hlist![x_test]);
    println!("At x = {x_test}: result = {natural_result}");
    println!("✓ Results match: {}", traditional_result == natural_result);
    println!();

    // APPROACH 3: Complex nested composition
    println!("=== Complex Nested Composition ===");
    let complex_traditional = sine.compose(&square_plus_one.compose(&linear));
    println!("Traditional: sine.compose(&square_plus_one.compose(&linear))");

    let complex_natural = MathFunction::<f64>::from_lambda("h_f_g_natural", |builder| {
        builder.lambda(|x| h.call(f.call(g.call(x)))) // h(f(g(x))) - pure mathematical notation!
    });
    println!("Natural: builder.lambda(|x| h.call(f.call(g.call(x))))");

    let complex_traditional_result = complex_traditional.eval(hlist![x_test]);
    let complex_natural_result = complex_natural.eval(hlist![x_test]);

    println!("At x = {x_test}:");
    println!("  Traditional: {complex_traditional_result}");
    println!("  Natural: {complex_natural_result}");
    println!(
        "  ✓ Results match: {}",
        (complex_traditional_result - complex_natural_result).abs() < 1e-15
    );
    println!();

    // APPROACH 4: Mixed with regular operations
    println!("=== Mixed with Regular Operations ===");
    let mixed_expression = MathFunction::<f64>::from_lambda("mixed", |builder| {
        builder.lambda(|x| {
            let intermediate = f.call(x.clone()); // f(x) = x² + 1
            intermediate.clone() * 2.0 + g.call(x) // 2*f(x) + g(x)
        })
    });
    println!("Mixed: 2*f(x) + g(x) = 2*(x² + 1) + (2x + 3)");

    let mixed_result = mixed_expression.eval(hlist![x_test]);
    let expected_mixed = 2.0 * (x_test * x_test + 1.0) + (2.0 * x_test + 3.0);
    println!(
        "At x = {x_test}: result = {mixed_result}, expected = {expected_mixed}"
    );
    println!(
        "✓ Calculation correct: {}",
        (mixed_result - expected_mixed).abs() < 1e-15
    );
    println!();

    // Show the benefits
    println!("=== Benefits of Natural Syntax ===");
    println!("❌ Old way: square_plus_one.compose(&linear.compose(&sine))");
    println!("   - Requires learning .compose() method");
    println!("   - Reads backwards (right to left)");
    println!("   - Hard to see the mathematical structure");
    println!();
    println!("✅ New way: builder.lambda(|x| f.call(g.call(h.call(x))))");
    println!("   - Reads like mathematical notation f(g(h(x)))");
    println!("   - Intuitive left-to-right reading");
    println!("   - Natural for anyone familiar with function calls");
    println!("   - Easily mixable with arithmetic operations");
    println!();

    // Verify lambda infrastructure is still used
    println!("=== Lambda Infrastructure Verification ===");
    let lambda_ref = natural_composed.lambda();
    println!("✓ Natural syntax compiles to proper Lambda struct");
    println!("  Lambda arity: {}", lambda_ref.arity());
    println!("  Variable indices: {:?}", lambda_ref.var_indices);
    println!("  Body type: Lambda expression with substituted function calls");
    println!();

    println!("🎉 Natural function call syntax achieved!");
    println!("   Mathematical expressions now read like mathematics:");
    println!("   f(g(h(x))) instead of f.compose(&g.compose(&h))");

    // 5. Mixed Variables and Lambda composition
    println!("\n=== Mixed Usage: Variables + Lambda ===");

    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    let variable_expr = &x + 42.0; // Regular Variable expression

    let lambda_wrapper = MathFunction::from_lambda("wrapper", |builder| {
        builder.lambda(|input| input + 1.0) // Simple lambda
    });

    // We can evaluate them separately and combine the results
    let x_test = 3.0;
    let variable_result = ctx.eval(&variable_expr, hlist![x_test]); // Variable: use HList
    let lambda_result = lambda_wrapper.eval(hlist![x_test]); // Lambda: use hlist!
    let mixed_result = variable_result + lambda_result;

    println!("Variable expr (x + 42): {variable_result}");
    println!("Lambda expr (x + 1): {lambda_result}");
    println!("Combined result: {mixed_result}");
    println!("Expected: {} + {} = {}", 45.0, 4.0, 49.0);
    println!("✓ Correct: {}\n", mixed_result == 49.0);

    Ok(())
}
