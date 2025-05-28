//! Polynomial Evaluation Demo using Horner's Method
//!
//! This example demonstrates the polynomial evaluation capabilities of `MathJIT`
//! using the efficient Horner's method. It shows how the final tagless approach
//! enables the same polynomial definition to work with different interpreters.

use mathjit::final_tagless::{polynomial, DirectEval, MathExpr, PrettyPrint};
use mathjit::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MathJIT Polynomial Evaluation Demo ===\n");

    // Example 1: Basic Horner evaluation
    println!("1. Basic Horner's Method:");

    // Polynomial: 5 + 4x + 3x² + 2x³
    let coeffs = [5.0, 4.0, 3.0, 2.0]; // [constant, x, x², x³]
    let x_val = 2.0;

    // Direct evaluation
    let x = DirectEval::var("x", x_val);
    let result = polynomial::horner::<DirectEval, f64>(&coeffs, x);

    // Manual calculation for verification: 5 + 4(2) + 3(4) + 2(8) = 5 + 8 + 12 + 16 = 41
    println!("   Polynomial: 5 + 4x + 3x² + 2x³");
    println!("   At x = {x_val}: {result}");
    println!("   Expected: 5 + 4(2) + 3(4) + 2(8) = 41");
    println!("   ✓ Correct: {}\n", result == 41.0);

    // Example 2: Pretty printing the Horner structure
    println!("2. Horner's Method Structure:");

    let x_pretty = PrettyPrint::var("x");
    let pretty_result = polynomial::horner::<PrettyPrint, f64>(&coeffs, x_pretty);
    println!("   Horner form: {pretty_result}\n");

    // Example 3: Polynomial from roots
    println!("3. Polynomial from Roots:");

    // Create polynomial with roots at 1, 2, and 3: (x-1)(x-2)(x-3)
    let roots = [1.0, 2.0, 3.0];

    // Test at x = 0: (0-1)(0-2)(0-3) = (-1)(-2)(-3) = -6
    let poly_at_0 = polynomial::from_roots::<DirectEval, f64>(&roots, DirectEval::var("x", 0.0));
    println!("   Polynomial with roots [1, 2, 3]: (x-1)(x-2)(x-3)");
    println!("   At x = 0: {poly_at_0}");
    println!("   Expected: (0-1)(0-2)(0-3) = -6");
    println!("   ✓ Correct: {}", poly_at_0 == -6.0);

    // Test at the roots (should be 0)
    for &root in &roots {
        let poly_at_root =
            polynomial::from_roots::<DirectEval, f64>(&roots, DirectEval::var("x", root));
        println!("   At x = {root}: {poly_at_root} (should be 0)");
    }
    println!();

    // Example 4: Polynomial derivative
    println!("4. Polynomial Derivative:");

    // Derivative of 5 + 4x + 3x² + 2x³ is 4 + 6x + 6x²
    let deriv_at_2 =
        polynomial::horner_derivative::<DirectEval, f64>(&coeffs, DirectEval::var("x", 2.0));
    println!("   Original: 5 + 4x + 3x² + 2x³");
    println!("   Derivative: 4 + 6x + 6x²");
    println!("   At x = 2: {deriv_at_2}");
    println!("   Expected: 4 + 6(2) + 6(4) = 4 + 12 + 24 = 40");
    println!("   ✓ Correct: {}\n", deriv_at_2 == 40.0);

    // Example 5: Comparison with naive evaluation
    println!("5. Efficiency Comparison:");

    // Define the same polynomial using naive method (traditional syntax)
    fn naive_polynomial_traditional<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        // 5 + 4x + 3x² + 2x³
        let x2 = E::pow(x.clone(), E::constant(2.0));
        let x3 = E::pow(x.clone(), E::constant(3.0));

        E::add(
            E::add(
                E::add(E::constant(5.0), E::mul(E::constant(4.0), x.clone())),
                E::mul(E::constant(3.0), x2),
            ),
            E::mul(E::constant(2.0), x3),
        )
    }

    // Define the same polynomial using modern operator overloading
    fn naive_polynomial_modern(x: Expr<DirectEval, f64>) -> Expr<DirectEval, f64> {
        // 5 + 4x + 3x² + 2x³ - much more readable!
        let c0 = Expr::constant(5.0);
        let c1 = Expr::constant(4.0);
        let c2 = Expr::constant(3.0);
        let c3 = Expr::constant(2.0);
        let two = Expr::constant(2.0);
        let three = Expr::constant(3.0);

        c0 + c1 * x.clone() + c2 * x.clone().pow(two) + c3 * x.pow(three)
    }

    let naive_result_traditional =
        naive_polynomial_traditional::<DirectEval>(DirectEval::var("x", 2.0));
    let naive_result_modern = naive_polynomial_modern(Expr::var_with_value("x", 2.0));
    let naive_result_modern_val = naive_result_modern.eval();
    let horner_result = polynomial::horner::<DirectEval, f64>(&coeffs, DirectEval::var("x", 2.0));

    println!("   Traditional naive method: {naive_result_traditional}");
    println!("   Modern naive method: {naive_result_modern_val}");
    println!("   Horner method result: {horner_result}");
    println!(
        "   ✓ All methods agree: {}",
        naive_result_traditional == naive_result_modern_val
            && naive_result_traditional == horner_result
    );

    // Show the structure difference
    let naive_pretty = naive_polynomial_traditional::<PrettyPrint>(PrettyPrint::var("x"));
    let horner_pretty = polynomial::horner::<PrettyPrint, f64>(&coeffs, PrettyPrint::var("x"));

    println!("\n   Naive structure (many multiplications):");
    println!("   {naive_pretty}");
    println!("\n   Horner structure (fewer multiplications):");
    println!("   {horner_pretty}\n");

    // Example 6: Working with different numeric types
    println!("6. Different Numeric Types:");

    // Same polynomial with f32
    let coeffs_f32 = [5.0_f32, 4.0_f32, 3.0_f32, 2.0_f32];
    let result_f32 =
        polynomial::horner::<DirectEval, f32>(&coeffs_f32, DirectEval::var("x", 2.0_f32));
    println!("   f32 result: {result_f32}");

    // Same polynomial with f64
    let result_f64 = polynomial::horner::<DirectEval, f64>(&coeffs, DirectEval::var("x", 2.0_f64));
    println!("   f64 result: {result_f64}");
    println!("   ✓ Type flexibility: Both work seamlessly\n");

    // Example 7: Edge cases
    println!("7. Edge Cases:");

    // Empty polynomial
    let empty_coeffs: [f64; 0] = [];
    let empty_result =
        polynomial::horner::<DirectEval, f64>(&empty_coeffs, DirectEval::var("x", 5.0));
    println!("   Empty polynomial: {empty_result}");

    // Single coefficient (constant)
    let constant_coeffs = [42.0];
    let constant_result =
        polynomial::horner::<DirectEval, f64>(&constant_coeffs, DirectEval::var("x", 5.0));
    println!("   Constant polynomial (42): {constant_result}");

    // Linear polynomial
    let linear_coeffs = [1.0, 2.0]; // 1 + 2x
    let linear_result =
        polynomial::horner::<DirectEval, f64>(&linear_coeffs, DirectEval::var("x", 3.0));
    println!("   Linear polynomial (1 + 2x) at x=3: {linear_result}");
    println!("   Expected: 1 + 2(3) = 7");
    println!("   ✓ Correct: {}\n", linear_result == 7.0);

    println!("=== Demo Complete ===");
    println!("\nKey Benefits of Horner's Method:");
    println!("• Reduces multiplications from O(n²) to O(n)");
    println!("• Better numerical stability");
    println!("• Works with any final tagless interpreter");
    println!("• Type-safe and zero-cost abstractions");

    Ok(())
}
