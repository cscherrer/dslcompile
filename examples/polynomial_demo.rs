//! Polynomial Demo - All About Polynomials
//!
//! This example demonstrates polynomial operations using `MathCompile`'s
//! final tagless approach and polynomial utility functions.

use mathcompile::final_tagless::{DirectEval, MathExpr, PrettyPrint};
use mathcompile::polynomial;
use mathcompile::prelude::*;

fn main() -> Result<()> {
    println!("🚀 MathCompile Polynomial Demo");
    println!("==============================");

    // Create variables using the new index-based API
    let x_val = 2.0;
    let x = DirectEval::var_with_value(0, x_val);

    println!("Working with polynomials using DirectEval and PrettyPrint interpreters");
    println!("x = {x_val}");
    println!();

    // 1. Basic polynomial evaluation
    println!("1. Basic Polynomial Evaluation");
    println!("------------------------------");

    // Create a simple polynomial: 3x² + 2x + 1
    let poly_result = 3.0 * x * x + 2.0 * x + 1.0;
    println!("Polynomial: 3x² + 2x + 1");
    println!("Result at x={x_val}: {poly_result}");
    println!();

    // 2. Polynomial from roots
    println!("2. Polynomial from Roots");
    println!("------------------------");
    let roots = vec![1.0, 2.0, 3.0];
    println!("Roots: {roots:?}");

    // Evaluate polynomial from roots at x=0
    let poly_at_0 =
        polynomial::from_roots::<DirectEval, f64>(&roots, DirectEval::var_with_value(0, 0.0));
    println!("Polynomial from roots at x=0: {poly_at_0}");

    // Check that each root gives 0
    for &root in &roots {
        let poly_at_root =
            polynomial::from_roots::<DirectEval, f64>(&roots, DirectEval::var_with_value(0, root));
        println!("Polynomial at root x={root}: {poly_at_root}");
        assert!(poly_at_root.abs() < 1e-10);
    }
    println!();

    // 3. Horner's method for polynomial evaluation
    println!("3. Horner's Method");
    println!("-----------------");
    let coeffs = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x² (coefficients in ascending degree order)
    let horner_result =
        polynomial::horner::<DirectEval, f64>(&coeffs, DirectEval::var_with_value(0, 2.0));
    println!("Coefficients: {coeffs:?} (1 + 2x + 3x²)");
    println!("Horner evaluation at x=2.0: {horner_result}");
    println!("Expected: 1 + 2(2) + 3(4) = 1 + 4 + 12 = 17");
    assert_eq!(horner_result, 17.0);

    // Derivative using Horner's method
    let derivative_result = polynomial::horner_derivative::<DirectEval, f64>(
        &coeffs,
        DirectEval::var_with_value(0, 2.0),
    );
    println!("Derivative at x=2.0: {derivative_result}");
    println!("Expected derivative (2 + 6x): 2 + 6(2) = 14");
    assert_eq!(derivative_result, 14.0);
    println!();

    // 4. Traditional final tagless polynomial
    println!("4. Traditional Final Tagless Polynomial");
    println!("---------------------------------------");

    fn naive_polynomial_traditional<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        // x³ + 2x² + 3x + 4
        let x_cubed = E::pow(x.clone(), E::constant(3.0));
        let x_squared = E::pow(x.clone(), E::constant(2.0));
        let linear = E::mul(E::constant(3.0), x.clone());
        let constant = E::constant(4.0);

        E::add(
            E::add(E::add(x_cubed, E::mul(E::constant(2.0), x_squared)), linear),
            constant,
        )
    }

    let poly_direct_result =
        naive_polynomial_traditional::<DirectEval>(DirectEval::var_with_value(0, 2.0));
    println!("Traditional polynomial x³ + 2x² + 3x + 4 at x=2.0: {poly_direct_result}");

    let coeffs = vec![4.0, 3.0, 2.0, 1.0]; // 4 + 3x + 2x² + x³
    let horner_result =
        polynomial::horner::<DirectEval, f64>(&coeffs, DirectEval::var_with_value(0, 2.0));
    println!("Same polynomial using Horner method: {horner_result}");
    assert_eq!(poly_direct_result, horner_result);
    println!("Both approaches give the same result!");
    println!();

    // 5. Pretty printing
    println!("5. Pretty Printing");
    println!("------------------");
    let naive_pretty = naive_polynomial_traditional::<PrettyPrint>(PrettyPrint::var(0));
    println!("Traditional polynomial expression: {naive_pretty}");

    let horner_pretty = polynomial::horner::<PrettyPrint, f64>(&coeffs, PrettyPrint::var(0));
    println!("Horner method expression: {horner_pretty}");
    println!();

    // 6. Working with different numeric types
    println!("6. Different Numeric Types");
    println!("---------------------------");

    // f32 polynomials
    let coeffs_f32 = vec![1.0_f32, 2.0_f32, 3.0_f32];
    let result_f32 =
        polynomial::horner::<DirectEval, f32>(&coeffs_f32, DirectEval::var_with_value(0, 2.0_f32));
    println!("f32 polynomial (1 + 2x + 3x²) at x=2.0: {result_f32}");

    // f64 polynomials
    let coeffs = vec![1.0_f64, 2.0_f64, 3.0_f64];
    let result_f64 =
        polynomial::horner::<DirectEval, f64>(&coeffs, DirectEval::var_with_value(0, 2.0_f64));
    println!("f64 polynomial (1 + 2x + 3x²) at x=2.0: {result_f64}");
    println!();

    // 7. Edge cases
    println!("7. Edge Cases");
    println!("-------------");

    // Empty polynomial (should be 0)
    let empty_coeffs: Vec<f64> = vec![];
    let empty_result =
        polynomial::horner::<DirectEval, f64>(&empty_coeffs, DirectEval::var_with_value(0, 5.0));
    println!("Empty polynomial: {empty_result}");
    assert_eq!(empty_result, 0.0);

    // Constant polynomial
    let constant_coeffs = vec![42.0];
    let constant_result =
        polynomial::horner::<DirectEval, f64>(&constant_coeffs, DirectEval::var_with_value(0, 5.0));
    println!("Constant polynomial (42): {constant_result}");
    assert_eq!(constant_result, 42.0);

    // Linear polynomial
    let linear_coeffs = vec![1.0, 2.0]; // 1 + 2x
    let linear_result =
        polynomial::horner::<DirectEval, f64>(&linear_coeffs, DirectEval::var_with_value(0, 3.0));
    println!("Linear polynomial (1 + 2x) at x=3.0: {linear_result}");
    assert_eq!(linear_result, 7.0); // 1 + 2*3 = 7
    println!();

    println!("🎉 Polynomial Demo Complete!");
    println!();
    println!("Key Takeaways:");
    println!("- Polynomials can be evaluated using traditional final tagless or utility functions");
    println!("- Horner's method is more efficient for polynomial evaluation");
    println!("- Works with different numeric types (f32, f64)");
    println!("- Pretty printing shows the structure of polynomial expressions");
    println!("- Polynomial from roots generates polynomials with specified zeros");

    Ok(())
}
