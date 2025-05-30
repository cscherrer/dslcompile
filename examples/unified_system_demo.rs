//! Unified System Demo
//!
//! This demo showcases the new unified typed variable system that combines:
//! - Type-safe variable creation with compile-time type checking
//! - Beautiful operator overloading syntax
//! - High-level mathematical functions (polynomials, Gaussian, logistic, etc.)
//! - Simple evaluation interface
//! - Full backward compatibility

use mathcompile::MathBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unified Typed Variable System Demo ===\n");

    // Create the unified builder
    let math = MathBuilder::new();

    // ============================================================================
    // 1. Beautiful Operator Overloading
    // ============================================================================
    println!("1. Beautiful Operator Overloading:");

    let x = math.var("x");
    let y = math.var("y");

    // Natural mathematical syntax
    let expr1 = &x * &x + 2.0 * &x + &y;
    println!("   Expression: x² + 2x + y");

    let result1 = math.eval(&expr1, &[("x", 3.0), ("y", 1.0)]);
    println!("   Result at x=3, y=1: {result1}"); // 3² + 2*3 + 1 = 16

    // ============================================================================
    // 2. Transcendental Functions
    // ============================================================================
    println!("\n2. Transcendental Functions:");

    let expr2 = x.clone().sin() * y.clone().cos() + x.clone().exp();
    println!("   Expression: sin(x) * cos(y) + exp(x)");

    let result2 = math.eval(&expr2, &[("x", 0.0), ("y", 0.0)]);
    println!("   Result at x=0, y=0: {result2}"); // sin(0) * cos(0) + exp(0) = 0 * 1 + 1 = 1

    // ============================================================================
    // 3. High-Level Mathematical Functions
    // ============================================================================
    println!("\n3. High-Level Mathematical Functions:");

    // Polynomial: 2x² + 3x + 1
    let poly = math.poly(&[1.0, 3.0, 2.0], &x);
    println!("   Polynomial: 2x² + 3x + 1");
    let poly_result = math.eval(&poly, &[("x", 2.0)]);
    println!("   Result at x=2: {poly_result}"); // 2*4 + 3*2 + 1 = 15

    // Quadratic: x² - 4x + 3
    let quad = math.quadratic(1.0, -4.0, 3.0, &x);
    println!("   Quadratic: x² - 4x + 3");
    let quad_result = math.eval(&quad, &[("x", 1.0)]);
    println!("   Result at x=1: {quad_result}"); // 1 - 4 + 3 = 0

    // Gaussian distribution (mean=0, std=1)
    let gaussian = math.gaussian(0.0, 1.0, &x);
    println!("   Gaussian: N(0,1)");
    let gauss_result = math.eval(&gaussian, &[("x", 0.0)]);
    println!("   Result at x=0: {gauss_result:.6}"); // Should be ~0.398942 (1/√(2π))

    // Logistic function
    let logistic = math.logistic(&x);
    println!("   Logistic: 1/(1 + exp(-x))");
    let logistic_result = math.eval(&logistic, &[("x", 0.0)]);
    println!("   Result at x=0: {logistic_result}"); // Should be 0.5

    // ============================================================================
    // 4. Complex Expressions
    // ============================================================================
    println!("\n4. Complex Expressions:");

    // Combine high-level functions with operators
    let complex = &poly + &logistic * math.constant(10.0);
    println!("   Expression: (2x² + 3x + 1) + 10 * logistic(x)");
    let complex_result = math.eval(&complex, &[("x", 1.0)]);
    println!("   Result at x=1: {complex_result:.6}");

    // ============================================================================
    // 5. Type Safety Demo
    // ============================================================================
    println!("\n5. Type Safety:");

    // Create typed variables
    let x_f64 = math.typed_var::<f64>("x_f64");
    let y_f32 = math.typed_var::<f32>("y_f32");

    let x_expr = math.expr_from(x_f64);
    let y_expr = math.expr_from(y_f32);

    // Cross-type operations work with automatic promotion
    let mixed = &x_expr + y_expr; // f32 automatically promotes to f64
    println!("   Mixed types: f64 + f32 → f64");
    let mixed_result = math.eval(&mixed, &[("x_f64", 2.5), ("y_f32", 1.5)]);
    println!("   Result: {mixed_result}");

    println!("\n=== Demo Complete ===");
    println!("\nKey Benefits:");
    println!("✓ Beautiful syntax: x * x + 2.0 * x + y");
    println!("✓ Type safety with automatic promotion");
    println!("✓ High-level mathematical functions");
    println!("✓ Simple evaluation interface");
    println!("✓ Full backward compatibility");

    Ok(())
}
