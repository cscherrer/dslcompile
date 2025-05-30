//! Special Functions Example
//!
//! This example demonstrates the use of special mathematical functions
//! like gamma, beta, error functions, and Bessel functions.
//!
//! To run this example:
//! ```bash
//! cargo run --example special_functions --features special_functions
//! ```

#[cfg(feature = "special_functions")]
fn main() {
    use mathcompile::ast::{ASTRepr, function_categories::SpecialCategory};

    println!("=== Special Functions Example ===\n");

    // Create some special function expressions
    let x = ASTRepr::variable("x".to_string());
    let two = ASTRepr::constant(2.0);
    let half = ASTRepr::constant(0.5);

    // Gamma function: Γ(x)
    let gamma_expr = ASTRepr::Function(Box::new(SpecialCategory::gamma(x.clone())));
    println!("Gamma function: Γ(x) = {:?}", gamma_expr);

    // Beta function: B(2, 0.5)
    let beta_expr = ASTRepr::Function(Box::new(SpecialCategory::beta(two.clone(), half.clone())));
    println!("Beta function: B(2, 0.5) = {:?}", beta_expr);

    // Error function: erf(x)
    let erf_expr = ASTRepr::Function(Box::new(SpecialCategory::erf(x.clone())));
    println!("Error function: erf(x) = {:?}", erf_expr);

    // Complementary error function: erfc(x)
    let erfc_expr = ASTRepr::Function(Box::new(SpecialCategory::erfc(x.clone())));
    println!("Complementary error function: erfc(x) = {:?}", erfc_expr);

    // Bessel function of the first kind: J₀(x)
    let bessel_j0_expr = ASTRepr::Function(Box::new(SpecialCategory::bessel_j0(x.clone())));
    println!("Bessel J₀ function: J₀(x) = {:?}", bessel_j0_expr);

    // Lambert W function: W₀(x)
    let lambert_w0_expr = ASTRepr::Function(Box::new(SpecialCategory::lambert_w0(x.clone())));
    println!("Lambert W₀ function: W₀(x) = {:?}", lambert_w0_expr);

    println!("\n=== Egglog Representations ===\n");

    // Show egglog representations
    println!("Γ(x) in egglog: {}", gamma_expr.to_egglog());
    println!("B(2, 0.5) in egglog: {}", beta_expr.to_egglog());
    println!("erf(x) in egglog: {}", erf_expr.to_egglog());
    println!("J₀(x) in egglog: {}", bessel_j0_expr.to_egglog());
    println!("W₀(x) in egglog: {}", lambert_w0_expr.to_egglog());

    println!("\n=== Mathematical Identities ===\n");

    // Demonstrate some mathematical identities
    println!("The special functions support mathematical identities:");
    println!("• Γ(1) = 1");
    println!("• Γ(1/2) = √π ≈ 1.7724538509055159");
    println!("• B(a,b) = B(b,a) (symmetry)");
    println!("• erf(0) = 0");
    println!("• erf(-x) = -erf(x) (odd function)");
    println!("• erfc(x) = 1 - erf(x)");
    println!("• J₀(0) = 1");
    println!("• W₀(0) = 0");
    println!("• W₀(e) = 1");

    println!("\n=== Integration with 'special' crate ===\n");

    println!("For f64 and f32 types, these functions use the high-quality");
    println!("implementations from the 'special' crate:");
    println!("https://docs.rs/special/latest/special/");

    println!("\nThis provides:");
    println!("• Accurate numerical implementations");
    println!("• Proper handling of edge cases");
    println!("• Performance-optimized algorithms");
    println!("• Well-tested mathematical functions");
}

#[cfg(not(feature = "special_functions"))]
fn main() {
    println!("Special functions are not enabled!");
    println!("To run this example, use:");
    println!("cargo run --example special_functions --features special_functions");
}
