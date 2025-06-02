//! Autodiff + Rust Code Generation Demo
//!
//! This example demonstrates symbolic automatic differentiation with Rust code generation.
//! Note: This example requires the '`ad_trait`' and 'optimization' features.

#[cfg(all(feature = "ad_trait", feature = "optimization"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use mathcompile::backends::RustCodeGenerator;
    use mathcompile::final_tagless::{ASTEval, ASTMathExpr};
    use mathcompile::symbolic::symbolic::SymbolicOptimizer;
    use mathcompile::symbolic::symbolic_ad::SymbolicAD;

    println!("🚀 MathCompile: Symbolic Autodiff + Rust Code Generation Demo");
    println!("=============================================================\n");

    // 1. Define a mathematical function
    println!("1️⃣  Defining Mathematical Function");
    println!("----------------------------------");

    // f(x) = x² + 2x + 1
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)),
            ASTEval::mul(ASTEval::constant(2.0), ASTEval::var(0)),
        ),
        ASTEval::constant(1.0),
    );

    println!("Function: f(x) = x² + 2x + 1");
    println!("Expected derivative: f'(x) = 2x + 2\n");

    // 2. Generate Rust code for the original function
    println!("2️⃣  Generating Rust Code for Original Function");
    println!("-----------------------------------------------");

    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&expr, "quadratic_function")?;

    println!("Generated Rust code:");
    println!("{rust_code}\n");

    // 3. Use symbolic differentiation
    println!("3️⃣  Computing Derivatives with Symbolic AD");
    println!("-------------------------------------------");

    let mut symbolic_ad = SymbolicAD::new()?;
    let result = symbolic_ad.compute_with_derivatives(&expr)?;

    println!("Function: {:?}", result.function);
    println!("Derivatives: {:?}", result.first_derivatives);
    println!("Optimization stats: {:?}\n", result.stats);

    // 4. Apply symbolic optimization
    println!("4️⃣  Applying Symbolic Optimization");
    println!("----------------------------------");

    let mut optimizer = SymbolicOptimizer::new()?;
    let optimized_expr = optimizer.optimize(&expr)?;

    println!("Original expression: {expr:?}");
    println!("Optimized expression: {optimized_expr:?}\n");

    // 5. Generate optimized Rust code
    let optimized_rust = codegen.generate_function(&optimized_expr, "optimized_quadratic")?;
    println!("Optimized Rust code:");
    println!("{optimized_rust}");

    println!("\n✅ Symbolic autodiff and code generation completed successfully!");

    Ok(())
}

#[cfg(not(all(feature = "ad_trait", feature = "optimization")))]
fn main() {
    println!("❌ This demo requires both 'ad_trait' and 'optimization' features!");
    println!(
        "Run with: cargo run --example autodiff_rust_codegen_demo --features \"ad_trait optimization\""
    );
}
