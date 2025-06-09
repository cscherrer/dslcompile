//! Demo: Sqrt Pre/Post Processing
//!
//! This demo shows how sqrt() method calls are transparently converted to
//! power operations internally while generating efficient .sqrt() calls in the output.
//!
//! **Pre-processing**: `x.sqrt()` → `x.pow(0.5)` during AST construction  
//! **Post-processing**: `x.pow(0.5)` → `x.sqrt()` during code generation
//!
//! This approach provides:
//! - Unified power optimization infrastructure  
//! - Familiar sqrt() API for users
//! - Efficient .sqrt() calls in generated code

use dslcompile::ast::VariableRegistry;
use dslcompile::ast::ast_repr::ASTRepr;
use dslcompile::backends::rust_codegen::RustCodeGenerator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sqrt Pre/Post Processing Demo ===\n");

    // Create a variable registry
    let mut registry = VariableRegistry::new();
    let _x_idx = registry.register_variable();

    // Create an expression: x.sqrt()
    let x = ASTRepr::<f64>::Variable(0);
    let sqrt_expr = x.sqrt(); // This is pre-processed into x^0.5

    println!("1. User writes: x.sqrt()");

    // Show internal representation (should be Power)
    println!("2. Internal AST: {sqrt_expr:#?}");

    // Verify it's actually a power operation
    match &sqrt_expr {
        ASTRepr::Pow(base, exp) => {
            println!("3. ✅ Pre-processing successful: sqrt() → Pow(x, 0.5)");
            if let (ASTRepr::Variable(0), ASTRepr::Constant(exp_val)) =
                (base.as_ref(), exp.as_ref())
            {
                println!("   Base: Variable(0) = x");
                println!("   Exponent: Constant({exp_val}) ≈ 0.5");

                if (exp_val - 0.5).abs() < 1e-15 {
                    println!("   ✅ Exponent is exactly 0.5");
                } else {
                    println!("   ❌ Exponent is not 0.5: {exp_val}");
                }
            }
        }
        _ => {
            println!("3. ❌ Pre-processing failed: expected Pow, got {sqrt_expr:?}");
            return Ok(());
        }
    }

    // Generate Rust code (should use .sqrt() optimization)
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&sqrt_expr, "test_sqrt")?;

    println!("4. Generated Rust code: {rust_code}");

    // Verify post-processing worked
    if rust_code.contains(".sqrt()") {
        println!("5. ✅ Post-processing successful: Pow(x, 0.5) → .sqrt()");
    } else if rust_code.contains(".powf(0.5)") {
        println!("5. ⚠️  Post-processing not applied: still using .powf(0.5)");
    } else {
        println!("5. ❓ Unexpected output: {rust_code}");
    }

    println!("\n=== Complex Example: sqrt(x^2 + 1) ===");

    // Create a more complex expression: sqrt(x^2 + 1)
    let x = ASTRepr::<f64>::Variable(0);
    let x_squared = x.clone().pow(ASTRepr::Constant(2.0));
    let x_squared_plus_one = ASTRepr::Add(Box::new(x_squared), Box::new(ASTRepr::Constant(1.0)));
    let complex_sqrt = x_squared_plus_one.sqrt();

    println!("User expression: sqrt(x^2 + 1)");

    // Generate code for complex example
    let complex_code = codegen.generate_function(&complex_sqrt, "test_complex_sqrt")?;
    println!("Generated code: {complex_code}");

    // Check if it contains .sqrt() call
    if complex_code.contains(".sqrt()") {
        println!("✅ Complex sqrt optimization working");
    } else {
        println!("⚠️  Complex sqrt not optimized");
    }

    println!("\n=== Benefits of This Approach ===");
    println!("✅ User-friendly API: x.sqrt() just works");
    println!("✅ Internal unification: all powers handled consistently");
    println!("✅ Optimal code generation: .sqrt() is fast");
    println!("✅ No code duplication: sqrt logic unified with power logic");
    println!("✅ Future-proof: can add more power optimizations easily");

    Ok(())
}
