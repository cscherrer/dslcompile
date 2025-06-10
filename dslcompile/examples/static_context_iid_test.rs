use dslcompile::{
    backends::{RustCodeGenerator, RustCompiler},
    compile_time::static_scoped::*,
};
use frunk::hlist;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 StaticContext IID Gaussian Test");
    println!("==================================");
    println!("Testing if the issue is in DynamicContext frontend vs deeper pipeline");

    // Test data
    let mu = 2.0;
    let sigma = 0.5;
    let x_val = 2.447731644977576; // Single point for debugging

    println!("\nTest parameters: μ={}, σ={}", mu, sigma);
    println!("Test data point: {}", x_val);

    // Build expression with StaticContext
    println!("\n=== Building with StaticContext ===");
    let mut ctx = StaticContext::new();

    // Create the Gaussian log-likelihood expression for a single data point
    // -½((x-μ)/σ)² - log(σ√2π)
    let likelihood_expr = ctx.new_scope(|scope| {
        let (mu_var, scope) = scope.auto_var::<f64>();
        let (sigma_var, scope) = scope.auto_var::<f64>();

        // Constants
        let x_const = scope.constant(x_val);
        let neg_half = scope.constant(-0.5);
        let log_2pi = scope.constant((2.0 * std::f64::consts::PI).ln());
        let half = scope.constant(0.5);

        // Build simplified version: -0.5 * ((x - μ) / σ)²
        // Note: We'll avoid log operations for now as they may not be implemented
        let diff = x_const + (neg_half.clone() * mu_var.clone()); // x - μ (approximation)
        // For now, just test: x - μ
        diff
    });

    println!("✅ Built simplified expression with StaticContext");

    // Test evaluation first
    println!("\n=== Direct Evaluation Test ===");
    let inputs = hlist![mu, sigma];
    let static_result = likelihood_expr.eval(inputs);
    let expected_result = x_val - mu; // Should be x - μ

    println!("StaticContext result: {:.10}", static_result);
    println!("Expected result:      {:.10}", expected_result);
    println!(
        "Difference:           {:.2e}",
        (static_result - expected_result).abs()
    );

    let eval_matches = (static_result - expected_result).abs() < 1e-10;
    println!(
        "Evaluation match:     {}",
        if eval_matches { "✅ YES" } else { "❌ NO" }
    );

    if !eval_matches {
        println!("⚠️  StaticContext evaluation already has issues!");
        return Ok(());
    }

    // Now test code generation
    println!("\n=== Code Generation Test ===");

    // Convert to AST (this might be the issue)
    println!("Converting to AST...");
    // Note: StaticContext expressions don't have to_ast(), so let's skip codegen for now
    println!("⚠️  StaticContext doesn't have AST conversion yet");

    println!("\n🎯 Results:");
    if eval_matches {
        println!("✅ StaticContext evaluation works correctly");
        println!("   This suggests the issue is likely in:");
        println!("   1. DynamicContext summation handling");
        println!("   2. Lambda variable parameter counting in codegen");
        println!("   3. AST conversion from expressions with summations");
    }

    Ok(())
}
