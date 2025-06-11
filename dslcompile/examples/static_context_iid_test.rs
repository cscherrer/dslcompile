use dslcompile::{
    backends::{RustCodeGenerator, RustCompiler},
    ast::runtime::expression_builder::DynamicContext,
};
use frunk::hlist;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Simple DynamicContext Test (No Summation)");
    println!("=============================================");
    println!("Testing if the issue is in DynamicContext frontend vs deeper pipeline");

    // Test data
    let mu = 2.0_f64;
    let sigma = 0.5_f64;
    let x_val = 2.447731644977576_f64; // Single point for debugging

    println!("\nTest parameters: μ={}, σ={}", mu, sigma);
    println!("Test data point: {}", x_val);

    // Build expression with DynamicContext
    println!("\n=== Building with DynamicContext ===");
    let mut ctx = DynamicContext::new();

    // Create variables
    let mu_var = ctx.var(); // Variable(0)
    let sigma_var = ctx.var(); // Variable(1)

    // Build simplified version: x - μ (where x is a constant)
    let x_const = ctx.constant(x_val);
    let diff = &x_const - &mu_var; // x - μ

    println!("✅ Built simplified expression: x - μ");

    // Test evaluation first
    println!("\n=== Direct Evaluation Test ===");
    let inputs = hlist![mu, sigma];
    let dynamic_result = ctx.eval(&diff, inputs);
    let expected_result = x_val - mu; // Should be x - μ

    println!("DynamicContext result: {:.10}", dynamic_result);
    println!("Expected result:       {:.10}", expected_result);
    println!(
        "Difference:            {:.2e}",
        (dynamic_result - expected_result).abs()
    );

    let eval_matches = (dynamic_result - expected_result).abs() < 1e-10;
    println!(
        "Evaluation match:      {}",
        if eval_matches { "✅ YES" } else { "❌ NO" }
    );

    if !eval_matches {
        println!("⚠️  DynamicContext evaluation already has issues!");
        return Ok(());
    }

    // Now test code generation
    println!("\n=== Code Generation Test ===");

    // Convert to AST
    println!("Converting to AST...");
    let ast = diff.as_ast();
    println!("✅ AST conversion successful");

    // Generate Rust code
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(ast, "simple_diff")?;
    println!("✅ Code generation successful");
    println!("Generated code:\n{}", rust_code);

    // Try to compile if rustc is available
    if RustCompiler::is_available() {
        println!("\n=== Compilation Test ===");
        let compiler = RustCompiler::new();
        match compiler.compile_and_load(&rust_code, "simple_diff") {
            Ok(compiled_func) => {
                let compiled_result = compiled_func.call(hlist![mu, sigma])?;
                println!("Compiled result: {:.10}", compiled_result);
                
                let compiled_matches = (compiled_result - expected_result).abs() < 1e-10;
                println!(
                    "Compiled match:  {}",
                    if compiled_matches { "✅ YES" } else { "❌ NO" }
                );
                
                if eval_matches && compiled_matches {
                    println!("\n🎉 SUCCESS: Basic DynamicContext compilation works!");
                    println!("   This confirms the issue is specifically in summation handling,");
                    println!("   not in the core compilation pipeline.");
                } else {
                    println!("\n⚠️  Code generation has issues even without summation.");
                }
            }
            Err(e) => {
                println!("⚠️  Compilation failed: {e}");
                println!("   This might be expected in some environments");
            }
        }
    } else {
        println!("\n=== Compilation Test Skipped ===");
        println!("rustc not available");
    }

    println!("\n🎯 Results:");
    if eval_matches {
        println!("✅ DynamicContext evaluation works correctly");
        println!("   This suggests the issue is likely in:");
        println!("   1. DynamicContext summation handling");
        println!("   2. Lambda variable parameter counting in codegen");
        println!("   3. AST conversion from expressions with summations");
    }

    Ok(())
}
