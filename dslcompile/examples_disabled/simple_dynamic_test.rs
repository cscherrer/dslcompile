use dslcompile::{
    ast::DynamicContext,
    backends::{RustCodeGenerator, RustCompiler},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Simple DynamicContext Test (No Summation)");
    println!("=============================================");
    println!("Testing if basic DynamicContext compilation works without summation");

    // Test data
    let mu = 2.0;
    let sigma = 0.5;
    let x_val = 2.447731644977576;

    println!("\nTest parameters: μ={mu}, σ={sigma}, x={x_val}");

    // Build simple expression with DynamicContext (NO summation)
    println!("\n=== Building Simple Expression ===");
    let mut ctx = DynamicContext::new();

    // Create variables
    let mu_var: dslcompile::DynamicExpr<f64, 0> = ctx.var(); // Variable(0)
    let sigma_var: dslcompile::DynamicExpr<f64, 0> = ctx.var(); // Variable(1)
    let x_var: dslcompile::DynamicExpr<f64, 0> = ctx.var(); // Variable(2)

    // Build: (x - μ) / σ
    let standardized = (&x_var - &mu_var) / &sigma_var;

    println!("✅ Built expression: (x - μ) / σ");
    println!("   Variables: mu=var_0, sigma=var_1, x=var_2");

    // Test direct evaluation first
    println!("\n=== Direct Evaluation Test ===");
    use frunk::hlist;
    let direct_result: f64 = ctx.eval(&standardized, hlist![mu, sigma, x_val]);
    let expected_result: f64 = (x_val - mu) / sigma;

    println!("DynamicContext result: {direct_result:.10}");
    println!("Expected result:       {expected_result:.10}");
    println!(
        "Difference:            {:.2e}",
        (direct_result - expected_result).abs()
    );

    let eval_matches = (direct_result - expected_result).abs() < 1e-10;
    println!(
        "Evaluation match:      {}",
        if eval_matches { "✅ YES" } else { "❌ NO" }
    );

    if !eval_matches {
        println!("⚠️  DynamicContext evaluation already has issues!");
        return Ok(());
    }

    // Test code generation
    println!("\n=== Code Generation Test ===");

    // Convert to AST
    let ast = dslcompile::ast::advanced::ast_from_expr(&standardized);
    println!("AST: {ast:?}");

    // Generate code
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(ast, "simple_test")?;

    println!("\nGenerated Rust code:");
    println!("{}", "=".repeat(50));
    println!("{rust_code}");
    println!("{}", "=".repeat(50));

    // Compile
    let compiler = RustCompiler::new();
    let compiled_fn = compiler.compile_and_load(&rust_code, "simple_test")?;

    // Test compiled function
    println!("\n=== Compiled Function Test ===");
    let params = vec![mu, sigma, x_val];
    let compiled_result = compiled_fn.call(params)?;

    println!("Compiled result:       {compiled_result:.10}");
    println!("Expected result:       {expected_result:.10}");
    println!(
        "Difference:            {:.2e}",
        (compiled_result - expected_result).abs()
    );

    let compiled_matches = (compiled_result - expected_result).abs() < 1e-10;
    println!(
        "Compilation match:     {}",
        if compiled_matches {
            "✅ YES"
        } else {
            "❌ NO"
        }
    );

    // Summary
    println!("\n🎯 Test Results:");
    println!(
        "   Direct evaluation:  {}",
        if eval_matches { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "   Code generation:    {}",
        if compiled_matches {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    if eval_matches && compiled_matches {
        println!("\n🎉 SUCCESS: Basic DynamicContext compilation works!");
        println!("   This confirms the issue is specifically in summation handling,");
        println!("   not in the core compilation pipeline.");
    } else if eval_matches && !compiled_matches {
        println!("\n⚠️  Code generation has issues even without summation.");
        println!("   The problem is deeper in the compilation pipeline.");
    } else {
        println!("\n❌ DynamicContext has fundamental evaluation issues.");
    }

    Ok(())
}
