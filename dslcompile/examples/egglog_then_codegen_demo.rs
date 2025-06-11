//! EggLog + Collection Code Generation Pipeline Demo
//!
//! This demonstrates the complete optimization pipeline:
//! 1. Collection-based AST construction
//! 2. EggLog optimization (should simplify to constants)
//! 3. Rust code generation (should generate constants, not iterators)

use dslcompile::{
    backends::rust_codegen::RustCodeGenerator, contexts::DynamicContext,
    symbolic::native_egglog::NativeEgglogOptimizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 EggLog + Collection Code Generation Pipeline Demo");
    println!("==================================================\n");

    let mut ctx: DynamicContext<f64> = DynamicContext::new();
    let codegen = RustCodeGenerator::new();
    let mut optimizer = NativeEgglogOptimizer::new()?;

    // Test case: sum(i for i in 1..=5) should optimize to 15.0
    println!("📊 Test Case: sum(i for i in 1..=5)");
    println!("Expected: AST → EggLog → Constant(15.0) → Rust constant\n");

    // Step 1: Build Collection AST
    let sum_expr = ctx.sum(1..=5, |i| i);
    let original_ast = sum_expr.to_f64().as_ast().clone();

    println!("🔸 Step 1: Original Collection AST");
    println!("{original_ast:#?}\n");

    // Step 2: Apply EggLog optimization
    println!("🔸 Step 2: Applying EggLog optimization...");
    let optimized_ast = optimizer.optimize(&original_ast)?;

    println!("Optimized AST:");
    println!("{optimized_ast:#?}\n");

    // Check if EggLog successfully optimized to constant
    match &optimized_ast {
        dslcompile::ast::ASTRepr::Constant(value) => {
            println!("✅ SUCCESS: EggLog optimized to constant: {value}");
            if (*value - 15.0).abs() < 1e-10 {
                println!("✅ CORRECT: Value is 15.0 as expected");
            } else {
                println!("❌ WRONG: Expected 15.0, got {value}");
            }
        }
        _ => {
            println!("❌ FAILED: EggLog did not optimize to constant");
            println!("   Still has collection structure - rules not firing");
        }
    }
    println!();

    // Step 3: Generate Rust code from optimized AST
    println!("🔸 Step 3: Generating Rust code from optimized AST...");
    let rust_code = codegen.generate_function(&optimized_ast, "optimized_sum")?;

    println!("Generated Rust code:");
    println!("{rust_code}");
    println!();

    // Check if the generated code is a constant
    if rust_code.contains("return 15") || rust_code.contains("return 15.0") {
        println!("✅ SUCCESS: Generated code contains constant 15!");
    } else if rust_code.contains(".map(") || rust_code.contains(".sum(") {
        println!("❌ PARTIAL: Generated code still contains iterators");
        println!("   This means EggLog optimization didn't work properly");
    } else {
        println!("⚠️  UNKNOWN: Generated code pattern not recognized");
    }

    println!("\n🎯 Pipeline Analysis:");
    println!("Collection AST ✅ → EggLog ? → Rust Codegen ?");

    Ok(())
}
