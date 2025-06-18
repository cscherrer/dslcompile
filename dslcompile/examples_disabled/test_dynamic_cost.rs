//! Test Dynamic Cost Functionality
//!
//! Simple test to verify dynamic cost integration works.

use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Testing Dynamic Cost Integration");

    // Test creating the optimizer with dynamic cost support
    match NativeEgglogOptimizer::new() {
        Ok(mut optimizer) => {
            println!("✅ Successfully created optimizer with dynamic cost support");

            // Test setting a simple dynamic cost
            let simple_expr = dslcompile::ast::ASTRepr::Constant(42.0);
            match optimizer.set_dynamic_cost(&simple_expr, 100) {
                Ok(()) => println!("✅ Successfully set dynamic cost"),
                Err(e) => println!("❌ Failed to set dynamic cost: {e}"),
            }

            // Test basic optimization
            let test_expr = dslcompile::ast::ASTRepr::Add(vec![
                dslcompile::ast::ASTRepr::Variable(0),
                dslcompile::ast::ASTRepr::Constant(0.0),
            ]);

            match optimizer.optimize(&test_expr) {
                Ok(result) => {
                    println!("✅ Successfully optimized expression");
                    println!("   Input:  {test_expr:?}");
                    println!("   Output: {result:?}");
                }
                Err(e) => println!("❌ Failed to optimize: {e}"),
            }
        }
        Err(e) => {
            println!("❌ Failed to create optimizer: {e}");
            return Err(e.into());
        }
    }

    println!("🎯 Dynamic cost integration test complete");
    Ok(())
}
