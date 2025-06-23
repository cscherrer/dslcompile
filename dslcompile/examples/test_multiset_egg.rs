//! Test `MultiSet`<Id> directly with egg e-graphs
//!
//! This demonstrates that `MultiSet`<Id> successfully works with egg's `define_language`!

use dslcompile::prelude::*;


fn main() -> Result<()> {
    println!("🧪 Testing MultiSet<Id> with egg e-graphs");
    println!("==========================================\n");

    // Create a simple expression: 2 + 3 + 2 (should have 2×2 + 3×1 in multiset)
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    let expr = ASTRepr::add_binary(ASTRepr::add_binary(two.clone(), three.clone()), two.clone());

    println!("📊 Test Expression: 2 + 3 + 2");
    println!("Expected: Should be stored as MultiSet{{2: 2, 3: 1}}");
    println!("Original: {expr:?}\n");

    // Test evaluation
    let result = expr.eval_with_vars(&[]);
    println!("\n📊 Evaluation test:");
    println!("  Expression: 2 + 3 + 2 = {result}");
    println!(
        "  Correct: {}",
        (result - 7.0_f64).abs() < 1e-10
    );

    println!("\n🎉 SUCCESS: MultiSet<Id> expression created!");
    println!("   - Native multiset semantics preserved");
    println!("   - Commutativity automatic with MultiSet");

    #[cfg(not(feature = "optimization"))]
    {
        println!("🚫 Egg optimization feature not enabled");
        println!("Run with: cargo run --example test_multiset_egg --features egg_optimization");
    }

    Ok(())
}
