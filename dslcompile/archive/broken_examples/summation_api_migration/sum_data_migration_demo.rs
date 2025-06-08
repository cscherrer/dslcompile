//! Migration Demo: From sum_data() to Unified sum() API
//!
//! This example demonstrates how to migrate from the deprecated `sum_data()` method
//! to the new unified `sum()` API that handles both mathematical and data summation
//! with the same interface.

use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("🔄 Migration Demo: sum_data() → Unified sum() API");
    println!("==================================================\n");

    // Demo 1: Basic data summation migration
    basic_migration_demo()?;

    // Demo 2: Complex expression migration
    complex_expression_migration()?;

    // Demo 3: Performance comparison
    performance_comparison_demo()?;

    println!("🎉 Migration demo completed!");
    println!("💡 Use the unified sum() API for all new code!");
    Ok(())
}

fn basic_migration_demo() -> Result<()> {
    println!("📊 Demo 1: Basic Data Summation Migration");
    println!("==========================================");

    let ctx = DynamicContext::new();
    let param = ctx.var();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    println!("Data: {:?}", data);
    println!("Expression: Σ(x * param for x in data)\n");

    // ❌ OLD WAY: sum_data() + eval_with_data()
    println!("❌ OLD (Deprecated) API:");
    #[allow(deprecated)]
    let old_expr = ctx.sum_data(|x| x * param.clone())?;
    
    #[allow(deprecated)]
    let old_result = ctx.eval_with_data(&old_expr, &[2.0], &[data.clone()]);
    println!("   sum_data() + eval_with_data(): {}", old_result);

    // ✅ NEW WAY: unified sum() + eval_hlist()
    println!("\n✅ NEW (Unified) API:");
    let new_expr = ctx.sum(data.clone(), |x| x * param.clone())?;
    let new_result = ctx.eval_hlist(&new_expr, hlist![2.0, data.clone()]);
    println!("   sum() + eval_hlist(): {}", new_result);

    // Verify identical results
    assert!((old_result - new_result).abs() < 1e-10);
    println!("\n✅ Both approaches produce identical results: {}", new_result);

    println!("\n🎯 Migration Benefits:");
    println!("  • No artificial API distinction");
    println!("  • Type-safe evaluation with HLists");
    println!("  • Same sum() method for mathematical and data summation");
    println!("  • Better performance through direct data binding\n");

    Ok(())
}

fn complex_expression_migration() -> Result<()> {
    println!("🧮 Demo 2: Complex Expression Migration");
    println!("=======================================");

    let ctx = DynamicContext::new();
    let scale = ctx.var();
    let offset = ctx.var();
    let data = vec![1.5, 2.5, 3.5, 4.5];

    println!("Data: {:?}", data);
    println!("Expression: Σ((x + offset) * scale for x in data)\n");

    // ❌ OLD WAY: Complex expression with sum_data
    println!("❌ OLD (Deprecated) API:");
    #[allow(deprecated)]
    let old_complex = ctx.sum_data(|x| (x + offset.clone()) * scale.clone())?;
    
    #[allow(deprecated)]
    let old_result = ctx.eval_with_data(&old_complex, &[2.0, 1.0], &[data.clone()]);
    println!("   Complex sum_data(): {}", old_result);

    // ✅ NEW WAY: Same expression with unified API
    println!("\n✅ NEW (Unified) API:");
    let new_complex = ctx.sum(data.clone(), |x| (x + offset.clone()) * scale.clone())?;
    let new_result = ctx.eval_hlist(&new_complex, hlist![2.0, 1.0, data.clone()]);
    println!("   Complex sum(): {}", new_result);

    // Verify identical results
    assert!((old_result - new_result).abs() < 1e-10);
    println!("\n✅ Complex expressions work identically: {}", new_result);

    // Show the mathematical equivalence
    let expected = data.iter().map(|&x| (x + 1.0) * 2.0).sum::<f64>();
    println!("📐 Mathematical verification: {}", expected);
    assert!((new_result - expected).abs() < 1e-10);

    println!("\n🎯 Complex Expression Benefits:");
    println!("  • Same syntax for complex mathematical expressions");
    println!("  • No need to learn different APIs for data vs mathematical sums");
    println!("  • Consistent parameter ordering in HLists\n");

    Ok(())
}

fn performance_comparison_demo() -> Result<()> {
    println!("⚡ Demo 3: Performance Comparison");
    println!("=================================");

    let ctx = DynamicContext::new();
    let param = ctx.var();
    
    // Use larger dataset for performance testing
    let large_data: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.1).collect();
    println!("Dataset size: {} elements", large_data.len());

    // Build expressions
    #[allow(deprecated)]
    let old_expr = ctx.sum_data(|x| x * param.clone())?;
    let new_expr = ctx.sum(large_data.clone(), |x| x * param.clone())?;

    // Time the old approach
    let start = std::time::Instant::now();
    #[allow(deprecated)]
    let old_result = ctx.eval_with_data(&old_expr, &[2.0], &[large_data.clone()]);
    let old_time = start.elapsed();

    // Time the new approach
    let start = std::time::Instant::now();
    let new_result = ctx.eval_hlist(&new_expr, hlist![2.0, large_data.clone()]);
    let new_time = start.elapsed();

    println!("\n📊 Performance Results:");
    println!("   Old API time: {:?}", old_time);
    println!("   New API time: {:?}", new_time);
    
    let speedup = old_time.as_nanos() as f64 / new_time.as_nanos() as f64;
    println!("   Speedup: {:.2}x", speedup);

    // Verify identical results
    assert!((old_result - new_result).abs() < 1e-10);
    println!("   Results match: {:.2}", new_result);

    println!("\n🚀 Performance Benefits:");
    println!("  • Direct data binding eliminates indirection");
    println!("  • HList evaluation is optimized for type safety");
    println!("  • Unified API reduces code complexity");
    println!("  • Better compiler optimization opportunities\n");

    Ok(())
} 