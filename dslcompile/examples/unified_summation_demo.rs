use dslcompile::unified_context::UnifiedContext;
use dslcompile::symbolic::symbolic::OptimizationConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 UNIFIED SUMMATION INTEGRATION DEMO");
    println!("=====================================");
    println!();

    // ========================================================================
    // MATHEMATICAL RANGE SUMMATIONS
    // ========================================================================
    
    println!("📊 MATHEMATICAL RANGE SUMMATIONS");
    println!("---------------------------------");
    
    // Test with different optimization strategies
    let strategies = [
        ("StaticCodegen", OptimizationConfig::zero_overhead()),
        ("DynamicCodegen", OptimizationConfig::dynamic_performance()),
        ("Interpretation", OptimizationConfig::dynamic_flexible()),
        ("Adaptive", OptimizationConfig::adaptive()),
    ];
    
    for (name, config) in &strategies {
        println!("🎯 Strategy: {name}");
        let ctx = UnifiedContext::with_config(config.clone());
        
        // Simple summation: Σ(i=1 to 5) i = 15
        let sum_expr = ctx.sum(1..=5, |i| i)?;
        let result = ctx.eval(&sum_expr, &[])?;
        println!("  Σ(i=1 to 5) i = {result} (expected: 15.0)");
        
        // Constant factor: Σ(i=1 to 5) 2*i = 30  
        let sum_expr = ctx.sum(1..=5, |i| {
            let two = ctx.constant(2.0);
            i * two
        })?;
        let result = ctx.eval(&sum_expr, &[])?;
        println!("  Σ(i=1 to 5) 2*i = {result} (expected: 30.0)");
        
        // Complex expression: Σ(i=1 to 3) (i² + 2*i + 1) = (1+2+1) + (4+4+1) + (9+6+1) = 4 + 9 + 16 = 29
        let sum_expr = ctx.sum(1..=3, |i| {
            let two = ctx.constant(2.0);
            let one = ctx.constant(1.0);
            i.clone() * i.clone() + two * i + one
        })?;
        let result = ctx.eval(&sum_expr, &[])?;
        println!("  Σ(i=1 to 3) (i² + 2*i + 1) = {result} (expected: 29.0)");
        
        println!();
    }
    
    // ========================================================================
    // DATA ITERATION SUMMATIONS  
    // ========================================================================
    
    println!("📈 DATA ITERATION SUMMATIONS");
    println!("-----------------------------");
    
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    for (name, config) in &strategies {
        println!("🎯 Strategy: {name}");
        let ctx = UnifiedContext::with_config(config.clone());
        
        // Simple data sum: Σ(x in data) x = 15
        let sum_expr = ctx.sum(data.clone(), |x| x)?;
        let result = ctx.eval(&sum_expr, &[])?;
        println!("  Σ(x in [1,2,3,4,5]) x = {result} (expected: 15.0)");
        
        // Data transformation: Σ(x in data) 2*x = 30
        let sum_expr = ctx.sum(data.clone(), |x| {
            let two = ctx.constant(2.0);
            x * two
        })?;
        let result = ctx.eval(&sum_expr, &[])?;
        println!("  Σ(x in [1,2,3,4,5]) 2*x = {result} (expected: 30.0)");
        
        // Complex data transformation: Σ(x in data) (x² + 1)
        let sum_expr = ctx.sum(data.clone(), |x| {
            let one = ctx.constant(1.0);
            x.clone() * x + one
        })?;
        let result = ctx.eval(&sum_expr, &[])?;
        println!("  Σ(x in [1,2,3,4,5]) (x² + 1) = {result} (expected: 60.0)"); // 1+1 + 4+1 + 9+1 + 16+1 + 25+1 = 2+5+10+17+26 = 60
        
        println!();
    }
    
    // ========================================================================
    // UNIFIED API DEMONSTRATION
    // ========================================================================
    
    println!("🔄 UNIFIED API DEMONSTRATION");
    println!("-----------------------------");
    println!("✅ Same `sum()` method handles both:");
    println!("   • Mathematical ranges (1..=10)");
    println!("   • Data vectors (vec![1.0, 2.0, 3.0])");
    println!("✅ Same closure syntax for both:");
    println!("   • |i| i * ctx.constant(2.0)");
    println!("   • |x| x * ctx.constant(2.0)");
    println!("✅ Strategy-based optimization:");
    println!("   • StaticCodegen: Compile-time closed-form evaluation");
    println!("   • DynamicCodegen: JIT compilation (TODO)");
    println!("   • Interpretation: AST-based evaluation");
    println!("   • Adaptive: Smart strategy selection");
    println!();
    
    // ========================================================================
    // FEATURE PARITY STATUS
    // ========================================================================
    
    println!("🎯 FEATURE PARITY STATUS");
    println!("-------------------------");
    println!("✅ COMPLETE: Basic summation API");
    println!("✅ COMPLETE: Mathematical range summation");
    println!("✅ COMPLETE: Data iteration summation");
    println!("✅ COMPLETE: Strategy-based optimization");
    println!("✅ COMPLETE: Unified `sum()` method");
    println!("✅ COMPLETE: Same closure syntax");
    println!("✅ COMPLETE: Drop-in replacement for DynamicContext");
    println!();
    println!("🚀 NEXT STEPS:");
    println!("   • Implement JIT compilation for DynamicCodegen strategy");
    println!("   • Add symbolic data parameter summation");
    println!("   • Optimize closed-form evaluation for more patterns");
    println!("   • Add summation-specific optimizations (splitting, factoring)");
    
    Ok(())
} 