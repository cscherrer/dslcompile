use dslcompile::ast::ASTRepr;
use dslcompile::ast::runtime::expression_builder::{DynamicContext, SummationOptimizer};
use dslcompile::symbolic::symbolic::OptimizationConfig;
use dslcompile::unified_context::UnifiedContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 DEBUGGING UNIFIED SUMMATION vs WORKING SUMMATION");
    println!("===================================================");
    println!();

    // ========================================================================
    // TEST 1: Direct SummationOptimizer (WORKING)
    // ========================================================================

    println!("🎯 TEST 1: Direct SummationOptimizer (should work)");
    println!("--------------------------------------------------");

    let optimizer = SummationOptimizer::new();

    // Test: Σ(i) for i=1..5 = 15
    let var_expr = ASTRepr::Variable(0);
    let result1 = optimizer.optimize_summation(1, 5, var_expr)?;
    println!("  Σ(i=1 to 5) i = {result1} (expected: 15)");

    // Test: Σ(2*i) for i=1..5 = 30
    let two_i_expr = ASTRepr::Mul(
        Box::new(ASTRepr::Constant(2.0)),
        Box::new(ASTRepr::Variable(0)),
    );
    let result2 = optimizer.optimize_summation(1, 5, two_i_expr)?;
    println!("  Σ(i=1 to 5) 2*i = {result2} (expected: 30)");

    println!();

    // ========================================================================
    // TEST 2: Unified Context (BROKEN)
    // ========================================================================

    println!("🎯 TEST 2: Unified Context (currently broken)");
    println!("----------------------------------------------");

    let ctx = UnifiedContext::with_config(OptimizationConfig::zero_overhead());

    // Test: Σ(i) for i=1..5 = 15
    let result3 = ctx.sum(1..=5, |i| i)?;
    let eval_result3 = ctx.eval(&result3, &[])?;
    println!("  Σ(i=1 to 5) i = {eval_result3} (expected: 15)");

    // Test: Σ(2*i) for i=1..5 = 30
    let result4 = ctx.sum(1..=5, |i| {
        let two = ctx.constant(2.0);
        i * two
    })?;
    let eval_result4 = ctx.eval(&result4, &[])?;
    println!("  Σ(i=1 to 5) 2*i = {eval_result4} (expected: 30)");

    println!();

    // ========================================================================
    // TEST 3: Debug AST Conversion
    // ========================================================================

    println!("🎯 TEST 3: Debug AST Conversion");
    println!("--------------------------------");

    // Let's see what AST the unified context is generating
    let debug_expr = ctx.sum(1..=5, |i| i)?;
    println!("  Generated AST: {:?}", debug_expr.ast());

    let debug_expr2 = ctx.sum(1..=5, |i| {
        let two = ctx.constant(2.0);
        i * two
    })?;
    println!("  Generated AST (2*i): {:?}", debug_expr2.ast());

    println!();

    // ========================================================================
    // TEST 4: DynamicContext (WORKING REFERENCE)
    // ========================================================================

    println!("🎯 TEST 4: DynamicContext (working reference)");
    println!("----------------------------------------------");

    let dyn_ctx = DynamicContext::new();

    // Test: Σ(i) for i=1..5 = 15
    let dyn_result1 = dyn_ctx.sum(1..=5, |i| i)?;
    let dyn_eval1 = dyn_ctx.eval(&dyn_result1, &[]);
    println!("  Σ(i=1 to 5) i = {dyn_eval1} (expected: 15)");

    // Test: Σ(2*i) for i=1..5 = 30
    let dyn_result2 = dyn_ctx.sum(1..=5, |i| {
        let two = dyn_ctx.constant(2.0);
        i * two
    })?;
    let dyn_eval2 = dyn_ctx.eval(&dyn_result2, &[]);
    println!("  Σ(i=1 to 5) 2*i = {dyn_eval2} (expected: 30)");

    println!();
    println!("🔍 ANALYSIS:");
    println!("  - Direct SummationOptimizer: Should work perfectly");
    println!("  - Unified Context: Currently broken - need to find the bug");
    println!("  - DynamicContext: Working reference implementation");

    Ok(())
}
