//! Egglog Data Semantics Test
//!
//! This test demonstrates the fundamental semantic mismatch when converting
//! data iteration to mathematical ranges in the egglog collection system.

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🔍 Egglog Data Semantics Test");
    println!("=============================\n");

    let ctx = DynamicContext::new();
    let param = ctx.var();

    // Test data with specific values
    let data = hlist![10.0, 20.0, 30.0];
    println!("Test data: {:?}", data);
    println!("Expected sum with param=1.0: {} (10*1 + 20*1 + 30*1 = 60)", 10.0 + 20.0 + 30.0);
    println!("Expected sum with param=2.0: {} (10*2 + 20*2 + 30*2 = 120)", 2.0 * (10.0 + 20.0 + 30.0));

    // Test 1: Old range-based system (disabled but we can see what it would do)
    println!("\n📊 Test 1: What the old system would do");
    #[allow(deprecated)]
    let old_expr = ctx.sum_data(|x| x * param.clone())?;
    #[allow(deprecated)]
    let old_result = ctx.eval_with_data(&old_expr, &[1.0], &[data.clone()]);
    println!("Old system result (param=1.0): {}", old_result);

    // Test 2: New egglog system (currently active)
    println!("\n🧪 Test 2: What the egglog system does");
    let new_expr = ctx.sum(data.clone(), |x| x * param.clone())?;
    let new_result = ctx.eval(&new_expr, &[1.0]);
    println!("Egglog system result (param=1.0): {}", new_result);
    println!("Pretty print: {}", new_expr.pretty_print());

    // Test 3: Manual verification of what egglog is actually computing
    println!("\n🔍 Test 3: Manual verification of egglog computation");
    println!("Egglog converts data=[10,20,30] to range 0..=2");
    println!("Then computes: Σ(i=0 to 2) i * param");
    let manual_egglog = 0.0 * 1.0 + 1.0 * 1.0 + 2.0 * 1.0;
    println!("Manual egglog calculation: 0*1 + 1*1 + 2*1 = {}", manual_egglog);

    // Test 4: Show the semantic difference clearly
    println!("\n❌ Semantic Mismatch Analysis:");
    println!("  Data iteration semantics: Σ(f(data_value) for data_value in [10,20,30])");
    println!("  Range conversion semantics: Σ(f(index) for index in 0..=2)");
    println!("  Expected (data values): 10*1 + 20*1 + 30*1 = 60");
    println!("  Actual (indices): 0*1 + 1*1 + 2*1 = 3");
    println!("  Egglog result: {}", new_result);
    println!("  Old system result: {}", old_result);

    if (old_result - 60.0).abs() < 1e-10 {
        println!("  ✅ Old system: CORRECT semantics");
    } else {
        println!("  ❌ Old system: INCORRECT");
    }

    if (new_result - 60.0).abs() < 1e-10 {
        println!("  ✅ Egglog system: CORRECT semantics");
    } else {
        println!("  ❌ Egglog system: INCORRECT semantics (converts data to indices)");
    }

    println!("\n🎯 Conclusion:");
    println!("The egglog collection system has a fundamental semantic mismatch:");
    println!("- It converts data iteration to mathematical range iteration");
    println!("- This changes the meaning from 'iterate over data values' to 'iterate over indices'");
    println!("- This breaks the unified API promise that both approaches should be equivalent");

    Ok(())
} 