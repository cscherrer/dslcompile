//! Unified Sum Semantics Test
//!
//! This example demonstrates the unified sum semantics where evaluation strategy
//! is automatically determined based on variable binding:
//! 
//! 1. No unbound vars → Immediate evaluation
//! 2. Has unbound vars → Rewrite rules and symbolic representation
//! 3. Unbound data → Full symbolic deferral

use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🎯 Unified Sum Semantics Test");
    println!("=============================\n");

    let ctx = DynamicContext::new();

    // Test 1: No unbound variables → Immediate evaluation
    println!("📊 Test 1: No Unbound Variables (Immediate Evaluation)");
    
    let expr1 = ctx.sum(1..=10, |i| i * ctx.constant(2.0))?;
    let unbound_vars1 = ctx.find_unbound_variables(&expr1);
    let has_unbound1 = ctx.has_unbound_variables(&expr1);
    
    println!("Expression: Σ(i=1 to 10) 2*i");
    println!("Unbound variables: {:?}", unbound_vars1);
    println!("Has unbound variables: {}", has_unbound1);
    println!("Result: {}", ctx.eval(&expr1, &[]));
    println!("Expected: 110 (2 * 55)\n");

    // Test 2: Has unbound variables → Should apply rewrite rules
    println!("📊 Test 2: Has Unbound Variables (Rewrite Rules)");
    
    let x = ctx.var(); // Unbound variable
    let expr2 = ctx.sum(1..=10, |i| i * ctx.constant(2.0) * x.clone())?;
    let unbound_vars2 = ctx.find_unbound_variables(&expr2);
    let has_unbound2 = ctx.has_unbound_variables(&expr2);
    
    println!("Expression: Σ(i=1 to 10) 2*i*x where x is unbound");
    println!("Unbound variables: {:?}", unbound_vars2);
    println!("Has unbound variables: {}", has_unbound2);
    println!("Pretty print: {}", expr2.pretty_print());
    
    // This should be optimizable to: x * 110
    let result2_x1 = ctx.eval(&expr2, &[1.0]); // x = 1
    let result2_x2 = ctx.eval(&expr2, &[2.0]); // x = 2
    println!("Result with x=1.0: {}", result2_x1);
    println!("Result with x=2.0: {}", result2_x2);
    println!("Expected: 110, 220 (should be linear in x)\n");

    // Test 3: Unbound data → Symbolic representation
    println!("📊 Test 3: Unbound Data (Symbolic Representation)");
    
    let data_expr = ctx.sum_data(|x| x.pow(ctx.constant(2.0)))?;
    let unbound_vars3 = ctx.find_unbound_variables(&data_expr);
    let has_unbound3 = ctx.has_unbound_variables(&data_expr);
    
    println!("Expression: Σ(x² for x in data) where data is unbound");
    println!("Unbound variables: {:?}", unbound_vars3);
    println!("Has unbound variables: {}", has_unbound3);
    println!("Pretty print: {}", data_expr.pretty_print());
    
    // Evaluate with different data arrays
    let result3_data1 = ctx.eval_with_data(&data_expr, &[], &[vec![1.0, 2.0, 3.0]]);
    let result3_data2 = ctx.eval_with_data(&data_expr, &[], &[vec![2.0, 3.0]]);
    println!("Result with data=[1,2,3]: {}", result3_data1);
    println!("Result with data=[2,3]: {}", result3_data2);
    println!("Expected: 14 (1+4+9), 13 (4+9)\n");

    // Test 4: Complex case with both unbound variables and data
    println!("📊 Test 4: Complex Case (Unbound Variables + Data)");
    
    let scale = ctx.var(); // Unbound variable
    let complex_expr = ctx.sum_data(|x| x * scale.clone())?;
    let unbound_vars4 = ctx.find_unbound_variables(&complex_expr);
    let has_unbound4 = ctx.has_unbound_variables(&complex_expr);
    
    println!("Expression: Σ(x*scale for x in data) where both scale and data are unbound");
    println!("Unbound variables: {:?}", unbound_vars4);
    println!("Has unbound variables: {}", has_unbound4);
    println!("Pretty print: {}", complex_expr.pretty_print());
    
    // This should be optimizable to: scale * Σ(data)
    let result4 = ctx.eval_with_data(&complex_expr, &[2.0], &[vec![1.0, 2.0, 3.0]]);
    println!("Result with scale=2.0, data=[1,2,3]: {}", result4);
    println!("Expected: 12 (2 * (1+2+3))\n");

    println!("✅ Unified Sum Semantics Test Complete!");
    println!("Key insight: Evaluation strategy is automatically determined by variable binding");
    
    Ok(())
} 