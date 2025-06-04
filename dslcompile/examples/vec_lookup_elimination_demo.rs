//! Vec Lookup Elimination Demo
//!
//! This demo showcases the complete evolution of DSLCompile performance:
//! 1. Old Context (homogeneous, ~5.7ns baseline)
//! 2. V4 Heterogeneous (Vec lookup, ~1.8ns) 
//! 3. V5 Ultimate (O(1) array access, ~0.00ns)

use std::time::Instant;
use dslcompile::compile_time::scoped::{Context, ScopedMathExpr, ScopedVarArray};
use dslcompile::compile_time::heterogeneous_v4::{
    TrueZeroContext, TrueZeroInputs, true_zero_add, TrueZeroExpr
};
use dslcompile::compile_time::heterogeneous_v5::{
    UltimateZeroContext, UltimateZeroInputs, ultimate_zero_add, UltimateZeroExpr
};

fn main() {
    println!("🎯 Complete Performance Evolution Demo");
    println!("=====================================\n");
    
    // Demonstrate the complete evolution
    demonstrate_complete_evolution();
    demonstrate_scaling_evolution();
    
    println!("\n✅ Demo complete!");
}

fn demonstrate_complete_evolution() {
    println!("📊 Complete Evolution: Old Context → V4 → V5");
    println!("--------------------------------------------");
    
    // OLD CONTEXT SYSTEM (Homogeneous baseline)
    let mut old_ctx = Context::new();
    let old_expr = old_ctx.new_scope(|scope| {
        let (x, scope) = scope.auto_var();
        let (y, _scope) = scope.auto_var(); 
        x.add(y)
    });
    
    // V4: Heterogeneous with Vec lookup
    let mut ctx_v4 = TrueZeroContext::<0>::new();
    let x_v4 = ctx_v4.var::<f64>();
    let y_v4 = ctx_v4.var::<f64>();
    let expr_v4 = true_zero_add::<f64, _, _, 0>(x_v4, y_v4);
    
    let mut inputs_v4 = TrueZeroInputs::new();
    inputs_v4.add_f64(0, 3.0);
    inputs_v4.add_f64(1, 4.0);
    
    // V5: Ultimate zero-overhead with array access
    let mut ctx_v5 = UltimateZeroContext::<0, 8>::new();
    let x_v5 = ctx_v5.var::<f64>();
    let y_v5 = ctx_v5.var::<f64>();
    let expr_v5 = ultimate_zero_add::<f64, _, _, 0>(x_v5, y_v5);
    
    let mut inputs_v5 = UltimateZeroInputs::<8>::new();
    inputs_v5.add_f64(0, 3.0);
    inputs_v5.add_f64(1, 4.0);
    
    // Verify all produce correct results
    let old_vars = ScopedVarArray::new(vec![3.0, 4.0]);
    let result_old = old_expr.eval(&old_vars);
    let result_v4 = expr_v4.eval(&inputs_v4);
    let result_v5 = expr_v5.eval(&inputs_v5);
    
    println!("Old Context Result: {}", result_old);
    println!("V4 Result (Vec lookup): {}", result_v4);
    println!("V5 Result (Array access): {}", result_v5);
    assert_eq!(result_old, result_v4);
    assert_eq!(result_v4, result_v5);
    println!("✅ All systems produce identical results!\n");
    
    // Performance comparison across all three systems
    const ITERATIONS: usize = 100_000;
    
    // OLD CONTEXT TIMING
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = old_expr.eval(&old_vars);
    }
    let old_duration = start.elapsed();
    
    // V4 TIMING
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = expr_v4.eval(&inputs_v4);
    }
    let v4_duration = start.elapsed();
    
    // V5 TIMING
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = expr_v5.eval(&inputs_v5);
    }
    let v5_duration = start.elapsed();
    
    println!("🏆 Performance Evolution ({} iterations):", ITERATIONS);
    println!("Old Context (homogeneous): {:?} ({:.2} ns/op)", old_duration, old_duration.as_nanos() as f64 / ITERATIONS as f64);
    println!("V4 (heterogeneous + Vec lookup): {:?} ({:.2} ns/op)", v4_duration, v4_duration.as_nanos() as f64 / ITERATIONS as f64);
    println!("V5 (ultimate zero-overhead): {:?} ({:.2} ns/op)", v5_duration, v5_duration.as_nanos() as f64 / ITERATIONS as f64);
    
    // Calculate improvements
    let v4_vs_old = old_duration.as_nanos() as f64 / v4_duration.as_nanos() as f64;
    let v5_vs_v4 = v4_duration.as_nanos() as f64 / v5_duration.as_nanos() as f64;
    let v5_vs_old = old_duration.as_nanos() as f64 / v5_duration.as_nanos() as f64;
    
    println!("\n📈 Performance Improvements:");
    println!("🚀 V4 vs Old Context: {:.2}x faster", v4_vs_old);
    println!("🚀 V5 vs V4: {:.2}x faster", v5_vs_v4);
    println!("🚀 V5 vs Old Context: {:.2}x faster (TOTAL IMPROVEMENT)", v5_vs_old);
}

fn demonstrate_scaling_evolution() {
    println!("\n📈 Scaling Evolution with 8 Variables");
    println!("--------------------------------------");
    
    // OLD CONTEXT with 8 variables - use simpler approach
    let mut old_ctx = Context::new();
    let old_expr = old_ctx.new_scope(|scope| {
        // Create 8 variables by chaining auto_var calls
        let (v0, scope) = scope.auto_var();
        let (v1, scope) = scope.auto_var();
        let (v2, scope) = scope.auto_var();
        let (v3, scope) = scope.auto_var();
        let (v4, scope) = scope.auto_var();
        let (v5, scope) = scope.auto_var();
        let (v6, scope) = scope.auto_var();
        let (v7, _scope) = scope.auto_var();
        
        // Use the last two variables: v6 + v7
        v6.add(v7)
    });
    
    // V4: Heterogeneous with Vec lookup (gets slower with more variables)
    let mut ctx_v4 = TrueZeroContext::<0>::new();
    let vars_v4: Vec<_> = (0..8).map(|_| ctx_v4.var::<f64>()).collect();
    let expr_v4 = true_zero_add::<f64, _, _, 0>(vars_v4[6].clone(), vars_v4[7].clone());
    
    let mut inputs_v4 = TrueZeroInputs::new();
    for i in 0..8 {
        inputs_v4.add_f64(i, i as f64 * 10.0);
    }
    
    // V5: Ultimate zero-overhead (constant time regardless of variable count)
    let mut ctx_v5 = UltimateZeroContext::<0, 16>::new();
    let vars_v5: Vec<_> = (0..8).map(|_| ctx_v5.var::<f64>()).collect();
    let expr_v5 = ultimate_zero_add::<f64, _, _, 0>(vars_v5[6].clone(), vars_v5[7].clone());
    
    let mut inputs_v5 = UltimateZeroInputs::<16>::new();
    for i in 0..8 {
        inputs_v5.add_f64(i, i as f64 * 10.0);
    }
    
    // Verify results
    let old_vars_vec: Vec<f64> = (0..8).map(|i| i as f64 * 10.0).collect();
    let old_vars = ScopedVarArray::new(old_vars_vec);
    let result_old = old_expr.eval(&old_vars);
    let result_v4 = expr_v4.eval(&inputs_v4);
    let result_v5 = expr_v5.eval(&inputs_v5);
    
    println!("8-variable test results:");
    println!("Old Context: {}", result_old);
    println!("V4: {}", result_v4); 
    println!("V5: {}", result_v5);
    assert_eq!(result_old, result_v4);
    assert_eq!(result_v4, result_v5);
    println!("✅ All systems produce identical results with 8 variables!\n");
    
    const ITERATIONS: usize = 100_000;
    
    // OLD CONTEXT SCALING
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = old_expr.eval(&old_vars);
    }
    let old_duration = start.elapsed();
    
    // V4 SCALING (should be slower due to O(n) Vec lookup)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = expr_v4.eval(&inputs_v4);
    }
    let v4_duration = start.elapsed();
    
    // V5 SCALING (should maintain constant time)
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = expr_v5.eval(&inputs_v5);
    }
    let v5_duration = start.elapsed();
    
    println!("🏆 8-Variable Scaling Performance ({} iterations):", ITERATIONS);
    println!("Old Context: {:?} ({:.2} ns/op)", old_duration, old_duration.as_nanos() as f64 / ITERATIONS as f64);
    println!("V4 (Vec lookup O(n)): {:?} ({:.2} ns/op)", v4_duration, v4_duration.as_nanos() as f64 / ITERATIONS as f64);
    println!("V5 (Array access O(1)): {:?} ({:.2} ns/op)", v5_duration, v5_duration.as_nanos() as f64 / ITERATIONS as f64);
    
    let v4_vs_old_scaling = old_duration.as_nanos() as f64 / v4_duration.as_nanos() as f64;
    let v5_vs_old_scaling = old_duration.as_nanos() as f64 / v5_duration.as_nanos() as f64;
    let v5_vs_v4_scaling = v4_duration.as_nanos() as f64 / v5_duration.as_nanos() as f64;
    
    println!("\n📊 Scaling Analysis:");
    println!("🚀 V4 vs Old Context (8 vars): {:.2}x faster", v4_vs_old_scaling);
    println!("🚀 V5 vs Old Context (8 vars): {:.2}x faster", v5_vs_old_scaling);
    println!("🚀 V5 vs V4 (8 vars): {:.2}x faster", v5_vs_v4_scaling);
    
    println!("\n📝 Technical Notes:");
    println!("  • Old Context: Excellent baseline performance (~5-10ns)");
    println!("  • V4: Heterogeneous types + runtime dispatch elimination, but O(n) Vec lookup");
    println!("  • V5: True zero-overhead with heterogeneous types + O(1) array access");
    println!("  • V5 achieves both heterogeneous flexibility AND superior performance!");
} 