//! New Iterator-Based Summation API Demo
//!
//! This demo shows the new egglog-based summation system that uses
//! standard Rust iterator syntax but builds AST expressions when
//! working with symbolic variables.

use dslcompile::ast::runtime::expression_builder::{DynamicContext, SymbolicRangeExt};

fn main() {
    println!("🚀 New Iterator-Based Summation API Demo");
    println!("=========================================\n");

    let ctx = DynamicContext::new();

    // ============================================================================
    // DEMO 1: Range summation with symbolic mapping
    // ============================================================================
    println!("📊 Demo 1: Range Summation");
    println!("---------------------------");
    
    // Example: Σ(i=1 to 5) 2*i
    let range_sum = SymbolicRangeExt::map(1..=5, |i| i * 2.0).sum();
    println!("Expression: (1..=5).map(|i| i * 2.0).sum()");
    println!("AST: {:?}", range_sum.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&range_sum));

    // ============================================================================
    // DEMO 2: Data variable summation
    // ============================================================================
    println!("📈 Demo 2: Data Variable Summation");
    println!("-----------------------------------");
    
    // Example: Σ(x in data) ln(x)
    let data_var = ctx.data_var();
    let data_sum = data_var.map(|x| x.ln()).sum();
    println!("Expression: data_var.map(|x| x.ln()).sum()");
    println!("AST: {:?}", data_sum.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&data_sum));

    // ============================================================================
    // DEMO 3: Parametric summation (range with parameters)
    // ============================================================================
    println!("🔧 Demo 3: Parametric Summation");
    println!("--------------------------------");
    
    // Example: Σ(i=1 to n) i * param
    let param = ctx.var();
    let param_sum = SymbolicRangeExt::map(1..=10, |i| i * param.clone()).sum();
    println!("Expression: (1..=10).map(|i| i * param).sum()");
    println!("AST: {:?}", param_sum.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&param_sum));

    // ============================================================================
    // DEMO 4: Complex expressions
    // ============================================================================
    println!("🧮 Demo 4: Complex Expressions");
    println!("-------------------------------");
    
    // Example: Σ(i=1 to 3) (i^2 + sin(i))
    let complex_sum = SymbolicRangeExt::map(1..=3, |i| {
        let i_squared = i.clone().pow(ctx.constant(2.0));
        let sin_i = i.sin();
        i_squared + sin_i
    }).sum();
    println!("Expression: (1..=3).map(|i| i^2 + sin(i)).sum()");
    println!("AST: {:?}", complex_sum.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&complex_sum));

    // ============================================================================
    // DEMO 5: Multiple data variables
    // ============================================================================
    println!("📊 Demo 5: Multiple Data Variables");
    println!("----------------------------------");
    
    // Example: Σ(x in data1) x^2 + Σ(y in data2) y
    let data1 = ctx.data_var();
    let data2 = ctx.data_var();
    
    let sum1 = data1.map(|x| x.clone().pow(ctx.constant(2.0))).sum();
    let sum2 = data2.map(|y| y).sum();
    let combined = sum1 + sum2;
    
    println!("Expression: data1.map(|x| x^2).sum() + data2.map(|y| y).sum()");
    println!("AST: {:?}", combined.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&combined));

    println!("✅ Demo completed! The new iterator API successfully builds AST expressions.");
    println!("🎯 Next steps: Integrate with egglog for optimization and evaluation.");
} 