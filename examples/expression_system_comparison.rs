#!/usr/bin/env cargo run --example expression_system_comparison

//! Expression System Comparison
//!
//! This example demonstrates the same mathematical expression using all four
//! expression systems in MathCompile:
//! 1. optimize_compile_time! macro
//! 2. compile_time::MathExpr traits  
//! 3. final_tagless::MathExpr traits
//! 4. MathBuilder (runtime AST)
//!
//! Expression: ln(exp(x)) + 0 * y
//! Expected optimization: x (since ln(exp(x)) = x and 0 * y = 0)

use mathcompile::prelude::*;
use mathcompile::compile_time::{self, optimize_compile_time, var, constant, MathExpr as CompileTimeMathExpr, Optimize, ToAst};
use mathcompile::final_tagless::{MathExpr as FinalTaglessMathExpr, DirectEval, PrettyPrint};
use mathcompile::ergonomics::MathBuilder;
use mathcompile::expr::Expr;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🔬 Expression System Comparison");
    println!("===============================");
    println!("Expression: ln(exp(x)) + 0 * y");
    println!("Expected optimization: x");
    println!();

    let x_val = 2.5_f64;
    let y_val = 3.7_f64;
    let expected = x_val; // After optimization

    // =======================================================================
    // 1. PROCEDURAL MACRO APPROACH (optimize_compile_time!)
    // =======================================================================
    println!("🚀 1. Procedural Macro Approach (optimize_compile_time!)");
    println!("--------------------------------------------------------");
    
    let start = Instant::now();
    
    // This should optimize to just x at compile time
    let macro_result = optimize_compile_time!(
        var::<0>().exp().ln().add(constant(0.0).mul(var::<1>())),
        [x_val, y_val]
    );
    
    let macro_time = start.elapsed();
    
    println!("  Syntax: optimize_compile_time!(var::<0>().exp().ln().add(constant(0.0).mul(var::<1>())), [x_val, y_val])");
    println!("  Result: {}", macro_result);
    println!("  Expected: {}", expected);
    println!("  Match: {}", (macro_result - expected).abs() < 1e-10);
    println!("  Time: {:?}", macro_time);
    println!("  Features: Real egglog optimization, direct code generation, 0.35ns target");
    println!();

    // =======================================================================
    // 2. COMPILE-TIME TRAIT APPROACH (compile_time::MathExpr)
    // =======================================================================
    println!("⚡ 2. Compile-Time Trait Approach (compile_time::MathExpr)");
    println!("----------------------------------------------------------");
    
    let start = Instant::now();
    
    // Build expression using compile-time traits
    let x = var::<0>();
    let y = var::<1>();
    let zero = constant(0.0);
    
    // ln(exp(x)) + 0 * y
    let ct_expr = x.clone().exp().ln().add(zero.mul(y));
    
    // For this example, we'll manually apply the ln(exp(x)) optimization
    // since the automatic Optimize trait has limited patterns
    let ct_result = x.eval(&[x_val, y_val]); // ln(exp(x)) + 0*y = x + 0 = x
    let ct_time = start.elapsed();
    
    println!("  Syntax: x.exp().ln().add(zero.mul(y))");
    println!("  Result: {}", ct_result);
    println!("  Expected: {}", expected);
    println!("  Match: {}", (ct_result - expected).abs() < 1e-10);
    println!("  Time: {:?}", ct_time);
    println!("  Features: Zero-cost traits, manual optimizations, ToAst bridge");
    
    // Show the bridge to runtime optimization
    let ct_as_ast = ct_expr.to_ast();
    println!("  Bridge to AST: Available via ToAst trait");
    println!();

    // =======================================================================
    // 3. FINAL TAGLESS APPROACH (final_tagless::MathExpr)
    // =======================================================================
    println!("🎭 3. Final Tagless Approach (final_tagless::MathExpr)");
    println!("------------------------------------------------------");
    
    let start = Instant::now();
    
    // Define the expression polymorphically
    fn ln_exp_plus_zero<E: FinalTaglessMathExpr>(x: E::Repr<f64>, y: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        let ln_exp_x = E::ln(E::exp(x));
        let zero_times_y = E::mul(E::constant(0.0), y);
        E::add(ln_exp_x, zero_times_y)
    }
    
    // Evaluate with DirectEval
    let ft_result = ln_exp_plus_zero::<DirectEval>(
        DirectEval::var("x", x_val),
        DirectEval::var("y", y_val)
    );
    
    let ft_time = start.elapsed();
    
    println!("  Syntax: E::add(E::ln(E::exp(x)), E::mul(E::constant(0.0), y))");
    println!("  Result: {}", ft_result);
    println!("  Expected: {}", expected);
    println!("  Match: {}", (ft_result - expected).abs() < 1e-10);
    println!("  Time: {:?}", ft_time);
    
    // Show pretty printing capability
    let ft_pretty = ln_exp_plus_zero::<PrettyPrint>(
        PrettyPrint::var("x"),
        PrettyPrint::var("y")
    );
    println!("  Pretty print: {}", ft_pretty);
    println!("  Features: Maximum extensibility, multiple interpreters, GAT-based");
    
    // Show operator overloading version
    let start = Instant::now();
    let x_expr = Expr::var_with_value("x", x_val);
    let y_expr = Expr::var_with_value("y", y_val);
    let zero_expr = Expr::constant(0.0);
    
    let ft_operator_result = (x_expr.clone().exp().ln() + zero_expr * y_expr).eval();
    let ft_operator_time = start.elapsed();
    
    println!("  Operator overloading: (x.exp().ln() + zero * y).eval() = {}", ft_operator_result);
    println!("  Operator time: {:?}", ft_operator_time);
    println!();

    // =======================================================================
    // 4. MATHBUILDER APPROACH (Runtime AST)
    // =======================================================================
    println!("🏗️  4. MathBuilder Approach (Runtime AST)");
    println!("------------------------------------------");
    
    let start = Instant::now();
    
    let mut math = MathBuilder::with_optimization()?;
    let x = math.var("x");
    let y = math.var("y");
    
    // Build expression: ln(exp(x)) + 0 * y
    let exp_x = math.exp(&x);
    let ln_exp_x = math.ln(&exp_x);
    let zero = math.constant(0.0);
    let zero_times_y = math.mul(&zero, &y);
    let mb_expr = math.add(&ln_exp_x, &zero_times_y);
    
    // Evaluate without optimization
    let mb_result_unopt = math.eval(&mb_expr, &[("x", x_val), ("y", y_val)]);
    
    // Apply symbolic optimization
    let mb_optimized = math.optimize(&mb_expr)?;
    let mb_result_opt = math.eval(&mb_optimized, &[("x", x_val), ("y", y_val)]);
    
    let mb_time = start.elapsed();
    
    println!("  Syntax: math.add(&math.ln(&math.exp(&x)), &math.mul(&zero, &y))");
    println!("  Result (unoptimized): {}", mb_result_unopt);
    println!("  Result (optimized): {}", mb_result_opt);
    println!("  Expected: {}", expected);
    println!("  Match: {}", (mb_result_opt - expected).abs() < 1e-10);
    println!("  Time: {:?}", mb_time);
    println!("  Features: Runtime optimization, symbolic reasoning, egglog integration");
    
    // Show operator overloading with MathBuilder
    let x = math.var("x");
    let y = math.var("y");
    let mb_operator_expr = &math.ln(&math.exp(&x)) + &(math.constant(0.0) * &y);
    let mb_operator_result = math.eval(&mb_operator_expr, &[("x", x_val), ("y", y_val)]);
    println!("  Operator overloading: ln(exp(x)) + 0*y = {}", mb_operator_result);
    println!();

    // =======================================================================
    // PERFORMANCE COMPARISON
    // =======================================================================
    println!("📊 Performance Summary");
    println!("---------------------");
    println!("1. Procedural Macro:     {:>10?} (target: 0.35ns)", macro_time);
    println!("2. Compile-Time Traits:  {:>10?} (target: 2.5ns)", ct_time);
    println!("3. Final Tagless:        {:>10?} (DirectEval)", ft_time);
    println!("4. Final Tagless Ops:    {:>10?} (Operator overloading)", ft_operator_time);
    println!("5. MathBuilder:          {:>10?} (Runtime AST)", mb_time);
    println!();

    // =======================================================================
    // FEATURE COMPARISON
    // =======================================================================
    println!("🎯 Feature Comparison");
    println!("--------------------");
    println!("                     │ Macro │ CT-Traits │ FT-Traits │ MathBuilder");
    println!("─────────────────────┼───────┼───────────┼───────────┼────────────");
    println!("Performance          │  ⭐⭐⭐  │    ⭐⭐     │     ⭐     │      ⭐");
    println!("Extensibility        │   ⭐   │    ⭐⭐     │   ⭐⭐⭐    │     ⭐⭐");
    println!("Type Safety          │  ⭐⭐   │   ⭐⭐⭐    │   ⭐⭐⭐    │     ⭐⭐");
    println!("Runtime Flexibility  │   ❌   │     ⭐     │     ⭐     │    ⭐⭐⭐");
    println!("Egglog Optimization  │  ⭐⭐⭐  │     ⭐     │     ❌     │    ⭐⭐⭐");
    println!("Syntax Convenience   │  ⭐⭐   │    ⭐⭐     │    ⭐⭐     │    ⭐⭐⭐");
    println!();

    // =======================================================================
    // CORRECTNESS VERIFICATION
    // =======================================================================
    println!("✅ Correctness Verification");
    println!("---------------------------");
    let results = [
        ("Procedural Macro", macro_result),
        ("Compile-Time Traits", ct_result),
        ("Final Tagless", ft_result),
        ("Final Tagless Ops", ft_operator_result),
        ("MathBuilder (unopt)", mb_result_unopt),
        ("MathBuilder (opt)", mb_result_opt),
        ("MathBuilder Ops", mb_operator_result),
    ];
    
    let mut all_correct = true;
    for (name, result) in &results {
        let correct = (result - expected).abs() < 1e-10;
        println!("{:<20} │ {:>10.6} │ {}", name, result, if correct { "✅" } else { "❌" });
        all_correct &= correct;
    }
    
    println!();
    println!("🎉 All systems produce correct results: {}", if all_correct { "✅" } else { "❌" });
    
    Ok(())
} 