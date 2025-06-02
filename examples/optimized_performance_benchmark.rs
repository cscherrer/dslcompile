#!/usr/bin/env cargo run --example optimized_performance_benchmark

//! Performance Benchmark
//!
//! This benchmark measures the evaluation performance of different expression systems
//! for comparison purposes.
//!
//! Expression: ln(exp(x)) + 0 * y
//! Expected optimization: x (since ln(exp(x)) = x and 0 * y = 0)
//!
//! What we measure:
//! 1. Procedural Macro: Compile-time egglog optimization → direct Rust code
//! 2. Compile-Time Traits: Manual optimization (ln(exp(x)) → x, 0*y → 0)
//! 3. Final Tagless + Egglog: AST → egglog optimization → optimized evaluation
//! 4. MathBuilder + Egglog: Runtime egglog optimization → optimized evaluation
//! 5. Manual Rust: Just `x` (the baseline)

use mathcompile::prelude::*;
use mathcompile::compile_time::{var, constant, MathExpr as CompileTimeMathExpr, ToAst};
use mathcompile::final_tagless::{ASTEval, ASTMathExpr, DirectEval, MathExpr as FinalTaglessMathExpr};
use mathcompile::ergonomics::MathBuilder;
use mathcompile::symbolic::native_egglog::optimize_with_native_egglog;
use mathcompile_macros::optimize_compile_time;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Optimized Performance Benchmark");
    println!("===================================");
    println!("Expression: ln(exp(x)) + 0 * y");
    println!("Expected optimization: x");
    println!();

    let x_val = 2.5_f64;
    let y_val = 3.7_f64;
    let expected = x_val; // After optimization
    let iterations = 1_000_000;

    println!("Benchmark parameters:");
    println!("  x = {}, y = {}", x_val, y_val);
    println!("  iterations = {}", iterations);
    println!("  expected result = {}", expected);
    println!();

    // =======================================================================
    // 1. MANUAL RUST (THEORETICAL OPTIMUM)
    // =======================================================================
    println!("🎯 1. Manual Rust (Theoretical Optimum)");
    println!("---------------------------------------");
    
    let start = Instant::now();
    let mut manual_sum = 0.0;
    for _ in 0..iterations {
        manual_sum += x_val; // The optimal result: just x
    }
    let manual_time = start.elapsed();
    let manual_ns_per_eval = manual_time.as_nanos() as f64 / iterations as f64;
    
    println!("  Code: x_val");
    println!("  Result: {}", manual_sum / iterations as f64);
    println!("  Time: {:?} ({:.2} ns/eval)", manual_time, manual_ns_per_eval);
    println!("  Features: Theoretical optimum, hand-optimized");
    println!();

    // =======================================================================
    // 2. PROCEDURAL MACRO (COMPILE-TIME EGGLOG)
    // =======================================================================
    println!("🚀 2. Procedural Macro (Compile-Time Egglog)");
    println!("---------------------------------------------");
    
    let start = Instant::now();
    let mut macro_sum = 0.0;
    for _ in 0..iterations {
        macro_sum += optimize_compile_time!(
            var::<0>().exp().ln().add(constant(0.0).mul(var::<1>())),
            [x_val, y_val]
        );
    }
    let macro_time = start.elapsed();
    let macro_ns_per_eval = macro_time.as_nanos() as f64 / iterations as f64;
    
    println!("  Code: optimize_compile_time!(var::<0>().exp().ln().add(constant(0.0).mul(var::<1>())), [x_val, y_val])");
    println!("  Result: {}", macro_sum / iterations as f64);
    println!("  Time: {:?} ({:.2} ns/eval)", macro_time, macro_ns_per_eval);
    println!("  Features: Compile-time egglog, direct code generation");
    println!();

    // =======================================================================
    // 3. COMPILE-TIME TRAITS (MANUAL OPTIMIZATION)
    // =======================================================================
    println!("⚡ 3. Compile-Time Traits (Manual Optimization)");
    println!("-----------------------------------------------");
    
    // Build the expression
    let x = var::<0>();
    let y = var::<1>();
    let zero = constant(0.0);
    let ct_expr = x.clone().exp().ln().add(zero.mul(y));
    
    // Apply manual optimization: ln(exp(x)) + 0*y → x + 0 → x
    let start = Instant::now();
    let mut ct_sum = 0.0;
    for _ in 0..iterations {
        ct_sum += x.eval(&[x_val, y_val]); // Manually optimized to just x
    }
    let ct_time = start.elapsed();
    let ct_ns_per_eval = ct_time.as_nanos() as f64 / iterations as f64;
    
    println!("  Code: x.eval(&[x_val, y_val]) // manually optimized from ln(exp(x)) + 0*y");
    println!("  Result: {}", ct_sum / iterations as f64);
    println!("  Time: {:?} ({:.2} ns/eval)", ct_time, ct_ns_per_eval);
    println!("  Features: Zero-cost traits, manual optimization");
    
    // Show the bridge to egglog optimization
    let ct_as_ast = ct_expr.to_ast();
    println!("  Bridge to egglog: Available via ToAst trait");
    println!();

    // =======================================================================
    // 4. COMPILE-TIME TRAITS + EGGLOG (RUNTIME OPTIMIZATION)
    // =======================================================================
    println!("⚡ 4. Compile-Time Traits + Egglog (Runtime Optimization)");
    println!("---------------------------------------------------------");
    
    // Convert to AST and optimize with egglog
    let ct_ast = ct_expr.to_ast();
    let ct_optimized = optimize_with_native_egglog(&ct_ast)?;
    
    let start = Instant::now();
    let mut ct_egglog_sum = 0.0;
    for _ in 0..iterations {
        ct_egglog_sum += ct_optimized.eval_with_vars(&[x_val, y_val]);
    }
    let ct_egglog_time = start.elapsed();
    let ct_egglog_ns_per_eval = ct_egglog_time.as_nanos() as f64 / iterations as f64;
    
    println!("  Code: ct_expr.to_ast() → egglog → optimized.eval_with_vars(&[x_val, y_val])");
    println!("  Result: {}", ct_egglog_sum / iterations as f64);
    println!("  Time: {:?} ({:.2} ns/eval)", ct_egglog_time, ct_egglog_ns_per_eval);
    println!("  Features: Compile-time traits + runtime egglog optimization");
    println!("  Optimized AST: {:?}", ct_optimized);
    println!();

    // =======================================================================
    // 5. FINAL TAGLESS + EGGLOG (RUNTIME OPTIMIZATION)
    // =======================================================================
    println!("🎭 5. Final Tagless + Egglog (Runtime Optimization)");
    println!("---------------------------------------------------");
    
    // Build expression using ASTEval interpreter
    let ft_ast = <ASTEval as ASTMathExpr>::add(
        <ASTEval as ASTMathExpr>::ln(<ASTEval as ASTMathExpr>::exp(<ASTEval as ASTMathExpr>::var(0))),
        <ASTEval as ASTMathExpr>::mul(<ASTEval as ASTMathExpr>::constant(0.0), <ASTEval as ASTMathExpr>::var(1))
    );
    
    // Optimize with egglog
    let ft_optimized = optimize_with_native_egglog(&ft_ast)?;
    
    let start = Instant::now();
    let mut ft_sum = 0.0;
    for _ in 0..iterations {
        ft_sum += ft_optimized.eval_with_vars(&[x_val, y_val]);
    }
    let ft_time = start.elapsed();
    let ft_ns_per_eval = ft_time.as_nanos() as f64 / iterations as f64;
    
    println!("  Code: ASTMathExpr::add(...) → egglog → optimized.eval_with_vars(&[x_val, y_val])");
    println!("  Result: {}", ft_sum / iterations as f64);
    println!("  Time: {:?} ({:.2} ns/eval)", ft_time, ft_ns_per_eval);
    println!("  Features: Final tagless + runtime egglog optimization");
    println!("  Optimized AST: {:?}", ft_optimized);
    println!();

    // =======================================================================
    // 6. FINAL TAGLESS DIRECTEVAL (NO OPTIMIZATION)
    // =======================================================================
    println!("🎭 6. Final Tagless DirectEval (No Optimization)");
    println!("------------------------------------------------");
    
    // Define the expression polymorphically
    fn ln_exp_plus_zero<E: FinalTaglessMathExpr>(x: E::Repr<f64>, y: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        let ln_exp_x = E::ln(E::exp(x));
        let zero_times_y = E::mul(E::constant(0.0), y);
        E::add(ln_exp_x, zero_times_y)
    }
    
    let start = Instant::now();
    let mut ft_direct_sum = 0.0;
    for _ in 0..iterations {
        ft_direct_sum += ln_exp_plus_zero::<DirectEval>(
            DirectEval::var("x", x_val),
            DirectEval::var("y", y_val)
        );
    }
    let ft_direct_time = start.elapsed();
    let ft_direct_ns_per_eval = ft_direct_time.as_nanos() as f64 / iterations as f64;
    
    println!("  Code: ln_exp_plus_zero::<DirectEval>(DirectEval::var(\"x\", x_val), DirectEval::var(\"y\", y_val))");
    println!("  Result: {}", ft_direct_sum / iterations as f64);
    println!("  Time: {:?} ({:.2} ns/eval)", ft_direct_time, ft_direct_ns_per_eval);
    println!("  Features: Direct evaluation, no optimization, computes ln(exp(x)) + 0*y literally");
    println!();

    // =======================================================================
    // 7. MATHBUILDER + EGGLOG (RUNTIME OPTIMIZATION)
    // =======================================================================
    println!("🏗️  7. MathBuilder + Egglog (Runtime Optimization)");
    println!("--------------------------------------------------");
    
    let mut math = MathBuilder::with_optimization()?;
    let x = math.var("x");
    let y = math.var("y");
    
    // Build expression: ln(exp(x)) + 0 * y
    let exp_x = math.exp(&x);
    let ln_exp_x = math.ln(&exp_x);
    let zero = math.constant(0.0);
    let zero_times_y = math.mul(&zero, &y);
    let mb_expr = math.add(&ln_exp_x, &zero_times_y);
    
    // Apply symbolic optimization
    let mb_optimized = math.optimize(&mb_expr)?;
    
    let start = Instant::now();
    let mut mb_sum = 0.0;
    for _ in 0..iterations {
        mb_sum += math.eval(&mb_optimized, &[("x", x_val), ("y", y_val)]);
    }
    let mb_time = start.elapsed();
    let mb_ns_per_eval = mb_time.as_nanos() as f64 / iterations as f64;
    
    println!("  Code: math.optimize(&expr) → math.eval(&optimized, &[(\"x\", x_val), (\"y\", y_val)])");
    println!("  Result: {}", mb_sum / iterations as f64);
    println!("  Time: {:?} ({:.2} ns/eval)", mb_time, mb_ns_per_eval);
    println!("  Features: Runtime optimization, symbolic reasoning, egglog integration");
    println!();

    // =======================================================================
    // PERFORMANCE COMPARISON
    // =======================================================================
    println!("📊 Optimized Performance Comparison");
    println!("===================================");
    
    let results = [
        ("Manual Rust", manual_ns_per_eval, manual_sum / iterations as f64),
        ("Procedural Macro", macro_ns_per_eval, macro_sum / iterations as f64),
        ("Compile-Time Traits", ct_ns_per_eval, ct_sum / iterations as f64),
        ("CT-Traits + Egglog", ct_egglog_ns_per_eval, ct_egglog_sum / iterations as f64),
        ("Final Tagless + Egglog", ft_ns_per_eval, ft_sum / iterations as f64),
        ("Final Tagless Direct", ft_direct_ns_per_eval, ft_direct_sum / iterations as f64),
        ("MathBuilder + Egglog", mb_ns_per_eval, mb_sum / iterations as f64),
    ];
    
    println!("{:<25} │ {:>12} │ {:>12} │ {:>8}", "System", "ns/eval", "Result", "Speedup");
    println!("─────────────────────────┼──────────────┼──────────────┼─────────");
    
    for (name, ns_per_eval, result) in &results {
        let speedup = ft_direct_ns_per_eval / ns_per_eval;
        let correct = (result - expected).abs() < 1e-10;
        println!("{:<25} │ {:>12.2} │ {:>12.6} │ {:>7.1}x {}", 
                 name, ns_per_eval, result, speedup, if correct { "✅" } else { "❌" });
    }
    
    println!();
    println!("🎯 Key Insights:");
    println!("  • Manual Rust is the theoretical optimum");
    println!("  • Speedup is relative to unoptimized Final Tagless DirectEval");
    println!("  • All optimized systems should produce the same result: {}", expected);
    println!();

    // =======================================================================
    // FAT CUTTING ANALYSIS
    // =======================================================================
    println!("✂️  Fat Cutting Analysis");
    println!("========================");
    
    println!("Based on optimized performance:");
    println!();
    
    // Find the fastest optimized systems
    let mut optimized_systems = vec![
        ("Manual Rust", manual_ns_per_eval),
        ("Procedural Macro", macro_ns_per_eval),
        ("Compile-Time Traits", ct_ns_per_eval),
        ("CT-Traits + Egglog", ct_egglog_ns_per_eval),
        ("Final Tagless + Egglog", ft_ns_per_eval),
        ("MathBuilder + Egglog", mb_ns_per_eval),
    ];
    
    optimized_systems.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    println!("🚀 Performance Ranking (fastest to slowest):");
    for (i, (name, ns_per_eval)) in optimized_systems.iter().enumerate() {
        let rank = i + 1;
        let emoji = match rank {
            1 => "🥇",
            2 => "🥈", 
            3 => "🥉",
            _ => "  ",
        };
        println!("  {} {}. {:<25} ({:.2} ns/eval)", emoji, rank, name, ns_per_eval);
    }
    
    println!();
    println!("🎯 Recommendations:");
    println!("  • Keep the top 3 performers for different use cases");
    println!("  • Consider removing systems that are >10x slower than the fastest");
    println!("  • Unoptimized Final Tagless DirectEval shows the cost of no optimization");
    
    Ok(())
} 