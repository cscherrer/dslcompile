#!/usr/bin/env cargo run --example expression_system_comparison

//! Expression System Comparison
//!
//! This example demonstrates how `DSLCompile` is designed for high-performance mathematical computing
//! by comparing different approaches to the same computation: ln(exp(x)) + 0 * y
//!
//! 1. Final Tagless `DirectEval` - immediate evaluation
//! 2. Final Tagless `PrettyPrint` - string representation
//! 3. Procedural Macro - compile-time optimization
//! 4. `ASTEval` - traditional abstract syntax tree approach
//! 5. Compile-time traits - zero-cost abstractions

use dslcompile::compile_time::optimized::ToAst;
use dslcompile::compile_time::{MathExpr as CompileTimeMathExpr, constant, var};
use dslcompile::final_tagless::{ASTEval, ASTMathExpr, DirectEval, MathExpr, PrettyPrint};
use dslcompile::prelude::*;
use dslcompile_macros::optimize_compile_time;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 DSLCompile Expression System Comparison");
    println!("===========================================");
    println!("Expression: ln(exp(x)) + 0 * y");
    println!("Expected result: x (after optimization)");
    println!();

    let x_val = 2.5;
    let y_val = 3.7;

    // Test function for final tagless systems
    fn ln_exp_plus_zero<E: MathExpr>(x: E::Repr<f64>, y: E::Repr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        let ln_exp_x = E::ln(E::exp(x));
        let zero_times_y = E::mul(E::constant(0.0), y);
        E::add(ln_exp_x, zero_times_y)
    }

    println!("=== 1. Final Tagless DirectEval (Immediate Evaluation) ===");
    let start = Instant::now();

    // DirectEval needs immediate values since it evaluates directly
    let direct_result = ln_exp_plus_zero::<DirectEval>(
        DirectEval::var_with_value(0, x_val),
        DirectEval::var_with_value(1, y_val),
    );
    let direct_time = start.elapsed();

    println!("Result: {direct_result}");
    println!("Time: {direct_time:?}");
    println!("Notes: Immediate evaluation, no intermediate representation");
    println!();

    println!("=== 2. Final Tagless PrettyPrint (String Representation) ===");
    let start = Instant::now();

    let pretty_result = ln_exp_plus_zero::<PrettyPrint>(PrettyPrint::var(0), PrettyPrint::var(1));
    let pretty_time = start.elapsed();

    println!("Expression: {pretty_result}");
    println!("Time: {pretty_time:?}");
    println!("Notes: Generates human-readable mathematical expressions");
    println!();

    println!("=== 3. Procedural Macro (Compile-time Optimization) ===");
    let start = Instant::now();

    let macro_result = optimize_compile_time!(
        var::<0>().exp().ln().add(constant(0.0).mul(var::<1>())),
        [x_val, y_val]
    );
    let macro_time = start.elapsed();

    println!("Result: {macro_result}");
    println!("Time: {macro_time:?}");
    println!("Notes: Compile-time optimization using egglog, direct code generation");
    println!();

    println!("=== 4. ASTEval (Abstract Syntax Tree) ===");
    let start = Instant::now();

    // Build AST using ASTEval interpreter
    let ast_expr = <ASTEval as ASTMathExpr>::add(
        <ASTEval as ASTMathExpr>::ln(<ASTEval as ASTMathExpr>::exp(
            <ASTEval as ASTMathExpr>::var(0),
        )),
        <ASTEval as ASTMathExpr>::mul(
            <ASTEval as ASTMathExpr>::constant(0.0),
            <ASTEval as ASTMathExpr>::var(1),
        ),
    );

    let ast_result = ast_expr.eval_with_vars(&[x_val, y_val]);
    let ast_time = start.elapsed();

    println!("Result: {ast_result}");
    println!("Time: {ast_time:?}");
    println!("Notes: Traditional AST approach with runtime evaluation");
    println!("AST structure: {ast_expr:?}");
    println!();

    println!("=== 5. Compile-time Traits (Zero-cost Abstractions) ===");
    let start = Instant::now();

    // Build expression using compile-time traits
    let x = var::<0>();
    let y = var::<1>();
    let zero = constant(0.0);
    let ct_expr = x.clone().exp().ln().add(zero.mul(y));

    let ct_result = ct_expr.eval(&[x_val, y_val]);
    let ct_time = start.elapsed();

    println!("Result: {ct_result}");
    println!("Time: {ct_time:?}");
    println!("Notes: Zero-cost abstractions, compile-time optimization potential");

    // Show bridge to egglog optimization
    let ct_as_ast = ct_expr.to_ast();
    println!("Can convert to AST: {ct_as_ast:?}");
    println!();

    println!("=== Performance Summary ===");
    let results = [
        ("DirectEval", direct_result, direct_time),
        ("ASTEval", ast_result, ast_time),
        ("Procedural Macro", macro_result, macro_time),
        ("Compile-time Traits", ct_result, ct_time),
    ];

    println!("{:<20} │ {:>12} │ {:>12}", "System", "Result", "Time (ns)");
    println!("────────────────────┼──────────────┼─────────────");

    for (name, result, time) in &results {
        let time_ns = time.as_nanos();
        let correct = (result - x_val).abs() < 1e-10;
        println!(
            "{:<20} │ {:>12.6} │ {:>12} {}",
            name,
            result,
            time_ns,
            if correct { "✅" } else { "❌" }
        );
    }
    println!();

    println!("=== System Characteristics ===");
    println!();

    println!("🏃 DirectEval:");
    println!("  • Immediate evaluation, no IR");
    println!("  • Zero allocation overhead");
    println!("  • No optimization potential");
    println!("  • Best for simple computations");
    println!();

    println!("🌳 ASTEval:");
    println!("  • Traditional AST representation");
    println!("  • Runtime interpretation");
    println!("  • Supports optimization via egglog");
    println!("  • Good for complex expressions");
    println!();

    println!("🚀 Procedural Macro:");
    println!("  • Compile-time egglog optimization");
    println!("  • Direct Rust code generation");
    println!("  • Zero runtime overhead");
    println!("  • Best for performance-critical code");
    println!();

    println!("⚡ Compile-time Traits:");
    println!("  • Zero-cost abstractions");
    println!("  • Compile-time expression building");
    println!("  • Bridge to runtime optimization");
    println!("  • Most flexible approach");
    println!();

    println!("=== Use Case Recommendations ===");
    println!();
    println!("Choose DirectEval when:");
    println!("  • Simple mathematical computations");
    println!("  • No need for expression manipulation");
    println!("  • Minimum overhead required");
    println!();
    println!("Choose ASTEval when:");
    println!("  • Complex expressions need optimization");
    println!("  • Runtime expression building");
    println!("  • Symbolic manipulation required");
    println!();
    println!("Choose Procedural Macro when:");
    println!("  • Performance is critical");
    println!("  • Expressions known at compile time");
    println!("  • Maximum optimization needed");
    println!();
    println!("Choose Compile-time Traits when:");
    println!("  • Building expression libraries");
    println!("  • Need both performance and flexibility");
    println!("  • Want zero-cost abstractions");

    Ok(())
}
