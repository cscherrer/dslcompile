#!/usr/bin/env cargo run --example debug_optimization

//! Debug Demo - Tracing MathCompile Optimization Issues
//!
//! This debug version traces the optimization pipeline to identify
//! where correctness issues are occurring.

use mathcompile::compile_time::{MathExpr, var, constant, eval_ast};
use mathcompile::compile_time::optimized::{ToAst, equality_saturation, generate_direct_code};
use mathcompile::optimize_compile_time;

fn debug_optimization<T: ToAst>(expr: &T, name: &str, var_names: &[&str]) {
    println!("🔍 Debugging: {}", name);
    
    // Step 1: Convert to AST
    let ast = expr.to_ast();
    println!("  Original AST: {:?}", ast);
    
    // Step 2: Apply optimization
    let optimized_ast = equality_saturation(&ast, 10);
    println!("  Optimized AST: {:?}", optimized_ast);
    
    // Step 3: Generate code
    let generated_code = generate_direct_code(&optimized_ast, var_names);
    println!("  Generated code: {}", generated_code);
    
    // Step 4: Test evaluation
    let test_vars = [2.5, 1.5, 0.8];
    let original_result = eval_ast(&ast, &test_vars);
    let optimized_result = eval_ast(&optimized_ast, &test_vars);
    println!("  Original result: {:.10}", original_result);
    println!("  Optimized result: {:.10}", optimized_result);
    println!("  Results match: {}", (original_result - optimized_result).abs() < 1e-10);
    println!();
}

fn main() {
    println!("🐛 MathCompile Debug Session");
    println!("============================");
    println!();
    
    // Debug Test 1: Simple expression
    println!("📊 Debug Test 1: sin(x) + cos(y)");
    println!("----------------------------------");
    
    let expr1 = var::<0>().sin().add(var::<1>().cos());
    debug_optimization(&expr1, "sin(x) + cos(y)", &["x", "y"]);
    
    // Manual verification
    let x = 2.5;
    let y = 1.5;
    let manual_result = (x as f64).sin() + (y as f64).cos();
    let macro_result = optimize_compile_time!(expr1, [x, y]);
    println!("Manual result: {:.10}", manual_result);
    println!("Macro result:  {:.10}", macro_result);
    println!("Match: {}", (manual_result - macro_result).abs() < 1e-10);
    println!();
    
    // Debug Test 2: The problematic expression
    println!("📊 Debug Test 2: ln(exp(x)) + y * 1 + 0 * z");
    println!("---------------------------------------------");
    
    let expr2 = var::<0>().exp().ln()  // ln(exp(x)) -> x
        .add(var::<1>().mul(constant(1.0)))  // y * 1 -> y  
        .add(var::<2>().mul(constant(0.0))); // 0 * z -> 0
    
    debug_optimization(&expr2, "ln(exp(x)) + y * 1 + 0 * z", &["x", "y", "z"]);
    
    // Manual verification
    let z = 0.8;
    let manual_result2 = x + y; // Expected after optimization
    let macro_result2 = optimize_compile_time!(expr2, [x, y, z]);
    println!("Expected (x + y): {:.10}", manual_result2);
    println!("Macro result:     {:.10}", macro_result2);
    println!("Match: {}", (manual_result2 - macro_result2).abs() < 1e-10);
    println!();
    
    // Debug Test 3: Individual components
    println!("📊 Debug Test 3: Breaking down the components");
    println!("----------------------------------------------");
    
    // Test ln(exp(x)) alone
    let ln_exp_x = var::<0>().exp().ln();
    debug_optimization(&ln_exp_x, "ln(exp(x))", &["x"]);
    
    // Test y * 1 alone
    let y_times_1 = var::<1>().mul(constant(1.0));
    debug_optimization(&y_times_1, "y * 1", &["x", "y"]);
    
    // Test 0 * z alone
    let zero_times_z = var::<2>().mul(constant(0.0));
    debug_optimization(&zero_times_z, "0 * z", &["x", "y", "z"]);
    
    // Debug Test 4: Constants investigation
    println!("📊 Debug Test 4: Constant values");
    println!("---------------------------------");
    
    let const_0 = constant(0.0);
    let const_1 = constant(1.0);
    let const_2 = constant(2.0);
    
    println!("constant(0.0) AST: {:?}", const_0.to_ast());
    println!("constant(1.0) AST: {:?}", const_1.to_ast());
    println!("constant(2.0) AST: {:?}", const_2.to_ast());
    
    println!("constant(0.0) eval: {}", eval_ast(&const_0.to_ast(), &[]));
    println!("constant(1.0) eval: {}", eval_ast(&const_1.to_ast(), &[]));
    println!("constant(2.0) eval: {}", eval_ast(&const_2.to_ast(), &[]));
    println!();
    
    println!("🎯 Debug Summary");
    println!("================");
    println!("Look for:");
    println!("1. AST conversion errors (wrong structure)");
    println!("2. Optimization rule bugs (incorrect transformations)");
    println!("3. Constant handling issues (wrong values)");
    println!("4. Evaluation errors (wrong results)");
} 