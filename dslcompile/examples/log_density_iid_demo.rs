//! Log-Density IID Sampling Demo
//!
//! This demo demonstrates:
//! 1. Normal log-density function using DynamicContext ergonomic API
//! 2. IID combinator using summation (not unrolled)
//! 3. Expression complexity analysis
//! 4. Egglog symbolic optimization
//! 5. Code generation and performance benchmarking

use dslcompile::prelude::*;
use dslcompile::ast::ast_repr::Collection;
use dslcompile::backends::{RustCodeGenerator, RustCompiler};
use dslcompile::SymbolicOptimizer;
use frunk::hlist;
use std::time::Instant;

fn main() -> Result<()> {
    println!("📊 Log-Density IID Sampling Demo");
    println!("=================================\n");

    // =======================================================================
    // 1. Define Normal Log-Density Function using ergonomic DynamicContext
    // =======================================================================
    
    println!("1️⃣ Creating Normal Log-Density Function");
    println!("----------------------------------------");
    
    let mut ctx = DynamicContext::<f64>::new();
    
    // Variables: x (observation), mu (mean), sigma (std dev)
    let x = ctx.var();     // Variable(0) - observation
    let mu = ctx.var();    // Variable(1) - mean
    let sigma = ctx.var(); // Variable(2) - standard deviation
    
    // Normal log-density: -0.5 * log(2π) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
    let log_2pi = ctx.constant((2.0 * std::f64::consts::PI).ln()); // ln(2π)
    let half = ctx.constant(0.5);
    let neg_half = ctx.constant(-0.5);
    
    let centered = &x - &mu;                    // (x - mu)
    let standardized = &centered / &sigma;      // (x - mu) / sigma
    let squared = &standardized * &standardized; // ((x - mu) / sigma)^2
    
    // Complete log-density formula using natural syntax
    let log_density = &neg_half * &log_2pi - sigma.clone().ln() + &neg_half * &squared;
    
    println!("✅ Normal log-density: -0.5*ln(2π) - ln(σ) - 0.5*((x-μ)/σ)²");
    println!("   Variables: x={}, μ={}, σ={}", x.var_id(), mu.var_id(), sigma.var_id());
    
    // Test single evaluation
    let single_result = ctx.eval(&log_density, hlist![1.0, 0.0, 1.0]); // N(0,1) at x=1
    println!("   Test: log_density(x=1, μ=0, σ=1) = {:.6}", single_result);
    
    // =======================================================================
    // 2. IID Combinator using Summation (NOT unrolled)
    // =======================================================================
    
    println!("\n2️⃣ Creating IID Combinator using Summation");
    println!("--------------------------------------------");
    
    // Create a new context for the IID combinator
    let mut iid_ctx = DynamicContext::<f64>::new();
    
    // Parameters that are shared across all observations
    let mu_iid = iid_ctx.var();    // Variable(0) - shared mean
    let sigma_iid = iid_ctx.var(); // Variable(1) - shared std dev
    
    // Create the actual IID expression using summation over EXTERNAL data
    // This demonstrates the correct approach: build expression, pass data at runtime
    
    // Create constants for the lambda body
    let log_2pi_const = iid_ctx.constant((2.0 * std::f64::consts::PI).ln());
    let neg_half_const = iid_ctx.constant(-0.5);
    
    // Create IID expression that operates over external data 
    let external_data_placeholder: &[f64] = &[];
    let iid_expr = iid_ctx.sum(external_data_placeholder, |x_data| {
        // x_data represents each element from the external data array
        let centered = &x_data - &mu_iid;                     // (x - μ)
        let standardized = &centered / &sigma_iid;            // (x - μ) / σ
        let squared = &standardized * &standardized;          // ((x - μ) / σ)²
        
        // Complete log-density: -0.5*ln(2π) - ln(σ) - 0.5*((x-μ)/σ)²
        &neg_half_const * &log_2pi_const - sigma_iid.clone().ln() + &neg_half_const * &squared
    });
    
    println!("✅ IID combinator expressions created!");
    println!("   External data expr: Uses Collection::Variable for runtime data");
    println!("   Parameters: μ={}, σ={}", mu_iid.var_id(), sigma_iid.var_id());
    println!("   The external version will expect data to be passed at evaluation time");
    
    // =======================================================================
    // 3. Expression Complexity Analysis
    // =======================================================================
    
    println!("\n3️⃣ Expression Complexity Analysis");
    println!("----------------------------------");
    
    // Convert to AST for analysis
    let log_density_ast = ctx.to_ast(&log_density);
    let external_iid_ast = iid_ctx.to_ast(&iid_expr);
    
    let single_ops = log_density_ast.count_operations();
    let single_vars = count_variables(&log_density_ast);
    let single_depth = compute_depth(&log_density_ast);
    
    let external_ops = external_iid_ast.count_operations();
    let external_vars = count_variables(&external_iid_ast);
    let external_depth = compute_depth(&external_iid_ast);
    let external_sums = external_iid_ast.count_summations();
    
    println!("Single Log-Density Expression:");
    println!("   • Operations: {}", single_ops);
    println!("   • Variables: {}", single_vars);
    println!("   • Depth: {}", single_depth);
    println!("   • Summations: {}", log_density_ast.count_summations());
    println!("   • Structure: {}", summarize_ast_structure(&log_density_ast));
    
    println!("\nExternal IID Expression (for runtime data):");
    println!("   • Operations: {}", external_ops);
    println!("   • Variables: {} (parameters only)", external_vars);
    println!("   • Depth: {}", external_depth);
    println!("   • Summations: {}", external_sums);
    println!("   • Structure: Sum(Map(Lambda, Collection::Variable))");
    println!("   • Data: External (passed at evaluation time)");
    
    // =======================================================================
    // 4. Symbolic Optimization with Egglog
    // =======================================================================
    
    println!("\n4️⃣ Symbolic Optimization");
    println!("-------------------------");
    
    #[cfg(feature = "optimization")]
    {
        let mut optimizer = SymbolicOptimizer::new()?;
        
        println!("🔧 Optimizing expressions...");
        
        // Optimize single log-density
        let optimized_single = optimizer.optimize(&log_density_ast)?;
        let single_reduction = if single_ops > optimized_single.count_operations() {
            format!("{} operations reduced", single_ops - optimized_single.count_operations())
        } else {
            "No reduction".to_string()
        };
        
        // Optimize IID expression  
        let optimized_iid = optimizer.optimize(&external_iid_ast)?;
        let iid_reduction = if external_ops > optimized_iid.count_operations() {
            format!("{} operations reduced", external_ops - optimized_iid.count_operations())
        } else {
            "No reduction".to_string()
        };
        
        println!("   Single log-density:");
        println!("     • Original: {} ops → Optimized: {} ops", single_ops, optimized_single.count_operations());
        println!("     • Result: {}", single_reduction);
        
        println!("   IID expression:");
        println!("     • Original: {} ops → Optimized: {} ops", external_ops, optimized_iid.count_operations());
        println!("     • Result: {}", iid_reduction);
        
        // =======================================================================
        // 5. Code Generation
        // =======================================================================
        
        println!("\n5️⃣ Code Generation");
        println!("------------------");
        
        let codegen = RustCodeGenerator::new();
        let compiler = RustCompiler::new();
        
        // Generate code for optimized expressions
        let single_code = codegen.generate_function(&optimized_single, "single_log_density")?;
        let iid_code = codegen.generate_function(&optimized_iid, "iid_log_density")?;
        
        println!("✅ Generated Rust code for both expressions");
        
        // Compile to native functions
        let single_fn = compiler.compile_and_load(&single_code, "single_log_density")?;
        let iid_fn = compiler.compile_and_load(&iid_code, "iid_log_density")?;
        
        println!("✅ Compiled to native functions");
        
        // =======================================================================
        // 6. Performance Benchmarking
        // =======================================================================
        
        println!("\n6️⃣ Performance Benchmarking");
        println!("----------------------------");
        
        // Test data
        let test_mu = 0.0;
        let test_sigma = 1.0;
        let test_x = 1.0;
        
        // Single evaluation benchmark
        let iterations = 1_000_000;
        
        // Interpreted evaluation
        let start = Instant::now();
        let mut interpreted_sum = 0.0;
        for _ in 0..iterations {
            interpreted_sum += ctx.eval(&log_density, hlist![test_x, test_mu, test_sigma]);
        }
        let interpreted_time = start.elapsed();
        
        // Compiled evaluation
        let start = Instant::now();
        let mut compiled_sum = 0.0;
        for _ in 0..iterations {
            compiled_sum += single_fn.call(vec![test_x, test_mu, test_sigma])?;
        }
        let compiled_time = start.elapsed();
        
        println!("Single Log-Density Performance ({} iterations):", iterations);
        println!("   • Interpreted: {:.2?} ({:.2} ns/eval)", interpreted_time, interpreted_time.as_nanos() as f64 / iterations as f64);
        println!("   • Compiled:    {:.2?} ({:.2} ns/eval)", compiled_time, compiled_time.as_nanos() as f64 / iterations as f64);
        println!("   • Speedup:     {:.2}x", interpreted_time.as_nanos() as f64 / compiled_time.as_nanos() as f64);
        println!("   • Results match: {}", (interpreted_sum - compiled_sum).abs() < 1e-6);
    }
    
    #[cfg(not(feature = "optimization"))]
    {
        println!("⚠️  Optimization features disabled - compile with --features optimization");
    }
    
    println!("\n🎉 Demo completed successfully!");
    println!("   • Natural mathematical syntax: &x + &y, &x * &y");
    println!("   • Ergonomic DynamicContext API");
    println!("   • Zero-cost HList evaluation");
    println!("   • Symbolic optimization and code generation");
    
    Ok(())
}

// =======================================================================
// Helper Functions for Analysis
// =======================================================================

fn count_variables<T>(ast: &ASTRepr<T>) -> usize {
    let mut vars = std::collections::HashSet::new();
    collect_variables(ast, &mut vars);
    vars.len()
}

fn collect_variables<T>(ast: &ASTRepr<T>, vars: &mut std::collections::HashSet<usize>) {
    match ast {
        ASTRepr::Variable(index) => {
            vars.insert(*index);
        }
        ASTRepr::BoundVar(index) => {
            vars.insert(*index);
        }
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        ASTRepr::Let(_, expr, body) => {
            collect_variables(expr, vars);
            collect_variables(body, vars);
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => {
            collect_variables(inner, vars);
        }
        ASTRepr::Sum(collection) => {
            collect_variables_from_collection(collection, vars);
        }
        ASTRepr::Lambda(lambda) => {
            collect_variables(&lambda.body, vars);
        }
        ASTRepr::Constant(_) => {}
    }
}

fn collect_variables_from_collection<T>(collection: &Collection<T>, vars: &mut std::collections::HashSet<usize>) {
    match collection {
        Collection::Empty => {}
        Collection::Singleton(expr) => collect_variables(expr, vars),
        Collection::Range { start, end } => {
            collect_variables(start, vars);
            collect_variables(end, vars);
        }
        Collection::Union { left, right } | Collection::Intersection { left, right } => {
            collect_variables_from_collection(left, vars);
            collect_variables_from_collection(right, vars);
        }
        Collection::Variable(index) => {
            vars.insert(*index);
        }
        Collection::Filter { collection, predicate } => {
            collect_variables_from_collection(collection, vars);
            collect_variables(predicate, vars);
        }
        Collection::Map { lambda, collection } => {
            collect_variables(&lambda.body, vars);
            collect_variables_from_collection(collection, vars);
        }
    }
}

fn compute_depth<T>(ast: &ASTRepr<T>) -> usize {
    match ast {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => 1,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => 1 + compute_depth(left).max(compute_depth(right)),
        ASTRepr::Let(_, expr, body) => 1 + compute_depth(expr).max(compute_depth(body)),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + compute_depth(inner),
        ASTRepr::Sum(_) => 2, // Summation adds depth
        ASTRepr::Lambda(lambda) => 1 + compute_depth(&lambda.body),
    }
}

fn summarize_ast_structure<T>(ast: &ASTRepr<T>) -> String {
    match ast {
        ASTRepr::Constant(_) => "Constant".to_string(),
        ASTRepr::Variable(i) => format!("Variable({})", i),
        ASTRepr::BoundVar(i) => format!("BoundVar({})", i),
        ASTRepr::Add(_, _) => "Add(...)".to_string(),
        ASTRepr::Sub(_, _) => "Sub(...)".to_string(),
        ASTRepr::Mul(_, _) => "Mul(...)".to_string(),
        ASTRepr::Div(_, _) => "Div(...)".to_string(),
        ASTRepr::Pow(_, _) => "Pow(...)".to_string(),
        ASTRepr::Neg(_) => "Neg(...)".to_string(),
        ASTRepr::Ln(_) => "Ln(...)".to_string(),
        ASTRepr::Exp(_) => "Exp(...)".to_string(),
        ASTRepr::Sin(_) => "Sin(...)".to_string(),
        ASTRepr::Cos(_) => "Cos(...)".to_string(),
        ASTRepr::Sqrt(_) => "Sqrt(...)".to_string(),
        ASTRepr::Sum(_) => "Sum(Collection)".to_string(),
        ASTRepr::Lambda(_) => "Lambda(...)".to_string(),
        ASTRepr::Let(_, _, _) => "Let(...)".to_string(),
    }
}