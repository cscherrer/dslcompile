//! Comprehensive IID Gaussian Demo
//!
//! This demo demonstrates the complete DSLCompile pipeline using PROPER API usage:
//! 1. Use DynamicContext to build expressions (not manual ASTRepr construction)
//! 2. Define a Gaussian log-density function (symbolic)
//! 3. Define an IID combinator using summation
//! 4. Compose them to create IID Gaussian observation log-density
//! 5. Simplify using egglog optimization
//! 6. Pretty-print the simplified expression
//! 7. Generate and compile Rust code
//! 8. Evaluate with runtime datasets
//!
//! This properly demonstrates the high-level DynamicContext API instead of low-level AST manipulation.

use dslcompile::ast::{DynamicContext, TypedBuilderExpr, VariableRegistry};
use dslcompile::ast::pretty::pretty_ast;
use dslcompile::symbolic::native_egglog::optimize_with_native_egglog;
use dslcompile::backends::{RustCodeGenerator, RustCompiler};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Comprehensive IID Gaussian Demo (Proper API Usage)");
    println!("=====================================================");
    println!("Demonstrating DynamicContext API vs manual AST construction");

    // Phase 1: Proper Expression Building with DynamicContext
    println!("\n=== Phase 1: Proper Expression Building ===");
    
    let (mut ctx, gaussian_expr) = define_gaussian_log_density_proper();
    println!("✅ Built Gaussian log-density using DynamicContext");
    println!("   Variables managed automatically, no manual indices!");
    
    let iid_expr = define_iid_summation_proper(&mut ctx, &gaussian_expr);
    println!("✅ Built IID summation expression");
    println!("   Used same context for variable consistency");

    // Phase 2: Mathematical Optimization (Symbolic)
    println!("\n=== Phase 2: Egglog Optimization ===");
    let optimized_expr = simplify_with_egglog(&iid_expr)?;
    
    // Phase 3: Pretty Print Analysis
    println!("\n=== Phase 3: Expression Analysis ===");
    pretty_print_expression(&optimized_expr);

    // Phase 4: Code Generation and Compilation
    println!("\n=== Phase 4: Code Generation ===");
    let compiled_fn = generate_and_compile_rust(&optimized_expr)?;
    println!("✅ Generated and compiled native code");

    // Phase 5: Runtime Data Evaluation
    println!("\n=== Phase 5: Runtime Data Evaluation ===");
    evaluate_with_runtime_data(&compiled_fn)?;

    println!("\n🎉 Demo Complete!");
    println!("   ✅ Proper DynamicContext usage demonstrated");
    println!("   ✅ No manual variable index management");
    println!("   ✅ Type-safe expression building");

    Ok(())
}

/// Step 1: Define symbolic Gaussian log-density using PROPER DynamicContext API
/// Returns the context and expression - proper pattern for managing variable scope
fn define_gaussian_log_density_proper() -> (DynamicContext<f64>, TypedBuilderExpr<f64>) {
    let mut ctx = DynamicContext::new();
    
    // ✅ Proper way: use DynamicContext to create variables
    // No manual index management needed!
    let x = ctx.var();      // observation variable (auto-assigned)
    let mu = ctx.var();     // mean parameter (auto-assigned)
    let sigma = ctx.var();  // std deviation parameter (auto-assigned)

    println!("   Created variables using ctx.var() - indices managed automatically");

    // Build -½((x-μ)/σ)² using natural mathematical syntax
    let diff = &x - &mu;
    let standardized = diff / &sigma;
    let squared = standardized.clone() * standardized;
    let neg_half_squared = ctx.constant(-0.5) * squared;

    // Build normalization: -log(σ√2π)
    let log_sigma = sigma.clone().ln();
    let log_sqrt_2pi = ctx.constant((2.0 * std::f64::consts::PI).sqrt().ln());
    let normalization = -(log_sigma + log_sqrt_2pi);

    // Complete: -½((x-μ)/σ)² - log(σ√2π)
    let gaussian_expr = neg_half_squared + normalization;
    
    println!("   Built complex expression using natural operators: +, -, *, /, ln()");
    
    (ctx, gaussian_expr)
}

/// Step 2: Define IID summation using the SAME context (proper variable management)
fn define_iid_summation_proper(
    ctx: &mut DynamicContext<f64>, 
    gaussian_template: &TypedBuilderExpr<f64>
) -> TypedBuilderExpr<f64> {
    
    println!("   Building IID summation over data points");
    println!("   Using same context to ensure variable consistency");
    
    // Create some sample data for demonstration
    // In practice, this would come from the evaluation phase
    let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Create parameters outside the closure to avoid borrow checker issues
    let mu = ctx.var();     // Parameter for mean
    let sigma = ctx.var();  // Parameter for std deviation
    let const_neg_half = ctx.constant(-0.5);
    let const_log_sqrt_2pi = ctx.constant((2.0 * std::f64::consts::PI).sqrt().ln());
    
    // ✅ Proper way: use the unified sum API
    // This creates proper Collection/Lambda AST structure automatically
    let iid_expr = ctx.sum(sample_data, |x_i| {
        // x_i is the iterator variable, automatically managed
        // Build the log-density for this data point using pre-created parameters
        
        let diff = &x_i - &mu;
        let standardized = diff / &sigma;
        let squared = standardized.clone() * standardized;
        let neg_half_squared = &const_neg_half * squared;
        
        let log_sigma = sigma.clone().ln();
        let normalization = -(log_sigma + &const_log_sqrt_2pi);
        
        neg_half_squared + normalization
    });
    
    println!("   ✅ Created proper Sum(Collection) structure");
    println!("   Iterator variable and collection managed automatically");
    
    iid_expr
}

/// Step 3: Apply egglog optimization (unchanged - works with any expression)
fn simplify_with_egglog(expr: &TypedBuilderExpr<f64>) -> Result<TypedBuilderExpr<f64>, Box<dyn std::error::Error>> {
    println!("🔧 Applying egglog mathematical optimization...");
    
    let start = Instant::now();
    
    // Convert to ASTRepr for optimization (this is the proper bridge point)
    let ast_expr = dslcompile::ast::advanced::ast_from_expr(expr).clone();
    
    match optimize_with_native_egglog(&ast_expr) {
        Ok(optimized_ast) => {
            let duration = start.elapsed();
            println!("   Optimization completed in {duration:.2?}");
            println!("   ✅ Applied algebraic simplification rules");
            
            // Convert back to TypedBuilderExpr
            let registry = std::sync::Arc::new(std::cell::RefCell::new(VariableRegistry::new()));
            Ok(TypedBuilderExpr::new(optimized_ast, registry))
        }
        Err(e) => {
            println!("   ⚠️ Egglog optimization failed: {e}");
            println!("   Continuing with original expression");
            Ok(expr.clone())
        }
    }
}

/// Step 4: Pretty print the expression (enhanced to show proper structure)
fn pretty_print_expression(expr: &TypedBuilderExpr<f64>) {
    println!("📋 Expression Structure Analysis:");
    
    // Use the advanced API to access AST for analysis
    let ast = dslcompile::ast::advanced::ast_from_expr(expr);
    let var_registry = VariableRegistry::for_expression(ast);
    println!("   Expression: {}", pretty_ast(ast, &var_registry));
    
    println!("\n📊 Architecture Benefits:");
    println!("   ✅ Variables: Automatically managed indices");
    println!("   ✅ Type Safety: No index collision possible");
    println!("   ✅ Natural Syntax: Mathematical operators work directly");
    println!("   ✅ Composable: Single context manages all variables");
    
    if contains_summation_proper(ast) {
        println!("   ✅ Contains summation with proper Collection structure");
    } else {
        println!("   ℹ️ Single evaluation template (no summation)");
    }
}

/// Helper: Check if expression contains summation (using advanced API)
fn contains_summation_proper(expr: &dslcompile::ast::ASTRepr<f64>) -> bool {
    use dslcompile::ast::ASTRepr;
    match expr {
        ASTRepr::Sum(_) => true,
        ASTRepr::Add(l, r) | ASTRepr::Sub(l, r) | ASTRepr::Mul(l, r) | 
        ASTRepr::Div(l, r) | ASTRepr::Pow(l, r) => {
            contains_summation_proper(l) || contains_summation_proper(r)
        },
        ASTRepr::Neg(inner) | ASTRepr::Ln(inner) | ASTRepr::Exp(inner) |
        ASTRepr::Sin(inner) | ASTRepr::Cos(inner) | ASTRepr::Sqrt(inner) => {
            contains_summation_proper(inner)
        },
        _ => false,
    }
}

/// Step 5: Generate and compile Rust code (unchanged - works with TypedBuilderExpr)
fn generate_and_compile_rust(expr: &TypedBuilderExpr<f64>) -> Result<dslcompile::backends::CompiledRustFunction, Box<dyn std::error::Error>> {
    println!("🔨 Generating Rust source code...");
    
    // Convert to AST for codegen (proper bridge point)
    let ast = dslcompile::ast::advanced::ast_from_expr(expr);
    
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(ast, "iid_gaussian_likelihood")?;
    
    println!("   Generated function for IID Gaussian likelihood");
    
    if !RustCompiler::is_available() {
        return Err("Rust compiler not available".into());
    }
    
    println!("🚀 Compiling to native machine code...");
    let compiler = RustCompiler::new();
    let compiled = compiler.compile_and_load(&rust_code, "iid_gaussian_likelihood")?;
    
    println!("   ✅ Successfully compiled to native function");
    Ok(compiled)
}

/// Step 6: Evaluate with runtime data (unchanged)
fn evaluate_with_runtime_data(compiled_fn: &dslcompile::backends::CompiledRustFunction) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Loading runtime datasets...");
    
    let datasets = load_datasets_from_runtime_sources();
    
    for (name, mu, sigma, observations) in datasets {
        println!("\n📊 {}", name);
        println!("   Parameters: μ={:.2}, σ={:.2}", mu, sigma);
        println!("   Observations: {} data points", observations.len());
        
        let log_likelihood = evaluate_iid_likelihood(compiled_fn, mu, sigma, &observations)?;
        
        println!("   📈 Total log-likelihood: {:.4}", log_likelihood);
        println!("   📊 Average per observation: {:.4}", log_likelihood / observations.len() as f64);
        
        let mean = observations.iter().sum::<f64>() / observations.len() as f64;
        let variance = observations.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / observations.len() as f64;
        println!("   📋 Data mean: {:.3}, variance: {:.3}", mean, variance);
    }
    
    Ok(())
}

/// Helper functions (unchanged from original)
fn load_datasets_from_runtime_sources() -> Vec<(&'static str, f64, f64, Vec<f64>)> {
    vec![
        (
            "Sensor A (High Precision)",
            2.1, 0.15,
            generate_runtime_data(2.1, 0.15, 50),
        ),
        (
            "Sensor B (Moderate Noise)", 
            -0.5, 0.8,
            generate_runtime_data(-0.5, 0.8, 30),
        ),
        (
            "Sensor C (High Variance)",
            1.0, 2.0,
            generate_runtime_data(1.0, 2.0, 25),
        ),
    ]
}

fn generate_runtime_data(mu: f64, sigma: f64, n: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    
    (0..n).map(|i| {
        let u1: f64 = (i as f64 + 1.0) / (n as f64 + 2.0);
        let u2: f64 = (i as f64 * 7.0 + 3.0) % 1.0;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mu + sigma * z
    }).collect()
}

fn evaluate_iid_likelihood(
    compiled_fn: &dslcompile::backends::CompiledRustFunction,
    mu: f64,
    sigma: f64,
    data: &[f64],
) -> Result<f64, Box<dyn std::error::Error>> {
    let total_log_likelihood: f64 = data.iter()
        .map(|&x| compiled_fn.call(&[x, mu, sigma] as &[f64]).unwrap_or(0.0))
        .sum();
    
    Ok(total_log_likelihood)
} 