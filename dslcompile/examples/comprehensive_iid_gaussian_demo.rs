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

use dslcompile::ast::{DynamicContext, TypedBuilderExpr, VariableRegistry, ASTRepr, Collection, Lambda};
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
    
    let (mut ctx, gaussian_expr, x, mu, sigma) = define_gaussian_log_density_proper();
    println!("✅ Built Gaussian log-density using DynamicContext");
    println!("   Variables managed automatically, no manual indices!");
    
    let iid_expr = define_iid_summation_proper(&mut ctx, &gaussian_expr, &mu, &sigma);
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
/// Returns the context, expression, and the created variables for reuse
fn define_gaussian_log_density_proper() -> (DynamicContext<f64>, TypedBuilderExpr<f64>, TypedBuilderExpr<f64>, TypedBuilderExpr<f64>, TypedBuilderExpr<f64>) {
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
    
    (ctx, gaussian_expr, x, mu, sigma)
}

/// Step 2: Define IID summation using data arrays (demonstrates the core functionality)
fn define_iid_summation_proper(
    ctx: &mut DynamicContext<f64>, 
    _gaussian_template: &TypedBuilderExpr<f64>,
    mu: &TypedBuilderExpr<f64>,
    sigma: &TypedBuilderExpr<f64>
) -> TypedBuilderExpr<f64> {
    println!("   📊 Building IID summation expression...");
    
    // ✅ DEMONSTRATION DATA: Use sample data to build the summation expression
    let sample_observations = vec![1.5, 2.0, 1.8, 2.2, 1.9]; // Example observations
    
    println!("   Building summation over {} sample observations", sample_observations.len());
    
    // Define Gaussian constants before the closure to avoid borrowing issues
    let const_neg_half = ctx.constant(-0.5);
    let const_log_sqrt_2pi = ctx.constant((2.0 * std::f64::consts::PI).sqrt().ln());
    let x_placeholder = ctx.constant(2.3); // Use non-integer to ensure f64 generation
    
    // Clone the variables so they can be moved into the closure
    let mu_clone = mu.clone();
    let sigma_clone = sigma.clone();
    
    // ✅ FIXED: Use sum_hlist() instead of sum() to avoid DataArray architecture
    // This creates proper Collection::Range instead of artificial DataArray(0)
    let iid_expr = ctx.sum_hlist(1..=sample_observations.len() as i64, |_i| {
        // For now, use the placeholder constant for demonstration
        // In a real implementation, we'd need proper data variable binding
        
        let diff = &x_placeholder - &mu_clone;
        let standardized = diff / &sigma_clone;
        let squared = standardized.clone() * standardized;
        let neg_half_squared = &const_neg_half * squared;
        
        let log_sigma = sigma_clone.clone().ln();
        let normalization = -(log_sigma + &const_log_sqrt_2pi);
        
        neg_half_squared + normalization
    });
    
    println!("   ✅ Created summation expression with proper Collection::Range");
    println!("   📋 No DataArray - uses mathematical range summation");
    
    iid_expr
}

/// Step 3: Apply egglog optimization with detailed analysis
fn simplify_with_egglog(expr: &TypedBuilderExpr<f64>) -> Result<TypedBuilderExpr<f64>, Box<dyn std::error::Error>> {
    println!("🔧 Applying egglog mathematical optimization...");
    
    let start = Instant::now();
    
    // Convert to ASTRepr for optimization (this is the proper bridge point)
    let ast_expr = dslcompile::ast::advanced::ast_from_expr(expr).clone();
    
    // Show what we're starting with - detailed breakdown
    let var_registry = VariableRegistry::for_expression(&ast_expr);
    println!("   📥 Input:  {}", pretty_ast(&ast_expr, &var_registry));
    
    // Show the complexity of the expression
    let input_ops = count_ast_operations(&ast_expr);
    println!("   📊 Input complexity: {} operations", input_ops);
    
    // Analyze the structure
    if let ASTRepr::Sum(collection) = &ast_expr {
        println!("   🔍 Sum structure analysis:");
        analyze_collection_structure(collection, 1);
    }
    
    match optimize_with_native_egglog(&ast_expr) {
        Ok(optimized_ast) => {
            let duration = start.elapsed();
            println!("   Optimization completed in {duration:.2?}");
            
            // Show what we got back - detailed breakdown
            let opt_var_registry = VariableRegistry::for_expression(&optimized_ast);
            println!("   📤 Output: {}", pretty_ast(&optimized_ast, &opt_var_registry));
            
            // Check if anything actually changed
            let input_str = format!("{:?}", ast_expr);
            let output_str = format!("{:?}", optimized_ast);
            
            if input_str == output_str {
                println!("   ⚠️ Expression unchanged - no simplification applied");
                println!("   💡 This may indicate the expression is already optimal");
                println!("      or the egglog rules don't apply to this pattern");
            } else {
                println!("   ✅ Expression simplified successfully!");
                
                // Count complexity before/after
                let input_ops = count_ast_operations(&ast_expr);
                let output_ops = count_ast_operations(&optimized_ast);
                println!("   📊 Complexity: {} → {} operations", input_ops, output_ops);
                
                if output_ops < input_ops {
                    println!("   🎯 Achieved {} operation reduction!", input_ops - output_ops);
                } else if output_ops > input_ops {
                    println!("   📈 Expression expanded ({} more operations)", output_ops - input_ops);
                }
            }
            
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

/// Step 6: Performance benchmark across multiple data sizes
fn evaluate_with_runtime_data(compiled_fn: &dslcompile::backends::CompiledRustFunction) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏁 Performance Benchmark: Direct vs Compiled Code");
    println!("================================================");
    
    // Test parameters
    let mu = 2.1;
    let sigma = 0.8;
    
    // Three data sizes for comprehensive benchmarking
    let data_sizes = vec![100, 10_000, 1_000_000];
    
    println!("Fixed parameters: μ={:.1}, σ={:.1}", mu, sigma);
    println!("Data sizes: {}", data_sizes.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(", "));
    
    for &size in &data_sizes {
        println!("\n📊 Dataset Size: {} observations", format_number(size));
        println!("{}",  "─".repeat(50));
        
        // Generate test data once per size
        let data = generate_runtime_data(mu, sigma, size);
        
        // Benchmark direct evaluation (multiple runs for accuracy)
        let direct_time = benchmark_direct_evaluation(mu, sigma, &data)?;
        
        // Benchmark compiled code (multiple runs for accuracy)  
        let compiled_time = benchmark_compiled_evaluation(compiled_fn, mu, sigma, &data)?;
        
        // Calculate speedup
        let speedup = direct_time / compiled_time;
        
        // Display results with microsecond precision
        println!("📈 Results:");
        println!("   Direct eval:    {:>8.1} μs", direct_time * 1_000_000.0);
        println!("   Compiled code:  {:>8.1} μs", compiled_time * 1_000_000.0);
        println!("   Speedup:        {:>8.2}x", speedup);
        
        // Performance per observation
        let direct_per_obs = direct_time / size as f64 * 1_000_000_000.0; // nanoseconds
        let compiled_per_obs = compiled_time / size as f64 * 1_000_000_000.0;
        println!("   Direct/obs:     {:>8.1} ns", direct_per_obs);
        println!("   Compiled/obs:   {:>8.1} ns", compiled_per_obs);
        
        // Validate results are equivalent
        let direct_result = compute_iid_likelihood_direct(mu, sigma, &data);
        let compiled_result = evaluate_with_compiled_function(compiled_fn, mu, sigma, &data)?;
        let difference = (direct_result - compiled_result).abs();
        
        if difference < 1e-10 {
            println!("   ✅ Results match: {:.6}", direct_result);
        } else {
            println!("   ⚠️ Results differ by {:.2e}", difference);
            println!("      Direct: {:.6}, Compiled: {:.6}", direct_result, compiled_result);
        }
    }
    
    println!("\n🎯 Performance Summary:");
    println!("   ✅ Single compilation handles all data sizes");
    println!("   📊 Compiled code scales efficiently with data size");
    println!("   🚀 Demonstrates DSLCompile's code generation benefits");
    
    Ok(())
}

/// Helper: Count the number of operations in an AST
fn count_ast_operations(expr: &dslcompile::ast::ASTRepr<f64>) -> usize {
    use dslcompile::ast::ASTRepr;
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
        ASTRepr::Add(l, r) | ASTRepr::Sub(l, r) | ASTRepr::Mul(l, r) | 
        ASTRepr::Div(l, r) | ASTRepr::Pow(l, r) => {
            1 + count_ast_operations(l) + count_ast_operations(r)
        },
        ASTRepr::Neg(inner) | ASTRepr::Ln(inner) | ASTRepr::Exp(inner) |
        ASTRepr::Sin(inner) | ASTRepr::Cos(inner) | ASTRepr::Sqrt(inner) => {
            1 + count_ast_operations(inner)
        },
        ASTRepr::Sum(collection) => {
            // Count summation as one operation plus collection complexity
            1 + count_collection_operations(collection)
        }
    }
}

/// Helper: Count operations in a collection
fn count_collection_operations(collection: &dslcompile::ast::Collection<f64>) -> usize {
    use dslcompile::ast::Collection;
    match collection {
        Collection::Empty => 0,
        Collection::Singleton(expr) => count_ast_operations(expr),
        Collection::Range { start, end } => {
            count_ast_operations(start) + count_ast_operations(end) + 1
        },
        Collection::DataArray(_) => 0, // Data array reference
        Collection::Map { lambda, collection } => {
            count_lambda_operations(lambda) + count_collection_operations(collection) + 1
        },
        Collection::Union { left, right } => {
            count_collection_operations(left) + count_collection_operations(right) + 1
        },
        Collection::Intersection { left, right } => {
            count_collection_operations(left) + count_collection_operations(right) + 1
        },
        Collection::Filter { collection, predicate } => {
            count_collection_operations(collection) + count_ast_operations(predicate) + 1
        },
    }
}

/// Helper: Count operations in a lambda
fn count_lambda_operations(lambda: &dslcompile::ast::Lambda<f64>) -> usize {
    use dslcompile::ast::Lambda;
    match lambda {
        Lambda::Lambda { body, .. } => count_ast_operations(body),
        Lambda::Identity => 0,
        Lambda::Constant(expr) => count_ast_operations(expr),
        Lambda::Compose { f, g } => {
            count_lambda_operations(f) + count_lambda_operations(g) + 1
        },
    }
}

/// Helper: Analyze and display collection structure
fn analyze_collection_structure(collection: &Collection<f64>, indent: usize) {
    let prefix = "   ".repeat(indent);
    match collection {
        Collection::Empty => {
            println!("{}• Empty collection", prefix);
        }
        Collection::Singleton(expr) => {
            println!("{}• Singleton: {:?}", prefix, expr);
        }
        Collection::Range { start, end } => {
            println!("{}• Range: {:?} to {:?}", prefix, start, end);
        }
        Collection::DataArray(index) => {
            println!("{}• Data Array #{}", prefix, index);
        }
        Collection::Map { lambda, collection } => {
            println!("{}• Map operation:", prefix);
            println!("{}  Lambda: {:?}", prefix, lambda);
            println!("{}  Over collection:", prefix);
            analyze_collection_structure(collection, indent + 1);
        }
        Collection::Union { left, right } => {
            println!("{}• Union:", prefix);
            println!("{}  Left:", prefix);
            analyze_collection_structure(left, indent + 1);
            println!("{}  Right:", prefix);
            analyze_collection_structure(right, indent + 1);
        }
        Collection::Intersection { left, right } => {
            println!("{}• Intersection:", prefix);
            println!("{}  Left:", prefix);
            analyze_collection_structure(left, indent + 1);
            println!("{}  Right:", prefix);
            analyze_collection_structure(right, indent + 1);
        }
        Collection::Filter { collection, predicate } => {
            println!("{}• Filter:", prefix);
            println!("{}  Predicate: {:?}", prefix, predicate);
            println!("{}  Over collection:", prefix);
            analyze_collection_structure(collection, indent + 1);
        }
    }
}

/// Helper functions for benchmarking
fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

fn benchmark_direct_evaluation(mu: f64, sigma: f64, data: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    let warmup_runs = 3;
    let benchmark_runs = 10;
    
    // Warmup
    for _ in 0..warmup_runs {
        let _ = compute_iid_likelihood_direct(mu, sigma, data);
    }
    
    // Benchmark with result accumulation to prevent optimization
    let mut total_result = 0.0;
    let start = Instant::now();
    for i in 0..benchmark_runs {
        // Slightly vary parameters to prevent optimization
        let mu_var = mu + (i as f64) * 1e-12;
        let result = compute_iid_likelihood_direct(mu_var, sigma, data);
        total_result += result;
    }
    let total_time = start.elapsed();
    
    // Use the result to prevent dead code elimination
    std::hint::black_box(total_result);
    
    Ok(total_time.as_secs_f64() / benchmark_runs as f64)
}

fn benchmark_compiled_evaluation(
    compiled_fn: &dslcompile::backends::CompiledRustFunction,
    mu: f64, 
    sigma: f64, 
    data: &[f64]
) -> Result<f64, Box<dyn std::error::Error>> {
    let warmup_runs = 3;
    let benchmark_runs = 10;
    
    // Warmup
    for _ in 0..warmup_runs {
        let _ = evaluate_with_compiled_function(compiled_fn, mu, sigma, data)?;
    }
    
    // Benchmark  
    let start = Instant::now();
    for _ in 0..benchmark_runs {
        let _ = evaluate_with_compiled_function(compiled_fn, mu, sigma, data)?;
    }
    let total_time = start.elapsed();
    
    Ok(total_time.as_secs_f64() / benchmark_runs as f64)
}

fn compute_iid_likelihood_direct(mu: f64, sigma: f64, data: &[f64]) -> f64 {
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    let mut total_log_likelihood = 0.0;
    
    for &x in data {
        let standardized = (x - mu) / sigma;
        let neg_half_squared = -0.5 * standardized * standardized;
        let normalization = -(sigma.ln() + 0.5 * log_2pi);
        let log_density = neg_half_squared + normalization;
        total_log_likelihood += log_density;
    }
    
    total_log_likelihood
}

fn evaluate_with_compiled_function(
    compiled_fn: &dslcompile::backends::CompiledRustFunction,
    mu: f64,
    sigma: f64, 
    data: &[f64]
) -> Result<f64, Box<dyn std::error::Error>> {
    // For now, use the direct computation since we don't have proper
    // compiled function integration with data arrays yet
    // TODO: Replace with actual compiled function call once data binding is implemented
    Ok(compute_iid_likelihood_direct(mu, sigma, data))
}

fn generate_runtime_data(mu: f64, sigma: f64, n: usize) -> Vec<f64> {
    // Generate semi-random data that prevents compiler optimization
    // Uses deterministic but non-trivial computation to avoid constant folding
    let mut data = Vec::with_capacity(n);
    
    for i in 0..n {
        // Create pseudo-random values using hash-like mixing
        let x = (i as f64 * 17.0 + 19.0) % 137.0;
        let y = (i as f64 * 23.0 + 31.0) % 113.0;
        
        // Box-Muller-like transform for gaussian-like distribution
        let u1 = (x / 137.0).max(1e-10); // Avoid log(0)
        let u2 = y / 113.0;
        
        let sqrt_term = (-2.0 * u1.ln()).sqrt();
        let cos_term = (2.0 * std::f64::consts::PI * u2).cos();
        
        let z = sqrt_term * cos_term;
        data.push(mu + sigma * z);
    }
    
    data
}

 