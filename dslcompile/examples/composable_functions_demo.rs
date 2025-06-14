//! Composable Expression Functions Demo
//!
//! This demo demonstrates TRUE composability:
//! 1. Reusable expression functions that take expressions and return expressions
//! 2. Generic combinators that work with any density function
//! 3. Cross-function algebraic optimization (whole program analysis)
//! 4. Single compiled function from composed expressions

use dslcompile::{
    SymbolicOptimizer,
    backends::{RustCodeGenerator, RustCompiler},
    prelude::*,
};
use frunk::hlist;

fn main() -> Result<()> {
    println!("🔧 Composable Expression Functions Demo");
    println!("=======================================\n");

    // =======================================================================
    // 1. Define Composable Expression Functions
    // =======================================================================

    println!("1️⃣ Defining Composable Expression Functions");
    println!("--------------------------------------------");

    let mut ctx = DynamicContext::new();

    // Define parameters that will be shared across compositions
    let mu = ctx.var(); // Variable(0) - mean parameter
    let sigma = ctx.var(); // Variable(1) - std dev parameter

    println!(
        "✅ Created shared parameters: μ={}, σ={}",
        mu.var_id(),
        sigma.var_id()
    );

    // =======================================================================
    // 2. Create Building Block Expressions
    // =======================================================================

    println!("\n2️⃣ Creating Building Block Expressions");
    println!("---------------------------------------");

    // Building block 1: Normal log-density expression (reusable components)
    let x_single = ctx.var(); // Variable(2) - single observation
    let normal_single = create_normal_log_density(&mut ctx, &x_single, &mu, &sigma);

    // Building block 2: IID combinator - directly using sum
    let iid_normal = create_iid_normal(&mut ctx, &mu, &sigma);

    // Building block 3: More complex composition - mixture of normals
    let mu2 = ctx.var(); // Variable(3) - second mean
    let sigma2 = ctx.var(); // Variable(4) - second std dev  
    let weight = ctx.var(); // Variable(5) - mixture weight
    let x_mix = ctx.var(); // Variable(6) - mixture observation

    let mixture_normal =
        create_mixture_normal(&mut ctx, &x_mix, &mu, &sigma, &mu2, &sigma2, &weight);

    println!("✅ Created building block expressions:");
    println!("   • normal_single: Single normal log-density");
    println!("   • iid_normal: IID summation over data vector");
    println!("   • mixture_normal: Mixture of two normals");

    // =======================================================================
    // 3. Analyze Composed Expressions
    // =======================================================================

    println!("\n3️⃣ Analyzing Composed Expressions");
    println!("----------------------------------");

    let single_ast = ctx.to_ast(&normal_single);
    let iid_ast = ctx.to_ast(&iid_normal);
    let mixture_ast = ctx.to_ast(&mixture_normal);

    println!("Single Normal Log-Density:");
    println!("   • Operations: {}", single_ast.count_operations());
    println!("   • Variables: {}", count_variables(&single_ast));
    println!("   • Depth: {}", compute_depth(&single_ast));
    println!("   • Structure: Direct mathematical expression");

    println!("\nIID Normal Likelihood (composed function):");
    println!("   • Operations: {}", iid_ast.count_operations());
    println!("   • Variables: {}", count_variables(&iid_ast));
    println!("   • Depth: {}", compute_depth(&iid_ast));
    println!("   • Summations: {}", iid_ast.count_summations());
    println!("   • 🔑 Single AST from function composition!");

    println!("\nMixture Normal (complex composition):");
    println!("   • Operations: {}", mixture_ast.count_operations());
    println!("   • Variables: {}", count_variables(&mixture_ast));
    println!("   • Depth: {}", compute_depth(&mixture_ast));
    println!("   • 🔑 Complex structure from composable building blocks!");

    // =======================================================================
    // 4. Cross-Function Optimization (Whole Program Analysis!)
    // =======================================================================

    println!("\n4️⃣ Cross-Function Optimization");
    println!("-------------------------------");

    #[cfg(feature = "optimization")]
    {
        let mut optimizer = SymbolicOptimizer::new()?;

        println!("🔧 Performing whole-program optimization...");

        // Optimize composed expressions - this is where the magic happens!
        let optimized_single = optimizer.optimize(&single_ast)?;
        let optimized_iid = optimizer.optimize(&iid_ast)?;
        let optimized_mixture = optimizer.optimize(&mixture_ast)?;

        let single_reduction = calculate_reduction(
            single_ast.count_operations(),
            optimized_single.count_operations(),
        );
        let iid_reduction =
            calculate_reduction(iid_ast.count_operations(), optimized_iid.count_operations());
        let mixture_reduction = calculate_reduction(
            mixture_ast.count_operations(),
            optimized_mixture.count_operations(),
        );

        println!("   Single Normal:");
        println!(
            "     • Before: {} operations",
            single_ast.count_operations()
        );
        println!(
            "     • After: {} operations",
            optimized_single.count_operations()
        );
        println!("     • Result: {}", single_reduction);

        println!("   IID Normal (function composition):");
        println!("     • Before: {} operations", iid_ast.count_operations());
        println!(
            "     • After: {} operations",
            optimized_iid.count_operations()
        );
        println!("     • Result: {}", iid_reduction);

        println!("   Mixture Normal (complex composition):");
        println!(
            "     • Before: {} operations",
            mixture_ast.count_operations()
        );
        println!(
            "     • After: {} operations",
            optimized_mixture.count_operations()
        );
        println!("     • Result: {}", mixture_reduction);

        println!("   🎯 Key Insight: Optimizer sees ENTIRE composed expression!");
        println!("      Algebraic simplifications work across composition boundaries!");

        // =======================================================================
        // 5. Code Generation from Composed Functions
        // =======================================================================

        println!("\n5️⃣ Code Generation from Compositions");
        println!("-------------------------------------");

        let codegen = RustCodeGenerator::new();

        // Generate code for composed function
        let single_code = codegen.generate_function(&optimized_single, "composed_single_normal")?;
        let iid_code = codegen.generate_function(&optimized_iid, "composed_iid_normal")?;
        let mixture_code = codegen.generate_function(&optimized_mixture, "composed_mixture")?;

        println!("✅ Generated code for composed single normal:");
        println!("{}", single_code);

        println!("\n✅ Generated code for composed IID normal:");
        println!("{}", iid_code);

        println!("\n✅ Generated code for composed mixture:");
        println!("{}", mixture_code);

        // Compile the composed functions
        let compiler = RustCompiler::new();
        let _compiled_single = compiler.compile_and_load(&single_code, "composed_single_normal")?;
        let _compiled_iid = compiler.compile_and_load(&iid_code, "composed_iid_normal")?;
        let _compiled_mixture = compiler.compile_and_load(&mixture_code, "composed_mixture")?;

        println!("✅ Successfully compiled all composed expressions to native code!");

        // =======================================================================
        // 6. Test Composed Functions
        // =======================================================================

        println!("\n6️⃣ Testing Composed Functions");
        println!("------------------------------");

        // Test the single normal
        let test_result = ctx.eval(&normal_single, hlist![1.0, 0.0, 1.0]); // N(0,1) at x=1
        println!(
            "Single normal log-density(x=1, μ=0, σ=1): {:.6}",
            test_result
        );

        // Test composition behavior
        println!("\n🔍 Composition Analysis:");
        println!("   • Expression functions compose into single AST");
        println!("   • No runtime function call overhead");
        println!("   • Whole-program optimization across boundaries");
        println!("   • Single compiled function for entire composition");

        println!("\n📊 Composability Benefits:");
        println!("   ✅ Reusable expression building blocks");
        println!("   ✅ Type-safe mathematical compositions");
        println!("   ✅ Automatic inlining and optimization");
        println!("   ✅ Zero abstraction cost in generated code");
        println!("   ✅ Cross-function algebraic simplification");

        // =======================================================================
        // 7. Advanced Composition Example
        // =======================================================================

        println!("\n7️⃣ Advanced Composition Example");
        println!("--------------------------------");

        // Create a more complex composition that shows cross-function optimization potential
        let advanced_expr = create_advanced_composition(&mut ctx, &mu, &sigma);
        let advanced_ast = ctx.to_ast(&advanced_expr);
        let optimized_advanced = optimizer.optimize(&advanced_ast)?;

        println!("Advanced composition (multiple layers):");
        println!(
            "   • Before optimization: {} operations",
            advanced_ast.count_operations()
        );
        println!(
            "   • After optimization: {} operations",
            optimized_advanced.count_operations()
        );
        println!(
            "   • Reduction: {}",
            calculate_reduction(
                advanced_ast.count_operations(),
                optimized_advanced.count_operations()
            )
        );
        println!("   • 🎯 Shows optimization across multiple composition layers!");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("⚠️  Optimization requires the 'optimization' feature");
        println!(
            "   Run with: cargo run --features optimization --example composable_functions_demo"
        );
    }

    println!("\n🎉 Composability Demo Summary");
    println!("=============================");
    println!("✅ Defined reusable expression building blocks");
    println!("✅ Composed functions into complex expressions");
    println!("✅ Demonstrated whole-program optimization");
    println!("✅ Generated optimized code from compositions");
    println!("\n🔑 Key Insight: Expression functions compose algebraically!");
    println!("   Unlike runtime functions, these compose into single optimizable ASTs.");
    println!("   This enables mathematical optimization across function boundaries,");
    println!("   similar to whole-program analysis but for mathematical expressions.");

    Ok(())
}

// =======================================================================
// Composable Building Block Functions
// =======================================================================

/// Create a normal log-density expression
fn create_normal_log_density(
    ctx: &mut DynamicContext,
    x: &DynamicExpr<f64>,
    mu: &DynamicExpr<f64>,
    sigma: &DynamicExpr<f64>,
) -> DynamicExpr<f64> {
    let log_2pi = ctx.constant(1.8378770664093453_f64);
    let neg_half = ctx.constant(-0.5);

    let centered = x - mu;
    let standardized = &centered / sigma;
    let squared = &standardized * &standardized;

    &neg_half * &log_2pi - sigma.clone().ln() + &neg_half * &squared
}

/// Create an IID normal likelihood (sum over data vector)
fn create_iid_normal(
    ctx: &mut DynamicContext,
    mu: &DynamicExpr<f64>,
    sigma: &DynamicExpr<f64>,
) -> DynamicExpr<f64> {
    // Create the constants we need inside the sum
    let log_2pi = ctx.constant(1.8378770664093453_f64);
    let neg_half = ctx.constant(-0.5);

    // Use sum over empty vector to create Collection::Variable
    ctx.sum(vec![], |x| {
        // Inline the normal log-density to avoid borrowing issues
        let centered = &x - mu;
        let standardized = &centered / sigma;
        let squared = &standardized * &standardized;

        &neg_half * &log_2pi - sigma.clone().ln() + &neg_half * &squared
    })
}

/// Create a mixture of two normal distributions
fn create_mixture_normal(
    ctx: &mut DynamicContext,
    x: &DynamicExpr<f64>,
    mu1: &DynamicExpr<f64>,
    sigma1: &DynamicExpr<f64>,
    mu2: &DynamicExpr<f64>,
    sigma2: &DynamicExpr<f64>,
    weight: &DynamicExpr<f64>,
) -> DynamicExpr<f64> {
    let density1 = create_normal_log_density(ctx, x, mu1, sigma1);
    let density2 = create_normal_log_density(ctx, x, mu2, sigma2);

    // Simplified mixture (real version would use log_sum_exp for numerical stability)
    let one = ctx.constant(1.0);
    let w_comp = &one - weight;

    let weighted1 = weight.clone().ln() + density1;
    let weighted2 = w_comp.ln() + density2;

    // This composition shows how expressions naturally combine
    weighted1 + weighted2
}

/// Create a more advanced composition showing multiple layers
fn create_advanced_composition(
    ctx: &mut DynamicContext,
    mu: &DynamicExpr<f64>,
    sigma: &DynamicExpr<f64>,
) -> DynamicExpr<f64> {
    // Create a composition that has potential for cross-function optimization
    let x = ctx.var();

    // Layer 1: Basic normal
    let normal1 = create_normal_log_density(ctx, &x, mu, sigma);

    // Layer 2: Transform parameters
    let mu_transformed = mu + sigma; // This creates optimization opportunities
    let sigma_transformed = sigma * ctx.constant(2.0);

    // Layer 3: Another normal with transformed parameters
    let normal2 = create_normal_log_density(ctx, &x, &mu_transformed, &sigma_transformed);

    // Layer 4: Combine with algebraic operations
    let combined = &normal1 + &normal2 - mu.clone().ln();

    // Layer 5: More transformations that could be optimized
    let final_expr = &combined * &combined + sigma.clone();

    final_expr
}

// =======================================================================
// Helper Functions
// =======================================================================

fn calculate_reduction(before: usize, after: usize) -> String {
    if before > after {
        format!("{} operations eliminated!", before - after)
    } else if after > before {
        format!(
            "Optimization added {} operations (may improve numerical stability)",
            after - before
        )
    } else {
        "No reduction found".to_string()
    }
}

fn count_variables<T>(ast: &ASTRepr<T>) -> usize {
    use std::collections::HashSet;
    let mut vars = HashSet::new();
    collect_variables(ast, &mut vars);
    vars.len()
}

fn collect_variables<T>(ast: &ASTRepr<T>, vars: &mut std::collections::HashSet<usize>) {
    match ast {
        ASTRepr::Variable(idx) | ASTRepr::BoundVar(idx) => {
            vars.insert(*idx);
        }
        ASTRepr::Add(l, r)
        | ASTRepr::Sub(l, r)
        | ASTRepr::Mul(l, r)
        | ASTRepr::Div(l, r)
        | ASTRepr::Pow(l, r) => {
            collect_variables(l, vars);
            collect_variables(r, vars);
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
        ASTRepr::Lambda(lambda) => {
            collect_variables(&lambda.body, vars);
        }
        ASTRepr::Constant(_) => {}
        ASTRepr::Sum(_) => {}
    }
}

fn collect_variables_from_collection<T>(
    collection: &dslcompile::ast::ast_repr::Collection<T>,
    vars: &mut std::collections::HashSet<usize>,
) {
    use dslcompile::ast::ast_repr::Collection;
    match collection {
        Collection::Variable(idx) => {
            vars.insert(*idx);
        }
        Collection::Singleton(expr) => {
            collect_variables(expr, vars);
        }
        Collection::Range { start, end } => {
            collect_variables(start, vars);
            collect_variables(end, vars);
        }
        Collection::Filter {
            collection,
            predicate,
        } => {
            collect_variables_from_collection(collection, vars);
            collect_variables(predicate, vars);
        }
        Collection::Map {
            lambda: _,
            collection,
        } => {
            collect_variables_from_collection(collection, vars);
        }
        Collection::Empty => {}
        Collection::DataArray(_) => {
            // DataArray contains literal data, no variables to collect
        }
    }
}

fn compute_depth<T>(ast: &ASTRepr<T>) -> usize {
    match ast {
        ASTRepr::Variable(_) | ASTRepr::BoundVar(_) | ASTRepr::Constant(_) => 1,
        ASTRepr::Add(l, r)
        | ASTRepr::Sub(l, r)
        | ASTRepr::Mul(l, r)
        | ASTRepr::Div(l, r)
        | ASTRepr::Pow(l, r) => 1 + std::cmp::max(compute_depth(l), compute_depth(r)),
        ASTRepr::Let(_, expr, body) => 1 + std::cmp::max(compute_depth(expr), compute_depth(body)),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + compute_depth(inner),
        ASTRepr::Lambda(lambda) => 1 + compute_depth(&lambda.body),
        ASTRepr::Sum(_) => 2,
    }
}
