//! Egglog Optimization Test
//!
//! This test specifically targets the bidirectional optimization patterns
//! in the egglog collection system to verify they're working correctly.

use dslcompile::prelude::*;
use dslcompile::symbolic::collection_summation::*;

fn main() -> Result<()> {
    println!("🔬 Egglog Bidirectional Optimization Test");
    println!("=========================================\n");

    // Test 1: Arithmetic series pattern (should trigger n(n+1)/2 optimization)
    test_arithmetic_series_optimization()?;

    // Test 2: Constant sum pattern (should trigger c*size optimization)
    test_constant_sum_optimization()?;

    // Test 3: Direct collection API (bypass conversion issues)
    test_direct_collection_api()?;

    // Test 4: Lambda composition optimization
    test_lambda_composition()?;

    println!("✅ Egglog optimization test completed!");
    Ok(())
}

fn test_arithmetic_series_optimization() -> Result<()> {
    println!("🧮 Test 1: Arithmetic Series Optimization");
    println!("==========================================");

    let ctx = DynamicContext::new();

    // Create range collection directly
    let range = Collection::Range {
        start: Box::new(ASTRepr::Constant(1.0)),
        end: Box::new(ASTRepr::Constant(10.0)),
    };

    // Create identity lambda (should trigger arithmetic series optimization)
    let identity = Lambda::Identity;

    println!("Input: Sum(Range(1, 10), Identity)");
    println!("Expected optimization: n(n+1)/2 = 10*11/2 = 55");

    // Apply optimization directly
    let mut optimizer = CollectionSummationOptimizer::new();
    let collection_expr = CollectionExpr::Sum {
        collection: range,
        lambda: identity,
    };

    let optimized = optimizer.optimize_collection_expr(&collection_expr)?;
    println!("Optimized expression: {:?}", optimized);

    // Convert back to AST and evaluate
    let ast = optimizer.to_ast(&optimized)?;
    let expr = TypedBuilderExpr::new(ast, ctx.registry());
    let result = ctx.eval(&expr, &[]);

    println!("Result: {} (expected: 55)", result);
    println!("Pretty print: {}", ctx.pretty_print(&expr));
    println!();

    Ok(())
}

fn test_constant_sum_optimization() -> Result<()> {
    println!("🔢 Test 2: Constant Sum Optimization");
    println!("====================================");

    let ctx = DynamicContext::new();

    // Create range collection
    let range = Collection::Range {
        start: Box::new(ASTRepr::Constant(1.0)),
        end: Box::new(ASTRepr::Constant(5.0)),
    };

    // Create constant lambda (should trigger c*size optimization)
    let constant_lambda = Lambda::Constant(Box::new(ASTRepr::Constant(7.0)));

    println!("Input: Sum(Range(1, 5), Constant(7))");
    println!("Expected optimization: 7 * size = 7 * 5 = 35");

    // Apply optimization directly
    let mut optimizer = CollectionSummationOptimizer::new();
    let collection_expr = CollectionExpr::Sum {
        collection: range,
        lambda: constant_lambda,
    };

    let optimized = optimizer.optimize_collection_expr(&collection_expr)?;
    println!("Optimized expression: {:?}", optimized);

    // Convert back to AST and evaluate
    let ast = optimizer.to_ast(&optimized)?;
    let expr = TypedBuilderExpr::new(ast, ctx.registry());
    let result = ctx.eval(&expr, &[]);

    println!("Result: {} (expected: 35)", result);
    println!("Pretty print: {}", ctx.pretty_print(&expr));
    println!();

    Ok(())
}

fn test_direct_collection_api() -> Result<()> {
    println!("🎯 Test 3: Direct Collection API");
    println!("=================================");

    let ctx = DynamicContext::new();

    // Test the direct collection API to bypass conversion issues
    let range = ctx.range_collection(
        ctx.constant(1.0),
        ctx.constant(3.0),
    );

    let identity = ctx.identity_lambda();

    println!("Input: Direct collection API with identity lambda");
    println!("Range: [1, 3], Lambda: Identity");

    let result_expr = ctx.sum_collection(range, identity)?;
    let result = ctx.eval(&result_expr, &[]);

    println!("Result: {} (expected: 6)", result); // 1+2+3 = 6
    println!("Pretty print: {}", ctx.pretty_print(&result_expr));
    println!();

    Ok(())
}

fn test_lambda_composition() -> Result<()> {
    println!("🔗 Test 4: Lambda Composition");
    println!("=============================");

    let ctx = DynamicContext::new();

    // Create a composed lambda: f(g(x)) where g(x) = x and f(x) = 2*x
    let g = Lambda::Identity;
    let f = Lambda::Lambda {
        var: "x".to_string(),
        body: Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(2.0)),
            Box::new(ASTRepr::Variable(0)),
        )),
    };

    let composed = Lambda::Compose {
        f: Box::new(f),
        g: Box::new(g),
    };

    let range = Collection::Range {
        start: Box::new(ASTRepr::Constant(1.0)),
        end: Box::new(ASTRepr::Constant(4.0)),
    };

    println!("Input: Sum(Range(1, 4), Compose(f, Identity)) where f(x) = 2*x");
    println!("Expected: Sum(Range(1, 4), f) = 2*(1+2+3+4) = 20");

    // Apply optimization
    let mut optimizer = CollectionSummationOptimizer::new();
    let collection_expr = CollectionExpr::Sum {
        collection: range,
        lambda: composed,
    };

    let optimized = optimizer.optimize_collection_expr(&collection_expr)?;
    println!("Optimized expression: {:?}", optimized);

    // Convert and evaluate
    let ast = optimizer.to_ast(&optimized)?;
    let expr = TypedBuilderExpr::new(ast, ctx.registry());
    let result = ctx.eval(&expr, &[]);

    println!("Result: {} (expected: 20)", result);
    println!("Pretty print: {}", ctx.pretty_print(&expr));
    println!();

    Ok(())
} 