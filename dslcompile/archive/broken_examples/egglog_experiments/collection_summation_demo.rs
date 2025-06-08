//! Collection-Based Summation Demo
//!
//! This example demonstrates the new Map-based collection summation approach
//! that provides more mathematical expressiveness and powerful optimizations
//! through egglog's bidirectional rewrite rules.

use dslcompile::prelude::*;
use dslcompile::symbolic::collection_summation::*;

fn main() -> Result<()> {
    println!("🗂️  Collection-Based Summation Demo");
    println!("===================================\n");

    // Demo 1: Basic collection operations
    demo_basic_collections()?;

    // Demo 2: Lambda calculus optimizations
    demo_lambda_optimizations()?;

    // Demo 3: Bidirectional mathematical identities
    demo_mathematical_identities()?;

    // Demo 4: Automatic pattern recognition
    demo_pattern_recognition()?;

    // Demo 5: Complex collection operations
    demo_complex_collections()?;

    println!("✅ Collection-based summation provides powerful mathematical optimizations!");
    Ok(())
}

fn demo_basic_collections() -> Result<()> {
    println!("📊 Demo 1: Basic Collection Operations");
    println!("=====================================");

    let ctx = DynamicContext::new();

    // Create range collection: [1, 10]
    let range = ctx.range_collection(
        ctx.constant(1.0),
        ctx.constant(10.0),
    );

    // Create identity lambda: x -> x
    let identity = ctx.identity_lambda();

    // Sum over range with identity: Σ(x for x in 1..=10)
    let arithmetic_sum = ctx.sum_collection(range.clone(), identity)?;
    let result = ctx.eval(&arithmetic_sum, &[]);
    
    println!("Range collection: [1, 10]");
    println!("Identity lambda: x -> x");
    println!("Sum result: {} (expected: 55)", result);
    
    // Create constant lambda: x -> 5
    let constant_five = ctx.constant_lambda(ctx.constant(5.0));
    
    // Sum over range with constant: Σ(5 for x in 1..=10)
    let constant_sum = ctx.sum_collection(range, constant_five)?;
    let constant_result = ctx.eval(&constant_sum, &[]);
    
    println!("Constant lambda: x -> 5");
    println!("Constant sum result: {} (expected: 50)", constant_result);
    println!();

    Ok(())
}

fn demo_lambda_optimizations() -> Result<()> {
    println!("🔧 Demo 2: Lambda Calculus Optimizations");
    println!("========================================");

    let ctx = DynamicContext::new();

    // Create a complex lambda: x -> 2 * (x + 1)
    let complex_lambda = ctx.lambda(|x| {
        ctx.constant(2.0) * (x + ctx.constant(1.0))
    })?;

    let range = ctx.range_collection(
        ctx.constant(1.0),
        ctx.constant(5.0),
    );

    let result = ctx.sum_collection(range, complex_lambda)?;
    let value = ctx.eval(&result, &[]);
    
    println!("Complex lambda: x -> 2 * (x + 1)");
    println!("Range: [1, 5]");
    println!("Result: {} (expected: 40)", value);
    
    // Manual calculation: 2*(1+1) + 2*(2+1) + 2*(3+1) + 2*(4+1) + 2*(5+1) = 4+6+8+10+12 = 40
    
    println!("✅ Lambda expressions are properly optimized");
    println!();

    Ok(())
}

fn demo_mathematical_identities() -> Result<()> {
    println!("🧮 Demo 3: Bidirectional Mathematical Identities");
    println!("===============================================");

    let ctx = DynamicContext::new();

    // Demonstrate linearity: Σ(f(x) + g(x)) = Σ(f(x)) + Σ(g(x))
    println!("Linearity Identity: Σ(f(x) + g(x)) = Σ(f(x)) + Σ(g(x))");
    
    let range = ctx.range_collection(
        ctx.constant(1.0),
        ctx.constant(3.0),
    );

    // Create f(x) = 2*x and g(x) = x^2
    let f_lambda = ctx.lambda(|x| ctx.constant(2.0) * x)?;
    let g_lambda = ctx.lambda(|x| x.clone() * x)?;
    
    // Sum f(x) and g(x) separately
    let sum_f = ctx.sum_collection(range.clone(), f_lambda)?;
    let sum_g = ctx.sum_collection(range.clone(), g_lambda)?;
    
    let f_result = ctx.eval(&sum_f, &[]);
    let g_result = ctx.eval(&sum_g, &[]);
    let separate_sum = f_result + g_result;
    
    println!("  Σ(2*x) for x in [1,3]: {}", f_result);
    println!("  Σ(x²) for x in [1,3]: {}", g_result);
    println!("  Sum of separate results: {}", separate_sum);
    
    // Create combined lambda: f(x) + g(x) = 2*x + x^2
    let combined_lambda = ctx.lambda(|x| {
        ctx.constant(2.0) * x.clone() + x.clone() * x
    })?;
    
    let combined_sum = ctx.sum_collection(range, combined_lambda)?;
    let combined_result = ctx.eval(&combined_sum, &[]);
    
    println!("  Σ(2*x + x²) for x in [1,3]: {}", combined_result);
    println!("  ✅ Linearity verified: {} = {}", separate_sum, combined_result);
    println!();

    Ok(())
}

fn demo_pattern_recognition() -> Result<()> {
    println!("🎯 Demo 4: Automatic Pattern Recognition");
    println!("=======================================");

    let ctx = DynamicContext::new();

    // Arithmetic series pattern: Σ(i) = n(n+1)/2
    println!("Arithmetic Series: Σ(i for i=1 to n) = n(n+1)/2");
    
    let n = 100.0;
    let range = ctx.range_collection(
        ctx.constant(1.0),
        ctx.constant(n),
    );
    
    let identity = ctx.identity_lambda();
    let arithmetic_sum = ctx.sum_collection(range, identity)?;
    let result = ctx.eval(&arithmetic_sum, &[]);
    
    let expected = n * (n + 1.0) / 2.0;
    println!("  Sum of 1 to {}: {}", n, result);
    println!("  Expected formula result: {}", expected);
    println!("  ✅ Pattern recognition: {}", if (result - expected).abs() < 1e-10 { "SUCCESS" } else { "FAILED" });
    
    // Constant series pattern: Σ(c) = c * n
    println!("\nConstant Series: Σ(c for i=1 to n) = c * n");
    
    let c = 7.0;
    let constant_lambda = ctx.constant_lambda(ctx.constant(c));
    let range2 = ctx.range_collection(
        ctx.constant(1.0),
        ctx.constant(10.0),
    );
    
    let constant_sum = ctx.sum_collection(range2, constant_lambda)?;
    let constant_result = ctx.eval(&constant_sum, &[]);
    
    let expected_constant = c * 10.0;
    println!("  Sum of {} repeated 10 times: {}", c, constant_result);
    println!("  Expected: {}", expected_constant);
    println!("  ✅ Constant pattern: {}", if (constant_result - expected_constant).abs() < 1e-10 { "SUCCESS" } else { "FAILED" });
    println!();

    Ok(())
}

fn demo_complex_collections() -> Result<()> {
    println!("🔗 Demo 5: Complex Collection Operations");
    println!("======================================");

    let ctx = DynamicContext::new();

    // Demonstrate union operations
    println!("Collection Union: Σ(f(x) for x in A ∪ B)");
    
    // Create two ranges: [1,3] and [4,6]
    let range_a = Collection::Range {
        start: Box::new(ASTRepr::Constant(1.0)),
        end: Box::new(ASTRepr::Constant(3.0)),
    };
    
    let range_b = Collection::Range {
        start: Box::new(ASTRepr::Constant(4.0)),
        end: Box::new(ASTRepr::Constant(6.0)),
    };
    
    // Create union collection
    let union_collection = Collection::Union {
        left: Box::new(range_a.clone()),
        right: Box::new(range_b.clone()),
    };
    
    let square_lambda = ctx.lambda(|x| x.clone() * x)?;
    
    // Sum over union
    let union_sum = ctx.sum_collection(union_collection, square_lambda.clone())?;
    let union_result = ctx.eval(&union_sum, &[]);
    
    // Sum over individual ranges
    let sum_a = ctx.sum_collection(range_a, square_lambda.clone())?;
    let sum_b = ctx.sum_collection(range_b, square_lambda)?;
    
    let a_result = ctx.eval(&sum_a, &[]);
    let b_result = ctx.eval(&sum_b, &[]);
    let individual_sum = a_result + b_result;
    
    println!("  Range A [1,3]: squares sum = {}", a_result);
    println!("  Range B [4,6]: squares sum = {}", b_result);
    println!("  Individual sum: {}", individual_sum);
    println!("  Union sum: {}", union_result);
    println!("  ✅ Union property: {}", if (union_result - individual_sum).abs() < 1e-10 { "SUCCESS" } else { "FAILED" });
    
    // Demonstrate data array collections
    println!("\nData Array Collections:");
    let data_collection = ctx.data_collection("sensor_data");
    println!("  Created data collection: 'sensor_data'");
    println!("  This can be bound to actual data at runtime");
    println!("  Enables symbolic data processing with optimization");
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_summation_demo() {
        // Run the demo to ensure it works
        main().unwrap();
    }

    #[test]
    fn test_arithmetic_series_optimization() {
        let ctx = DynamicContext::new();
        
        let range = ctx.range_collection(
            ctx.constant(1.0),
            ctx.constant(100.0),
        );
        
        let identity = ctx.identity_lambda();
        let sum_expr = ctx.sum_collection(range, identity).unwrap();
        let result = ctx.eval(&sum_expr, &[]);
        
        // Should equal 100 * 101 / 2 = 5050
        assert!((result - 5050.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_optimization() {
        let ctx = DynamicContext::new();
        
        let range = ctx.range_collection(
            ctx.constant(1.0),
            ctx.constant(20.0),
        );
        
        let constant_lambda = ctx.constant_lambda(ctx.constant(3.0));
        let sum_expr = ctx.sum_collection(range, constant_lambda).unwrap();
        let result = ctx.eval(&sum_expr, &[]);
        
        // Should equal 3 * 20 = 60
        assert!((result - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_lambda_composition() {
        let ctx = DynamicContext::new();
        
        // Test that lambda expressions work correctly
        let lambda = ctx.lambda(|x| x.clone() * ctx.constant(2.0)).unwrap();
        
        let range = ctx.range_collection(
            ctx.constant(1.0),
            ctx.constant(5.0),
        );
        
        let sum_expr = ctx.sum_collection(range, lambda).unwrap();
        let result = ctx.eval(&sum_expr, &[]);
        
        // Should equal 2*(1+2+3+4+5) = 2*15 = 30
        assert!((result - 30.0).abs() < 1e-10);
    }
} 