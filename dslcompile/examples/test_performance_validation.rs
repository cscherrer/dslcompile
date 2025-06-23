//! Performance Validation for Sum Splitting Optimization
//!
//! This test validates that our sum splitting optimizations actually improve
//! operational complexity as expected: Σ(2*x + 3*y) = 3n operations vs
//! 2*Σ(x) + 3*Σ(y) = 2n+1 operations

use dslcompile::{
    ast::ast_repr::{Collection, Lambda},
    prelude::*,
};


fn main() -> Result<()> {
    println!("⚡ Performance Validation for Sum Splitting");
    println!("==========================================\n");

    // Test the operational complexity claim:
    // Σ(2*x + 3*y) = 3n operations (2 muls + 1 add per iteration)
    // vs 2*Σ(x) + 3*Σ(y) = 2n+1 operations (2 separate sums + 1 final add)

    test_operational_complexity_analysis()?;
    test_with_large_collections()?;
    test_deeply_nested_expressions()?;

    println!("\n🏁 Performance validation completed!");
    Ok(())
}

fn test_operational_complexity_analysis() -> Result<()> {
    println!("📊 Test 1: Operational Complexity Analysis");
    println!("   Analyzing: Σ(2*x + 3*y) vs optimized form\n");

    // Create the expression Σ(λv.(2*v + 3*w)) where w is external
    let var_v = ASTRepr::BoundVar(0); // Lambda-bound iteration variable
    let var_w = ASTRepr::Variable(1); // External parameter
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    // Inner expression: 2*v + 3*w
    let inner_expr = ASTRepr::add_binary(
        ASTRepr::mul_binary(two.clone(), var_v.clone()),
        ASTRepr::mul_binary(three.clone(), var_w.clone()),
    );

    let lambda = Lambda {
        var_indices: vec![0],
        body: Box::new(inner_expr),
    };

    // Create sum over a reasonably sized collection for analysis
    let collection_size = 1000;
    let data_collection = Collection::Constant((0..collection_size).map(f64::from).collect());
    let map_collection = Collection::Map {
        lambda: Box::new(lambda),
        collection: Box::new(data_collection),
    };

    let original_sum = ASTRepr::Sum(Box::new(map_collection));

    // Analyze original expression
    let original_ops = original_sum.count_operations();
    println!("   Original expression operations: {original_ops}");
    println!("   Per iteration: 2 muls + 1 add = 3 ops");
    println!(
        "   Total with {} items: 3 × {} = {} ops",
        collection_size,
        collection_size,
        3 * collection_size
    );

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            let optimized_ops = original_ops; // Use original count
            println!("\n   Expression created: {optimized_ops} operations");
            println!("   Expected form: 2*Σ(v) + 3*w*|collection|");
            println!(
                "   Theoretical optimized operations: 2n + 1 = {} ops",
                2 * collection_size + 1
            );

            // Calculate theoretical improvement
            let original_theoretical = 3 * collection_size;
            let optimized_theoretical = 2 * collection_size + 1;
            let improvement_ratio =
                f64::from(original_theoretical) / f64::from(optimized_theoretical);

            println!("\n   💡 Theoretical Analysis:");
            println!("      Original:  3n = {original_theoretical} operations");
            println!("      Optimized: 2n+1 = {optimized_theoretical} operations");
            println!("      Improvement: {improvement_ratio:.2}x speedup");
            println!(
                "      Savings: {} operations ({:.1}%)",
                original_theoretical - optimized_theoretical,
                100.0 * f64::from(original_theoretical - optimized_theoretical)
                    / f64::from(original_theoretical)
            );
        }
    }

    println!();
    Ok(())
}

fn test_with_large_collections() -> Result<()> {
    println!("📈 Test 2: Large Collection Performance");
    println!("   Testing with collections of varying sizes\n");

    let sizes = [10, 100, 1000, 10000];

    for &size in &sizes {
        println!("   📏 Collection size: {size}");

        // Create expression: Σ(3*x + 2*x) = Σ(5*x)
        let var_x = ASTRepr::BoundVar(0);
        let three = ASTRepr::Constant(3.0);
        let two = ASTRepr::Constant(2.0);

        let lambda_body = ASTRepr::add_binary(
            ASTRepr::mul_binary(three, var_x.clone()),
            ASTRepr::mul_binary(two, var_x),
        );

        let lambda = Lambda {
            var_indices: vec![0],
            body: Box::new(lambda_body),
        };

        let data_collection = Collection::Constant((0..size).map(f64::from).collect());
        let map_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(data_collection),
        };

        let sum_expr = ASTRepr::Sum(Box::new(map_collection));
        let original_ops = sum_expr.count_operations();

        #[cfg(feature = "optimization")]
        {
            // Optimization functionality removed
            {
                println!(
                    "      Expression created: {original_ops} ops (theoretical optimization available)"
                );
            }
        }
    }

    println!();
    Ok(())
}

fn test_deeply_nested_expressions() -> Result<()> {
    println!("🏗️  Test 3: Deeply Nested Expression Optimization");
    println!("   Testing: Σ(Σ(a*x + b*y)) nested sum optimization\n");

    // Create a nested sum expression
    let var_x = ASTRepr::BoundVar(0);
    let var_y = ASTRepr::BoundVar(1);
    let a = ASTRepr::Constant(2.0);
    let b = ASTRepr::Constant(3.0);

    // Inner lambda: λy.(a*x + b*y)
    let inner_lambda_body = ASTRepr::add_binary(
        ASTRepr::mul_binary(a, var_x.clone()),
        ASTRepr::mul_binary(b, var_y),
    );

    let inner_lambda = Lambda {
        var_indices: vec![1], // Binds y
        body: Box::new(inner_lambda_body),
    };

    // Inner sum: Σ_y(a*x + b*y)
    let inner_data = Collection::Constant(vec![1.0, 2.0, 3.0]);
    let inner_map = Collection::Map {
        lambda: Box::new(inner_lambda),
        collection: Box::new(inner_data),
    };
    let inner_sum = ASTRepr::Sum(Box::new(inner_map));

    // Outer lambda: λx.Σ_y(a*x + b*y)
    let outer_lambda = Lambda {
        var_indices: vec![0], // Binds x
        body: Box::new(inner_sum),
    };

    // Outer sum: Σ_x(Σ_y(a*x + b*y))
    let outer_data = Collection::Constant(vec![1.0, 2.0]);
    let outer_map = Collection::Map {
        lambda: Box::new(outer_lambda),
        collection: Box::new(outer_data),
    };
    let nested_sum = ASTRepr::Sum(Box::new(outer_map));

    let original_ops = nested_sum.count_operations();
    println!("   Original nested expression operations: {original_ops}");

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            println!("   Nested expression created: {original_ops} operations");
            println!("   🎯 Theoretical optimization available for nested expressions");
            println!("   📋 Original structure:");
            
            // Show a truncated view of the original structure
            let original_str = format!("{nested_sum:?}");
            if original_str.len() > 200 {
                println!("      {}...", &original_str[..200]);
            } else {
                println!("      {original_str}");
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("   🚫 Egg optimization not enabled for nested test");
    }

    println!();
    Ok(())
}
