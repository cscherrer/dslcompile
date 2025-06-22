//! Test to demonstrate the DataArray equality issue in MultiSet

use dslcompile::{
    ast::{
        ast_repr::{ASTRepr, Collection},
        multiset::MultiSet,
    },
    prelude::*,
};

#[test]
fn test_dataarray_equality_issue() {
    println!("Testing DataArray equality issue in MultiSet...");

    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Create two Sum expressions with identical lambda structures but different data
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0]; // Different data

    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);

    // Get the AST representations
    let ast1 = ctx.to_ast(&sum1);
    let ast2 = ctx.to_ast(&sum2);

    println!("AST1: {:#?}", ast1);
    println!("AST2: {:#?}", ast2);

    // Check if they are equal (they shouldn't be due to different data)
    println!("AST1 == AST2: {}", ast1 == ast2);

    // Now let's see what happens when we put them in a MultiSet
    let mut multiset = MultiSet::new();
    multiset.insert(ast1.clone());
    multiset.insert(ast2.clone());

    println!("MultiSet length: {}", multiset.len());
    println!("MultiSet distinct length: {}", multiset.distinct_len());

    // If the bug exists, distinct_len will be 1 instead of 2
    // because the two Sum expressions are treated as equal

    if multiset.distinct_len() == 1 {
        println!("❌ BUG CONFIRMED: Two Sum expressions with different data are treated as equal!");
        println!("   This means they will be merged in the MultiSet instead of kept separate.");
    } else {
        println!(
            "✅ No bug: Two Sum expressions with different data are correctly treated as distinct."
        );
    }

    // Let's also test the individual Collection equality
    if let (ASTRepr::Sum(coll1), ASTRepr::Sum(coll2)) = (&ast1, &ast2) {
        println!("Collection1 == Collection2: {}", coll1 == coll2);

        // Extract the lambda and data components for detailed comparison
        if let (
            Collection::Map {
                lambda: lambda1,
                collection: coll_inner1,
            },
            Collection::Map {
                lambda: lambda2,
                collection: coll_inner2,
            },
        ) = (coll1.as_ref(), coll2.as_ref())
        {
            println!("Lambda1 == Lambda2: {}", lambda1 == lambda2);
            println!(
                "Inner Collection1 == Inner Collection2: {}",
                coll_inner1 == coll_inner2
            );

            // Extract the actual data arrays
            if let (Collection::DataArray(data1), Collection::DataArray(data2)) =
                (coll_inner1.as_ref(), coll_inner2.as_ref())
            {
                println!("Data1: {:?}", data1);
                println!("Data2: {:?}", data2);
                println!("Data1 == Data2: {}", data1 == data2);
            }
        }
    }

    // The assertions that should pass if the fix is correct
    assert_ne!(
        ast1, ast2,
        "Two Sum expressions with different data should NOT be equal"
    );
    assert_eq!(
        multiset.distinct_len(),
        2,
        "MultiSet should contain 2 distinct elements"
    );
}
