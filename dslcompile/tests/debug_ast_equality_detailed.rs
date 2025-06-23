//! Detailed debugging of AST equality issue with Sum expressions

use dslcompile::{
    ast::{
        ast_repr::{ASTRepr, Collection, Lambda},
        multiset::MultiSet,
    },
    prelude::*,
};

#[test]
fn debug_sum_ast_equality_detailed() {
    println!("=== DETAILED AST EQUALITY DEBUGGING ===");

    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Create two different data arrays
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    println!("Data1: {:?}", data1);
    println!("Data2: {:?}", data2);

    // Create individual sums
    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);

    // Get their ASTs
    let ast1 = ctx.to_ast(&sum1);
    let ast2 = ctx.to_ast(&sum2);

    println!("\n=== AST STRUCTURE ANALYSIS ===");

    // Detailed inspection of AST1
    if let ASTRepr::Sum(coll1) = &ast1 {
        if let Collection::Map {
            lambda: lambda1,
            collection: inner1,
        } = coll1.as_ref()
        {
            println!("AST1 Lambda: {:?}", lambda1);
            if let Collection::Constant(data1_ast) = inner1.as_ref() {
                println!("AST1 DataArray: {:?}", data1_ast);
            }
        }
    }

    // Detailed inspection of AST2
    if let ASTRepr::Sum(coll2) = &ast2 {
        if let Collection::Map {
            lambda: lambda2,
            collection: inner2,
        } = coll2.as_ref()
        {
            println!("AST2 Lambda: {:?}", lambda2);
            if let Collection::Constant(data2_ast) = inner2.as_ref() {
                println!("AST2 DataArray: {:?}", data2_ast);
            }
        }
    }

    println!("\n=== EQUALITY CHECKS ===");
    println!("AST1 == AST2: {}", ast1 == ast2);

    // Test what happens when we create a compound expression
    let compound = &sum1 + &sum2;
    let compound_ast = ctx.to_ast(&compound);

    println!("\n=== COMPOUND ANALYSIS ===");
    if let ASTRepr::Add(terms) = &compound_ast {
        println!("Compound has {} terms", terms.len());
        println!("Compound has {} distinct terms", terms.distinct_len());

        for (i, (term, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}", i, count);
            if let ASTRepr::Sum(coll) = term {
                if let Collection::Map {
                    lambda: _,
                    collection: inner,
                } = coll.as_ref()
                {
                    if let Collection::Constant(data) = inner.as_ref() {
                        println!("  Data: {:?}", data);
                    }
                }
            }
        }
    }

    // The critical test - this should be 2 if the bug is fixed
    if let ASTRepr::Add(terms) = &compound_ast {
        assert_eq!(
            terms.distinct_len(),
            2,
            "Should have 2 distinct terms, but MultiSet is merging them!"
        );
    }

    println!("\n=== MANUAL MULTISET TEST ===");
    let mut manual_multiset = MultiSet::new();
    manual_multiset.insert(ast1.clone());
    manual_multiset.insert(ast2.clone());

    println!("Manual MultiSet length: {}", manual_multiset.len());
    println!(
        "Manual MultiSet distinct length: {}",
        manual_multiset.distinct_len()
    );

    assert_eq!(
        manual_multiset.distinct_len(),
        2,
        "Manual MultiSet should have 2 distinct elements"
    );
}
