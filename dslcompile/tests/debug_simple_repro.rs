//! Simple repro test to isolate the data corruption issue

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

#[cfg(feature = "optimization")]
#[test]
fn debug_simple_repro() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Create two different data arrays
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    println!("=== ORIGINAL DATA ===");
    println!("data1: {:?}", data1);
    println!("data2: {:?}", data2);

    // Create individual sums
    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);

    println!("\n=== INDIVIDUAL EVALUATION ===");
    let test_x = 2.0;
    let sum1_eval = ctx.eval(&sum1, hlist![test_x]);
    let sum2_eval = ctx.eval(&sum2, hlist![test_x]);
    println!("sum1 eval: {} (expected: {})", sum1_eval, test_x * 3.0);
    println!("sum2 eval: {} (expected: {})", sum2_eval, test_x * 7.0);

    // Create compound manually to avoid any issues
    let compound = &sum1 + &sum2;
    let compound_eval = ctx.eval(&compound, hlist![test_x]);
    println!(
        "compound eval: {} (expected: {})",
        compound_eval,
        test_x * 10.0
    );

    // Check the AST before optimization
    let compound_ast = ctx.to_ast(&compound);
    println!("\n=== COMPOUND AST DETAILS ===");

    if let dslcompile::ast::ast_repr::ASTRepr::Add(terms) = &compound_ast {
        for (i, (sum_expr, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}", i, count);
            if let dslcompile::ast::ast_repr::ASTRepr::Sum(collection) = sum_expr {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map {
                        lambda,
                        collection: inner,
                    } => {
                        println!("  Lambda: {:?}", lambda);
                        match inner.as_ref() {
                            dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                                println!("  Data: {:?}", data);
                            }
                            _ => println!("  Not a DataArray: {:?}", inner),
                        }
                    }
                    _ => println!("  Not a Map collection: {:?}", collection),
                }
            } else {
                println!("  Not a Sum: {:?}", sum_expr);
            }
        }
    } else {
        println!("Not an Add: {:?}", compound_ast);
    }

    // Now test optimization
    println!("\n=== OPTIMIZATION ===");
    let result = optimize_simple_sum_splitting(&compound_ast).unwrap();
    let optimized_eval = result.eval_with_vars(&[test_x]);
    println!(
        "optimized eval: {} (expected: {})",
        optimized_eval,
        test_x * 10.0
    );

    assert!(
        (optimized_eval - test_x * 10.0).abs() < 1e-10,
        "Optimization should preserve semantics: {} vs {}",
        optimized_eval,
        test_x * 10.0
    );
}
