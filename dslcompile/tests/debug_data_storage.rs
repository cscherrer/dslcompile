//! Debug test to check data storage mapping

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

#[cfg(feature = "optimization")]
#[test]
fn debug_data_storage_mapping() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Create two sums with different data
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    let compound = &sum1 + &sum2;

    println!("=== INDIVIDUAL SUMS ===");

    // Test sum1 alone
    let sum1_ast = ctx.to_ast(&sum1);
    println!("Sum1 AST: {:?}", sum1_ast);

    // Test sum2 alone
    let sum2_ast = ctx.to_ast(&sum2);
    println!("Sum2 AST: {:?}", sum2_ast);

    // Test compound
    let compound_ast = ctx.to_ast(&compound);
    println!("Compound AST: {:?}", compound_ast);

    // Let's also check what data is in each sum within compound
    if let dslcompile::ast::ast_repr::ASTRepr::Add(terms) = &compound_ast {
        for (i, (sum_expr, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}, expr={:?}", i, count, sum_expr);
            if let dslcompile::ast::ast_repr::ASTRepr::Sum(collection) = sum_expr {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map {
                        lambda: _,
                        collection: inner,
                    } => match inner.as_ref() {
                        dslcompile::ast::ast_repr::Collection::Constant(data) => {
                            println!("  Data: {:?}", data);
                        }
                        _ => println!("  Not a DataArray"),
                    },
                    _ => println!("  Not a Map collection"),
                }
            }
        }
    }

    // Now test optimization
    println!("\n=== OPTIMIZATION ===");

    let sum1_result = optimize_simple_sum_splitting(&sum1_ast).unwrap();
    println!("Sum1 optimized: {:?}", sum1_result);

    let sum2_result = optimize_simple_sum_splitting(&sum2_ast).unwrap();
    println!("Sum2 optimized: {:?}", sum2_result);

    let compound_result = optimize_simple_sum_splitting(&compound_ast).unwrap();
    println!("Compound optimized: {:?}", compound_result);

    // Test evaluation
    let test_x = 2.0;

    let sum1_eval = sum1_result.eval_with_vars(&[test_x]);
    let sum2_eval = sum2_result.eval_with_vars(&[test_x]);
    let compound_eval = compound_result.eval_with_vars(&[test_x]);

    println!("\n=== EVALUATION ===");
    println!("Sum1 eval: {} (expected: {})", sum1_eval, test_x * 3.0);
    println!("Sum2 eval: {} (expected: {})", sum2_eval, test_x * 7.0);
    println!(
        "Compound eval: {} (expected: {})",
        compound_eval,
        test_x * 10.0
    );

    assert!((sum1_eval - test_x * 3.0).abs() < 1e-10);
    assert!((sum2_eval - test_x * 7.0).abs() < 1e-10);
    assert!((compound_eval - test_x * 10.0).abs() < 1e-10);
}
