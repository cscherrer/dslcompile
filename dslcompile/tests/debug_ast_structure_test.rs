//! Debug test to examine AST structure

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

#[cfg(feature = "optimization")]
#[test]
fn debug_ast_structure() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Create the failing case
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    let compound = &sum1 + &sum2;

    println!("=== AST STRUCTURE DEBUG ===");

    // Create all ASTs first
    let compound_ast = ctx.to_ast(&compound);
    let sum1_ast = ctx.to_ast(&sum1);
    let sum2_ast = ctx.to_ast(&sum2);

    // Print original AST structure
    println!("Original AST: {:#?}", compound_ast);

    let test_x = 2.0;
    let original_result = ctx.eval(&compound, hlist![test_x]);
    println!("Original evaluation: {}", original_result);

    // Convert and print optimized AST structure
    let compound_result = optimize_simple_sum_splitting(&compound_ast).unwrap();
    println!("Optimized AST: {:#?}", compound_result);

    let optimized_result = compound_result.eval_with_vars(&[test_x]);
    println!("Optimized evaluation: {}", optimized_result);

    // Let's also test the individual sum conversions to see if they create unique data IDs
    println!("\n=== INDIVIDUAL CONVERSIONS ===");
    let sum1_result = optimize_simple_sum_splitting(&sum1_ast).unwrap();
    let sum2_result = optimize_simple_sum_splitting(&sum2_ast).unwrap();
    println!("Sum1 conversion: {:#?}", sum1_result);
    println!("Sum2 conversion: {:#?}", sum2_result);

    // Print the individual sum ASTs for comparison
    println!("Sum1 AST: {:#?}", sum1_ast);
    println!("Sum2 AST: {:#?}", sum2_ast);
}
