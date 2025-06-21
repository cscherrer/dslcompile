//! Comprehensive test for DataArray equality in various scenarios

use dslcompile::prelude::*;
use dslcompile::ast::{ast_repr::{ASTRepr, Collection, Lambda}, multiset::MultiSet};

#[test]
fn test_comprehensive_dataarray_equality() {
    println!("Testing comprehensive DataArray equality scenarios...");
    
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    
    // Scenario 1: Same lambda, different data (should be different)
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];
    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    
    test_distinct_sums(&ctx, &sum1, &sum2, "Scenario 1: Same lambda, different data");
    
    // Scenario 2: Different lambdas, same data (should be different)
    let data3 = vec![1.0, 2.0];
    let sum3 = ctx.sum(data3.clone(), |item| &x * item);
    let sum4 = ctx.sum(data3.clone(), |item| &x + item);
    
    test_distinct_sums(&ctx, &sum3, &sum4, "Scenario 2: Different lambdas, same data");
    
    // Scenario 3: Same lambda, same data (should be equal)
    let data4 = vec![1.0, 2.0];
    let sum5 = ctx.sum(data4.clone(), |item| &x * item);
    let sum6 = ctx.sum(data4.clone(), |item| &x * item);
    
    test_identical_sums(&ctx, &sum5, &sum6, "Scenario 3: Same lambda, same data");
    
    // Scenario 4: Test with addition of multiple sums with different data
    let compound = &sum1 + &sum2;
    test_compound_expression(&ctx, &compound, "Scenario 4: Addition of sums with different data");
    
    // Scenario 5: Complex lambda with different data
    let complex_data1 = vec![1.0, 2.0, 3.0];
    let complex_data2 = vec![4.0, 5.0, 6.0];
    let complex_sum1 = ctx.sum(complex_data1.clone(), |item| (&x * item) + (&x * &x));
    let complex_sum2 = ctx.sum(complex_data2.clone(), |item| (&x * item) + (&x * &x));
    
    test_distinct_sums(&ctx, &complex_sum1, &complex_sum2, "Scenario 5: Complex lambda, different data");
    
    println!("All comprehensive tests completed!");
}

fn test_distinct_sums<const S: usize>(
    ctx: &DynamicContext<S>,
    sum1: &DynamicExpr<f64, S>,
    sum2: &DynamicExpr<f64, S>,
    scenario_name: &str,
) {
    println!("\n--- {} ---", scenario_name);
    
    let ast1 = ctx.to_ast(sum1);
    let ast2 = ctx.to_ast(sum2);
    
    println!("AST1 == AST2: {}", ast1 == ast2);
    
    let mut multiset = MultiSet::new();
    multiset.insert(ast1.clone());
    multiset.insert(ast2.clone());
    
    println!("MultiSet distinct length: {}", multiset.distinct_len());
    
    assert_ne!(ast1, ast2, "{}: ASTs should be different", scenario_name);
    assert_eq!(multiset.distinct_len(), 2, "{}: MultiSet should contain 2 distinct elements", scenario_name);
}

fn test_identical_sums<const S: usize>(
    ctx: &DynamicContext<S>,
    sum1: &DynamicExpr<f64, S>,
    sum2: &DynamicExpr<f64, S>,
    scenario_name: &str,
) {
    println!("\n--- {} ---", scenario_name);
    
    let ast1 = ctx.to_ast(sum1);
    let ast2 = ctx.to_ast(sum2);
    
    println!("AST1 == AST2: {}", ast1 == ast2);
    
    let mut multiset = MultiSet::new();
    multiset.insert(ast1.clone());
    multiset.insert(ast2.clone());
    
    println!("MultiSet distinct length: {}", multiset.distinct_len());
    
    assert_eq!(ast1, ast2, "{}: ASTs should be identical", scenario_name);
    assert_eq!(multiset.distinct_len(), 1, "{}: MultiSet should contain 1 element (merged)", scenario_name);
}

fn test_compound_expression<const S: usize>(
    ctx: &DynamicContext<S>,
    compound: &DynamicExpr<f64, S>,
    scenario_name: &str,
) {
    println!("\n--- {} ---", scenario_name);
    
    let compound_ast = ctx.to_ast(compound);
    
    if let ASTRepr::Add(terms) = &compound_ast {
        println!("Compound expression has {} distinct terms", terms.distinct_len());
        
        for (i, (sum_expr, multiplicity)) in terms.iter_with_multiplicity().enumerate() {
            println!("Term {}: multiplicity={:?}", i, multiplicity);
            if let ASTRepr::Sum(collection) = sum_expr {
                if let Collection::Map { lambda: _, collection: inner } = collection.as_ref() {
                    if let Collection::DataArray(data) = inner.as_ref() {
                        println!("  Data: {:?}", data);
                    }
                }
            }
        }
        
        assert_eq!(terms.distinct_len(), 2, "{}: Should have 2 distinct terms", scenario_name);
    } else {
        panic!("{}: Expected Add expression", scenario_name);
    }
}

#[test] 
fn test_dataarray_specific_edge_cases() {
    println!("Testing DataArray specific edge cases...");
    
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    
    // Edge case 1: Empty arrays
    let empty1 = vec![];
    let empty2 = vec![];
    let sum_empty1 = ctx.sum(empty1, |item| &x * item);
    let sum_empty2 = ctx.sum(empty2, |item| &x * item);
    
    let ast_empty1 = ctx.to_ast(&sum_empty1);
    let ast_empty2 = ctx.to_ast(&sum_empty2);
    
    println!("Empty arrays equal: {}", ast_empty1 == ast_empty2);
    assert_eq!(ast_empty1, ast_empty2, "Empty arrays should be equal");
    
    // Edge case 2: Arrays with same values but different order
    let ordered1 = vec![1.0, 2.0, 3.0];
    let ordered2 = vec![3.0, 2.0, 1.0];
    let sum_ordered1 = ctx.sum(ordered1, |item| &x * item);
    let sum_ordered2 = ctx.sum(ordered2, |item| &x * item);
    
    let ast_ordered1 = ctx.to_ast(&sum_ordered1);
    let ast_ordered2 = ctx.to_ast(&sum_ordered2);
    
    println!("Different order arrays equal: {}", ast_ordered1 == ast_ordered2);
    assert_ne!(ast_ordered1, ast_ordered2, "Different order arrays should NOT be equal");
    
    // Edge case 3: Arrays with duplicate values  
    let duplicates1 = vec![1.0, 1.0, 2.0];
    let duplicates2 = vec![1.0, 2.0, 2.0];
    let sum_dup1 = ctx.sum(duplicates1, |item| &x * item);
    let sum_dup2 = ctx.sum(duplicates2, |item| &x * item);
    
    let ast_dup1 = ctx.to_ast(&sum_dup1);
    let ast_dup2 = ctx.to_ast(&sum_dup2);
    
    println!("Different duplicates equal: {}", ast_dup1 == ast_dup2);
    assert_ne!(ast_dup1, ast_dup2, "Arrays with different duplicate patterns should NOT be equal");
    
    println!("All edge case tests passed!");
}