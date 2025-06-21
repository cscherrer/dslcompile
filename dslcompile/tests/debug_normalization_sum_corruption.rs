//! Test normalization of compound Add expressions with Sum terms

use dslcompile::prelude::*;
use dslcompile::ast::{ast_repr::ASTRepr, normalization::normalize};

#[test]
fn test_normalization_sum_corruption() {
    println!("=== TESTING NORMALIZATION OF SUM EXPRESSIONS ===");
    
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    
    // Create two different data arrays
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];
    
    println!("Original data1: {:?}", data1);
    println!("Original data2: {:?}", data2);
    
    // Create individual sums  
    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);
    
    // Create compound expression
    let compound = &sum1 + &sum2;
    let compound_ast = ctx.to_ast(&compound);
    
    println!("\n=== BEFORE NORMALIZATION ===");
    if let ASTRepr::Add(terms) = &compound_ast {
        println!("Terms count: {}", terms.len());
        println!("Distinct terms: {}", terms.distinct_len());
        
        for (i, (term, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}", i, count);
            if let ASTRepr::Sum(collection) = term {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map { lambda: _, collection: inner } => {
                        match inner.as_ref() {
                            dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                                println!("  Data: {:?}", data);
                            }
                            _ => println!("  Not DataArray"),
                        }
                    }
                    _ => println!("  Not Map"),
                }
            }
        }
    }
    
    // Now test normalization
    println!("\n=== APPLYING NORMALIZATION ===");
    let normalized = normalize(&compound_ast);
    
    println!("\n=== AFTER NORMALIZATION ===");
    if let ASTRepr::Add(terms) = &normalized {
        println!("Terms count: {}", terms.len());
        println!("Distinct terms: {}", terms.distinct_len());
        
        for (i, (term, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}", i, count);
            if let ASTRepr::Sum(collection) = term {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map { lambda: _, collection: inner } => {
                        match inner.as_ref() {
                            dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                                println!("  Data: {:?}", data);
                            }
                            _ => println!("  Not DataArray"),
                        }
                    }
                    _ => println!("  Not Map"),
                }
            }
        }
        
        // Critical assertion - normalization should preserve distinct terms
        assert_eq!(terms.distinct_len(), 2, 
            "Normalization corrupted the MultiSet! Should have 2 distinct terms, got {}", 
            terms.distinct_len());
    } else {
        panic!("Expected Add expression after normalization");
    }
    
    println!("\n=== NORMALIZATION TEST PASSED ===");
}