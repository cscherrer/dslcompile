//! Test to debug AST creation process

use dslcompile::prelude::*;

#[test]
fn debug_ast_creation_process() {
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
    
    // Get individual ASTs
    let sum1_ast = ctx.to_ast(&sum1);
    let sum2_ast = ctx.to_ast(&sum2);
    
    println!("\n=== INDIVIDUAL ASTS ===");
    println!("sum1_ast == sum2_ast: {}", sum1_ast == sum2_ast);
    println!("sum1_ast: {:?}", sum1_ast);
    println!("sum2_ast: {:?}", sum2_ast);
    
    // Test direct add_binary 
    println!("\n=== DIRECT ADD_BINARY ===");
    let direct_compound = dslcompile::ast::ast_repr::ASTRepr::add_binary(sum1_ast.clone(), sum2_ast.clone());
    if let dslcompile::ast::ast_repr::ASTRepr::Add(terms) = &direct_compound {
        println!("Direct compound terms count: {}", terms.len());
        for (i, (expr, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}", i, count);
            if let dslcompile::ast::ast_repr::ASTRepr::Sum(collection) = expr {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map { lambda: _, collection: inner } => {
                        match inner.as_ref() {
                            dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                                println!("  Data: {:?}", data);
                            }
                            _ => println!("  Not a DataArray"),
                        }
                    }
                    _ => println!("  Not a Map collection"),
                }
            }
        }
    }
    
    // Now test the DynamicContext addition
    println!("\n=== DYNAMIC CONTEXT ADDITION ===");
    let compound = &sum1 + &sum2;
    let context_compound = ctx.to_ast(&compound);
    
    if let dslcompile::ast::ast_repr::ASTRepr::Add(terms) = &context_compound {
        println!("Context compound terms count: {}", terms.len());
        for (i, (expr, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}", i, count);
            if let dslcompile::ast::ast_repr::ASTRepr::Sum(collection) = expr {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map { lambda: _, collection: inner } => {
                        match inner.as_ref() {
                            dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                                println!("  Data: {:?}", data);
                            }
                            _ => println!("  Not a DataArray"),
                        }
                    }
                    _ => println!("  Not a Map collection"),
                }
            }
        }
    }
    
    // Compare the ASTs we already extracted
    println!("\n=== AST COMPARISON ===");
    println!("Already compared above: sum1_ast == sum2_ast: {}", sum1_ast == sum2_ast);
    
    // Test normalization step
    println!("\n=== NORMALIZATION TEST ===");
    let normalized = dslcompile::ast::normalization::normalize(&context_compound);
    if let dslcompile::ast::ast_repr::ASTRepr::Add(terms) = &normalized {
        println!("Normalized compound terms count: {}", terms.len());
        for (i, (expr, count)) in terms.iter().enumerate() {
            println!("Normalized Term {}: count={}", i, count);
            if let dslcompile::ast::ast_repr::ASTRepr::Sum(collection) = expr {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map { lambda: _, collection: inner } => {
                        match inner.as_ref() {
                            dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                                println!("  Normalized Data: {:?}", data);
                            }
                            _ => println!("  Not a DataArray"),
                        }
                    }
                    _ => println!("  Not a Map collection"),
                }
            }
        }
    }
}