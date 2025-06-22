//! Test to check MultiSet equality behavior for Sum expressions

use dslcompile::prelude::*;

#[test]
fn debug_multiset_equality() {
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();

    // Create two different data arrays
    let data1 = vec![1.0, 2.0];
    let data2 = vec![3.0, 4.0];

    // Create individual sums
    let sum1 = ctx.sum(data1.clone(), |item| &x * item);
    let sum2 = ctx.sum(data2.clone(), |item| &x * item);

    let sum1_ast = ctx.to_ast(&sum1);
    let sum2_ast = ctx.to_ast(&sum2);

    println!("=== EQUALITY CHECKS ===");
    println!("sum1_ast == sum2_ast: {}", sum1_ast == sum2_ast);
    println!("sum1_ast: {:?}", sum1_ast);
    println!("sum2_ast: {:?}", sum2_ast);

    // Test MultiSet behavior directly
    use dslcompile::ast::multiset::MultiSet;
    let mut multiset = MultiSet::new();
    multiset.insert(sum1_ast.clone());
    multiset.insert(sum2_ast.clone());

    println!("\n=== MULTISET BEHAVIOR ===");
    println!("multiset.len(): {}", multiset.len());
    println!("multiset unique count: {}", multiset.iter().count());

    for (i, (expr, count)) in multiset.iter().enumerate() {
        println!("Entry {}: count={}", i, count);
        if let dslcompile::ast::ast_repr::ASTRepr::Sum(collection) = expr {
            match collection.as_ref() {
                dslcompile::ast::ast_repr::Collection::Map {
                    lambda: _,
                    collection: inner,
                } => match inner.as_ref() {
                    dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                        println!("  Data: {:?}", data);
                    }
                    _ => println!("  Not a DataArray"),
                },
                _ => println!("  Not a Map collection"),
            }
        }
    }

    // Test add_binary result
    let compound_ast = dslcompile::ast::ast_repr::ASTRepr::add_binary(sum1_ast, sum2_ast);
    println!("\n=== ADD_BINARY RESULT ===");
    if let dslcompile::ast::ast_repr::ASTRepr::Add(terms) = &compound_ast {
        println!("Compound terms count: {}", terms.len());
        for (i, (expr, count)) in terms.iter().enumerate() {
            println!("Term {}: count={}", i, count);
            if let dslcompile::ast::ast_repr::ASTRepr::Sum(collection) = expr {
                match collection.as_ref() {
                    dslcompile::ast::ast_repr::Collection::Map {
                        lambda: _,
                        collection: inner,
                    } => match inner.as_ref() {
                        dslcompile::ast::ast_repr::Collection::DataArray(data) => {
                            println!("  Data: {:?}", data);
                        }
                        _ => println!("  Not a DataArray"),
                    },
                    _ => println!("  Not a Map collection"),
                }
            }
        }
    }
}
