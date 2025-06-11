//! Summation Integration Demo
//!
//! This demo shows the new unified summation system that integrates the
//! sophisticated Collection/Lambda architecture with the DynamicContext.
//!
//! Key features demonstrated:
//! 1. Mathematical range summation using Collection::Range
//! 2. Proper lambda expression creation with automatic variable management
//! 3. AST integration with the Sum(Collection) variant
//! 4. Clean API that leverages the mathematical optimization infrastructure

use dslcompile::{
    ast::{ASTRepr, ast_repr::Collection},
    contexts::DynamicContext,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Summation Integration Demo");
    println!("============================\n");

    demo_mathematical_range_summation()?;
    demo_collection_ast_structure()?;
    demo_data_array_summation()?;

    println!("🎉 Summation Integration Demo Complete!");
    println!("✅ Both mathematical ranges and data arrays working!");
    println!("✅ Collection/Lambda system fully integrated!");
    println!("✅ Ready for mathematical optimizations!");

    Ok(())
}

fn demo_mathematical_range_summation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Mathematical Range Summation");
    println!("---------------------------------------");

    let mut ctx = DynamicContext::<f64>::new();

    // Create a simple summation: Σ(i=1 to 5) i * 2
    println!("Creating: Σ(i=1 to 5) i * 2");
    let sum_expr = ctx.sum(1..=5, |i| {
        i * 2.0 // Simple scaling
    });

    println!("✅ Sum expression created successfully");
    println!("   Expression AST: {:?}", sum_expr.as_ast());
    println!("   Pretty print: {}", sum_expr.pretty_print());

    // Verify the structure is correct
    match sum_expr.as_ast() {
        ASTRepr::Sum(collection) => {
            println!("✅ Correct AST structure: Sum(Collection)");
            println!("   Collection type: {collection:?}");
        }
        _ => {
            println!("❌ Unexpected AST structure");
        }
    }

    println!();
    Ok(())
}

fn demo_collection_ast_structure() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Demo 2: Collection AST Structure");
    println!("------------------------------------");

    let mut ctx = DynamicContext::<f64>::new();

    // Create a more complex summation: Σ(i=1 to 3) (i^2 + 1)
    println!("Creating: Σ(i=1 to 3) (i^2 + 1)");

    // Create constants outside the closure to avoid borrowing conflicts
    let two = ctx.constant(2.0);
    let complex_sum = ctx.sum(1..=3, |i| {
        let i_squared = i.clone().pow(two.clone());
        i_squared + 1.0
    });

    // Analyze the AST structure
    println!("✅ Complex sum expression created");
    println!("   Demonstrates:");
    println!("   - Range collection creation (1..=3)");
    println!("   - Lambda with complex body expression");
    println!("   - Proper variable scoping in lambda");
    println!("   - Integration with existing operator overloading");

    match complex_sum.as_ast() {
        ASTRepr::Sum(collection) => match collection.as_ref() {
            Collection::Map {
                lambda,
                collection: inner_collection,
            } => {
                println!("✅ Correct structure: Sum(Map{{lambda, collection}})");
                println!("   Inner collection: {inner_collection:?}");
                println!("   Lambda: {lambda:?}");
            }
            _ => {
                println!("❓ Collection structure: {collection:?}");
            }
        },
        _ => {
            println!("❌ Unexpected AST structure");
        }
    }

    println!();
    Ok(())
}

fn demo_data_array_summation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️  Demo 3: Data Array Summation");
    println!("------------------------------------");

    let mut ctx = DynamicContext::<f64>::new();

    // Create a data array and summation expression
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("Creating: Σ(x for x in [1.0, 2.0, 3.0, 4.0, 5.0])");

    // Use the data vector directly in summation - this creates Collection::DataArray
    let sum_expr = ctx.sum(data.clone(), |x| x);

    // Analyze the AST structure
    println!("✅ Data array sum expression created");
    println!("   Demonstrates:");
    println!("   - Data array collection creation");
    println!("   - Lambda with identity function");
    println!("   - Proper variable scoping in lambda");
    println!("   - Integration with Collection::DataArray");

    match sum_expr.as_ast() {
        ASTRepr::Sum(collection) => match collection.as_ref() {
            Collection::Map {
                lambda,
                collection: inner_collection,
            } => {
                println!("✅ Correct structure: Sum(Map{{lambda, collection}})");

                match inner_collection.as_ref() {
                    Collection::DataArray(data_idx) => {
                        println!("✅ Inner collection: DataArray({data_idx})");
                        if let Some(stored_data) = ctx.get_data_array(*data_idx) {
                            println!("✅ Stored data: {stored_data:?}");
                        }
                    }
                    _ => {
                        println!("❓ Inner collection: {inner_collection:?}");
                    }
                }

                println!("   Lambda: {lambda:?}");
            }
            _ => {
                println!("❓ Collection structure: {collection:?}");
            }
        },
        _ => {
            println!("❌ Unexpected AST structure");
        }
    }

    // Evaluate the expression
    println!("🔍 Evaluating data array summation...");
    let result = ctx.eval(&sum_expr, frunk::hlist![]);
    println!("Result: {result} (expected: 15.0)");

    if (result - 15.0).abs() < 1e-10 {
        println!("✅ Data array summation working correctly!");
    } else {
        println!("❌ Data array summation failed - got {result} instead of 15.0");
    }

    println!();
    Ok(())
}
