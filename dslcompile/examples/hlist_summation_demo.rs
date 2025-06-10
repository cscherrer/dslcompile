use dslcompile::prelude::*;
use dslcompile::ast::{ASTRepr, Collection};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("🔧 HList Summation Demo");
    println!("======================");
    println!("Demonstrating the new unified approach that eliminates DataArray");
    println!("and treats all inputs as typed HList variables.");
    println!();

    // Create context
    let mut ctx = DynamicContext::<f64>::new();

    // 1. Mathematical Range Summation (no DataArray needed)
    println!("📊 Mathematical Range Summation:");
    println!("--------------------------------");
    
    let sum_expr = ctx.sum_hlist(1..=5, |i| i * 2.0);
    
    println!("✅ Created: sum_hlist(1..=5, |i| i * 2.0)");
    println!("   Expected result: (1+2+3+4+5) * 2 = 30");
    println!("   Expression: {}", sum_expr.pretty_print());
    
    // Analyze the AST to confirm it's using Range, not DataArray
    match sum_expr.as_ast() {
        ASTRepr::Sum(collection) => {
            match collection.as_ref() {
                Collection::Map { collection: inner, .. } => {
                    match inner.as_ref() {
                        Collection::Range { .. } => {
                            println!("✅ Correct: Uses Collection::Range (not DataArray)");
                        }
                        Collection::DataArray(idx) => {
                            println!("❌ Problem: Still uses DataArray({idx})");
                        }
                        other => {
                            println!("❓ Other: {other:?}");
                        }
                    }
                }
                other => {
                    println!("❓ Collection structure: {other:?}");
                }
            }
        }
        other => {
            println!("❌ Wrong AST type: {other:?}");
        }
    }

    println!();
    println!("🎯 Key Benefits of HList Approach:");
    println!("- No artificial DataArray vs Variable distinction");
    println!("- All inputs treated as typed HList variables"); 
    println!("- Code generation produces proper typed function signatures");
    println!("- Eliminates Vec<f64> flattening anti-pattern");
    println!("- Zero-cost heterogeneous operations");

    Ok(())
} 