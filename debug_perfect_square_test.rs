//! Debug test for perfect square expansion issue
//! Let's trace exactly what happens when we try to expand (x+y)²

use dslcompile::final_tagless::{ExpressionBuilder, ASTRepr};
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;
use dslcompile::Result;

fn main() -> Result<()> {
    println!("🔧 Debug: Perfect Square Expansion");
    println!("===================================\n");

    // Test 1: Direct AST construction of (x+y)²
    println!("🔬 Test 1: Direct AST (x+y)²");
    let x_plus_y_squared = ASTRepr::Pow(
        Box::new(ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        )),
        Box::new(ASTRepr::Constant(2.0)),
    );
    
    println!("   Input: {:?}", x_plus_y_squared);
    println!("   Operations count: {}", x_plus_y_squared.count_operations());

    // Try optimization with native egglog
    if let Ok(mut optimizer) = NativeEgglogOptimizer::new() {
        let optimized = optimizer.optimize(&x_plus_y_squared)?;
        println!("   Output: {:?}", optimized);
        println!("   Operations count: {}", optimized.count_operations());
        
        // Check if it expanded
        if optimized.count_operations() > x_plus_y_squared.count_operations() {
            println!("   ✅ Expansion occurred!");
        } else {
            println!("   ❌ No expansion detected");
        }
    } else {
        println!("   ⚠️ Native egglog optimization not available");
    }
    
    println!();

    // Test 2: Test (x+y)*(x+y) form 
    println!("🔬 Test 2: Multiplication form (x+y)*(x+y)");
    let x_plus_y = ASTRepr::Add(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Variable(1)),
    );
    let x_plus_y_mul = ASTRepr::Mul(
        Box::new(x_plus_y.clone()),
        Box::new(x_plus_y),
    );
    
    println!("   Input: {:?}", x_plus_y_mul);
    println!("   Operations count: {}", x_plus_y_mul.count_operations());

    if let Ok(mut optimizer) = NativeEgglogOptimizer::new() {
        let optimized = optimizer.optimize(&x_plus_y_mul)?;
        println!("   Output: {:?}", optimized);
        println!("   Operations count: {}", optimized.count_operations());
        
        // Check if it expanded
        if optimized.count_operations() > x_plus_y_mul.count_operations() {
            println!("   ✅ Expansion occurred!");
        } else {
            println!("   ❌ No expansion detected");
        }
    } else {
        println!("   ⚠️ Native egglog optimization not available");
    }
    
    println!();

    // Test 3: Using ExpressionBuilder
    println!("🔬 Test 3: ExpressionBuilder (x+y)²");
    let math = ExpressionBuilder::new();
    let x = math.var();
    let y = math.var();
    let expr = (x + y).pow(math.constant(2.0));
    let ast = expr.into_ast();
    
    println!("   Input: {:?}", ast);
    println!("   Operations count: {}", ast.count_operations());

    if let Ok(mut optimizer) = NativeEgglogOptimizer::new() {
        let optimized = optimizer.optimize(&ast)?;
        println!("   Output: {:?}", optimized);
        println!("   Operations count: {}", optimized.count_operations());
        
        // Check if it expanded
        if optimized.count_operations() > ast.count_operations() {
            println!("   ✅ Expansion occurred!");
        } else {
            println!("   ❌ No expansion detected");
        }
    } else {
        println!("   ⚠️ Native egglog optimization not available");
    }

    Ok(())
} 