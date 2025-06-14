//! Lambda Variable Indexing Test
//!
//! This test verifies that lambda variables and bound variables use consistent indexing
//! for egglog rule matching.

use dslcompile::prelude::*;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🔬 Lambda Variable Indexing Test");
    println!("================================\n");

    // Create a simple lambda expression manually
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    let _b = ctx.var::<f64>();
    
    // Create a simple sum: Σ(a * x) where x is the iterator variable
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(data, |x| a.clone() * x);
    
    println!("1️⃣ Simple Sum Expression:");
    println!("   AST: {:?}", sum_expr.as_ast());
    
    #[cfg(feature = "optimization")]
    {
        let mut optimizer = NativeEgglogOptimizer::new()?;
        let egglog_expr = optimizer.ast_to_egglog(sum_expr.as_ast())?;
        println!("   Egglog: {}", egglog_expr);
        
        // Test if this matches the constant factoring rule:
        // (rule ((= lhs (Sum (Map (LambdaFunc ?var (Mul ?k ?f)) ?collection))))
        //       ((union lhs (Mul ?k (Sum (Map (LambdaFunc ?var ?f) ?collection)))))
        //       :ruleset stage3_summation)
        
        println!("\n2️⃣ Testing Constant Factoring Rule:");
        println!("   This should apply constant factoring: Σ(a * x) → a * Σ(x)");
    }
    
    Ok(())
} 