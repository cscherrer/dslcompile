//! Debug Sum Splitting - See Exact Egglog Expression
//!
//! This debug script shows the exact egglog expression being generated
//! to help diagnose why sum splitting isn't working.

use dslcompile::prelude::*;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🔍 Debug Sum Splitting - Egglog Expression Analysis");
    println!("====================================================\n");

    // Create a simple sum splitting test case
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    let b = ctx.var::<f64>();
    let data = vec![1.0, 2.0, 3.0];

    // Create expression: Σ(a * x_i + b * x_i)
    let sum_expr = ctx.sum(&data, |x_i| &a * &x_i + &b * &x_i);
    let ast = ctx.to_ast(&sum_expr);

    println!("1️⃣ AST Structure:");
    println!("   {:?}", ast);

    #[cfg(feature = "optimization")]
    {
        let optimizer = NativeEgglogOptimizer::new()?;
        let egglog_expr = optimizer.ast_to_egglog(&ast)?;

        println!("\n2️⃣ Egglog Expression:");
        println!("   {}", egglog_expr);

        // Let's also check what the expected pattern should be
        println!("\n3️⃣ Expected Pattern for Sum Splitting Rule:");
        println!("   (Sum (Map (LambdaFunc ?var (Add ?f ?g)) ?collection))");

        println!("\n4️⃣ Analysis:");
        println!("   - Does our expression match the pattern?");
        println!("   - Lambda variable index vs BoundVar index alignment?");
        println!("   - Collection type compatibility?");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("⚠️  Optimization features disabled");
        println!("   Run with: cargo run --features optimization --example debug_sum_splitting");
    }

    Ok(())
}
