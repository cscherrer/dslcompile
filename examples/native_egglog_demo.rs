use mathcompile::final_tagless::{ASTRepr, ExpressionBuilder};

#[cfg(feature = "optimization")]
use mathcompile::symbolic::native_egglog::{NativeEgglogOptimizer, optimize_with_native_egglog};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Native egglog Integration Demo");
    println!("==================================");

    #[cfg(feature = "optimization")]
    {
        // Create some test expressions
        let builder = ExpressionBuilder::new();

        // Simple algebraic expression: x + 0
        let expr1 = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(0.0)),
        );

        // More complex: (x * 1) + (0 * y)
        let expr2 = ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(1.0)),
            )),
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Constant(0.0)),
                Box::new(ASTRepr::Variable(1)),
            )),
        );

        // Transcendental: ln(exp(x))
        let expr3 = ASTRepr::Ln(Box::new(ASTRepr::Exp(Box::new(ASTRepr::Variable(0)))));

        // Canonical form conversion: x - y (should become x + (-y))
        let expr4 = ASTRepr::Sub(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        println!("\n📊 Testing Native egglog Optimization:");
        println!("--------------------------------------");

        // Test each expression
        let expressions = vec![
            ("x + 0", expr1),
            ("(x * 1) + (0 * y)", expr2),
            ("ln(exp(x))", expr3),
            ("x - y", expr4),
        ];

        for (name, expr) in expressions {
            println!("\n🔍 Expression: {name}");

            // Create optimizer
            let mut optimizer = NativeEgglogOptimizer::new()?;

            // Show AST to egglog conversion
            let egglog_form = optimizer.ast_to_egglog(&expr)?;
            println!("   egglog form: {egglog_form}");

            // Optimize
            let optimized = optimizer.optimize(&expr)?;
            println!("   Original:    {expr:?}");
            println!("   Optimized:   {optimized:?}");

            // Note: Current implementation returns original expression
            // Future versions will implement proper extraction
        }

        println!("\n🎯 Using Helper Function:");
        println!("-------------------------");

        // Test the helper function
        let test_expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(0.0)),
        );

        let result = optimize_with_native_egglog(&test_expr)?;
        println!("Input:  {test_expr:?}");
        println!("Output: {result:?}");

        println!("\n✅ Native egglog Integration Working!");
        println!("🔮 Future: Domain-aware optimization with interval analysis");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("❌ Optimization feature not enabled.");
        println!("Run with: cargo run --example native_egglog_demo --features optimization");
    }

    Ok(())
}
