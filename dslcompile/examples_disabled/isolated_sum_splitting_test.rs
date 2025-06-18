//! Isolated Sum Splitting Test
//!
//! This test isolates the sum splitting issue to determine if:
//! 1. The egglog rule is matching the pattern
//! 2. The rule is being applied but not detected
//! 3. There's an issue with the evaluation system

use dslcompile::prelude::*;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🔬 Isolated Sum Splitting Test");
    println!("==============================\n");

    // Create the simplest possible sum splitting case
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    let b = ctx.var::<f64>();
    let data = vec![1.0, 2.0, 3.0];

    // Create: Σ(a * x + b * x) = Σ((a + b) * x) = (a + b) * Σ(x)
    let sum_expr = ctx.sum(&data, |x| &a * &x + &b * &x);
    let ast = ctx.to_ast(&sum_expr);

    println!("1️⃣ Original AST:");
    println!("   {ast:?}");

    #[cfg(feature = "optimization")]
    {
        let mut optimizer = NativeEgglogOptimizer::new()?;

        // Convert to egglog and show the exact expression
        let egglog_expr = optimizer.ast_to_egglog(&ast)?;
        println!("\n2️⃣ Egglog Expression:");
        println!("   {egglog_expr}");

        // Test the rule manually by creating a minimal egglog program
        println!("\n3️⃣ Testing Sum Splitting Rule Directly:");
        let test_program = format!(
            r"
; Load the datatypes and rules
{}

; Add our test expression
(let test_expr {})

; Show initial state
(query-extract test_expr)

; Run just the summation rules
(run-schedule (saturate stage3_summation))

; Show result after sum splitting
(query-extract test_expr)
",
            include_str!("../src/egglog_rules/staged_core_math.egg"),
            egglog_expr
        );

        // Create a fresh egglog instance to test the rule
        use egglog::EGraph;
        let mut test_egraph = EGraph::default();

        match test_egraph.parse_and_run_program(None, &test_program) {
            Ok(results) => {
                println!("   ✅ Egglog program executed successfully");
                println!("   Results: {results:?}");
            }
            Err(e) => {
                println!("   ❌ Egglog program failed: {e}");
            }
        }

        // Now test with the full optimizer
        println!("\n4️⃣ Full Optimization Test:");
        let optimized = optimizer.optimize(&ast)?;
        println!("   Original:  {ast:?}");
        println!("   Optimized: {optimized:?}");

        // Test evaluation
        println!("\n5️⃣ Evaluation Test:");
        use frunk::hlist;

        // Test parameters: a=2, b=3 (data is embedded in the expression)
        let params = hlist![2.0, 3.0];

        let original_result = ctx.eval(&sum_expr, params);
        println!("   Original evaluation: {original_result:?}");

        // Expected: (2+3) * (1+2+3) = 5 * 6 = 30
        println!("   Expected result: (2+3) * (1+2+3) = 5 * 6 = 30");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("⚠️  Optimization features disabled");
        println!(
            "   Run with: cargo run --features optimization --example isolated_sum_splitting_test"
        );
    }

    Ok(())
}
