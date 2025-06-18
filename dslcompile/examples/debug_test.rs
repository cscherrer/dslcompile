use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("🔍 Debug: Testing expression building and code generation");

    // Create the same expression as the failing test: x^2 + 2*x + 1
    let mut ctx = DynamicContext::new();
    let x = ctx.var();
    let x_squared = x.clone().pow(ctx.constant(2.0));
    let expr = &x_squared + 2.0 * &x + 1.0;

    println!("Expression: {}", ctx.pretty_print(&expr));

    // Test evaluation
    let result = ctx.eval(&expr, hlist![3.0]);
    println!("Evaluated at x=3: {result}");
    // Should be 3^2 + 2*3 + 1 = 9 + 6 + 1 = 16

    // Let's also test a simpler expression
    println!("\n🔍 Testing simpler expression: x + 1");
    let mut ctx2 = DynamicContext::new();
    let x2 = ctx2.var();
    let simple_expr = &x2 + 1.0;

    println!("Simple expression: {}", ctx2.pretty_print(&simple_expr));
    let simple_result = ctx2.eval(&simple_expr, hlist![5.0]);
    println!("Evaluated at x=5: {simple_result}");

    // Test trigonometric functions
    println!("\n🔍 Testing trigonometric expression: sin(2*x + cos(y))");
    let mut ctx3 = DynamicContext::new();
    let x3 = ctx3.var();
    let y3 = ctx3.var();
    let trig_expr = (2.0 * &x3 + y3.cos()).sin();

    println!(
        "Trigonometric expression: {}",
        ctx3.pretty_print(&trig_expr)
    );
    let trig_result = ctx3.eval(&trig_expr, hlist![1.0, 0.0]);
    println!("Evaluated at x=1, y=0: {trig_result}");

    #[cfg(feature = "optimization")]
    {
        // Test the optimizer with all expressions
        let optimizer = SymbolicOptimizer::new()?;

        let rust_code = optimizer.generate_rust_source(expr.as_ast(), "debug_func")?;
        println!("Generated Rust code:\n{rust_code}");

        let simple_rust_code =
            optimizer.generate_rust_source(simple_expr.as_ast(), "simple_func")?;
        println!("Simple generated Rust code:\n{simple_rust_code}");

        let trig_rust_code = optimizer.generate_rust_source(trig_expr.as_ast(), "trig_func")?;
        println!("Trigonometric generated Rust code:\n{trig_rust_code}");
    }

    println!("✅ Debug test completed successfully!");
    Ok(())
}
