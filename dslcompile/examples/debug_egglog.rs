use dslcompile::prelude::*;

fn main() -> Result<()> {
    println!("🔍 Debug Egglog Conversion");

    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    let b = ctx.var::<f64>();
    let test_data = vec![1.0, 2.0, 3.0];

    let sum_expr = ctx.sum(&test_data, |x_i| &a * &x_i + &b * &x_i);
    let original_ast = ctx.to_ast(&sum_expr);

    println!("AST Structure:");
    println!("{original_ast:#?}");

    #[cfg(feature = "optimization")]
    {
        use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;
        let optimizer = NativeEgglogOptimizer::new()?;
        let egglog_string = optimizer.ast_to_egglog(&original_ast)?;

        println!("\nEgglog Representation:");
        println!("{egglog_string}");

        println!("\nExpected pattern:");
        println!("(Sum (Map (LambdaFunc ?var (Add ?f ?g)) ?collection))");
    }

    Ok(())
}
