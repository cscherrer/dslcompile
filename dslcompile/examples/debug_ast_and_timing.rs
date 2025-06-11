use dslcompile::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🔍 Debug AST Structure and Optimization Timing");
    println!("==============================================");

    // Create a simple Gaussian expression like in the demo
    let mut ctx = DynamicContext::<f64>::new();
    let x: TypedBuilderExpr<f64> = ctx.var();
    let mu = ctx.var();
    let sigma = ctx.var();

    // Build the Gaussian log-density: -0.5 * ((x - mu) / sigma)^2
    let diff = &x - &mu;
    let standardized = &diff / &sigma;
    let squared = &standardized * &standardized;
    let log_density: Expr<f64> = -0.5 * &squared;

    let expr = log_density.into();

    println!("\n📋 AST Structure:");
    println!("{:#?}", expr);

    println!("\n📏 AST Size Analysis:");
    println!("   Expression depth: {}", count_depth(&expr));
    println!("   Node count: {}", count_nodes(&expr));

    // Test optimization timing
    println!("\n⏱️  Optimization Timing Test:");

    #[cfg(feature = "optimization")]
    {
        use dslcompile::symbolic::native_egglog::optimize_with_native_egglog;

        let start = Instant::now();
        match optimize_with_native_egglog(&expr) {
            Ok(optimized) => {
                let duration = start.elapsed();
                println!("   ✅ Optimization completed in: {:?}", duration);
                println!("   📋 Optimized AST:");
                println!("{:#?}", optimized);

                // Check if actually different
                let changed = !expressions_equal(&expr, &optimized);
                println!("   🔄 Expression changed: {}", changed);
            }
            Err(e) => {
                let duration = start.elapsed();
                println!("   ❌ Optimization failed after: {:?}", duration);
                println!("   Error: {}", e);
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("   ⚠️  Optimization feature not enabled");
    }

    // Test evaluation timing
    println!("\n⏱️  Evaluation Timing Test:");
    let test_values = [1.5, 1.0, 0.5]; // x, mu, sigma

    let start = Instant::now();
    let result = expr.eval_with_vars(&test_values);
    let eval_duration = start.elapsed();

    println!("   📊 Evaluation result: {}", result);
    println!("   ⏱️  Evaluation time: {:?}", eval_duration);

    Ok(())
}

fn count_depth(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(a, b) | ASTRepr::Sub(a, b) | ASTRepr::Mul(a, b) | ASTRepr::Div(a, b) => {
            1 + count_depth(a).max(count_depth(b))
        }
        ASTRepr::Neg(a) | ASTRepr::Sin(a) | ASTRepr::Cos(a) | ASTRepr::Ln(a) | ASTRepr::Exp(a) => {
            1 + count_depth(a)
        }
        ASTRepr::Pow(a, b) => 1 + count_depth(a).max(count_depth(b)),
        _ => 1, // For other variants
    }
}

fn count_nodes(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(a, b) | ASTRepr::Sub(a, b) | ASTRepr::Mul(a, b) | ASTRepr::Div(a, b) => {
            1 + count_nodes(a) + count_nodes(b)
        }
        ASTRepr::Neg(a) | ASTRepr::Sin(a) | ASTRepr::Cos(a) | ASTRepr::Ln(a) | ASTRepr::Exp(a) => {
            1 + count_nodes(a)
        }
        ASTRepr::Pow(a, b) => 1 + count_nodes(a) + count_nodes(b),
        _ => 1, // For other variants
    }
}

fn expressions_equal(a: &ASTRepr<f64>, b: &ASTRepr<f64>) -> bool {
    use std::mem::discriminant;

    if discriminant(a) != discriminant(b) {
        return false;
    }

    match (a, b) {
        (ASTRepr::Constant(a), ASTRepr::Constant(b)) => (a - b).abs() < 1e-10,
        (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,
        (ASTRepr::Add(a1, a2), ASTRepr::Add(b1, b2))
        | (ASTRepr::Sub(a1, a2), ASTRepr::Sub(b1, b2))
        | (ASTRepr::Mul(a1, a2), ASTRepr::Mul(b1, b2))
        | (ASTRepr::Div(a1, a2), ASTRepr::Div(b1, b2))
        | (ASTRepr::Pow(a1, a2), ASTRepr::Pow(b1, b2)) => {
            expressions_equal(a1, b1) && expressions_equal(a2, b2)
        }
        (ASTRepr::Neg(a), ASTRepr::Neg(b))
        | (ASTRepr::Sin(a), ASTRepr::Sin(b))
        | (ASTRepr::Cos(a), ASTRepr::Cos(b))
        | (ASTRepr::Ln(a), ASTRepr::Ln(b))
        | (ASTRepr::Exp(a), ASTRepr::Exp(b)) => expressions_equal(a, b),
        _ => false,
    }
}
