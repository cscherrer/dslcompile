// Debug the IID likelihood computation step by step
use dslcompile::prelude::*;
use frunk::hlist;

fn main() -> Result<()> {
    println!("Debugging IID Log-Likelihood computation");

    let sample_data = vec![1.0, 2.0, 0.5, 1.5, 0.8];
    println!("Sample data: {sample_data:?}");

    // Manual calculation step by step
    println!("\nManual calculation:");
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    println!("ln(2π) = {}", log_2pi);

    let mut manual_total = 0.0;
    for (i, &x) in sample_data.iter().enumerate() {
        let term = -0.5 * log_2pi - 0.0 - 0.5 * (x - 0.0_f64).powi(2);
        manual_total += term;
        println!(
            "  Point {}: x={}, term={}, running_total={}",
            i + 1,
            x,
            term,
            manual_total
        );
    }
    println!("Manual total: {}", manual_total);

    // StaticContext calculation
    println!("\nStaticContext calculation:");
    let mut ctx = StaticContext::new();
    let iid_expr = ctx.new_scope(|scope| {
        let (mu, scope) = scope.auto_var::<f64>();
        let (sigma, scope) = scope.auto_var::<f64>();

        let log_2pi = scope.constant((2.0 * std::f64::consts::PI).ln());
        let neg_half = scope.constant(-0.5);

        println!("  Constants created");

        let (sum_expr, _scope) = scope.sum(sample_data.clone(), |x| {
            let centered = x - mu.clone();
            let standardized = centered / sigma.clone();
            let squared = standardized.clone() * standardized;

            neg_half.clone() * log_2pi.clone() - sigma.clone().ln() + neg_half.clone() * squared
        });

        sum_expr
    });

    let static_result = iid_expr.eval(hlist![0.0, 1.0]);
    println!("StaticContext result: {}", static_result);

    // 1. AST analysis - skip for now due to missing Expr implementations
    println!("\n=== 1. AST STRUCTURE ANALYSIS ===");
    println!("Skipping AST analysis - need more Expr implementations");

    // 2. AST evaluation - skip for now
    println!("\n=== 2. AST EVALUATION TEST ===");
    println!("Skipping AST evaluation - need more Expr implementations");

    // 3. Verify with DynamicContext (control test)
    println!("\n=== 3. DYNAMICCONTEXT CONTROL TEST ===");
    let mut dyn_ctx = DynamicContext::new();
    let dyn_mu = dyn_ctx.var();
    let dyn_sigma = dyn_ctx.var();

    let dyn_iid = dyn_ctx.sum(sample_data.clone(), |x| {
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let neg_half = -0.5;

        let centered = x - &dyn_mu;
        let standardized = &centered / &dyn_sigma;
        let squared = &standardized * &standardized;

        neg_half * log_2pi - dyn_sigma.clone().ln() + neg_half * &squared
    });

    let dyn_result = dyn_ctx.eval(&dyn_iid, hlist![0.0, 1.0]);
    println!("DynamicContext result: {}", dyn_result);
    println!("Expected: {}", manual_total);

    println!("\n=== COMPARISON SUMMARY ===");
    println!("Manual:        {}", manual_total);
    println!("DynamicContext: {}", dyn_result);
    println!("StaticContext:  {}", static_result);

    // Try a simpler version to isolate the issue
    println!("\nSimpler test - just sum the squared terms:");
    let mut simple_ctx = StaticContext::new();
    let simple_expr = simple_ctx.new_scope(|scope| {
        let (sum_expr, _scope) = scope.sum(sample_data.clone(), |x| x.clone() * x.clone());
        sum_expr
    });

    let simple_result = simple_expr.eval(hlist![]);
    let expected_simple: f64 = sample_data.iter().map(|&x| x * x).sum();
    println!(
        "Simple squared sum: {} (expected: {})",
        simple_result, expected_simple
    );

    Ok(())
}
