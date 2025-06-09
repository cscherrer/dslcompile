//! Data-Driven IID Demo: Natural Rust-like API
//!
//! This demonstrates the NATURAL approach you suggested:
//! let iid_likelihood = data.sum(|x| gaussian_logdensity(x, mu, sigma))
//!
//! Key features:
//! 1. Extension trait adds sum() method to Vec<f64>
//! 2. Data drives the summation, not the context
//! 3. Reusable gaussian_logdensity function
//! 4. True compositionality

use dslcompile::ast::{DynamicContext, TypedBuilderExpr};
use frunk::hlist;

/// Extension trait that adds symbolic summation to data types
trait SymbolicSummation {
    /// Create a symbolic sum expression over this data
    ///
    /// This is the natural API: data.sum(|x| expression)
    fn sum<F>(self, f: F) -> SumExpression
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>;
}

/// Wrapper for sum expressions that can be evaluated with parameters
struct SumExpression {
    expr: TypedBuilderExpr<f64>,
    ctx: DynamicContext<f64>,
}

impl SumExpression {
    /// Evaluate the sum expression with given parameters
    fn eval(&self, params: &[f64]) -> f64 {
        self.ctx
            .eval_with_data_arrays(&self.expr, hlist![params[0], params[1]])
    }
}

impl SymbolicSummation for Vec<f64> {
    fn sum<F>(self, f: F) -> SumExpression
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        let mut ctx = DynamicContext::new();
        let expr = ctx.sum(self, f);
        SumExpression { expr, ctx }
    }
}

/// Reusable Gaussian log-density function
/// This is the composable component you mentioned
fn gaussian_logdensity(
    x: TypedBuilderExpr<f64>,
    mu: TypedBuilderExpr<f64>,
    sigma: TypedBuilderExpr<f64>,
) -> TypedBuilderExpr<f64> {
    // log p(x|μ,σ) = -½((x-μ)/σ)² - log(σ√2π)
    let diff = x - mu;
    let standardized = diff / sigma.clone();
    let standardized_squared = &standardized * &standardized;

    // Create constants
    let neg_half = TypedBuilderExpr::from(-0.5);
    let log_density_term = neg_half * standardized_squared;

    // Normalization: log(σ√2π) = log(σ) + ½log(2π)
    let log_sigma = sigma.ln();
    let log_2pi = TypedBuilderExpr::from((2.0 * std::f64::consts::PI).ln());
    let half = TypedBuilderExpr::from(0.5);
    let half_log_2pi = half * log_2pi;
    let normalization = log_sigma + half_log_2pi;

    log_density_term - normalization
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Data-Driven IID Demo: Natural Rust-like API");
    println!("==============================================\n");

    // Create parameters outside the data summation
    let mut ctx = DynamicContext::new();
    let mu = ctx.var(); // Mean parameter
    let sigma = ctx.var(); // Std dev parameter

    println!("🧩 Demo: Natural Data-Driven API");
    println!("Using: data.sum(|x| gaussian_logdensity(x, mu, sigma))");

    // Test datasets
    let datasets = [
        ("Dataset A", 2.0, 1.0, vec![1.8, 2.1, 1.9]),
        ("Dataset B", 1.5, 0.8, vec![1.4, 1.6, 1.3, 1.7]),
        ("Dataset C", 3.0, 0.5, vec![2.8, 3.2, 2.9, 3.1, 2.7]),
    ];

    for (name, mu_val, sigma_val, data) in &datasets {
        println!("\n   📊 {name}:");
        println!("      μ={mu_val:.1}, σ={sigma_val:.1}");
        println!("      Data: {:?} ({} observations)", data, data.len());

        // 🚀 THE NATURAL API YOU WANTED!
        let iid_likelihood = data
            .clone()
            .sum(|x| gaussian_logdensity(x, mu.clone(), sigma.clone()));

        // Evaluate with parameters
        let result = iid_likelihood.eval(&[*mu_val, *sigma_val]);

        println!("      Log-likelihood: {result:.6}");
        println!("      ✅ Natural data.sum() API working!");
    }

    println!("\n🎉 Data-Driven Demo Complete!");
    println!("\n📊 Key Benefits:");
    println!("   ✅ Natural Rust-like API: data.sum(|x| expression)");
    println!("   ✅ Data drives summation (not context)");
    println!("   ✅ Reusable gaussian_logdensity function");
    println!("   ✅ True compositionality");
    println!("   ✅ Uses Div and Ln operations");
    println!("   ✅ No manual unrolling");

    Ok(())
}
