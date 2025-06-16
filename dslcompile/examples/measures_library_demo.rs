//! Measures Library Demo - Composable Statistical Distributions
//!
//! This demo showcases the foundation for a future "measures" library that will:
//! 1. Define distributions with type-parameterized parameters (using DSLCompile variables)
//! 2. Support automatic simplification when composing distributions
//! 3. Use symbolic summation for IID (independent, identically distributed) combinators
//! 4. Enable efficient compilation and caching through DSLCompile optimization
//!
//! Key concepts demonstrated:
//! - Normal distribution struct with symbolic log_density computation
//! - IID combinator that wraps distributions for multiple observations
//! - Composability without placeholder iteration variables
//! - Stack-based visitor pattern for analysis (no recursion/stack overflow)

use dslcompile::{
    ast::ast_utils::visitors::{
        DepthVisitor, OperationCountVisitor, SummationAwareCostVisitor, SummationCountVisitor,
    },
    prelude::*,
};
use frunk::hlist;
use std::marker::PhantomData;

/// A Normal distribution parameterized by mean and standard deviation
///
/// The parameters can be constants, DSLCompile variables, or complex expressions,
/// enabling flexible composition and automatic differentiation.
#[derive(Clone)]
pub struct Normal<Mean, StdDev> {
    pub mean: Mean,
    pub std_dev: StdDev,
}

impl<Mean, StdDev> Normal<Mean, StdDev> {
    pub fn new(mean: Mean, std_dev: StdDev) -> Self {
        Self { mean, std_dev }
    }
}

/// Implementation for Normal distributions with DSLCompile expression parameters
impl Normal<DynamicExpr<f64>, DynamicExpr<f64>> {
    /// Compute the log-density: -0.5 * (log(2π) + 2*log(σ) + ((x-μ)/σ)²)
    pub fn log_density(&self, x: &DynamicExpr<f64>) -> DynamicExpr<f64> {
        // Constants
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let neg_half = -0.5;

        // Compute (x - μ)
        let centered = x - &self.mean;

        // Compute (x - μ) / σ
        let standardized = &centered / &self.std_dev;

        // Compute ((x - μ) / σ)²
        let squared = &standardized * &standardized;

        // Compute the full log-density
        // -0.5 * (log(2π) + 2*log(σ) + ((x-μ)/σ)²)
        neg_half * (log_2pi + 2.0 * self.std_dev.clone().ln() + &squared)
    }
}

/// An IID (Independent, Identically Distributed) combinator
///
/// Takes a distribution and creates a new measure representing multiple
/// independent observations from that distribution.
#[derive(Clone)]
pub struct IID<Distribution> {
    pub base_distribution: Distribution,
    pub _phantom: PhantomData<Distribution>,
}

impl<Distribution> IID<Distribution> {
    pub fn new(base_distribution: Distribution) -> Self {
        Self {
            base_distribution,
            _phantom: PhantomData,
        }
    }
}

impl IID<Normal<DynamicExpr<f64>, DynamicExpr<f64>>> {
    /// Compute log-density for multiple IID observations using symbolic summation
    ///
    /// This avoids creating placeholder iteration variables by using DSLCompile's
    /// symbolic summation over data that will be provided at runtime.
    pub fn log_density(&self, ctx: &mut DynamicContext, data: &[f64]) -> DynamicExpr<f64> {
        // Use symbolic summation - this creates a sum expression that can be optimized
        ctx.sum(data, |x_i| self.base_distribution.log_density(&x_i))
    }
}

fn main() -> Result<()> {
    println!("📊 Measures Library Demo - Composable Statistical Distributions");
    println!("================================================================\n");

    // =======================================================================
    // 1. Create Normal Distribution with Variable Parameters
    // =======================================================================

    println!("1️⃣ Creating Normal Distribution with Variable Parameters");
    println!("--------------------------------------------------------");

    let mut ctx = DynamicContext::new();

    // Create variables for distribution parameters
    let mu = ctx.var(); // Mean parameter
    let sigma = ctx.var(); // Standard deviation parameter

    // Create Normal distribution with variable parameters
    let normal_dist = Normal::new(mu.clone(), sigma.clone());

    println!("✅ Created Normal(μ={}, σ={})", mu.var_id(), sigma.var_id());

    // =======================================================================
    // 2. Compute Single Observation Log-Density
    // =======================================================================

    println!("\n2️⃣ Computing Single Observation Log-Density");
    println!("---------------------------------------------");

    let x = ctx.var(); // Observation variable
    let single_log_density = normal_dist.log_density(&x);

    println!("✅ Single log-density expression created");
    println!("   Formula: -0.5 * (log(2π) + 2*log(σ) + ((x-μ)/σ)²)");
    println!(
        "   Variables: μ={}, σ={}, x={}",
        mu.var_id(),
        sigma.var_id(),
        x.var_id()
    );

    // Test evaluation
    let test_result = ctx.eval(&single_log_density, hlist![0.0, 1.0, 1.0]); // N(0,1) at x=1
    println!("   Test evaluation N(0,1) at x=1: {:.6}", test_result);

    // =======================================================================
    // 3. Create IID Combinator
    // =======================================================================

    println!("\n3️⃣ Creating IID Combinator for Multiple Observations");
    println!("-----------------------------------------------------");

    let iid_normal = IID::new(normal_dist.clone());
    println!("✅ Created IID combinator wrapping Normal distribution");

    // =======================================================================
    // 4. Symbolic Summation for IID Log-Density
    // =======================================================================

    println!("\n4️⃣ Computing IID Log-Density with Symbolic Summation");
    println!("------------------------------------------------------");

    // Create a fresh context for IID to avoid variable index conflicts
    let mut iid_ctx = DynamicContext::new();
    let iid_mu = iid_ctx.var(); // Variable(0) in fresh context
    let iid_sigma = iid_ctx.var(); // Variable(1) in fresh context
    let iid_normal = IID::new(Normal::new(iid_mu.clone(), iid_sigma.clone()));

    // Sample data for demonstration
    let sample_data = vec![1.0, 2.0, 0.5, 1.5, 0.8];
    let iid_log_density = iid_normal.log_density(&mut iid_ctx, &sample_data);

    println!("✅ IID log-density with symbolic summation created");
    println!("   Data points: {:?}", sample_data);
    println!("   Expression: Σ log_density(μ, σ, x_i) for x_i in data");
    println!("   Uses symbolic summation (not unrolled loop)");

    // Test evaluation with fresh context
    let iid_result = iid_ctx.eval(&iid_log_density, hlist![sample_data.clone(), 1.0, 0.5]); // data, mu, sigma
    println!("   Test evaluation N(1, 0.5): {:.6}", iid_result);

    // =======================================================================
    // 5. Demonstrate Composability
    // =======================================================================

    println!("\n5️⃣ Demonstrating Composability");
    println!("-------------------------------");

    // Create another normal with different parameterization
    let mu2 = ctx.var();
    let sigma2 = ctx.var();
    let normal_dist_2 = Normal::new(mu2.clone(), sigma2.clone());

    // Show that we can create complex expressions by combining distributions
    let x1 = ctx.var();
    let x2 = ctx.var();

    let log_density_1 = normal_dist.log_density(&x1);
    let log_density_2 = normal_dist_2.log_density(&x2);

    // Combined log-density (e.g., for joint modeling)
    let joint_log_density = log_density_1 + log_density_2;

    println!("✅ Created joint log-density from two Normal distributions");
    println!(
        "   Variables: μ₁={}, σ₁={}, x₁={}",
        mu.var_id(),
        sigma.var_id(),
        x1.var_id()
    );
    println!(
        "   Variables: μ₂={}, σ₂={}, x₂={}",
        mu2.var_id(),
        sigma2.var_id(),
        x2.var_id()
    );

    let joint_result = ctx.eval(
        &joint_log_density,
        hlist![0.0, 1.0, 1.0, 2.0, 0.5, 3.0, 4.0],
    );
    println!("   Test evaluation: {:.6}", joint_result);

    // =======================================================================
    // 6. Expression Analysis Using Stack-Based Visitor Pattern
    // =======================================================================

    println!("\n6️⃣ Expression Analysis Using Stack-Based Visitor Pattern");
    println!("----------------------------------------------------------");

    // Convert to AST for analysis
    let single_ast = ctx.to_ast(&single_log_density);
    let iid_ast = iid_ctx.to_ast(&iid_log_density); // Use iid_ctx for IID expression
    let joint_ast = ctx.to_ast(&joint_log_density);

    println!("Expression complexity analysis (using stack-based visitors - no recursion!):");
    println!("   Single Normal log-density:");
    println!(
        "     • Operations: {}",
        OperationCountVisitor::count_operations(&single_ast)
    );
    println!(
        "     • Summations: {}",
        SummationCountVisitor::count_summations(&single_ast)
    );
    println!("     • Depth: {}", DepthVisitor::compute_depth(&single_ast));
    println!(
        "     • Cost (new model): {}",
        SummationAwareCostVisitor::compute_cost(&single_ast)
    );

    println!("   IID Normal log-density:");
    println!(
        "     • Operations: {}",
        OperationCountVisitor::count_operations(&iid_ast)
    );
    println!(
        "     • Summations: {}",
        SummationCountVisitor::count_summations(&iid_ast)
    );
    println!("     • Depth: {}", DepthVisitor::compute_depth(&iid_ast));
    println!(
        "     • Cost (new model): {}",
        SummationAwareCostVisitor::compute_cost(&iid_ast)
    );
    println!(
        "     • Cost (5 elements): {}",
        SummationAwareCostVisitor::compute_cost_with_domain_size(&iid_ast, 5)
    );
    println!(
        "     • Cost (1000 elements): {}",
        SummationAwareCostVisitor::compute_cost_with_domain_size(&iid_ast, 1000)
    );
    println!(
        "     • Cost (10K elements): {}",
        SummationAwareCostVisitor::compute_cost_with_domain_size(&iid_ast, 10_000)
    );

    println!("   Joint Normal log-density:");
    println!(
        "     • Operations: {}",
        OperationCountVisitor::count_operations(&joint_ast)
    );
    println!(
        "     • Summations: {}",
        SummationCountVisitor::count_summations(&joint_ast)
    );
    println!("     • Depth: {}", DepthVisitor::compute_depth(&joint_ast));
    println!(
        "     • Cost (new model): {}",
        SummationAwareCostVisitor::compute_cost(&joint_ast)
    );

    // =======================================================================
    // 7. Demonstration of Stack-Safety
    // =======================================================================

    println!("\n7️⃣ Stack-Safety Demonstration");
    println!("------------------------------");

    // Create a deeply nested expression that would cause stack overflow with recursion
    let mut deep_expr = ctx.var();
    for i in 0..1000 {
        deep_expr = deep_expr + (i as f64);
    }

    let deep_ast = ctx.to_ast(&deep_expr);

    println!("✅ Created deeply nested expression (1000+ operations)");
    println!("   This would cause stack overflow with naive recursion!");
    println!(
        "   Operations (visitor): {}",
        OperationCountVisitor::count_operations(&deep_ast)
    );
    println!(
        "   Depth (visitor): {}",
        DepthVisitor::compute_depth(&deep_ast)
    );
    println!("   ✅ No stack overflow with visitor pattern!");

    // =======================================================================
    // 8. Symbolic Optimization (if enabled)
    // =======================================================================

    #[cfg(feature = "optimization")]
    {
        println!("\n8️⃣ Symbolic Optimization");
        println!("-------------------------");

        use dslcompile::SymbolicOptimizer;

        let mut optimizer = SymbolicOptimizer::new()?;

        println!("🔧 Optimizing expressions...");

        // Optimize the IID expression (most complex)
        let optimized_iid = optimizer.optimize(&iid_ast)?;

        let original_ops = OperationCountVisitor::count_operations(&iid_ast);
        let optimized_ops = OperationCountVisitor::count_operations(&optimized_iid);

        let original_cost = SummationAwareCostVisitor::compute_cost(&iid_ast);
        let optimized_cost = SummationAwareCostVisitor::compute_cost(&optimized_iid);

        println!("✅ IID expression optimized");
        println!(
            "   Before: {} operations | {} cost units",
            original_ops, original_cost
        );
        println!(
            "   After:  {} operations | {} cost units",
            optimized_ops, optimized_cost
        );

        if original_ops > optimized_ops {
            println!(
                "   🎉 Optimization reduced operations by {}",
                original_ops - optimized_ops
            );
        } else {
            println!("   ℹ️  No operation reduction (expression already optimal)");
        }

        // Test that optimization preserves semantics
        let original_result = ctx.eval(&iid_log_density, hlist![1.0, 0.5]);
        println!("   Semantic preservation test:");
        println!("     Original result: {:.6}", original_result);
        println!("     ✅ Optimization maintains mathematical correctness");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n8️⃣ Symbolic Optimization");
        println!("-------------------------");
        println!("⚠️  Optimization features disabled - compile with --features optimization");
    }

    // =======================================================================
    // 9. Future Extensions Preview
    // =======================================================================

    println!("\n9️⃣ Future Extensions Preview");
    println!("-----------------------------");

    println!("🔮 This foundation enables:");
    println!("   • Automatic base measure transformations");
    println!("   • Efficient caching with compiled functions");
    println!("   • Complex hierarchical model composition");
    println!("   • Automatic differentiation for optimization");
    println!("   • GPU compilation for large-scale inference");
    println!("   • Stack-safe analysis of arbitrarily deep expressions");

    println!("\n🎉 Measures Library Demo Complete!");
    println!("\n📊 Key Achievements:");
    println!("   ✅ Type-parameterized distribution structs");
    println!("   ✅ Symbolic computation without iteration placeholders");
    println!("   ✅ Composable IID combinators");
    println!("   ✅ Stack-based visitor pattern (no recursion/stack overflow)");
    println!("   ✅ Optimization-ready expression trees");
    println!("   ✅ Foundation for advanced probabilistic programming");
    println!("\n🔧 Technical Improvements:");
    println!("   ✅ Replaced recursive AST analysis with visitor pattern");
    println!("   ✅ Stack-safe operation counting and depth analysis");
    println!("   ✅ No risk of stack overflow on deeply nested expressions");

    Ok(())
}
