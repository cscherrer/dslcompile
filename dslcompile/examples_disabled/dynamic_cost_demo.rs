//! Dynamic Cost Assignment Demo
//!
//! This example demonstrates how to use dynamic cost assignment from egglog-experimental
//! to control expression optimization based on runtime analysis of summation coupling.

use dslcompile::{ast::ASTRepr, prelude::*};

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🎯 Dynamic Cost Assignment Demo");
    println!("===============================");

    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    let z = ctx.var::<f64>();

    // Example 1: High coupling cost - summation with external parameter access
    println!("\n📊 Example 1: High Coupling Cost");
    println!("   Expression: Σ(i=1 to 10) (x * i²)");
    println!("   Analysis: External parameter 'x' accessed in each iteration");

    let high_coupling_expr = ctx.sum(1..=10, |i| &x * (&i * &i));
    println!("   Original: {}", ctx.pretty_print(&high_coupling_expr));

    // Create symbolic optimizer with dynamic cost support
    #[cfg(feature = "optimization")]
    let mut optimizer = NativeEgglogOptimizer::new()?;

    // Set high dynamic cost for expressions that couple external parameters with iteration
    // This will discourage keeping the summation in this form
    let coupling_expr = ASTRepr::Mul(vec![
        ASTRepr::Variable(0), // x (external parameter)
        ASTRepr::Variable(1), // i (iteration variable)
    ]);
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&coupling_expr, 5000)?;

    #[cfg(feature = "optimization")]
    let optimized_high_coupling = optimizer.optimize(&ctx.to_ast(&high_coupling_expr))?;
    #[cfg(feature = "optimization")]
    println!("   Optimized: {optimized_high_coupling:?}");

    // Example 2: Low coupling cost - summation with only range variables
    println!("\n📊 Example 2: Low Coupling Cost");
    println!("   Expression: Σ(i=1 to 10) i²");
    println!("   Analysis: Only range variable 'i' used, no external parameters");

    let low_coupling_expr = ctx.sum(1..=10, |i| &i * &i);
    println!("   Original: {}", ctx.pretty_print(&low_coupling_expr));

    // Set low dynamic cost for range-only expressions
    let range_only_expr = ASTRepr::Mul(vec![
        ASTRepr::Variable(0), // i (iteration variable)
        ASTRepr::Variable(0), // i (same iteration variable)
    ]);
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&range_only_expr, 10)?;

    #[cfg(feature = "optimization")]
    let optimized_low_coupling = optimizer.optimize(&ctx.to_ast(&low_coupling_expr))?;
    #[cfg(feature = "optimization")]
    println!("   Optimized: {optimized_low_coupling:?}");

    // Example 3: Factoring benefit - external parameter can be factored out
    println!("\n📊 Example 3: Factoring Benefit");
    println!("   Expression: Σ(i=1 to 10) (x * i²) = x * Σ(i=1 to 10) i²");
    println!("   Analysis: External parameter can be factored outside summation");

    // Set very low cost for factored expressions
    let factored_expr = dslcompile::ast::ASTRepr::Mul(vec![
        dslcompile::ast::ASTRepr::Variable(0), // x (external parameter)
        dslcompile::ast::ASTRepr::Sum(Box::new(dslcompile::ast::ast_repr::Collection::Range {
            start: Box::new(dslcompile::ast::ASTRepr::Constant(1.0)),
            end: Box::new(dslcompile::ast::ASTRepr::Constant(10.0)),
        })),
    ]);
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&factored_expr, 50)?;

    println!("   Factored form should be preferred due to lower dynamic cost");

    // Example 4: Simple multiplication with external variable
    println!("\n📊 Example 4: Simple Expression with External Variable");
    println!("   Expression: x * 5.0 + y");
    println!("   Analysis: Mixed operations with external parameters");

    let simple_expr = &x * 5.0 + &y;
    println!("   Original: {}", ctx.pretty_print(&simple_expr));

    // Set cost for simple expressions
    let simple_coupling_expr = dslcompile::ast::ASTRepr::Add(vec![
        dslcompile::ast::ASTRepr::Mul(vec![
            dslcompile::ast::ASTRepr::Variable(0), // x
            dslcompile::ast::ASTRepr::Constant(5.0),
        ]),
        dslcompile::ast::ASTRepr::Variable(1), // y
    ]);
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&simple_coupling_expr, 100)?;

    #[cfg(feature = "optimization")]
    let optimized_simple = optimizer.optimize(&ctx.to_ast(&simple_expr))?;
    #[cfg(feature = "optimization")]
    println!("   Optimized: {optimized_simple:?}");

    // Example 5: Cost-driven expansion vs. factoring
    println!("\n📊 Example 5: Cost-Driven Expansion vs. Factoring");
    println!("   Expression: (x + y)²");
    println!("   Analysis: Can expand to x² + 2xy + y² or keep factored");

    let square_expr = (&x + &y) * (&x + &y);
    println!("   Original: {}", ctx.pretty_print(&square_expr));

    // Set cost for expanded form (higher complexity but potentially more optimizable)
    let expanded_expr = dslcompile::ast::ASTRepr::Add(vec![
        dslcompile::ast::ASTRepr::Pow(
            Box::new(dslcompile::ast::ASTRepr::Variable(0)),
            Box::new(dslcompile::ast::ASTRepr::Constant(2.0)),
        ),
        dslcompile::ast::ASTRepr::Mul(vec![
            dslcompile::ast::ASTRepr::Constant(2.0),
            dslcompile::ast::ASTRepr::Variable(0),
            dslcompile::ast::ASTRepr::Variable(1),
        ]),
        dslcompile::ast::ASTRepr::Pow(
            Box::new(dslcompile::ast::ASTRepr::Variable(1)),
            Box::new(dslcompile::ast::ASTRepr::Constant(2.0)),
        ),
    ]);
    #[cfg(feature = "optimization")]
    optimizer.set_dynamic_cost(&expanded_expr, 100)?;

    #[cfg(feature = "optimization")]
    let optimized_square = optimizer.optimize(&ctx.to_ast(&square_expr))?;
    #[cfg(feature = "optimization")]
    println!("   Optimized: {optimized_square:?}");

    println!("\n✅ Dynamic Cost Assignment Demo Complete!");
    println!("   Dynamic costs enable fine-grained control over optimization decisions");
    println!("   Costs are set based on runtime analysis of expression patterns");
    println!("   This allows optimization to favor beneficial transformations while");
    println!("   discouraging expensive operations based on actual usage patterns");

    Ok(())
}
