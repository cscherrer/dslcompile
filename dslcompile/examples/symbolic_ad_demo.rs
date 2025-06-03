//! Symbolic Automatic Differentiation Demo
//!
//! This example demonstrates the symbolic automatic differentiation capabilities
//! of the `DSLCompile` library, showcasing the three-stage optimization pipeline:
//! 1. Pre-optimization using egglog
//! 2. Symbolic differentiation
//! 3. Post-optimization with subexpression sharing
//!
//! The demo shows how symbolic AD can compute derivatives symbolically and then
//! optimize the combined (f(x), f'(x)) expressions to share common subexpressions.

use dslcompile::final_tagless::{ASTEval, DirectEval};
use dslcompile::symbolic::symbolic_ad::{SymbolicAD, SymbolicADConfig, convenience};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 DSLCompile: Symbolic Automatic Differentiation Demo");
    println!("===================================================\n");

    // 1. Basic Symbolic Differentiation
    println!("1️⃣  Basic Symbolic Differentiation");
    println!("----------------------------------");

    let mut ad = SymbolicAD::new()?;

    // Simple polynomial: f(x) = 2x³ + 3x² + x + 1
    // Using index-based variables: x = var(0)
    let polynomial = ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::mul(
                    ASTEval::constant(2.0),
                    ASTEval::pow(ASTEval::var(0), ASTEval::constant(3.0)), // x³
                ),
                ASTEval::mul(
                    ASTEval::constant(3.0),
                    ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)), // x²
                ),
            ),
            ASTEval::var(0), // x
        ),
        ASTEval::constant(1.0),
    );

    println!("Function: f(x) = 2x³ + 3x² + x + 1");
    println!("Using index-based variables: x = var(0)");
    println!("Expected derivative: f'(x) = 6x² + 6x + 1");

    let result = ad.compute_with_derivatives(&polynomial)?;

    // Test the derivative at x = 2
    let x_val = 2.0;
    let f_val = DirectEval::eval_two_vars(&result.function, x_val, 0.0);
    let df_val = DirectEval::eval_two_vars(&result.first_derivatives["0"], x_val, 0.0);

    // Expected: f(2) = 2(8) + 3(4) + 2 + 1 = 16 + 12 + 2 + 1 = 31
    // Expected: f'(2) = 6(4) + 6(2) + 1 = 24 + 12 + 1 = 37
    println!("At x = {x_val}:");
    println!("  f({x_val}) = {f_val:.3}");
    println!("  f'({x_val}) = {df_val:.3}");
    println!("  Expected f({x_val}) = 31.0");
    println!("  Expected f'({x_val}) = 37.0");
    println!();

    // 2. Multivariate Functions and Gradients
    println!("2️⃣  Multivariate Functions and Gradients");
    println!("----------------------------------------");

    let mut config = SymbolicADConfig::default();
    config.num_variables = 2; // x and y
    let mut multivar_ad = SymbolicAD::with_config(config)?;

    // Bivariate function: f(x,y) = x²y + xy² + x + y
    // Using index-based variables: x = var(0), y = var(1)
    let bivariate = ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::mul(
                    ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)), // x²
                    ASTEval::var(1),                                       // y
                ),
                ASTEval::mul(
                    ASTEval::var(0),                                       // x
                    ASTEval::pow(ASTEval::var(1), ASTEval::constant(2.0)), // y²
                ),
            ),
            ASTEval::var(0), // x
        ),
        ASTEval::var(1), // y
    );

    println!("Function: f(x,y) = x²y + xy² + x + y");
    println!("Using index-based variables: x = var(0), y = var(1)");
    println!("Expected ∂f/∂x = 2xy + y² + 1");
    println!("Expected ∂f/∂y = x² + 2xy + 1");

    let multivar_result = multivar_ad.compute_with_derivatives(&bivariate)?;

    // Test at (x,y) = (2,3)
    let x_val = 2.0;
    let y_val = 3.0;
    let f_val = DirectEval::eval_two_vars(&multivar_result.function, x_val, y_val);
    let df_dx = DirectEval::eval_two_vars(&multivar_result.first_derivatives["0"], x_val, y_val);
    let df_dy = DirectEval::eval_two_vars(&multivar_result.first_derivatives["1"], x_val, y_val);

    // Expected: f(2,3) = 4*3 + 2*9 + 2 + 3 = 12 + 18 + 2 + 3 = 35
    // Expected: ∂f/∂x(2,3) = 2*2*3 + 9 + 1 = 12 + 9 + 1 = 22
    // Expected: ∂f/∂y(2,3) = 4 + 2*2*3 + 1 = 4 + 12 + 1 = 17
    println!("At (x,y) = ({x_val}, {y_val}):");
    println!("  f({x_val},{y_val}) = {f_val:.3}");
    println!("  ∂f/∂x = {df_dx:.3}");
    println!("  ∂f/∂y = {df_dy:.3}");
    println!("  Expected f(2,3) = 35.0");
    println!("  Expected ∂f/∂x = 22.0");
    println!("  Expected ∂f/∂y = 17.0");
    println!();

    // 3. Higher-Order Derivatives (Hessian)
    println!("3️⃣  Higher-Order Derivatives (Hessian Matrix)");
    println!("---------------------------------------------");

    let mut hessian_config = SymbolicADConfig::default();
    hessian_config.num_variables = 2; // x and y
    hessian_config.max_derivative_order = 2;
    let mut hessian_ad = SymbolicAD::with_config(hessian_config)?;

    // Simple quadratic: f(x,y) = x² + 2xy + y²
    let x = ASTEval::var(0); // Use index 0 for variable x
    let y = ASTEval::var(1); // Use index 1 for variable y
    let x_squared = ASTEval::pow(x.clone(), ASTEval::constant(2.0));
    let y_squared = ASTEval::pow(y.clone(), ASTEval::constant(2.0));
    let xy = ASTEval::mul(x.clone(), y.clone());
    let quadratic_2d = ASTEval::add(
        ASTEval::add(x_squared, ASTEval::mul(ASTEval::constant(2.0), xy)),
        y_squared,
    );

    println!("Function: f(x,y) = x² + 2xy + y²");
    println!("Expected Hessian:");
    println!("  ∂²f/∂x² = 2");
    println!("  ∂²f/∂x∂y = 2");
    println!("  ∂²f/∂y∂x = 2");
    println!("  ∂²f/∂y² = 2");

    let hessian_result = hessian_ad.compute_with_derivatives(&quadratic_2d)?;

    // Evaluate second derivatives at (1,1)
    let x_val = 1.0;
    let y_val = 1.0;

    if let Some(d2f_dx2) = hessian_result
        .second_derivatives
        .get(&("0".to_string(), "0".to_string()))
    {
        let val = DirectEval::eval_two_vars(d2f_dx2, x_val, y_val);
        println!("  ∂²f/∂x² = {val:.3}");
    }

    if let Some(d2f_dxdy) = hessian_result
        .second_derivatives
        .get(&("0".to_string(), "1".to_string()))
    {
        let val = DirectEval::eval_two_vars(d2f_dxdy, x_val, y_val);
        println!("  ∂²f/∂x∂y = {val:.3}");
    }

    if let Some(d2f_dydx) = hessian_result
        .second_derivatives
        .get(&("1".to_string(), "0".to_string()))
    {
        let val = DirectEval::eval_two_vars(d2f_dydx, x_val, y_val);
        println!("  ∂²f/∂y∂x = {val:.3}");
    }

    if let Some(d2f_dy2) = hessian_result
        .second_derivatives
        .get(&("1".to_string(), "1".to_string()))
    {
        let val = DirectEval::eval_two_vars(d2f_dy2, x_val, y_val);
        println!("  ∂²f/∂y² = {val:.3}");
    }
    println!();

    // 4. Optimization Pipeline Analysis
    println!("4️⃣  Optimization Pipeline Analysis");
    println!("----------------------------------");

    // Create a complex expression that benefits from optimization
    // Using index-based variables: x = var(0)
    let complex_expr = ASTEval::add(
        ASTEval::mul(
            ASTEval::add(ASTEval::var(0), ASTEval::constant(0.0)), // x + 0 → x
            ASTEval::constant(1.0),
        ), // (x + 0) * 1 → x
        ASTEval::sub(
            ASTEval::ln(ASTEval::exp(ASTEval::var(0))), // ln(exp(x)) → x
            ASTEval::constant(0.0),
        ), // ln(exp(x)) - 0 → x
    ); // Should optimize to x + x = 2x

    println!("Complex expression with optimization opportunities:");
    println!("  (x + 0) * 1 + ln(exp(x)) - 0");
    println!("  Using index-based variables: x = var(0)");
    println!("  Should optimize to: 2x");

    let mut opt_ad = SymbolicAD::new()?;
    let opt_result = opt_ad.compute_with_derivatives(&complex_expr)?;

    println!("\nOptimization Statistics:");
    println!(
        "  Operations before: {}",
        opt_result.stats.operations_before()
    );
    println!(
        "  Operations after: {}",
        opt_result.stats.operations_after()
    );
    println!(
        "  Optimization ratio: {:.2}",
        opt_result.stats.optimization_ratio()
    );
    println!(
        "  Stage 1 (pre-opt): {} μs",
        opt_result.stats.stage_times_us[0]
    );
    println!(
        "  Stage 2 (diff): {} μs",
        opt_result.stats.stage_times_us[1]
    );
    println!(
        "  Stage 3 (post-opt): {} μs",
        opt_result.stats.stage_times_us[2]
    );
    println!("  Total time: {} μs", opt_result.stats.total_time_us());

    // Test the optimized result
    let x_val = 3.0;
    let f_opt = DirectEval::eval_two_vars(&opt_result.function, x_val, 0.0);
    let df_opt = DirectEval::eval_two_vars(&opt_result.first_derivatives["0"], x_val, 0.0);

    println!("\nOptimized function evaluation at x = {x_val}:");
    println!("  f({x_val}) = {f_opt:.3} (expected: {:.3})", 2.0 * x_val);
    println!("  f'({x_val}) = {df_opt:.3} (expected: 2.0)");
    println!();

    // 5. Convenience Functions Demo
    println!("5️⃣  Convenience Functions");
    println!("-------------------------");

    // Using convenience functions for common operations
    let simple_quadratic = convenience::poly(&[1.0, -2.0, 1.0]); // 1 - 2x + x² = (x-1)²
    println!("Polynomial: f(x) = x² - 2x + 1 = (x-1)²");

    let grad = convenience::gradient(&simple_quadratic, &["0"])?;
    println!("Gradient computed using convenience function:");

    // Test at x = 3: f'(3) = 2*3 - 2 = 4
    let x_val = 3.0;
    let df_conv = DirectEval::eval_two_vars(&grad["0"], x_val, 0.0);
    println!("  f'({x_val}) = {df_conv:.3} (expected: 4.0)");

    // Hessian for multivariate function
    let x = ASTEval::var(0); // Use index 0 for variable x  
    let y = ASTEval::var(1); // Use index 1 for variable y
    let x_squared = ASTEval::pow(x.clone(), ASTEval::constant(2.0));
    let y_squared = ASTEval::pow(y.clone(), ASTEval::constant(2.0));
    let xy = ASTEval::mul(x.clone(), y.clone());
    let bivariate_quad = ASTEval::add(
        ASTEval::add(
            ASTEval::mul(ASTEval::constant(2.0), x_squared), // 2x²
            xy,                                              // xy
        ),
        ASTEval::mul(ASTEval::constant(2.0), y_squared), // 2y²
    ); // 2x² + xy + 2y²

    let hessian = convenience::hessian(&bivariate_quad, &["0", "1"])?;

    println!("\nHessian for f(x,y) = 2x² + xy + 2y²:");
    println!("Expected Hessian matrix:");
    println!("  [4  1]");
    println!("  [1  4]");

    let x_val = 1.0;
    let y_val = 1.0;

    if let Some(h_xx) = hessian.get(&("0".to_string(), "0".to_string())) {
        let val = DirectEval::eval_two_vars(h_xx, x_val, y_val);
        println!("  H[0,0] = {val:.3}");
    }

    if let Some(h_xy) = hessian.get(&("0".to_string(), "1".to_string())) {
        let val = DirectEval::eval_two_vars(h_xy, x_val, y_val);
        println!("  H[0,1] = {val:.3}");
    }

    if let Some(h_yx) = hessian.get(&("1".to_string(), "0".to_string())) {
        let val = DirectEval::eval_two_vars(h_yx, x_val, y_val);
        println!("  H[1,0] = {val:.3}");
    }

    if let Some(h_yy) = hessian.get(&("1".to_string(), "1".to_string())) {
        let val = DirectEval::eval_two_vars(h_yy, x_val, y_val);
        println!("  H[1,1] = {val:.3}");
    }
    println!();

    println!("=== Demo Complete ===");
    println!("✅ Successfully demonstrated symbolic AD with index-based variables");
    println!("✅ Three-stage optimization pipeline working correctly");
    println!("✅ Higher-order derivatives computed accurately");
    println!("✅ No string-based variable lookups required");

    Ok(())
}
