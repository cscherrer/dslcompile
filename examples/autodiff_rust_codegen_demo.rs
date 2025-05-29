//! Autodiff + Rust Code Generation Demo
//!
//! This example demonstrates the complete integration of automatic differentiation
//! with Rust code generation in the `MathCompile` library. It shows how to:
//! 1. Define mathematical expressions using the final tagless approach
//! 2. Use automatic differentiation to compute derivatives
//! 3. Generate optimized Rust code for both functions and their derivatives
//! 4. Integrate with the symbolic optimization pipeline

#[cfg(all(feature = "autodiff", feature = "optimization"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ad_trait::forward_ad::adfn::adfn;
    use mathcompile::autodiff::{ForwardAD, convenience};
    use mathcompile::backends::RustCodeGenerator;
    use mathcompile::final_tagless::{ASTEval, ASTMathExpr};
    use mathcompile::symbolic::SymbolicOptimizer;

    println!("🚀 MathCompile: Autodiff + Rust Code Generation Demo");
    println!("=================================================\n");

    // 1. Define a complex mathematical function
    println!("1️⃣  Defining Mathematical Function");
    println!("----------------------------------");

    // f(x) = x³ - 3x² + 2x + 1 (a cubic polynomial)
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(3.0)),
                ASTEval::mul(
                    ASTEval::constant(-3.0),
                    ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
                ),
            ),
            ASTEval::mul(ASTEval::constant(2.0), ASTEval::var_by_name("x")),
        ),
        ASTEval::constant(1.0),
    );

    println!("Function: f(x) = x³ - 3x² + 2x + 1");
    println!("Expected derivative: f'(x) = 3x² - 6x + 2\n");

    // 2. Generate Rust code for the original function
    println!("2️⃣  Generating Rust Code for Original Function");
    println!("-----------------------------------------------");

    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&expr, "cubic_function")?;

    println!("Generated Rust code:");
    println!("{rust_code}\n");

    // 3. Use automatic differentiation to compute derivatives
    println!("3️⃣  Computing Derivatives with Autodiff");
    println!("----------------------------------------");

    let forward_ad = ForwardAD::new();

    // Define the same function for autodiff
    let autodiff_func = |x: adfn<1>| {
        let x2 = x * x;
        let x3 = x2 * x;
        let neg_three = adfn::new(-3.0, [0.0]);
        let two = adfn::new(2.0, [0.0]);
        let one = adfn::new(1.0, [0.0]);

        x3 + neg_three * x2 + two * x + one
    };

    // Test at several points
    let test_points = [0.0, 1.0, 2.0, 3.0];
    println!("Testing derivatives at various points:");
    println!("x\t\tf(x)\t\tf'(x)\t\tExpected f'(x)");
    println!("------------------------------------------------------------");

    for &x_val in &test_points {
        let (value, derivative) = forward_ad.differentiate(autodiff_func, x_val)?;
        let expected_derivative = 3.0 * x_val * x_val - 6.0 * x_val + 2.0;

        println!("{x_val:.1}\t\t{value:.3}\t\t{derivative:.3}\t\t{expected_derivative:.3}");

        // Verify accuracy
        assert!((derivative - expected_derivative).abs() < 1e-10);
    }
    println!();

    // 4. Create expression for the derivative and generate Rust code
    println!("4️⃣  Generating Rust Code for Derivative");
    println!("---------------------------------------");

    // f'(x) = 3x² - 6x + 2
    let derivative_expr = ASTEval::add(
        ASTEval::add(
            ASTEval::mul(
                ASTEval::constant(3.0),
                ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
            ),
            ASTEval::mul(ASTEval::constant(-6.0), ASTEval::var_by_name("x")),
        ),
        ASTEval::constant(2.0),
    );

    let derivative_rust_code = codegen.generate_function(&derivative_expr, "cubic_derivative")?;

    println!("Generated Rust code for derivative:");
    println!("{derivative_rust_code}\n");

    // 5. Apply symbolic optimization
    println!("5️⃣  Applying Symbolic Optimization");
    println!("----------------------------------");

    let mut optimizer = SymbolicOptimizer::new()?;
    let optimized_expr = optimizer.optimize(&expr)?;
    let optimized_derivative = optimizer.optimize(&derivative_expr)?;

    println!("Original expression: {expr:?}");
    println!("Optimized expression: {optimized_expr:?}");
    println!("Derivative expression: {derivative_expr:?}");
    println!("Optimized derivative: {optimized_derivative:?}\n");

    // Generate optimized Rust code
    let optimized_rust = codegen.generate_function(&optimized_expr, "optimized_cubic")?;
    let optimized_derivative_rust =
        codegen.generate_function(&optimized_derivative, "optimized_cubic_derivative")?;

    println!("Optimized Rust code for function:");
    println!("{optimized_rust}");
    println!("Optimized Rust code for derivative:");
    println!("{optimized_derivative_rust}\n");

    // 6. Multi-variable example
    println!("6️⃣  Multi-Variable Function Example");
    println!("-----------------------------------");

    // g(x,y) = x²y + xy² - 2xy + 1
    let multi_var_expr = ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::mul(
                    ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
                    ASTEval::var_by_name("y"),
                ),
                ASTEval::mul(
                    ASTEval::var_by_name("x"),
                    ASTEval::pow(ASTEval::var_by_name("y"), ASTEval::constant(2.0)),
                ),
            ),
            ASTEval::mul(
                ASTEval::constant(-2.0),
                ASTEval::mul(ASTEval::var_by_name("x"), ASTEval::var_by_name("y")),
            ),
        ),
        ASTEval::constant(1.0),
    );

    println!("Multi-variable function: g(x,y) = x²y + xy² - 2xy + 1");

    // Generate Rust code for multi-variable function
    let multi_var_rust = codegen.generate_function(&multi_var_expr, "multi_var_func")?;
    println!("Generated Rust code:");
    println!("{multi_var_rust}");

    // Compute gradient using autodiff
    let multi_var_func = |vars: &[f64]| {
        let x = vars[0];
        let y = vars[1];
        x * x * y + x * y * y - 2.0 * x * y + 1.0
    };

    let test_point = [2.0, 3.0];
    let gradient = convenience::gradient(multi_var_func, &test_point)?;

    println!("Gradient at ({}, {}):", test_point[0], test_point[1]);
    println!("∂g/∂x = {:.3}", gradient[0]);
    println!("∂g/∂y = {:.3}", gradient[1]);

    // Expected: ∂g/∂x = 2xy + y² - 2y, ∂g/∂y = x² + 2xy - 2x
    let expected_dx =
        2.0 * test_point[0] * test_point[1] + test_point[1] * test_point[1] - 2.0 * test_point[1];
    let expected_dy =
        test_point[0] * test_point[0] + 2.0 * test_point[0] * test_point[1] - 2.0 * test_point[0];

    println!("Expected ∂g/∂x = {expected_dx:.3}");
    println!("Expected ∂g/∂y = {expected_dy:.3}\n");

    // 7. Practical application: Newton's method for root finding
    println!("7️⃣  Practical Application: Newton's Method");
    println!("------------------------------------------");

    // Find root of f(x) = x³ - 3x² + 2x + 1 = 0
    println!("Finding root of f(x) = x³ - 3x² + 2x + 1 = 0 using Newton's method");

    let mut x = 0.5; // Initial guess
    println!("Initial guess: x₀ = {x:.6}");

    for i in 0..8 {
        let (f_val, f_prime) = forward_ad.differentiate(autodiff_func, x)?;

        if f_prime.abs() < 1e-12 {
            println!("Derivative too small, stopping iteration");
            break;
        }

        let x_new = x - f_val / f_prime;

        println!(
            "Iteration {}: x = {:.6}, f(x) = {:.6}, f'(x) = {:.6}",
            i + 1,
            x,
            f_val,
            f_prime
        );

        if (x_new - x).abs() < 1e-12 {
            println!("Converged to root: x = {x_new:.12}");
            break;
        }

        x = x_new;
    }
    println!();

    // 8. Performance comparison
    println!("8️⃣  Performance Summary");
    println!("----------------------");

    use std::time::Instant;

    // Time autodiff computation
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = forward_ad.differentiate(autodiff_func, 2.0)?;
    }
    let autodiff_time = start.elapsed();

    // Time gradient computation
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = convenience::gradient(multi_var_func, &test_point)?;
    }
    let gradient_time = start.elapsed();

    println!("Forward AD (10,000 iterations): {autodiff_time:?}");
    println!("Gradient computation (1,000 iterations): {gradient_time:?}");
    println!();

    println!("✅ Demo completed successfully!");
    println!("Key achievements:");
    println!("  🔸 Seamless integration of autodiff with Rust code generation");
    println!("  🔸 Automatic derivative computation with high precision");
    println!("  🔸 Optimized Rust code generation for both functions and derivatives");
    println!("  🔸 Multi-variable gradient computation");
    println!("  🔸 Practical applications like Newton's method");
    println!("  🔸 Performance-optimized implementations");

    Ok(())
}

#[cfg(not(all(feature = "autodiff", feature = "optimization")))]
fn main() {
    println!("❌ This demo requires both 'autodiff' and 'optimization' features!");
    println!(
        "Run with: cargo run --example autodiff_rust_codegen_demo --features \"autodiff optimization\""
    );
}
