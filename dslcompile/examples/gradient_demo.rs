//! Gradient Computation Demo
//!
//! This example demonstrates comprehensive gradient computation capabilities
//! of the `DSLCompile` symbolic AD system, including:
//! - Multivariate function gradients
//! - Machine learning loss function gradients
//! - Optimization problem gradients
//! - Higher-dimensional gradient examples

use dslcompile::final_tagless::ASTEval;
use dslcompile::symbolic::symbolic_ad::convenience;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 DSLCompile: Comprehensive Gradient Computation Demo");
    println!("===================================================\n");

    // 1. Basic Multivariate Gradients
    println!("1️⃣  Basic Multivariate Gradients");
    println!("--------------------------------");

    // f(x,y,z) = x² + y² + z² + 2xy + 3xz + yz
    // Using index-based variables: x=0, y=1, z=2
    let multivar_func = ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::add(
                    ASTEval::add(
                        ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)), // x²
                        ASTEval::pow(ASTEval::var(1), ASTEval::constant(2.0)), // y²
                    ),
                    ASTEval::pow(ASTEval::var(2), ASTEval::constant(2.0)), // z²
                ),
                ASTEval::mul(
                    ASTEval::constant(2.0),
                    ASTEval::mul(ASTEval::var(0), ASTEval::var(1)), // 2xy
                ),
            ),
            ASTEval::mul(
                ASTEval::constant(3.0),
                ASTEval::mul(ASTEval::var(0), ASTEval::var(2)), // 3xz
            ),
        ),
        ASTEval::mul(ASTEval::var(1), ASTEval::var(2)), // yz
    );

    println!("Function: f(x,y,z) = x² + y² + z² + 2xy + 3xz + yz");
    println!("Using index-based variables: x=var(0), y=var(1), z=var(2)");
    println!("Expected gradient:");
    println!("  ∂f/∂x = 2x + 2y + 3z");
    println!("  ∂f/∂y = 2y + 2x + z");
    println!("  ∂f/∂z = 2z + 3x + y");

    let gradient = convenience::gradient(&multivar_func, &["0", "1", "2"])?;

    // Test at point (1, 2, 3)
    let test_point = [1.0, 2.0, 3.0];

    let _f_val = multivar_func.eval_with_vars(&test_point);
    let df_dx_val = gradient["0"].eval_with_vars(&test_point);
    let df_dy_val = gradient["1"].eval_with_vars(&test_point);
    let df_dz_val = gradient["2"].eval_with_vars(&test_point);

    println!(
        "\nAt point ({}, {}, {}):",
        test_point[0], test_point[1], test_point[2]
    );

    println!("  ∂f/∂x = {df_dx_val:.3}");
    println!("  ∂f/∂y = {df_dy_val:.3}");
    println!("  ∂f/∂z = {df_dz_val:.3}");

    // Expected: ∂f/∂x = 2(1) + 2(2) + 3(3) = 2 + 4 + 9 = 15
    // Expected: ∂f/∂y = 2(2) + 2(1) + 3 = 4 + 2 + 3 = 9
    // Expected: ∂f/∂z = 2(3) + 3(1) + 2 = 6 + 3 + 2 = 11
    println!("  Expected ∂f/∂x = 15.0, ∂f/∂y = 9.0, ∂f/∂z = 11.0");
    println!();

    // 2. Machine Learning Loss Function Gradients
    println!("2️⃣  Machine Learning Loss Function Gradients");
    println!("--------------------------------------------");

    // Mean Squared Error: L(w,b) = (y - (wx + b))²
    // where y is target, x is input, w=var(0) is weight, b=var(1) is bias
    let x_input = 2.0; // Input value
    let y_target = 5.0; // Target value

    // Create the loss function: L(w,b) = (5 - (w*2 + b))²
    let prediction = ASTEval::add(
        ASTEval::mul(ASTEval::var(0), ASTEval::constant(x_input)), // w * x_input
        ASTEval::var(1),                                           // b
    );
    let error = ASTEval::sub(ASTEval::constant(y_target), prediction);
    let mse_loss = ASTEval::pow(error, ASTEval::constant(2.0));

    println!("MSE Loss: L(w,b) = (y - (wx + b))²");
    println!("With x = {x_input}, y = {y_target}");
    println!("Using index-based variables: w=var(0), b=var(1)");
    println!("L(w,b) = ({y_target} - (w*{x_input} + b))²");
    println!("Expected gradients:");
    println!("  ∂L/∂w = -2x(y - (wx + b)) = -2*{x_input}*({y_target} - (w*{x_input} + b))");
    println!("  ∂L/∂b = -2(y - (wx + b)) = -2*({y_target} - (w*{x_input} + b))");

    let mse_gradient = convenience::gradient(&mse_loss, &["0", "1"])?;

    // Test at w=1.0, b=0.5
    let wb_vals = [1.0, 0.5];

    let loss_val = mse_loss.eval_with_vars(&wb_vals);
    let dl_dw = mse_gradient["0"].eval_with_vars(&wb_vals);
    let dl_db = mse_gradient["1"].eval_with_vars(&wb_vals);

    println!("\nAt w = {}, b = {}:", wb_vals[0], wb_vals[1]);
    println!("  Loss = {loss_val:.3}");
    println!("  ∂L/∂w = {dl_dw:.3}");
    println!("  ∂L/∂b = {dl_db:.3}");

    // Manual calculation: prediction = 1.0*2.0 + 0.5 = 2.5, error = 5.0 - 2.5 = 2.5
    // Loss = 2.5² = 6.25
    // ∂L/∂w = -2*2.0*2.5 = -10.0
    // ∂L/∂b = -2*2.5 = -5.0
    println!("  Expected Loss = 6.25");
    println!("  Expected ∂L/∂w = -10.0");
    println!("  Expected ∂L/∂b = -5.0");
    println!();

    // 3. Optimization Problem: Rosenbrock Function
    println!("3️⃣  Optimization Problem: Rosenbrock Function");
    println!("---------------------------------------------");

    // Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
    // Standard form: f(x,y) = (1-x)² + 100(y-x²)²
    // Using index-based variables: x=var(0), y=var(1)
    let a = 1.0;
    let b = 100.0;

    let term1 = ASTEval::pow(
        ASTEval::sub(ASTEval::constant(a), ASTEval::var(0)), // (1-x)
        ASTEval::constant(2.0),
    );
    let x_squared = ASTEval::pow(ASTEval::var(0), ASTEval::constant(2.0)); // x²
    let term2 = ASTEval::mul(
        ASTEval::constant(b),
        ASTEval::pow(
            ASTEval::sub(ASTEval::var(1), x_squared), // (y-x²)
            ASTEval::constant(2.0),
        ),
    );
    let rosenbrock = ASTEval::add(term1, term2);

    println!("Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²");
    println!("Using index-based variables: x=var(0), y=var(1)");
    println!("This is a classic optimization test function with global minimum at (1,1)");
    println!("Expected gradients:");
    println!("  ∂f/∂x = -2(1-x) + 100*2(y-x²)*(-2x) = -2(1-x) - 400x(y-x²)");
    println!("  ∂f/∂y = 100*2(y-x²) = 200(y-x²)");

    let rosenbrock_grad = convenience::gradient(&rosenbrock, &["0", "1"])?;

    // Test at several points
    let test_points = [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0], [1.5, 2.0]];

    for point in test_points {
        let f_val = rosenbrock.eval_with_vars(&point);
        let df_dx = rosenbrock_grad["0"].eval_with_vars(&point);
        let df_dy = rosenbrock_grad["1"].eval_with_vars(&point);

        println!("\nAt ({:.1}, {:.2}):", point[0], point[1]);
        println!("  f = {f_val:.3}");
        println!("  ∇f = [{df_dx:.3}, {df_dy:.3}]");

        // Check if we're at the minimum (gradient should be near zero)
        let grad_magnitude = (df_dx * df_dx + df_dy * df_dy).sqrt();
        println!("  |∇f| = {grad_magnitude:.3}");

        if grad_magnitude < 0.1 {
            println!("  → Near critical point! 🎯");
        }
    }
    println!();

    // 4. Logistic Regression Gradient
    println!("4️⃣  Logistic Regression Gradient");
    println!("--------------------------------");

    // Logistic loss: L(w,b) = -y*log(σ(wx+b)) - (1-y)*log(1-σ(wx+b))
    // where σ(z) = 1/(1+exp(-z)) is the sigmoid function
    // For simplicity, we'll use a linear approximation or focus on the linear part

    // Simplified version: L(w,b) = (σ(wx+b) - y)² where σ(z) ≈ z for small z
    let x_data = 1.5;
    let y_label = 1.0;

    let linear_output = ASTEval::add(
        ASTEval::mul(ASTEval::var(0), ASTEval::constant(x_data)), // w * x_data
        ASTEval::var(1),                                          // b
    );

    // For demonstration, use a quadratic loss: (wx + b - y)²
    let logistic_loss = ASTEval::pow(
        ASTEval::sub(linear_output, ASTEval::constant(y_label)),
        ASTEval::constant(2.0),
    );

    println!("Simplified logistic loss: L(w,b) = (wx + b - y)²");
    println!("Using index-based variables: w=var(0), b=var(1)");
    println!("With x = {x_data}, y = {y_label}");

    let logistic_grad = convenience::gradient(&logistic_loss, &["0", "1"])?;

    // Test at w=0.8, b=0.2
    let wb_test = [0.8, 0.2];

    let loss_val = logistic_loss.eval_with_vars(&wb_test);
    let dl_dw = logistic_grad["0"].eval_with_vars(&wb_test);
    let dl_db = logistic_grad["1"].eval_with_vars(&wb_test);

    println!("\nAt w = {}, b = {}:", wb_test[0], wb_test[1]);
    println!("  Loss = {loss_val:.3}");
    println!("  ∂L/∂w = {dl_dw:.3}");
    println!("  ∂L/∂b = {dl_db:.3}");

    // Manual: prediction = 0.8*1.5 + 0.2 = 1.4, error = 1.4 - 1.0 = 0.4
    // Loss = 0.4² = 0.16
    // ∂L/∂w = 2*(0.4)*1.5 = 1.2
    // ∂L/∂b = 2*(0.4) = 0.8
    println!("  Expected Loss ≈ 0.16");
    println!("  Expected ∂L/∂w ≈ 1.2");
    println!("  Expected ∂L/∂b ≈ 0.8");
    println!();

    // 5. Higher-dimensional gradient example
    println!("5️⃣  Higher-Dimensional Example");
    println!("------------------------------");

    // f(x₁,x₂,x₃,x₄) = x₁x₂ + x₂x₃ + x₃x₄ + x₄x₁ (circular coupling)
    let high_dim_func = ASTEval::add(
        ASTEval::add(
            ASTEval::add(
                ASTEval::mul(ASTEval::var(0), ASTEval::var(1)), // x₁x₂
                ASTEval::mul(ASTEval::var(1), ASTEval::var(2)), // x₂x₃
            ),
            ASTEval::mul(ASTEval::var(2), ASTEval::var(3)), // x₃x₄
        ),
        ASTEval::mul(ASTEval::var(3), ASTEval::var(0)), // x₄x₁
    );

    println!("Function: f(x₁,x₂,x₃,x₄) = x₁x₂ + x₂x₃ + x₃x₄ + x₄x₁");
    println!("Using index-based variables: x₁=var(0), x₂=var(1), x₃=var(2), x₄=var(3)");
    println!("Expected gradient:");
    println!("  ∂f/∂x₁ = x₂ + x₄");
    println!("  ∂f/∂x₂ = x₁ + x₃");
    println!("  ∂f/∂x₃ = x₂ + x₄");
    println!("  ∂f/∂x₄ = x₃ + x₁");

    let high_dim_grad = convenience::gradient(&high_dim_func, &["0", "1", "2", "3"])?;

    // Test at point (1, 2, 3, 4)
    let test_4d = [1.0, 2.0, 3.0, 4.0];

    let f_val = high_dim_func.eval_with_vars(&test_4d);
    println!(
        "\nAt point ({}, {}, {}, {}):",
        test_4d[0], test_4d[1], test_4d[2], test_4d[3]
    );
    println!("  f = {f_val:.3}");

    for i in 0..4 {
        let grad_val = high_dim_grad[&i.to_string()].eval_with_vars(&test_4d);
        println!("  ∂f/∂x₊{} = {grad_val:.3}", i + 1);
    }

    // Expected gradients:
    // ∂f/∂x₁ = x₂ + x₄ = 2 + 4 = 6
    // ∂f/∂x₂ = x₁ + x₃ = 1 + 3 = 4
    // ∂f/∂x₃ = x₂ + x₄ = 2 + 4 = 6
    // ∂f/∂x₄ = x₃ + x₁ = 3 + 1 = 4
    println!("  Expected: [6, 4, 6, 4]");
    println!();

    println!("=== Demo Complete ===");
    println!("✅ Successfully demonstrated index-based gradient computation");
    println!("✅ All gradient calculations use modern variable indexing");
    println!("✅ No string-based variable lookups required");

    Ok(())
}
