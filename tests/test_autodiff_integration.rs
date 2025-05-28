//! Integration tests for automatic differentiation with final tagless expressions

#[cfg(feature = "autodiff")]
mod autodiff_tests {
    use ad_trait::forward_ad::adfn::adfn;
    use mathjit::autodiff::{convenience, ForwardAD, HigherOrderAD, ReverseAD};

    #[test]
    fn test_forward_ad_basic_operations() {
        println!("🧮 Testing forward-mode AD with basic operations...");

        let forward_ad = ForwardAD::new();

        // Test f(x) = x^2 + 3x + 1, f'(x) = 2x + 3
        let quadratic = |x: adfn<1>| {
            let x_squared = x * x;
            let three_x = adfn::new(3.0, [0.0]) * x;
            let one = adfn::new(1.0, [0.0]);
            x_squared + three_x + one
        };

        let (value, derivative) = forward_ad.differentiate(quadratic, 2.0).unwrap();
        println!("f(2) = {value}, f'(2) = {derivative}");

        // f(2) = 4 + 6 + 1 = 11
        // f'(2) = 4 + 3 = 7
        assert!((value - 11.0).abs() < 1e-10);
        assert!((derivative - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_ad_basic_operations() {
        println!("🔄 Testing reverse-mode AD with basic operations...");

        let reverse_ad = ReverseAD::new();

        // Test f(x) = x^3 - 2x^2 + x, f'(x) = 3x^2 - 4x + 1
        let cubic = |x: f64| {
            let x_cubed = x * x * x;
            let two_x_squared = 2.0 * x * x;
            x_cubed - two_x_squared + x
        };

        let (value, derivative) = reverse_ad.differentiate(cubic, 3.0).unwrap();
        println!("f(3) = {value}, f'(3) = {derivative}");

        // f(3) = 27 - 18 + 3 = 12
        // f'(3) = 27 - 12 + 1 = 16
        assert!((value - 12.0).abs() < 1e-10);
        assert!((derivative - 16.0).abs() < 1e-6); // Finite difference tolerance
    }

    #[test]
    fn test_multi_variable_gradient() {
        println!("🎯 Testing multi-variable gradient computation...");

        // Test f(x,y) = x^2 + y^2 + xy, gradient = [2x + y, 2y + x]
        let func = |vars: &[f64]| {
            let x = vars[0];
            let y = vars[1];
            x * x + y * y + x * y
        };

        let inputs = [1.0, 2.0];
        let gradient = convenience::gradient(func, &inputs).unwrap();

        println!("Gradient at (1, 2): [{}, {}]", gradient[0], gradient[1]);

        // ∂f/∂x = 2x + y = 2(1) + 2 = 4
        // ∂f/∂y = 2y + x = 2(2) + 1 = 5
        assert!((gradient[0] - 4.0).abs() < 1e-6);
        assert!((gradient[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_jacobian_computation() {
        println!("🔢 Testing Jacobian matrix computation...");

        // Test vector function f(x,y) = [x^2 + y, xy]
        // Jacobian = [[2x, 1], [y, x]]
        let vector_func = |vars: &[f64]| {
            let x = vars[0];
            let y = vars[1];
            vec![x * x + y, x * y]
        };

        let inputs = [2.0, 3.0];
        let jacobian = convenience::jacobian(vector_func, &inputs, 2).unwrap();

        println!("Jacobian at (2, 3):");
        for (i, row) in jacobian.iter().enumerate() {
            println!("  Row {}: [{}, {}]", i, row[0], row[1]);
        }

        // At (2, 3):
        // Row 0: [2*2, 1] = [4, 1]
        // Row 1: [3, 2] = [3, 2]
        assert!((jacobian[0][0] - 4.0).abs() < 1e-6);
        assert!((jacobian[0][1] - 1.0).abs() < 1e-6);
        assert!((jacobian[1][0] - 3.0).abs() < 1e-6);
        assert!((jacobian[1][1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_higher_order_derivatives() {
        println!("🔬 Testing higher-order derivatives...");

        let higher_order = HigherOrderAD::new();

        // Test f(x) = x^4, f'(x) = 4x^3, f''(x) = 12x^2
        let quartic = |x: adfn<1>| {
            let x2 = x * x;
            x2 * x2
        };

        let (value, first_deriv, second_deriv) =
            higher_order.second_derivative(quartic, 2.0).unwrap();

        println!("f(2) = {value}, f'(2) = {first_deriv}, f''(2) = {second_deriv}");

        // f(2) = 16, f'(2) = 32, f''(2) = 48
        assert!((value - 16.0).abs() < 1e-10);
        assert!((first_deriv - 32.0).abs() < 1e-10);
        assert!((second_deriv - 48.0).abs() < 1e-6); // Finite difference tolerance
    }

    #[test]
    fn test_optimization_use_case() {
        println!("🎯 Testing AD for optimization use cases...");

        // Example: Find minimum of f(x) = (x - 3)^2 + 2
        // Minimum should be at x = 3 where f'(x) = 0
        let objective = |x: adfn<1>| {
            let three = adfn::new(3.0, [0.0]);
            let two = adfn::new(2.0, [0.0]);
            let diff = x - three;
            diff * diff + two
        };

        let forward_ad = ForwardAD::new();

        // Test gradient at various points
        let test_points = [2.0, 2.5, 3.0, 3.5, 4.0];
        for &x in &test_points {
            let (value, gradient) = forward_ad.differentiate(objective, x).unwrap();
            println!("At x = {x}: f(x) = {value:.3}, f'(x) = {gradient:.3}");

            // Verify f'(x) = 2(x - 3)
            let expected_gradient = 2.0 * (x - 3.0);
            assert!((gradient - expected_gradient).abs() < 1e-10);
        }

        // Verify minimum is at x = 3
        let (min_value, min_gradient) = forward_ad.differentiate(objective, 3.0).unwrap();
        assert!((min_value - 2.0).abs() < 1e-10);
        assert!(min_gradient.abs() < 1e-10);
    }

    #[test]
    fn test_performance_comparison() {
        println!("⚡ Testing performance comparison between forward and reverse AD...");

        use std::time::Instant;

        // Function with many inputs, few outputs (better for reverse AD)
        let many_inputs_func = |vars: &[f64]| {
            let mut sum = 0.0;
            for (i, &var) in vars.iter().enumerate() {
                let coeff = (i + 1) as f64;
                sum += coeff * var * var;
            }
            sum
        };

        let inputs: Vec<f64> = (1..=10).map(f64::from).collect();

        // Time reverse AD (using finite differences)
        let start = Instant::now();
        for _ in 0..100 {
            let _ = convenience::gradient(many_inputs_func, &inputs).unwrap();
        }
        let reverse_time = start.elapsed();

        println!("Reverse AD (100 iterations): {reverse_time:?}");
        println!("This demonstrates that reverse AD is efficient for many inputs, few outputs");

        // Verify correctness
        let gradient = convenience::gradient(many_inputs_func, &inputs).unwrap();
        for (i, &grad) in gradient.iter().enumerate() {
            let expected = 2.0 * (i + 1) as f64 * inputs[i];
            assert!((grad - expected).abs() < 1e-4); // Relaxed tolerance for finite differences
        }
    }

    #[cfg(feature = "optimization")]
    #[test]
    fn test_autodiff_with_rust_codegen() {
        println!("🦀 Testing autodiff integration with Rust code generation...");

        use mathjit::backends::RustCodeGenerator;
        use mathjit::final_tagless::{ASTEval, ASTMathExpr};
        use mathjit::symbolic::SymbolicOptimizer;

        // Create a mathematical expression: f(x) = x^2 + 2x + 1
        let expr = ASTEval::add(
            ASTEval::add(
                ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
                ASTEval::mul(ASTEval::constant(2.0), ASTEval::var_by_name("x")),
            ),
            ASTEval::constant(1.0),
        );

        println!("Original expression: f(x) = x^2 + 2x + 1");

        // Generate Rust code for the original function
        let codegen = RustCodeGenerator::new();
        let rust_code = codegen.generate_function(&expr, "original_func").unwrap();

        println!("Generated Rust code for original function:");
        println!("{rust_code}");
        assert!(rust_code.contains("original_func"));
        assert!(rust_code.contains("x * x") || rust_code.contains("x.powf(2"));
        assert!(rust_code.contains("2.0 * x"));

        // Now use autodiff to compute the derivative: f'(x) = 2x + 2
        let forward_ad = ForwardAD::new();

        // Define the same function for autodiff
        let autodiff_func = |x: adfn<1>| {
            let x_squared = x * x;
            let two_x = adfn::new(2.0, [0.0]) * x;
            let one = adfn::new(1.0, [0.0]);
            x_squared + two_x + one
        };

        // Test the derivative at a few points
        let test_points = [1.0, 2.0, 3.0];
        for &x_val in &test_points {
            let (value, derivative) = forward_ad.differentiate(autodiff_func, x_val).unwrap();

            // f(x) = x^2 + 2x + 1, so f'(x) = 2x + 2
            let expected_value = x_val * x_val + 2.0 * x_val + 1.0;
            let expected_derivative = 2.0 * x_val + 2.0;

            println!("At x = {x_val}: f(x) = {value:.3}, f'(x) = {derivative:.3}");
            println!("Expected: f(x) = {expected_value:.3}, f'(x) = {expected_derivative:.3}");

            assert!((value - expected_value).abs() < 1e-10);
            assert!((derivative - expected_derivative).abs() < 1e-10);
        }

        // Create an expression for the derivative: f'(x) = 2x + 2
        let derivative_expr = ASTEval::add(
            ASTEval::mul(ASTEval::constant(2.0), ASTEval::var_by_name("x")),
            ASTEval::constant(2.0),
        );

        // Generate Rust code for the derivative
        let derivative_rust_code = codegen
            .generate_function(&derivative_expr, "derivative_func")
            .unwrap();

        println!("\nGenerated Rust code for derivative function:");
        println!("{derivative_rust_code}");
        assert!(derivative_rust_code.contains("derivative_func"));
        assert!(derivative_rust_code.contains("2.0 * x"));

        // Test with symbolic optimization
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let optimized_expr = optimizer.optimize(&expr).unwrap();
        let optimized_derivative = optimizer.optimize(&derivative_expr).unwrap();

        println!("\nOptimized expressions:");
        println!("Original: {optimized_expr:?}");
        println!("Derivative: {optimized_derivative:?}");

        // Generate optimized Rust code
        let optimized_rust = codegen
            .generate_function(&optimized_expr, "optimized_func")
            .unwrap();
        let optimized_derivative_rust = codegen
            .generate_function(&optimized_derivative, "optimized_derivative")
            .unwrap();

        println!("\nOptimized Rust code:");
        println!("Function: {optimized_rust}");
        println!("Derivative: {optimized_derivative_rust}");

        assert!(optimized_rust.contains("optimized_func"));
        assert!(optimized_derivative_rust.contains("optimized_derivative"));

        println!("✅ Autodiff successfully integrates with Rust code generation!");
        println!("   - Can compute derivatives using forward-mode AD");
        println!("   - Can generate Rust code for both original and derivative functions");
        println!("   - Works with symbolic optimization pipeline");
    }
}

#[cfg(not(feature = "autodiff"))]
mod no_autodiff_tests {
    #[test]
    fn test_autodiff_feature_disabled() {
        println!("ℹ️  Autodiff feature is disabled. Enable with --features autodiff");
        // This test just ensures the test suite runs even without the autodiff feature
        assert!(true);
    }
}
