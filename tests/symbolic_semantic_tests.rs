//! Semantic tests for symbolic optimization system
//! 
//! These tests focus on the mathematical correctness of symbolic transformations,
//! optimization strategies, and preservation of mathematical equivalence.

use dslcompile::{
    ast::ASTRepr,
    contexts::dynamic::DynamicContext,
    symbolic::{SymbolicOptimizer, OptimizationConfig, OptimizationStrategy},
    frunk::hlist,
};
use proptest::prelude::*;
use std::collections::HashSet;

#[cfg(test)]
mod symbolic_optimization_semantics {
    use super::*;

    /// Test that symbolic optimization preserves mathematical equivalence
    #[test]
    fn test_optimization_preserves_mathematical_equivalence() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let y = ctx.var();
        
        // Create expression: x + x should optimize to 2*x
        let original = &x + &x;
        let ast = ctx.to_ast(&original);
        
        let optimizer = SymbolicOptimizer::new();
        let optimized = optimizer.optimize(&ast).expect("Optimization should succeed");
        
        // Test mathematical equivalence with multiple values
        let test_values = vec![
            (1.0, 2.0),
            (-3.5, 4.2),
            (0.0, -1.0),
            (100.0, -50.0),
        ];
        
        for (x_val, y_val) in test_values {
            let original_result = ctx.eval(&original, hlist![x_val, y_val]);
            let optimized_result = optimized.eval_with_vars(&[x_val, y_val]);
            
            assert!((original_result - optimized_result).abs() < 1e-12,
                   "Optimization changed mathematical result: {} vs {} for x={}, y={}",
                   original_result, optimized_result, x_val, y_val);
        }
    }

    /// Test algebraic identity preservation
    #[test]
    fn test_algebraic_identity_preservation() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let y = ctx.var();
        let z = ctx.var();
        
        let optimizer = SymbolicOptimizer::new();
        
        // Test distributive property: a * (b + c) = a*b + a*c
        let distributed = &x * (&y + &z);
        let expanded = &x * &y + &x * &z;
        
        let dist_ast = ctx.to_ast(&distributed);
        let exp_ast = ctx.to_ast(&expanded);
        
        let opt_distributed = optimizer.optimize(&dist_ast).expect("Should optimize distributed");
        let opt_expanded = optimizer.optimize(&exp_ast).expect("Should optimize expanded");
        
        // Both should be mathematically equivalent
        let test_vals = (2.0, 3.0, 4.0);
        let dist_result = opt_distributed.eval_with_vars(&[test_vals.0, test_vals.1, test_vals.2]);
        let exp_result = opt_expanded.eval_with_vars(&[test_vals.0, test_vals.1, test_vals.2]);
        
        assert!((dist_result - exp_result).abs() < 1e-12,
               "Distributive property not preserved: {} != {}", dist_result, exp_result);
    }

    /// Property test: Commutativity preservation
    proptest! {
        #[test]
        fn prop_commutativity_preservation(
            x_val in -100.0..100.0f64,
            y_val in -100.0..100.0f64
        ) {
            let mut ctx = DynamicContext::<f64>::new();
            let x = ctx.var();
            let y = ctx.var();
            
            // Test x + y = y + x
            let expr1 = &x + &y;
            let expr2 = &y + &x;
            
            let optimizer = SymbolicOptimizer::new();
            let opt1 = optimizer.optimize(&ctx.to_ast(&expr1)).expect("Should optimize");
            let opt2 = optimizer.optimize(&ctx.to_ast(&expr2)).expect("Should optimize");
            
            let result1 = opt1.eval_with_vars(&[x_val, y_val]);
            let result2 = opt2.eval_with_vars(&[x_val, y_val]);
            
            prop_assert!((result1 - result2).abs() < 1e-12,
                        "Commutativity not preserved: {} != {}", result1, result2);
        }
    }

    /// Property test: Associativity preservation
    proptest! {
        #[test]
        fn prop_associativity_preservation(
            x_val in -10.0..10.0f64,
            y_val in -10.0..10.0f64,
            z_val in -10.0..10.0f64
        ) {
            let mut ctx = DynamicContext::<f64>::new();
            let x = ctx.var();
            let y = ctx.var();
            let z = ctx.var();
            
            // Test (x + y) + z = x + (y + z)
            let expr1 = (&x + &y) + &z;
            let expr2 = &x + (&y + &z);
            
            let optimizer = SymbolicOptimizer::new();
            let opt1 = optimizer.optimize(&ctx.to_ast(&expr1)).expect("Should optimize");
            let opt2 = optimizer.optimize(&ctx.to_ast(&expr2)).expect("Should optimize");
            
            let result1 = opt1.eval_with_vars(&[x_val, y_val, z_val]);
            let result2 = opt2.eval_with_vars(&[x_val, y_val, z_val]);
            
            prop_assert!((result1 - result2).abs() < 1e-12,
                        "Associativity not preserved: {} != {}", result1, result2);
        }
    }

    /// Test trigonometric identity optimization
    #[test]
    fn test_trigonometric_identity_optimization() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        
        // sin²(x) + cos²(x) should potentially optimize toward 1
        let trig_identity = (&x).sin() * (&x).sin() + (&x).cos() * (&x).cos();
        let ast = ctx.to_ast(&trig_identity);
        
        let optimizer = SymbolicOptimizer::new();
        let optimized = optimizer.optimize(&ast).expect("Should optimize trigonometric expression");
        
        // Test with specific values
        let test_values = vec![0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];
        
        for x_val in test_values {
            let original_result = ctx.eval(&trig_identity, hlist![x_val]);
            let optimized_result = optimized.eval_with_vars(&[x_val]);
            
            // Should be very close to 1.0 (trigonometric identity)
            assert!((original_result - 1.0).abs() < 1e-12, 
                   "Original should satisfy trig identity: {}", original_result);
            assert!((optimized_result - 1.0).abs() < 1e-12,
                   "Optimized should satisfy trig identity: {}", optimized_result);
            assert!((original_result - optimized_result).abs() < 1e-12,
                   "Optimization should preserve trig identity");
        }
    }

    /// Test power rule optimization
    #[test]
    fn test_power_rule_optimization() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        
        let optimizer = SymbolicOptimizer::new();
        
        // Test x^1 = x
        let power_one = (&x).pow(1.0);
        let opt_power_one = optimizer.optimize(&ctx.to_ast(&power_one)).expect("Should optimize x^1");
        
        // Test x^0 = 1 (for x != 0)
        let power_zero = (&x).pow(0.0);
        let opt_power_zero = optimizer.optimize(&ctx.to_ast(&power_zero)).expect("Should optimize x^0");
        
        let test_val = 5.0;
        let power_one_result = opt_power_one.eval_with_vars(&[test_val]);
        let power_zero_result = opt_power_zero.eval_with_vars(&[test_val]);
        
        assert!((power_one_result - test_val).abs() < 1e-12,
               "x^1 should equal x: {} != {}", power_one_result, test_val);
        assert!((power_zero_result - 1.0).abs() < 1e-12,
               "x^0 should equal 1: {} != 1", power_zero_result);
    }

    /// Test logarithm and exponential optimization
    #[test]
    fn test_log_exp_optimization() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        
        let optimizer = SymbolicOptimizer::new();
        
        // Test ln(e^x) should optimize toward x
        let log_exp = (&x).exp().ln();
        let opt_log_exp = optimizer.optimize(&ctx.to_ast(&log_exp)).expect("Should optimize ln(e^x)");
        
        // Test e^(ln(x)) should optimize toward x (for x > 0)
        let exp_log = (&x).ln().exp();
        let opt_exp_log = optimizer.optimize(&ctx.to_ast(&exp_log)).expect("Should optimize e^(ln(x))");
        
        let test_val = 2.5; // Positive value for ln
        let log_exp_result = opt_log_exp.eval_with_vars(&[test_val]);
        let exp_log_result = opt_exp_log.eval_with_vars(&[test_val]);
        
        assert!((log_exp_result - test_val).abs() < 1e-10,
               "ln(e^x) should equal x: {} != {}", log_exp_result, test_val);
        assert!((exp_log_result - test_val).abs() < 1e-10,
               "e^(ln(x)) should equal x: {} != {}", exp_log_result, test_val);
    }

    /// Test different optimization strategies
    #[test]
    fn test_optimization_strategy_correctness() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let expr = &x * 2.0 + &x * 3.0; // Should optimize to x * 5.0
        let ast = ctx.to_ast(&expr);
        
        let strategies = vec![
            OptimizationStrategy::Aggressive,
            OptimizationStrategy::Conservative,
            OptimizationStrategy::Balanced,
        ];
        
        let test_val = 4.0;
        let expected = test_val * 2.0 + test_val * 3.0; // = 20.0
        
        for strategy in strategies {
            let config = OptimizationConfig {
                strategy,
                max_iterations: 100,
                ..Default::default()
            };
            
            let optimizer = SymbolicOptimizer::with_config(config);
            let optimized = optimizer.optimize(&ast).expect("Should optimize with strategy");
            let result = optimized.eval_with_vars(&[test_val]);
            
            assert!((result - expected).abs() < 1e-12,
                   "Strategy {:?} should preserve mathematical result: {} != {}",
                   strategy, result, expected);
        }
    }

    /// Test optimization configuration effects
    #[test]
    fn test_optimization_configuration_effects() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let complex_expr = (&x + 1.0) * (&x + 1.0) - (&x * &x + 2.0 * &x + 1.0);
        let ast = ctx.to_ast(&complex_expr);
        
        // This should optimize to 0
        let configs = vec![
            OptimizationConfig {
                max_iterations: 10,
                ..Default::default()
            },
            OptimizationConfig {
                max_iterations: 100,
                enable_algebraic_simplification: true,
                ..Default::default()
            },
            OptimizationConfig {
                max_iterations: 1000,
                enable_algebraic_simplification: true,
                enable_constant_folding: true,
                ..Default::default()
            },
        ];
        
        let test_val = 3.0;
        
        for (i, config) in configs.iter().enumerate() {
            let optimizer = SymbolicOptimizer::with_config(config.clone());
            let optimized = optimizer.optimize(&ast).expect("Should optimize");
            let result = optimized.eval_with_vars(&[test_val]);
            
            // All should give the same mathematical result (0)
            assert!(result.abs() < 1e-10,
                   "Config {} should optimize to 0: {}", i, result);
        }
    }

    /// Property test: Optimization should not change mathematical meaning
    proptest! {
        #[test]
        fn prop_optimization_preserves_mathematical_meaning(
            x_val in -50.0..50.0f64,
            y_val in -50.0..50.0f64,
            coeff in -10.0..10.0f64
        ) {
            let mut ctx = DynamicContext::<f64>::new();
            let x = ctx.var();
            let y = ctx.var();
            
            // Create a moderately complex expression
            let expr = (&x * coeff + &y) * (&x - &y) + coeff;
            let ast = ctx.to_ast(&expr);
            
            let optimizer = SymbolicOptimizer::new();
            let optimized = optimizer.optimize(&ast).expect("Should optimize");
            
            let original_result = ctx.eval(&expr, hlist![x_val, y_val]);
            let optimized_result = optimized.eval_with_vars(&[x_val, y_val]);
            
            prop_assert!((original_result - optimized_result).abs() < 1e-10,
                        "Optimization changed mathematical meaning: {} != {}",
                        original_result, optimized_result);
        }
    }

    /// Test adaptive optimization strategy semantics
    #[test]
    fn test_adaptive_optimization_strategy_semantics() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        
        // Simple expression that should be easy to optimize
        let simple_expr = &x + 0.0;
        // Complex expression that might need more iterations
        let complex_expr = ((&x + 1.0) * (&x - 1.0)) - (&x * &x - 1.0);
        
        let adaptive_config = OptimizationConfig {
            strategy: OptimizationStrategy::Adaptive,
            max_iterations: 50,
            ..Default::default()
        };
        
        let optimizer = SymbolicOptimizer::with_config(adaptive_config);
        
        let opt_simple = optimizer.optimize(&ctx.to_ast(&simple_expr)).expect("Should optimize simple");
        let opt_complex = optimizer.optimize(&ctx.to_ast(&complex_expr)).expect("Should optimize complex");
        
        let test_val = 7.0;
        let simple_result = opt_simple.eval_with_vars(&[test_val]);
        let complex_result = opt_complex.eval_with_vars(&[test_val]);
        
        // Simple should optimize to x
        assert!((simple_result - test_val).abs() < 1e-12,
               "Simple expression should optimize to x: {}", simple_result);
        
        // Complex should optimize to 0
        assert!(complex_result.abs() < 1e-12,
               "Complex expression should optimize to 0: {}", complex_result);
    }

    /// Test integer power conversion semantics
    #[test]
    fn test_integer_power_conversion_semantics() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        
        let optimizer = SymbolicOptimizer::new();
        
        // Test that integer powers are handled correctly
        let square = (&x).pow(2.0);
        let cube = (&x).pow(3.0);
        let fourth = (&x).pow(4.0);
        
        let opt_square = optimizer.optimize(&ctx.to_ast(&square)).expect("Should optimize x^2");
        let opt_cube = optimizer.optimize(&ctx.to_ast(&cube)).expect("Should optimize x^3");
        let opt_fourth = optimizer.optimize(&ctx.to_ast(&fourth)).expect("Should optimize x^4");
        
        let test_val = 2.0;
        let square_result = opt_square.eval_with_vars(&[test_val]);
        let cube_result = opt_cube.eval_with_vars(&[test_val]);
        let fourth_result = opt_fourth.eval_with_vars(&[test_val]);
        
        assert!((square_result - test_val.powi(2)).abs() < 1e-12,
               "x^2 optimization incorrect: {} != {}", square_result, test_val.powi(2));
        assert!((cube_result - test_val.powi(3)).abs() < 1e-12,
               "x^3 optimization incorrect: {} != {}", cube_result, test_val.powi(3));
        assert!((fourth_result - test_val.powi(4)).abs() < 1e-12,
               "x^4 optimization incorrect: {} != {}", fourth_result, test_val.powi(4));
    }

    /// Test compilation strategy equivalence
    #[test]
    fn test_compilation_strategy_equivalence() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let y = ctx.var();
        
        // Expression that should give same result regardless of compilation strategy
        let expr = (&x * &y + &x) / (&y + 1.0);
        let ast = ctx.to_ast(&expr);
        
        let strategies = vec![
            OptimizationStrategy::Aggressive,
            OptimizationStrategy::Conservative,
            OptimizationStrategy::Balanced,
            OptimizationStrategy::Adaptive,
        ];
        
        let test_vals = (3.0, 4.0);
        let mut results = Vec::new();
        
        for strategy in strategies {
            let config = OptimizationConfig {
                strategy,
                max_iterations: 100,
                ..Default::default()
            };
            
            let optimizer = SymbolicOptimizer::with_config(config);
            let optimized = optimizer.optimize(&ast).expect("Should optimize");
            let result = optimized.eval_with_vars(&[test_vals.0, test_vals.1]);
            results.push(result);
        }
        
        // All strategies should give mathematically equivalent results
        for (i, &result) in results.iter().enumerate() {
            for (j, &other_result) in results.iter().enumerate() {
                if i != j {
                    assert!((result - other_result).abs() < 1e-10,
                           "Strategy {} and {} gave different results: {} != {}",
                           i, j, result, other_result);
                }
            }
        }
    }

    /// Test optimization statistics validation
    #[test]
    fn test_optimization_statistics_validation() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        
        // Expression that should trigger multiple optimization passes
        let expr = (&x + 0.0) * 1.0 + (&x * 0.0);
        let ast = ctx.to_ast(&expr);
        
        let config = OptimizationConfig {
            collect_statistics: true,
            max_iterations: 50,
            ..Default::default()
        };
        
        let optimizer = SymbolicOptimizer::with_config(config);
        let optimized = optimizer.optimize(&ast).expect("Should optimize");
        
        // Should optimize to just x
        let test_val = 5.0;
        let result = optimized.eval_with_vars(&[test_val]);
        
        assert!((result - test_val).abs() < 1e-12,
               "Should optimize to x: {} != {}", result, test_val);
        
        // Statistics should be available if enabled
        if let Some(stats) = optimizer.get_statistics() {
            assert!(stats.iterations_performed > 0, "Should have performed some iterations");
            assert!(stats.rules_applied > 0, "Should have applied some rules");
        }
    }
} 