//! Semantic tests for the procedural macro system
//! 
//! These tests focus on the mathematical correctness and optimization semantics
//! of compile-time optimization, not just basic functionality.

use dslcompile_macros::optimize_compile_time;
use proptest::prelude::*;

#[cfg(test)]
mod compile_time_optimization_semantics {
    use super::*;

    /// Test that compile-time optimization preserves mathematical semantics
    #[test]
    fn test_optimization_preserves_mathematical_identity() {
        // Test additive identity: x + 0 = x
        let x_val = 3.14159;
        let optimized_result = optimize_compile_time!(x.add(constant(0.0)), [x]);
        let expected = x_val;
        
        // The macro should optimize x + 0 to just x
        assert!((optimized_result - expected).abs() < f64::EPSILON,
               "Additive identity not preserved: {} != {}", optimized_result, expected);
    }

    /// Test that multiplicative identity is preserved
    #[test] 
    fn test_multiplicative_identity_optimization() {
        let x_val = 2.71828;
        let optimized_result = optimize_compile_time!(x.mul(constant(1.0)), [x]);
        let expected = x_val;
        
        assert!((optimized_result - expected).abs() < f64::EPSILON,
               "Multiplicative identity not preserved: {} != {}", optimized_result, expected);
    }

    /// Test that zero multiplication is optimized correctly
    #[test]
    fn test_zero_multiplication_optimization() {
        let x_val = 42.0;
        let optimized_result = optimize_compile_time!(x.mul(constant(0.0)), [x]);
        let expected = 0.0;
        
        assert!((optimized_result - expected).abs() < f64::EPSILON,
               "Zero multiplication not optimized: {} != {}", optimized_result, expected);
    }

    /// Test constant folding semantics
    #[test]
    fn test_constant_folding_correctness() {
        // Should fold 2.0 + 3.0 at compile time
        let result = optimize_compile_time!(constant(2.0).add(constant(3.0)), []);
        let expected = 5.0;
        
        assert!((result - expected).abs() < f64::EPSILON,
               "Constant folding incorrect: {} != {}", result, expected);
    }

    /// Test trigonometric identity optimization
    #[test]
    fn test_trigonometric_identity_optimization() {
        let x_val = 1.0;
        // sin²(x) + cos²(x) should be optimized to 1.0 when possible
        let result = optimize_compile_time!(
            x.sin().mul(x.sin()).add(x.cos().mul(x.cos())), 
            [x]
        );
        
        // For x = 1.0, sin²(1) + cos²(1) = 1
        let expected = (x_val.sin().powi(2) + x_val.cos().powi(2));
        assert!((result - expected).abs() < 1e-10,
               "Trigonometric computation incorrect: {} != {}", result, expected);
    }

    /// Property test: Optimization should preserve commutativity
    proptest! {
        #[test]
        fn prop_optimization_preserves_commutativity(
            a_val in -100.0..100.0f64,
            b_val in -100.0..100.0f64
        ) {
            // Test that a + b = b + a after optimization
            let result1 = optimize_compile_time!(a.add(b), [a, b]);
            let result2 = optimize_compile_time!(b.add(a), [a, b]);
            
            prop_assert!((result1 - result2).abs() < f64::EPSILON,
                        "Commutativity not preserved: {} != {}", result1, result2);
        }
    }

    /// Property test: Optimization should preserve associativity
    proptest! {
        #[test]
        fn prop_optimization_preserves_associativity(
            a_val in -10.0..10.0f64,
            b_val in -10.0..10.0f64,
            c_val in -10.0..10.0f64
        ) {
            // Test that (a + b) + c = a + (b + c) after optimization
            let result1 = optimize_compile_time!(a.add(b).add(c), [a, b, c]);
            let result2 = optimize_compile_time!(a.add(b.add(c)), [a, b, c]);
            
            prop_assert!((result1 - result2).abs() < 1e-10,
                        "Associativity not preserved: {} != {}", result1, result2);
        }
    }

    /// Test power optimization semantics
    #[test]
    fn test_power_optimization_semantics() {
        let x_val = 2.0;
        
        // x^1 should optimize to x
        let result1 = optimize_compile_time!(x.pow(constant(1.0)), [x]);
        assert!((result1 - x_val).abs() < f64::EPSILON,
               "x^1 optimization failed: {} != {}", result1, x_val);
        
        // x^0 should optimize to 1
        let result0 = optimize_compile_time!(x.pow(constant(0.0)), [x]);
        assert!((result0 - 1.0).abs() < f64::EPSILON,
               "x^0 optimization failed: {} != 1.0", result0);
    }

    /// Test logarithm and exponential optimization
    #[test]
    fn test_log_exp_optimization_semantics() {
        let x_val = 2.718281828;
        
        // ln(e^x) should optimize toward x (within numerical precision)
        let result = optimize_compile_time!(x.exp().ln(), [x]);
        assert!((result - x_val).abs() < 1e-10,
               "ln(e^x) optimization failed: {} != {}", result, x_val);
    }

    /// Property test: Complex expression optimization correctness
    proptest! {
        #[test]
        fn prop_complex_expression_optimization_correctness(
            x_val in 0.1..10.0f64,
            y_val in 0.1..10.0f64
        ) {
            // Test complex expression: (x * y + x) / (y + 1)
            let optimized = optimize_compile_time!(
                x.mul(y).add(x).div(y.add(constant(1.0))), 
                [x, y]
            );
            
            // Manual calculation for verification
            let expected = (x_val * y_val + x_val) / (y_val + 1.0);
            
            prop_assert!((optimized - expected).abs() < 1e-10,
                        "Complex expression optimization incorrect: {} != {}", 
                        optimized, expected);
        }
    }

    /// Test that optimization handles edge cases correctly
    #[test]
    fn test_optimization_edge_cases() {
        // Division by very small numbers
        let result = optimize_compile_time!(
            constant(1.0).div(constant(1e-10)), 
            []
        );
        let expected = 1.0 / 1e-10;
        assert!((result - expected).abs() < 1e-5,
               "Division by small number failed: {} != {}", result, expected);
    }

    /// Test nested optimization semantics
    #[test]
    fn test_nested_optimization_semantics() {
        let x_val = 3.0;
        
        // Nested expression: sin(cos(x + 0) * 1)
        let result = optimize_compile_time!(
            x.add(constant(0.0)).cos().mul(constant(1.0)).sin(),
            [x]
        );
        
        let expected = (x_val.cos()).sin();
        assert!((result - expected).abs() < 1e-10,
               "Nested optimization failed: {} != {}", result, expected);
    }
}

#[cfg(test)]
mod egglog_integration_semantics {
    use super::*;

    /// Test that egglog rules preserve mathematical equivalence
    #[test]
    fn test_egglog_algebraic_equivalence() {
        let x_val = 2.0;
        let y_val = 3.0;
        
        // Test distributive property: a * (b + c) = a*b + a*c
        let distributed = optimize_compile_time!(
            x.mul(y.add(constant(1.0))),
            [x, y]
        );
        
        let expanded = optimize_compile_time!(
            x.mul(y).add(x.mul(constant(1.0))),
            [x, y]
        );
        
        // Both should give the same result
        assert!((distributed - expanded).abs() < f64::EPSILON,
               "Distributive property not preserved: {} != {}", distributed, expanded);
    }

    /// Property test: Egglog optimization should be deterministic
    proptest! {
        #[test]
        fn prop_egglog_optimization_deterministic(
            x_val in -10.0..10.0f64,
            y_val in -10.0..10.0f64
        ) {
            // Same expression should always optimize to same result
            let result1 = optimize_compile_time!(x.add(y).mul(constant(2.0)), [x, y]);
            let result2 = optimize_compile_time!(x.add(y).mul(constant(2.0)), [x, y]);
            
            prop_assert!((result1 - result2).abs() < f64::EPSILON,
                        "Egglog optimization not deterministic: {} != {}", result1, result2);
        }
    }
} 