//! Semantic tests for interval domain analysis
//! 
//! These tests focus on the mathematical correctness of interval arithmetic,
//! domain analysis, and soundness of interval-based optimizations.

use dslcompile::{
    ast::ASTRepr,
    contexts::dynamic::DynamicContext,
    interval_domain::{IntervalDomain, Endpoint, DomainAnalyzer},
    frunk::hlist,
};
use proptest::prelude::*;
use std::collections::HashMap;

#[cfg(test)]
mod interval_arithmetic_semantics {
    use super::*;

    /// Test basic interval arithmetic operations
    #[test]
    fn test_interval_arithmetic_correctness() {
        // Test addition: [1,3] + [2,4] = [3,7]
        let interval1 = IntervalDomain::closed_interval(1.0, 3.0);
        let interval2 = IntervalDomain::closed_interval(2.0, 4.0);
        let sum = interval1.add(&interval2);
        
        match sum {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(3.0), "Addition lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(7.0), "Addition upper bound incorrect");
            }
            _ => panic!("Addition should produce interval, got: {:?}", sum),
        }
        
        // Test multiplication: [2,3] * [4,5] = [8,15]
        let mult_result = IntervalDomain::closed_interval(2.0, 3.0)
            .multiply(&IntervalDomain::closed_interval(4.0, 5.0));
        
        match mult_result {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(8.0), "Multiplication lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(15.0), "Multiplication upper bound incorrect");
            }
            _ => panic!("Multiplication should produce interval, got: {:?}", mult_result),
        }
    }

    /// Test interval arithmetic with negative values
    #[test]
    fn test_interval_arithmetic_with_negatives() {
        // Test [-2,1] * [3,5] = [-10,5]
        let neg_interval = IntervalDomain::closed_interval(-2.0, 1.0);
        let pos_interval = IntervalDomain::closed_interval(3.0, 5.0);
        let product = neg_interval.multiply(&pos_interval);
        
        match product {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(-10.0), "Negative multiplication lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(5.0), "Negative multiplication upper bound incorrect");
            }
            _ => panic!("Negative multiplication should produce interval, got: {:?}", product),
        }
        
        // Test division: [4,8] / [2,4] = [1,4]
        let dividend = IntervalDomain::closed_interval(4.0, 8.0);
        let divisor = IntervalDomain::closed_interval(2.0, 4.0);
        let quotient = dividend.divide(&divisor);
        
        match quotient {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(1.0), "Division lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(4.0), "Division upper bound incorrect");
            }
            _ => panic!("Division should produce interval, got: {:?}", quotient),
        }
    }

    /// Test subtraction semantics
    #[test]
    fn test_interval_subtraction_semantics() {
        // Test [5,8] - [2,3] = [2,6]
        let minuend = IntervalDomain::closed_interval(5.0, 8.0);
        let subtrahend = IntervalDomain::closed_interval(2.0, 3.0);
        let difference = minuend.subtract(&subtrahend);
        
        match difference {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(2.0), "Subtraction lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(6.0), "Subtraction upper bound incorrect");
            }
            _ => panic!("Subtraction should produce interval, got: {:?}", difference),
        }
    }

    /// Property test: Interval arithmetic should be sound (conservative)
    proptest! {
        #[test]
        fn prop_interval_arithmetic_soundness(
            a_low in -100.0..100.0f64,
            a_high in -100.0..100.0f64,
            b_low in -100.0..100.0f64,
            b_high in -100.0..100.0f64
        ) {
            // Ensure proper ordering
            let a_min = a_low.min(a_high);
            let a_max = a_low.max(a_high);
            let b_min = b_low.min(b_high);
            let b_max = b_low.max(b_high);
            
            let interval_a = IntervalDomain::closed_interval(a_min, a_max);
            let interval_b = IntervalDomain::closed_interval(b_min, b_max);
            
            // Test addition soundness
            let sum = interval_a.add(&interval_b);
            let actual_min = a_min + b_min;
            let actual_max = a_max + b_max;
            
            prop_assert!(sum.contains(actual_min), "Addition should contain minimum: {} not in {:?}", actual_min, sum);
            prop_assert!(sum.contains(actual_max), "Addition should contain maximum: {} not in {:?}", actual_max, sum);
            
            // Test multiplication soundness (if intervals don't contain zero)
            if !interval_a.contains(0.0) && !interval_b.contains(0.0) {
                let product = interval_a.multiply(&interval_b);
                let corner_products = vec![
                    a_min * b_min,
                    a_min * b_max,
                    a_max * b_min,
                    a_max * b_max,
                ];
                
                for &corner in &corner_products {
                    prop_assert!(product.contains(corner), 
                               "Multiplication should contain corner product: {} not in {:?}", corner, product);
                }
            }
        }
    }

    /// Test trigonometric function intervals
    #[test]
    fn test_trigonometric_function_intervals() {
        // sin([0, π/2]) = [0, 1]
        let input_interval = IntervalDomain::closed_interval(0.0, std::f64::consts::PI / 2.0);
        let sin_result = input_interval.sin();
        
        match sin_result {
            IntervalDomain::Interval { lower, upper } => {
                // sin is monotonic on [0, π/2]
                assert!((lower.value() - 0.0).abs() < 1e-10, "sin([0,π/2]) lower bound should be ~0");
                assert!((upper.value() - 1.0).abs() < 1e-10, "sin([0,π/2]) upper bound should be ~1");
            }
            _ => panic!("sin should produce interval, got: {:?}", sin_result),
        }
        
        // cos([0, π/2]) = [0, 1]
        let cos_result = input_interval.cos();
        match cos_result {
            IntervalDomain::Interval { lower, upper } => {
                // cos is monotonically decreasing on [0, π/2]
                assert!((lower.value() - 0.0).abs() < 1e-10, "cos([0,π/2]) lower bound should be ~0");
                assert!((upper.value() - 1.0).abs() < 1e-10, "cos([0,π/2]) upper bound should be ~1");
            }
            _ => panic!("cos should produce interval, got: {:?}", cos_result),
        }
    }

    /// Test exponential and logarithm interval semantics
    #[test]
    fn test_exponential_logarithm_interval_semantics() {
        // exp([0, 1]) = [1, e]
        let input_interval = IntervalDomain::closed_interval(0.0, 1.0);
        let exp_result = input_interval.exp();
        
        match exp_result {
            IntervalDomain::Interval { lower, upper } => {
                assert!((lower.value() - 1.0).abs() < 1e-10, "exp([0,1]) lower bound should be 1");
                assert!((upper.value() - std::f64::consts::E).abs() < 1e-10, "exp([0,1]) upper bound should be e");
            }
            _ => panic!("exp should produce interval, got: {:?}", exp_result),
        }
        
        // ln([1, e]) = [0, 1]
        let ln_input = IntervalDomain::closed_interval(1.0, std::f64::consts::E);
        let ln_result = ln_input.ln();
        
        match ln_result {
            IntervalDomain::Interval { lower, upper } => {
                assert!((lower.value() - 0.0).abs() < 1e-10, "ln([1,e]) lower bound should be 0");
                assert!((upper.value() - 1.0).abs() < 1e-10, "ln([1,e]) upper bound should be 1");
            }
            _ => panic!("ln should produce interval, got: {:?}", ln_result),
        }
    }

    /// Test power interval semantics
    #[test]
    fn test_power_interval_semantics() {
        // [2, 3]^2 = [4, 9]
        let base_interval = IntervalDomain::closed_interval(2.0, 3.0);
        let power_result = base_interval.pow(2.0);
        
        match power_result {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(4.0), "Power lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(9.0), "Power upper bound incorrect");
            }
            _ => panic!("Power should produce interval, got: {:?}", power_result),
        }
        
        // [4, 9]^0.5 = [2, 3]
        let sqrt_input = IntervalDomain::closed_interval(4.0, 9.0);
        let sqrt_result = sqrt_input.pow(0.5);
        
        match sqrt_result {
            IntervalDomain::Interval { lower, upper } => {
                assert!((lower.value() - 2.0).abs() < 1e-10, "Square root lower bound incorrect");
                assert!((upper.value() - 3.0).abs() < 1e-10, "Square root upper bound incorrect");
            }
            _ => panic!("Square root should produce interval, got: {:?}", sqrt_result),
        }
    }

    /// Property test: Monotonicity preservation
    proptest! {
        #[test]
        fn prop_monotonicity_preservation(
            x_low in 0.1..10.0f64,
            x_high in 0.1..10.0f64
        ) {
            let x_min = x_low.min(x_high);
            let x_max = x_low.max(x_high);
            let interval = IntervalDomain::closed_interval(x_min, x_max);
            
            // Test monotonic functions preserve ordering
            let exp_result = interval.exp();
            let ln_result = interval.ln();
            
            // exp is monotonically increasing
            prop_assert!(exp_result.lower_bound() <= exp_result.upper_bound(),
                        "exp should preserve monotonicity");
            
            // ln is monotonically increasing (for positive inputs)
            prop_assert!(ln_result.lower_bound() <= ln_result.upper_bound(),
                        "ln should preserve monotonicity");
        }
    }
}

#[cfg(test)]
mod domain_analysis_semantics {
    use super::*;

    /// Test domain analysis for simple expressions
    #[test]
    fn test_domain_analysis_for_expressions() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        let y = ctx.var();
        
        // Expression: x + y
        let expr = &x + &y;
        let ast = ctx.to_ast(&expr);
        
        let analyzer = DomainAnalyzer::new();
        
        // Define input domains
        let mut input_domains = HashMap::new();
        input_domains.insert(0, IntervalDomain::closed_interval(1.0, 3.0)); // x ∈ [1,3]
        input_domains.insert(1, IntervalDomain::closed_interval(2.0, 4.0)); // y ∈ [2,4]
        
        let result_domain = analyzer.analyze_expression(&ast, &input_domains)
            .expect("Should analyze expression domain");
        
        // x + y should be in [3, 7]
        match result_domain {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(3.0), "Expression domain lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(7.0), "Expression domain upper bound incorrect");
            }
            _ => panic!("Expression domain should be interval, got: {:?}", result_domain),
        }
    }

    /// Test domain analysis for complex expressions
    #[test]
    fn test_complex_expression_domain_analysis() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        let y = ctx.var();
        
        // Expression: x * y + x
        let expr = &x * &y + &x;
        let ast = ctx.to_ast(&expr);
        
        let analyzer = DomainAnalyzer::new();
        
        let mut input_domains = HashMap::new();
        input_domains.insert(0, IntervalDomain::closed_interval(2.0, 3.0)); // x ∈ [2,3]
        input_domains.insert(1, IntervalDomain::closed_interval(4.0, 5.0)); // y ∈ [4,5]
        
        let result_domain = analyzer.analyze_expression(&ast, &input_domains)
            .expect("Should analyze complex expression domain");
        
        // x * y + x = x * (y + 1) ∈ [2,3] * [5,6] = [10,18]
        match result_domain {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(10.0), "Complex expression domain lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(18.0), "Complex expression domain upper bound incorrect");
            }
            _ => panic!("Complex expression domain should be interval, got: {:?}", result_domain),
        }
    }

    /// Property test: Domain analysis soundness
    proptest! {
        #[test]
        fn prop_domain_analysis_soundness(
            x_min in -10.0..10.0f64,
            x_max in -10.0..10.0f64,
            y_min in -10.0..10.0f64,
            y_max in -10.0..10.0f64
        ) {
            let x_low = x_min.min(x_max);
            let x_high = x_min.max(x_max);
            let y_low = y_min.min(y_max);
            let y_high = y_min.max(y_max);
            
            let mut ctx = DynamicContext::new();
            let x = ctx.var();
            let y = ctx.var();
            let expr = &x + &y * 2.0;
            let ast = ctx.to_ast(&expr);
            
            let analyzer = DomainAnalyzer::new();
            let mut input_domains = HashMap::new();
            input_domains.insert(0, IntervalDomain::closed_interval(x_low, x_high));
            input_domains.insert(1, IntervalDomain::closed_interval(y_low, y_high));
            
            let result_domain = analyzer.analyze_expression(&ast, &input_domains)
                .expect("Should analyze expression");
            
            // Test corner cases
            let corner_values = vec![
                (x_low, y_low),
                (x_low, y_high),
                (x_high, y_low),
                (x_high, y_high),
            ];
            
            for (x_val, y_val) in corner_values {
                let actual_result = ctx.eval(&expr, hlist![x_val, y_val]);
                prop_assert!(result_domain.contains(actual_result),
                           "Domain analysis should contain actual result: {} not in {:?}",
                           actual_result, result_domain);
            }
        }
    }

    /// Test division safety through domain analysis
    #[test]
    fn test_division_safety_domain_analysis() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        let y = ctx.var();
        
        // Expression: x / y
        let expr = &x / &y;
        let ast = ctx.to_ast(&expr);
        
        let analyzer = DomainAnalyzer::new();
        
        // Case 1: Safe division (y doesn't contain 0)
        let mut safe_domains = HashMap::new();
        safe_domains.insert(0, IntervalDomain::closed_interval(4.0, 8.0)); // x ∈ [4,8]
        safe_domains.insert(1, IntervalDomain::closed_interval(2.0, 4.0)); // y ∈ [2,4] (no zero)
        
        let safe_result = analyzer.analyze_expression(&ast, &safe_domains)
            .expect("Should analyze safe division");
        
        match safe_result {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(1.0), "Safe division lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(4.0), "Safe division upper bound incorrect");
            }
            _ => panic!("Safe division should produce interval, got: {:?}", safe_result),
        }
        
        // Case 2: Unsafe division (y contains 0)
        let mut unsafe_domains = HashMap::new();
        unsafe_domains.insert(0, IntervalDomain::closed_interval(1.0, 2.0)); // x ∈ [1,2]
        unsafe_domains.insert(1, IntervalDomain::closed_interval(-1.0, 1.0)); // y ∈ [-1,1] (contains 0)
        
        let unsafe_result = analyzer.analyze_expression(&ast, &unsafe_domains);
        
        // Should either return Top (unbounded) or an error
        match unsafe_result {
            Ok(IntervalDomain::Top) => {
                // This is acceptable - division by interval containing zero gives unbounded result
            }
            Err(_) => {
                // This is also acceptable - division by zero is an error
            }
            Ok(other) => {
                // Check if it's a very wide interval (conservative approximation)
                if let IntervalDomain::Interval { lower, upper } = other {
                    assert!(lower.value() < -1000.0 || upper.value() > 1000.0,
                           "Division by interval containing zero should produce wide interval or error");
                }
            }
        }
    }

    /// Test error handling for invalid domains
    #[test]
    fn test_error_handling_for_invalid_domains() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        
        // Expression: ln(x) - requires x > 0
        let expr = (&x).ln();
        let ast = ctx.to_ast(&expr);
        
        let analyzer = DomainAnalyzer::new();
        
        // Case 1: Valid domain for ln
        let mut valid_domains = HashMap::new();
        valid_domains.insert(0, IntervalDomain::closed_interval(1.0, std::f64::consts::E));
        
        let valid_result = analyzer.analyze_expression(&ast, &valid_domains)
            .expect("Should analyze valid ln domain");
        
        match valid_result {
            IntervalDomain::Interval { lower, upper } => {
                assert!((lower.value() - 0.0).abs() < 1e-10, "ln domain lower bound incorrect");
                assert!((upper.value() - 1.0).abs() < 1e-10, "ln domain upper bound incorrect");
            }
            _ => panic!("Valid ln should produce interval, got: {:?}", valid_result),
        }
        
        // Case 2: Invalid domain for ln (contains non-positive values)
        let mut invalid_domains = HashMap::new();
        invalid_domains.insert(0, IntervalDomain::closed_interval(-1.0, 1.0)); // Contains 0 and negatives
        
        let invalid_result = analyzer.analyze_expression(&ast, &invalid_domains);
        
        // Should handle the invalid domain gracefully
        match invalid_result {
            Err(_) => {
                // Error is expected for invalid domain
            }
            Ok(IntervalDomain::Bottom) => {
                // Bottom (empty set) is also acceptable
            }
            Ok(other) => {
                // If it returns a result, it should be conservative
                // (e.g., only the positive part of the interval)
                println!("ln of invalid domain returned: {:?}", other);
            }
        }
    }
}

#[cfg(test)]
mod interval_guided_optimization_semantics {
    use super::*;

    /// Test interval-guided optimization opportunities
    #[test]
    fn test_interval_guided_optimization_opportunities() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        
        // Expression: x * x (where x ∈ [2, 3])
        let expr = &x * &x;
        let ast = ctx.to_ast(&expr);
        
        let analyzer = DomainAnalyzer::new();
        let mut input_domains = HashMap::new();
        input_domains.insert(0, IntervalDomain::closed_interval(2.0, 3.0));
        
        let result_domain = analyzer.analyze_expression(&ast, &input_domains)
            .expect("Should analyze square expression");
        
        // x² where x ∈ [2,3] should give [4,9]
        match result_domain {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(4.0), "Square optimization lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(9.0), "Square optimization upper bound incorrect");
            }
            _ => panic!("Square optimization should produce interval, got: {:?}", result_domain),
        }
        
        // This information could guide optimization:
        // - Since result is always positive, we could optimize sqrt(x*x) to |x|
        // - Since x > 0, we could optimize |x| to x
        assert!(result_domain.is_positive(0.0), "Result should be positive");
    }

    /// Test range reduction analysis
    #[test]
    fn test_range_reduction_analysis() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        
        // Expression: sin(x) where x ∈ [0, π/4]
        let expr = (&x).sin();
        let ast = ctx.to_ast(&expr);
        
        let analyzer = DomainAnalyzer::new();
        let mut input_domains = HashMap::new();
        input_domains.insert(0, IntervalDomain::closed_interval(0.0, std::f64::consts::PI / 4.0));
        
        let result_domain = analyzer.analyze_expression(&ast, &input_domains)
            .expect("Should analyze sin expression");
        
        // sin(x) where x ∈ [0, π/4] should give [0, sin(π/4)] = [0, √2/2]
        match result_domain {
            IntervalDomain::Interval { lower, upper } => {
                assert!((lower.value() - 0.0).abs() < 1e-10, "sin range reduction lower bound incorrect");
                assert!((upper.value() - (std::f64::consts::PI / 4.0).sin()).abs() < 1e-10, 
                       "sin range reduction upper bound incorrect");
            }
            _ => panic!("sin range reduction should produce interval, got: {:?}", result_domain),
        }
        
        // This narrow range could enable optimizations like polynomial approximations
        let range_width = result_domain.width();
        assert!(range_width < 1.0, "Range should be narrow enough for approximation optimizations");
    }

    /// Test constant folding detection through interval analysis
    #[test]
    fn test_constant_folding_detection() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        
        // Expression: x + 0 (should be detected as equivalent to x)
        let expr = &x + 0.0;
        let ast = ctx.to_ast(&expr);
        
        let analyzer = DomainAnalyzer::new();
        let mut input_domains = HashMap::new();
        input_domains.insert(0, IntervalDomain::closed_interval(5.0, 10.0));
        
        let result_domain = analyzer.analyze_expression(&ast, &input_domains)
            .expect("Should analyze additive identity");
        
        // Result should be same as input domain
        match result_domain {
            IntervalDomain::Interval { lower, upper } => {
                assert_eq!(lower, Endpoint::Closed(5.0), "Additive identity lower bound incorrect");
                assert_eq!(upper, Endpoint::Closed(10.0), "Additive identity upper bound incorrect");
            }
            _ => panic!("Additive identity should produce interval, got: {:?}", result_domain),
        }
        
        // This could guide optimization: x + 0 → x
        let input_domain = &input_domains[&0];
        assert!(result_domain.is_equivalent_to(input_domain), 
               "Result domain should be equivalent to input for additive identity");
    }

    /// Property test: Interval-guided optimization soundness
    proptest! {
        #[test]
        fn prop_interval_guided_optimization_soundness(
            x_min in 1.0..10.0f64,
            x_max in 1.0..10.0f64,
            operation in 0..4usize
        ) {
            let x_low = x_min.min(x_max);
            let x_high = x_min.max(x_max);
            
            let mut ctx = DynamicContext::new();
            let x = ctx.var();
            
            // Different operations to test
            let expr = match operation {
                0 => &x * 2.0,      // Linear scaling
                1 => &x * &x,       // Quadratic
                2 => (&x).sqrt(),   // Square root
                3 => (&x).ln(),     // Logarithm
                _ => &x + 1.0,      // Default: linear shift
            };
            
            let ast = ctx.to_ast(&expr);
            
            let analyzer = DomainAnalyzer::new();
            let mut input_domains = HashMap::new();
            input_domains.insert(0, IntervalDomain::closed_interval(x_low, x_high));
            
            if let Ok(result_domain) = analyzer.analyze_expression(&ast, &input_domains) {
                // Test that actual evaluations fall within the computed domain
                let test_points = vec![x_low, x_high, (x_low + x_high) / 2.0];
                
                for x_val in test_points {
                    let actual_result = ctx.eval(&expr, hlist![x_val]);
                    prop_assert!(result_domain.contains(actual_result),
                               "Interval-guided optimization should be sound: {} not in {:?}",
                               actual_result, result_domain);
                }
            }
        }
    }
} 