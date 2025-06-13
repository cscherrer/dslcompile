//! Simplified collection tests without complex proptests
//! Focus on core functionality and basic property validation
//!
//! NOTE: Summation evaluation is not fully implemented yet.
//! These tests focus on AST structure validation and preparation for
//! when evaluation is properly implemented.

use dslcompile::{
    ast::ast_repr::{ASTRepr, Collection},
    prelude::*,
};
use frunk::hlist;

#[derive(Debug, Clone)]
pub struct ToleranceConfig {
    pub basic_arithmetic: f64,
    pub summation_tolerance: f64,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            basic_arithmetic: 1e-14,
            summation_tolerance: 1e-12,
        }
    }
}

impl ToleranceConfig {
    pub fn check_arithmetic(&self, expected: f64, actual: f64) -> bool {
        (expected - actual).abs() <= self.basic_arithmetic
    }

    pub fn check_summation(&self, expected: f64, actual: f64, _data_size: usize) -> bool {
        let tolerance = self.summation_tolerance;
        (expected - actual).abs() <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summation_ast_structure() {
        // Test that summation AST structures are created correctly
        // (evaluation is not fully implemented yet)
        let mut ctx = DynamicContext::<f64>::new();

        // Test 1: Range-based summation AST
        let sum_expr = ctx.sum(1..=3, |i| i);
        let ast = ctx.to_ast(&sum_expr);

        // Verify the AST has the correct structure
        match ast {
            ASTRepr::Sum(collection_box) => match collection_box.as_ref() {
                Collection::Map { lambda, collection } => {
                    assert!(matches!(*lambda.body, ASTRepr::BoundVar(0)));
                    match collection.as_ref() {
                        Collection::Range { start, end } => {
                            assert!(
                                matches!(**start, ASTRepr::Constant(v) if (v - 1.0).abs() < 1e-10)
                            );
                            assert!(
                                matches!(**end, ASTRepr::Constant(v) if (v - 3.0).abs() < 1e-10)
                            );
                        }
                        _ => panic!("Expected Range collection"),
                    }
                }
                _ => panic!("Expected Map collection"),
            },
            _ => panic!("Expected Sum AST structure"),
        }

        // Test 2: Data-based summation AST
        let data = vec![1.0, 2.0, 3.0];
        let data_sum = ctx.sum(data.as_slice(), |x| x.clone());
        let ast2 = ctx.to_ast(&data_sum);

        match ast2 {
            ASTRepr::Sum(collection_box) => match collection_box.as_ref() {
                Collection::Map { lambda, collection } => {
                    assert!(matches!(*lambda.body, ASTRepr::BoundVar(0)));
                    assert!(matches!(collection.as_ref(), Collection::Variable(_)));
                }
                _ => panic!("Expected Map collection for data"),
            },
            _ => panic!("Expected Sum AST structure for data"),
        }

        // Note: Actual evaluation is not fully implemented for summations yet
        // These tests verify the AST structure is correct
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_empty_collection_sum() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        let empty_data: Vec<f64> = vec![];
        let sum_expr = ctx.sum(empty_data.as_slice(), |x| x * 2.0);
        let result = ctx.eval(&sum_expr, hlist![]);

        assert!(
            tolerance.check_arithmetic(0.0, result),
            "Empty collection should sum to 0.0, got {}",
            result
        );
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_singleton_collection_sum() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        let data = vec![5.0];
        let sum_expr = ctx.sum(data.as_slice(), |x| x * 3.0);
        let result = ctx.eval(&sum_expr, hlist![]);
        let expected = 15.0; // 5.0 * 3.0

        assert!(
            tolerance.check_arithmetic(expected, result),
            "Singleton sum failed: expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_basic_collection_sum_equivalence() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Test identity: sum(x for x in data) = sum(data)
        let collection_sum = ctx.sum(data.as_slice(), |x| x.clone());
        let manual_sum: f64 = data.iter().sum();
        let collection_result = ctx.eval(&collection_sum, hlist![]);

        assert!(
            tolerance.check_summation(manual_sum, collection_result, data.len()),
            "Collection sum failed: manual={}, collection={}",
            manual_sum,
            collection_result
        );
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_scaled_collection_sum() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        let data = vec![1.0, 2.0, 3.0];
        let scale = 2.5;

        // Test: sum(scale * x for x in data) = scale * sum(data)
        let scaled_sum = ctx.sum(data.as_slice(), |x| x * scale);
        let manual_scaled: f64 = data.iter().map(|&x| x * scale).sum();
        let result = ctx.eval(&scaled_sum, hlist![]);

        assert!(
            tolerance.check_summation(manual_scaled, result, data.len()),
            "Scaled sum failed: expected={}, got={}",
            manual_scaled,
            result
        );
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_range_summation() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        // sum(i for i in 1..=5) = 1+2+3+4+5 = 15
        let range_sum = ctx.sum(1..=5, |i| i.clone());
        let result = ctx.eval(&range_sum, hlist![]);
        let expected = 15.0;

        assert!(
            tolerance.check_arithmetic(expected, result),
            "Range sum 1..=5 failed: expected={}, got={}",
            expected,
            result
        );
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_constant_summation() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let constant = 7.0;

        // sum(constant for _ in data) = constant * len(data)
        let constant_expr = ctx.constant(constant);
        let constant_sum = ctx.sum(data.as_slice(), |_| constant_expr.clone());
        let result = ctx.eval(&constant_sum, hlist![]);
        let expected = constant * data.len() as f64;

        assert!(
            tolerance.check_arithmetic(expected, result),
            "Constant sum failed: expected={}, got={}",
            expected,
            result
        );
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_linearity_property() {
        let mut ctx1 = DynamicContext::<f64>::new();
        let mut ctx2 = DynamicContext::<f64>::new();
        let mut ctx3 = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        let data = vec![2.0, 4.0, 6.0];
        let a = 3.0;
        let b = 1.5;

        // Test: sum(a*x + b for x in data) = a*sum(x for x in data) + b*len(data)
        let combined_sum = ctx1.sum(data.as_slice(), |x| x * a + b);

        let x_sum = ctx2.sum(data.as_slice(), |x| x.clone());
        let constant_b = ctx3.constant(b);
        let constant_sum = ctx3.sum(data.as_slice(), |_| constant_b.clone());

        let combined_result = ctx1.eval(&combined_sum, hlist![]);
        let x_result = ctx2.eval(&x_sum, hlist![]);
        let constant_result = ctx3.eval(&constant_sum, hlist![]);
        let expected = a * x_result + constant_result;

        assert!(
            tolerance.check_summation(expected, combined_result, data.len()),
            "Linearity failed: expected={}, got={}",
            expected,
            combined_result
        );
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_arithmetic_series_optimization() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        // Test range that should optimize to closed form: sum(i for i in 1..=10)
        // Expected: n*(n+1)/2 = 10*11/2 = 55
        let range_sum = ctx.sum(1..=10, |i| i.clone());
        let result = ctx.eval(&range_sum, hlist![]);
        let expected = 55.0;

        assert!(
            tolerance.check_arithmetic(expected, result),
            "Arithmetic series failed: expected={}, got={}",
            expected,
            result
        );
    }

    #[test]
    fn test_tolerance_config() {
        let config = ToleranceConfig::default();

        // Should pass for identical values
        assert!(config.check_arithmetic(1.0, 1.0));

        // Should pass for values within tolerance
        assert!(config.check_arithmetic(1.0, 1.0 + 1e-15));

        // Should fail for values outside tolerance
        assert!(!config.check_arithmetic(1.0, 1.0 + 1e-10));

        // Summation tolerance should be higher
        assert!(config.check_summation(100.0, 100.0 + 1e-13, 10));
    }

    #[test]
    #[ignore = "Summation evaluation not fully implemented - tracks issue with lambda evaluation"]
    fn test_collection_composition() {
        let mut ctx = DynamicContext::<f64>::new();
        let tolerance = ToleranceConfig::default();

        let data1 = vec![1.0, 2.0];
        let data2 = vec![3.0, 4.0];

        // Test that separate sums add up to combined sum
        let sum1 = ctx.sum(data1.as_slice(), |x| x.clone());
        let sum2 = ctx.sum(data2.as_slice(), |x| x.clone());

        let result1 = ctx.eval(&sum1, hlist![]);
        let result2 = ctx.eval(&sum2, hlist![]);
        let combined_expected = result1 + result2;

        // Manual verification
        let manual_combined = data1.iter().sum::<f64>() + data2.iter().sum::<f64>();

        assert!(
            tolerance.check_arithmetic(combined_expected, manual_combined),
            "Collection composition failed: combined={}, manual={}",
            combined_expected,
            manual_combined
        );
    }
}
