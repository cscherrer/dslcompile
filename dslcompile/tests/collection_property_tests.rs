use dslcompile::{
    ast::ast_repr::{ASTRepr, Collection},
    prelude::*,
};
use frunk::hlist;
use proptest::prelude::*;

proptest! {
    /// Test summation linearity: sum(a*x + b*x) = (a+b)*sum(x)
    #[test]
    fn prop_summation_linearity(
        start in 1i32..5,
        end in 6i32..10,
        a in -3.0..3.0f64,
        b in -3.0..3.0f64
    ) {
        let mut ctx = DynamicContext::new();

        // LHS: sum((a+b)*x) for x in start..=end
        let lhs_expr = ctx.sum(start..=end, |x| x * (a + b));
        let lhs_result = ctx.eval(&lhs_expr, hlist![]);

        // RHS: sum(a*x) + sum(b*x) = a*sum(x) + b*sum(x) = (a+b)*sum(x)
        let sum_ax = ctx.sum(start..=end, |x| x * a);
        let sum_bx = ctx.sum(start..=end, |x| x * b);
        let rhs_expr = &sum_ax + &sum_bx;
        let rhs_result = ctx.eval(&rhs_expr, hlist![]);

        prop_assert!((lhs_result - rhs_result).abs() < 1e-10);
    }

    /// Test summation commutativity with addition: sum(f) + sum(g) = sum(g) + sum(f)
    #[test]
    fn prop_summation_addition_commutativity(
        start in 1i32..5,
        end in 6i32..10,
        a in -3.0..3.0f64,
        b in -3.0..3.0f64
    ) {
        let mut ctx = DynamicContext::new();

        let sum_f = ctx.sum(start..=end, |x| x * a);
        let sum_g = ctx.sum(start..=end, |x| x * b);

        let lhs = &sum_f + &sum_g;
        let rhs = &sum_g + &sum_f;

        let lhs_result = ctx.eval(&lhs, hlist![]);
        let rhs_result = ctx.eval(&rhs, hlist![]);

        prop_assert!((lhs_result - rhs_result).abs() < 1e-10);
    }

    /// Test sum of sums equals sum of combined: sum(a*x) + sum(b*x) = sum((a+b)*x)
    #[test]
    fn prop_sum_of_sums_distributivity(
        start in 1i32..5,
        end in 6i32..10,
        a in -3.0..3.0f64,
        b in -3.0..3.0f64
    ) {
        let mut ctx = DynamicContext::new();

        // LHS: sum(a*x) + sum(b*x)
        let sum_f = ctx.sum(start..=end, |x| x * a);
        let sum_g = ctx.sum(start..=end, |x| x * b);
        let lhs = &sum_f + &sum_g;
        let lhs_result = ctx.eval(&lhs, hlist![]);

        // RHS: sum((a+b)*x)
        let sum_combined = ctx.sum(start..=end, |x| x * (a + b));
        let rhs_result = ctx.eval(&sum_combined, hlist![]);

        prop_assert!((lhs_result - rhs_result).abs() < 1e-10);
    }

    /// Test empty range behavior: sum from i to i-1 should be zero
    #[test]
    fn prop_empty_range_sum_zero(
        start in 2i32..10,
        a in -5.0..5.0f64
    ) {
        let mut ctx = DynamicContext::new();
        let end = start - 1; // Empty range

        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(start..=end, |x| x * a);
        let result = ctx.eval(&sum_expr, hlist![]);

        // Empty range should sum to zero
        prop_assert!(result.abs() < 1e-10);
    }

    /// Test quadratic summation formula: sum(i^2) for i=1..n = n(n+1)(2n+1)/6
    #[test]
    fn prop_quadratic_summation_formula(n in 1i32..8) {
        let mut ctx = DynamicContext::new();

        let sum_squares: DynamicExpr<f64, 0> = ctx.sum(1..=n, |i| &i * &i);
        let result = ctx.eval(&sum_squares, hlist![]);

        // Mathematical formula: n(n+1)(2n+1)/6
        let expected = f64::from(n * (n + 1) * (2 * n + 1)) / 6.0;

        prop_assert!((result - expected).abs() < 1e-10);
    }

    /// Test that evaluation and AST construction are consistent
    #[test]
    fn prop_evaluation_ast_consistency(
        start in 1i32..5,
        end in 6i32..10,
        coeff in -3.0..3.0f64
    ) {
        let mut ctx = DynamicContext::new();

        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(start..=end, |x| x * coeff);
        let result_direct = ctx.eval(&sum_expr, hlist![]);

        // Verify AST structure is as expected
        let ast = ctx.to_ast(&sum_expr);
        match ast {
            ASTRepr::Sum(collection_box) => {
                match collection_box.as_ref() {
                    Collection::Map { lambda: _, collection } => {
                        match collection.as_ref() {
                            Collection::Range { start: ast_start, end: ast_end } => {
                                // Verify range bounds in AST match our input
                                if let (ASTRepr::Constant(s), ASTRepr::Constant(e)) = (ast_start.as_ref(), ast_end.as_ref()) {
                                    prop_assert!((*s - f64::from(start)).abs() < 1e-10);
                                    prop_assert!((*e - f64::from(end)).abs() < 1e-10);
                                }
                            }
                            _ => prop_assert!(false, "Expected Range collection in AST"),
                        }
                    }
                    _ => prop_assert!(false, "Expected Map collection in AST"),
                }
            }
            _ => prop_assert!(false, "Expected Sum in AST"),
        }

        // Mathematical verification: sum(i * coeff) = coeff * sum(i)
        let expected = coeff * f64::from((start + end) * (end - start + 1)) / 2.0;
        prop_assert!((result_direct - expected).abs() < 1e-10);
    }

    /// Test multiplication distributivity: a * sum(f(x)) = sum(a * f(x))
    #[test]
    fn prop_multiplication_distributivity(
        start in 1i32..5,
        end in 6i32..10,
        a in -3.0..3.0f64,
        b in -3.0..3.0f64
    ) {
        let mut ctx = DynamicContext::new();

        // LHS: a * sum(b*x)
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(start..=end, |x| x * b);
        let lhs = &sum_expr * a;
        let lhs_result = ctx.eval(&lhs, hlist![]);

        // RHS: sum(a * b * x)
        let sum_distributed: DynamicExpr<f64, 0> = ctx.sum(start..=end, |x| x * a * b);
        let rhs_result = ctx.eval(&sum_distributed, hlist![]);

        prop_assert!((lhs_result - rhs_result).abs() < 1e-10);
    }
}

#[cfg(test)]
mod collection_unit_tests {
    use super::*;

    #[test]
    fn test_basic_range_summation() {
        let mut ctx = DynamicContext::new();

        // Test 1..3 identity sum: 1 + 2 + 3 = 6
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=3, |i| i);
        let result = ctx.eval(&sum_expr, hlist![]);
        println!("Expected: 6.0, Actual: {result}");
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_function_summation() {
        let mut ctx = DynamicContext::new();

        // Test sum(2*i + 1) for i = 1..3 = (2*1+1) + (2*2+1) + (2*3+1) = 3 + 5 + 7 = 15
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=3, |i| i * 2.0 + 1.0);
        let result = ctx.eval(&sum_expr, hlist![]);
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadratic_summation() {
        let mut ctx = DynamicContext::new();

        // Test sum(i^2) for i = 1..4 = 1 + 4 + 9 + 16 = 30
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=4, |i| &i * &i);
        let result = ctx.eval(&sum_expr, hlist![]);
        assert!((result - 30.0).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_empty_range_summation() {
        let mut ctx = DynamicContext::new();

        // Test empty range: sum from 5 to 4 should be 0
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(5..=4, |i| i);
        let result = ctx.eval(&sum_expr, hlist![]);
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_data_array_summation() {
        let mut ctx = DynamicContext::new();

        // Test summing over data array
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(data.as_slice(), |x| &x * 2.0);
        let result = ctx.eval(&sum_expr, hlist![]);

        // Expected: 2*1 + 2*2 + 2*3 + 2*4 = 2 + 4 + 6 + 8 = 20
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_nested_arithmetic_with_summation() {
        let mut ctx = DynamicContext::new();

        // Test: 3 * sum(i) + 10 where sum(i) for i=1..3 = 6
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=3, |i| i);
        let combined = &sum_expr * 3.0 + 10.0;
        let result = ctx.eval(&combined, hlist![]);

        // Expected: 3 * 6 + 10 = 28
        assert!((result - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_arithmetic_series_formula() {
        let mut ctx = DynamicContext::new();

        // Test arithmetic series: sum(i) for i=1..n = n(n+1)/2
        for n in 1..=10 {
            let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=n, |i| i);
            let result = ctx.eval(&sum_expr, hlist![]);
            let expected = f64::from(n * (n + 1)) / 2.0;

            assert!(
                (result - expected).abs() < 1e-10,
                "Failed for n={n}: got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_sum_linearity_property() {
        let mut ctx = DynamicContext::new();

        // Test that sum(a*f + b*g) = a*sum(f) + b*sum(g)
        let a = 2.5;
        let b = -1.5;

        let sum_f: DynamicExpr<f64, 0> = ctx.sum(1..=5, |i| i);
        let sum_g: DynamicExpr<f64, 0> = ctx.sum(1..=5, |i| &i * &i);
        let sum_combined: DynamicExpr<f64, 0> = ctx.sum(1..=5, |i| &i * a + &i * &i * b);

        let lhs_result = ctx.eval(&sum_combined, hlist![]);
        let rhs_result = a * ctx.eval(&sum_f, hlist![]) + b * ctx.eval(&sum_g, hlist![]);

        assert!((lhs_result - rhs_result).abs() < 1e-10);
    }
}
