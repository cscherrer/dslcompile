//! AST Structure and Property Tests
//!
//! This module implements comprehensive property-based tests for AST expression building,
//! structure validation, and mathematical correctness. Uses proptest to generate random
//! expressions and verify invariants.

use dslcompile::{
    ast::ast_repr::{ASTRepr, Collection, Lambda},
    prelude::*,
};
use frunk::hlist;
use proptest::prelude::*;
use std::collections::HashSet;

/// Configuration for expression complexity in property tests
#[derive(Debug, Clone)]
pub struct ExpressionComplexity {
    pub max_depth: usize,
    pub max_variables: usize,
    pub max_constants: usize,
    pub allow_transcendental: bool,
}

impl Default for ExpressionComplexity {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_variables: 3,
            max_constants: 10,
            allow_transcendental: true,
        }
    }
}

/// Wrapper for arbitrary AST expressions in property tests
#[derive(Debug, Clone)]
pub struct ArbitraryExpr {
    pub expr: ASTRepr<f64>,
    pub complexity: ExpressionComplexity,
}

impl ArbitraryExpr {
    pub fn simple() -> BoxedStrategy<Self> {
        let config = ExpressionComplexity {
            max_depth: 3,
            max_variables: 2,
            max_constants: 5,
            allow_transcendental: false,
        };
        Self::with_complexity(config)
    }

    pub fn complex() -> BoxedStrategy<Self> {
        let config = ExpressionComplexity {
            max_depth: 8,
            max_variables: 4,
            max_constants: 20,
            allow_transcendental: true,
        };
        Self::with_complexity(config)
    }

    pub fn with_complexity(complexity: ExpressionComplexity) -> BoxedStrategy<Self> {
        Self::generate_expr_recursive(0, &complexity)
            .prop_map(move |expr| ArbitraryExpr {
                expr,
                complexity: complexity.clone(),
            })
            .boxed()
    }

    fn generate_expr_recursive(
        depth: usize,
        config: &ExpressionComplexity,
    ) -> BoxedStrategy<ASTRepr<f64>> {
        if depth >= config.max_depth {
            // Base case: only constants and variables
            prop_oneof![
                (0.0..1000.0).prop_map(ASTRepr::Constant),
                (0..config.max_variables).prop_map(ASTRepr::Variable),
            ]
            .boxed()
        } else {
            let leaf_strategy = prop_oneof![
                (0.0..1000.0).prop_map(ASTRepr::Constant),
                (0..config.max_variables).prop_map(ASTRepr::Variable),
            ]
            .boxed();

            let config_clone = config.clone();
            let binary_ops = Self::generate_expr_recursive(depth + 1, config)
                .prop_flat_map(move |left| {
                    Self::generate_expr_recursive(depth + 1, &config_clone)
                        .prop_map(move |right| (left.clone(), right))
                })
                .prop_flat_map(|(left, right)| {
                    prop_oneof![
                        Just(ASTRepr::Add(vec![
                            left.clone(),
                            right.clone()
                        ])),
                        Just(ASTRepr::Sub(
                            Box::new(left.clone()),
                            Box::new(right.clone())
                        )),
                        Just(ASTRepr::Mul(vec![
                            left.clone(),
                            right.clone()
                        ])),
                        Just(ASTRepr::Div(
                            Box::new(left.clone()),
                            Box::new(right.clone())
                        )),
                        Just(ASTRepr::Pow(Box::new(left), Box::new(right))),
                    ]
                })
                .boxed();

            let allow_transcendental = config.allow_transcendental;
            let unary_ops = Self::generate_expr_recursive(depth + 1, config)
                .prop_flat_map(move |inner| {
                    if allow_transcendental {
                        prop_oneof![
                            Just(ASTRepr::Neg(Box::new(inner.clone()))),
                            Just(ASTRepr::Sin(Box::new(inner.clone()))),
                            Just(ASTRepr::Cos(Box::new(inner.clone()))),
                            Just(ASTRepr::Exp(Box::new(inner.clone()))),
                            Just(ASTRepr::Ln(Box::new(inner.clone()))),
                            Just(ASTRepr::Sqrt(Box::new(inner))),
                        ]
                        .boxed()
                    } else {
                        Just(ASTRepr::Neg(Box::new(inner))).boxed()
                    }
                })
                .boxed();

            prop_oneof![leaf_strategy, binary_ops, unary_ops,].boxed()
        }
    }
}

/// Utility functions for AST analysis
pub mod ast_utils {
    use super::*;

    pub fn collect_variable_indices<T>(expr: &ASTRepr<T>) -> HashSet<usize> {
        let mut indices = HashSet::new();
        collect_variables_recursive(expr, &mut indices);
        indices
    }

    fn collect_variables_recursive<T>(expr: &ASTRepr<T>, indices: &mut HashSet<usize>) {
        match expr {
            ASTRepr::Variable(idx) => {
                indices.insert(*idx);
            }
            ASTRepr::BoundVar(_) => {
                // BoundVar indices are local to their lambda scope
            }
            ASTRepr::Constant(_) => {}
            ASTRepr::Add(operands) => {
                for operand in operands {
                    collect_variables_recursive(operand, indices);
                }
            }
            ASTRepr::Sub(left, right) => {
                collect_variables_recursive(left, indices);
                collect_variables_recursive(right, indices);
            }
            ASTRepr::Mul(operands) => {
                for operand in operands {
                    collect_variables_recursive(operand, indices);
                }
            }
            ASTRepr::Div(left, right) => {
                collect_variables_recursive(left, indices);
                collect_variables_recursive(right, indices);
            }
            ASTRepr::Pow(left, right) => {
                collect_variables_recursive(left, indices);
                collect_variables_recursive(right, indices);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Sqrt(inner) => {
                collect_variables_recursive(inner, indices);
            }
            ASTRepr::Sum(collection) => {
                collect_variables_from_collection(collection, indices);
            }
            ASTRepr::Lambda(lambda) => {
                collect_variables_recursive(&lambda.body, indices);
            }
            ASTRepr::Let(_, expr, body) => {
                collect_variables_recursive(expr, indices);
                collect_variables_recursive(body, indices);
            }
        }
    }

    fn collect_variables_from_collection<T>(
        collection: &Collection<T>,
        indices: &mut HashSet<usize>,
    ) {
        match collection {
            Collection::Empty => {}
            Collection::Singleton(expr) => {
                collect_variables_recursive(expr, indices);
            }
            Collection::Range { start, end } => {
                collect_variables_recursive(start, indices);
                collect_variables_recursive(end, indices);
            }
            Collection::Variable(idx) => {
                indices.insert(*idx);
            }
            Collection::Filter {
                collection,
                predicate,
            } => {
                collect_variables_from_collection(collection, indices);
                collect_variables_recursive(predicate, indices);
            }
            Collection::Map {
                lambda: _,
                collection,
            } => {
                // Lambda variables are bound, only collect from collection
                collect_variables_from_collection(collection, indices);
            }
            Collection::DataArray(_) => {
                // DataArray contains literal data, no variables to collect
            }
        }
    }

    pub fn compute_expression_depth<T>(expr: &ASTRepr<T>) -> usize {
        match expr {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => 1,
            ASTRepr::Add(operands) => {
                1 + operands.iter().map(|operand| compute_expression_depth(operand)).max().unwrap_or(0)
            }
            ASTRepr::Sub(left, right) => {
                1 + compute_expression_depth(left).max(compute_expression_depth(right))
            }
            ASTRepr::Mul(operands) => {
                1 + operands.iter().map(|operand| compute_expression_depth(operand)).max().unwrap_or(0)
            }
            ASTRepr::Div(left, right) => {
                1 + compute_expression_depth(left).max(compute_expression_depth(right))
            }
            ASTRepr::Pow(left, right) => {
                1 + compute_expression_depth(left).max(compute_expression_depth(right))
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Sqrt(inner) => 1 + compute_expression_depth(inner),
            ASTRepr::Sum(collection) => 1 + compute_collection_depth(collection),
            ASTRepr::Lambda(lambda) => 1 + compute_expression_depth(&lambda.body),
            ASTRepr::Let(_, expr, body) => {
                1 + compute_expression_depth(expr).max(compute_expression_depth(body))
            }
        }
    }

    fn compute_collection_depth<T>(collection: &Collection<T>) -> usize {
        match collection {
            Collection::Empty => 1,
            Collection::Singleton(expr) => 1 + compute_expression_depth(expr),
            Collection::Range { start, end } => {
                1 + compute_expression_depth(start).max(compute_expression_depth(end))
            }
            Collection::Variable(_) => 1,
            Collection::Filter {
                collection,
                predicate,
            } => 1 + compute_collection_depth(collection).max(compute_expression_depth(predicate)),
            Collection::Map { lambda, collection } => {
                1 + compute_expression_depth(&lambda.body).max(compute_collection_depth(collection))
            }
            Collection::DataArray(_) => 1,
        }
    }

    pub fn contains_sub_or_div<T>(expr: &ASTRepr<T>) -> bool {
        match expr {
            ASTRepr::Sub(_, _) | ASTRepr::Div(_, _) => true,
            ASTRepr::Add(operands) => {
                operands.iter().any(|operand| contains_sub_or_div(operand))
            }
            ASTRepr::Mul(operands) => {
                operands.iter().any(|operand| contains_sub_or_div(operand))
            }
            ASTRepr::Pow(left, right) => {
                contains_sub_or_div(left) || contains_sub_or_div(right)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Sqrt(inner) => contains_sub_or_div(inner),
            ASTRepr::Sum(collection) => contains_sub_or_div_in_collection(collection),
            ASTRepr::Lambda(lambda) => contains_sub_or_div(&lambda.body),
            ASTRepr::Let(_, expr, body) => contains_sub_or_div(expr) || contains_sub_or_div(body),
            _ => false,
        }
    }

    fn contains_sub_or_div_in_collection<T>(collection: &Collection<T>) -> bool {
        match collection {
            Collection::Empty => false,
            Collection::Singleton(expr) => contains_sub_or_div(expr),
            Collection::Range { start, end } => {
                contains_sub_or_div(start) || contains_sub_or_div(end)
            }
            Collection::Variable(_) => false,
            Collection::Filter {
                collection,
                predicate,
            } => contains_sub_or_div_in_collection(collection) || contains_sub_or_div(predicate),
            Collection::Map { lambda, collection } => {
                contains_sub_or_div(&lambda.body) || contains_sub_or_div_in_collection(collection)
            }
            Collection::DataArray(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dslcompile::ast::advanced::{denormalize, is_canonical, normalize};

    /// Test basic properties of expression generation
    #[test]
    fn test_arbitrary_expression_generation() {
        let config = ExpressionComplexity::default();

        // Generate a few expressions to verify the generator works
        for _ in 0..10 {
            let strategy = ArbitraryExpr::with_complexity(config.clone());
            let mut runner = proptest::test_runner::TestRunner::default();
            let arbitrary_expr = strategy.new_tree(&mut runner).unwrap().current();

            // Basic validation
            let variables = ast_utils::collect_variable_indices(&arbitrary_expr.expr);
            assert!(variables.iter().all(|&idx| idx < config.max_variables));

            let depth = ast_utils::compute_expression_depth(&arbitrary_expr.expr);
            assert!(depth <= config.max_depth + 1); // Allow some flexibility for leaf nodes
        }
    }

    proptest! {
        #[test]
        fn prop_variable_indices_are_valid(arbitrary_expr in ArbitraryExpr::simple()) {
            let variables = ast_utils::collect_variable_indices(&arbitrary_expr.expr);

            // All variable indices should be within the allowed range
            for &var_idx in &variables {
                prop_assert!(var_idx < arbitrary_expr.complexity.max_variables,
                           "Variable index {} exceeds max {}", var_idx, arbitrary_expr.complexity.max_variables);
            }
        }

        #[test]
        fn prop_expression_depth_is_bounded(arbitrary_expr in ArbitraryExpr::simple()) {
            let depth = ast_utils::compute_expression_depth(&arbitrary_expr.expr);

            // Depth should be within reasonable bounds
            prop_assert!(depth <= arbitrary_expr.complexity.max_depth + 2,
                       "Expression depth {} exceeds expected maximum {}",
                       depth, arbitrary_expr.complexity.max_depth + 2);
        }

        #[test]
        fn prop_normalization_preserves_variable_set(arbitrary_expr in ArbitraryExpr::simple()) {
            let original_vars = ast_utils::collect_variable_indices(&arbitrary_expr.expr);
            let normalized = normalize(&arbitrary_expr.expr);
            let normalized_vars = ast_utils::collect_variable_indices(&normalized);

            // Normalization should not change the set of variables used
            prop_assert_eq!(original_vars, normalized_vars,
                          "Normalization changed variable set");
        }

        #[test]
        fn prop_canonical_forms_have_no_sub_div(arbitrary_expr in ArbitraryExpr::simple()) {
            let normalized = normalize(&arbitrary_expr.expr);

            // Canonical forms should not contain Sub or Div operations
            prop_assert!(!ast_utils::contains_sub_or_div(&normalized),
                       "Canonical form still contains Sub or Div operations");

            // Should be marked as canonical
            prop_assert!(is_canonical(&normalized), "Normalized expression is not canonical");
        }

        #[test]
        fn prop_denormalization_roundtrip_preserves_structure(arbitrary_expr in ArbitraryExpr::simple()) {
            let normalized = normalize(&arbitrary_expr.expr);
            let denormalized = denormalize(&normalized);
            let renormalized = normalize(&denormalized);

            // Double normalization should be idempotent
            prop_assert_eq!(format!("{:?}", normalized), format!("{:?}", renormalized),
                          "Double normalization is not idempotent");
        }

        #[test]
        fn prop_mathematical_identities_basic(x in -100.0..100.0f64, y in -100.0..100.0f64) {
            let mut ctx = DynamicContext::new();

            // Test commutativity: x + y = y + x
            let x_var = ctx.var();
            let y_var = ctx.var();

            let expr1 = &x_var + &y_var;
            let expr2 = &y_var + &x_var;

            let result1 = ctx.eval(&expr1, hlist![x, y]);
            let result2 = ctx.eval(&expr2, hlist![x, y]);

            prop_assert!((result1 - result2).abs() < 1e-12,
                       "Commutativity failed: {} + {} vs {} + {}", x, y, y, x);
        }

        #[test]
        fn prop_mathematical_identities_associativity(
            x in -10.0..10.0f64,
            y in -10.0..10.0f64,
            z in -10.0..10.0f64
        ) {
            let mut ctx = DynamicContext::new();

            // Test associativity: (x + y) + z = x + (y + z)
            let x_var = ctx.var();
            let y_var = ctx.var();
            let z_var = ctx.var();

            let expr1 = (&x_var + &y_var) + &z_var;
            let expr2 = &x_var + (&y_var + &z_var);

            let result1 = ctx.eval(&expr1, hlist![x, y, z]);
            let result2 = ctx.eval(&expr2, hlist![x, y, z]);

            prop_assert!((result1 - result2).abs() < 1e-12,
                       "Associativity failed: ({} + {}) + {} vs {} + ({} + {})",
                       x, y, z, x, y, z);
        }

        #[test]
        fn prop_distributive_law(
            x in -10.0..10.0f64,
            y in -10.0..10.0f64,
            z in -10.0..10.0f64
        ) {
            let mut ctx = DynamicContext::new();

            // Test distributivity: x * (y + z) = x*y + x*z
            let x_var = ctx.var();
            let y_var = ctx.var();
            let z_var = ctx.var();

            let expr1 = &x_var * (&y_var + &z_var);
            let expr2 = &x_var * &y_var + &x_var * &z_var;

            let result1 = ctx.eval(&expr1, hlist![x, y, z]);
            let result2 = ctx.eval(&expr2, hlist![x, y, z]);

            prop_assert!((result1 - result2).abs() < 1e-12,
                       "Distributivity failed: {} * ({} + {}) vs {}*{} + {}*{}",
                       x, y, z, x, y, x, z);
        }

        #[test]
        fn prop_identity_elements(x in -100.0..100.0f64) {
            let mut ctx = DynamicContext::new();
            let x_var = ctx.var();

            // Test additive identity: x + 0 = x
            let add_identity = &x_var + 0.0;
            let add_result = ctx.eval(&add_identity, hlist![x]);
            prop_assert!((add_result - x).abs() < 1e-12,
                       "Additive identity failed: {} + 0 = {}", x, add_result);

            // Test multiplicative identity: x * 1 = x
            let mul_identity = &x_var * 1.0;
            let mul_result = ctx.eval(&mul_identity, hlist![x]);
            prop_assert!((mul_result - x).abs() < 1e-12,
                       "Multiplicative identity failed: {} * 1 = {}", x, mul_result);
        }

        #[test]
        fn prop_inverse_functions(x in 0.1..100.0f64) {
            let mut ctx = DynamicContext::new();
            let x_var = ctx.var();

            // Test exp(ln(x)) = x for x > 0
            let exp_ln = x_var.clone().ln().exp();
            let result = ctx.eval(&exp_ln, hlist![x]);
            prop_assert!((result - x).abs() < 1e-10,
                       "exp(ln({})) = {} (expected {})", x, result, x);
        }

        #[test]
        fn prop_trigonometric_identity(x in -std::f64::consts::PI..std::f64::consts::PI) {
            let mut ctx = DynamicContext::new();
            let x_var = ctx.var();

            // Test sin^2(x) + cos^2(x) = 1
            let sin_x = x_var.clone().sin();
            let cos_x = x_var.cos();
            let identity = &sin_x * &sin_x + &cos_x * &cos_x;

            let result = ctx.eval(&identity, hlist![x]);
            prop_assert!((result - 1.0).abs() < 1e-10,
                       "sin^2({}) + cos^2({}) = {} (expected 1.0)", x, x, result);
        }
    }

    #[test]
    fn test_complex_nested_expression() {
        let mut ctx = DynamicContext::new();
        let x: DynamicExpr<f64, 0> = ctx.var();
        let y: DynamicExpr<f64, 0> = ctx.var();

        // Build a complex nested expression: sin(exp(x^2 + y^2))
        let x_squared = x.clone().pow(ctx.constant(2.0));
        let y_squared = y.clone().pow(ctx.constant(2.0));
        let sum_squares = x_squared + y_squared;
        let exp_sum = sum_squares.exp();
        let sin_exp = exp_sum.sin();

        // Verify it can be evaluated
        let result: f64 = ctx.eval(&sin_exp, hlist![1.0, 1.0]);
        assert!(
            result.is_finite(),
            "Complex expression should produce finite result"
        );

        // Verify AST structure is correct
        let ast = ctx.to_ast(&sin_exp);
        let depth = ast_utils::compute_expression_depth(&ast);
        assert!(
            depth >= 5,
            "Complex expression should have sufficient depth"
        );

        let variables = ast_utils::collect_variable_indices(&ast);
        assert_eq!(variables.len(), 2, "Should use exactly 2 variables");
        assert!(
            variables.contains(&0) && variables.contains(&1),
            "Should use variables 0 and 1"
        );
    }

    #[test]
    fn test_deep_nesting_stress() {
        let mut ctx = DynamicContext::new();
        let mut expr = ctx.var();

        // Build a deeply nested expression: ((((x + 1) + 1) + 1) + ... )
        for i in 1..=20 {
            expr = expr + ctx.constant(i as f64);
        }

        // Verify it evaluates correctly
        let result = ctx.eval(&expr, hlist![0.0]);
        let expected = (1..=20).sum::<i32>() as f64; // Sum of 1 to 20
        assert!(
            (result - expected).abs() < 1e-12,
            "Deep nesting evaluation incorrect: {} vs {}",
            result,
            expected
        );

        // Verify AST structure
        let ast = ctx.to_ast(&expr);
        let depth = ast_utils::compute_expression_depth(&ast);
        assert!(depth >= 20, "Deep expression should have appropriate depth");
    }
}
