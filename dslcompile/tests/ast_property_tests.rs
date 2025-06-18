//! AST Structure and Property Tests
//!
//! This module implements comprehensive property-based tests for AST expression building,
//! structure validation, and mathematical correctness. Uses proptest to generate random
//! expressions and verify invariants.

use dslcompile::{
    ast::{
        ast_repr::{ASTRepr, Collection},
        visitor::ASTVisitor,
    },
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
                        Just(ASTRepr::add_from_array([left.clone(), right.clone()])),
                        Just(ASTRepr::Sub(
                            Box::new(left.clone()),
                            Box::new(right.clone())
                        )),
                        Just(ASTRepr::mul_from_array([left.clone(), right.clone()])),
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

/// Utility functions for AST analysis using visitor pattern
pub mod ast_utils {
    use super::*;

    /// Visitor to collect all variable indices used in an expression
    struct VariableCollector {
        indices: HashSet<usize>,
    }

    impl ASTVisitor<f64> for VariableCollector {
        type Output = ();
        type Error = ();

        fn visit_constant(
            &mut self,
            _value: &f64,
        ) -> std::result::Result<Self::Output, Self::Error> {
            Ok(())
        }

        fn visit_variable(
            &mut self,
            index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            self.indices.insert(index);
            Ok(())
        }

        fn visit_bound_var(
            &mut self,
            _index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            // BoundVar indices are local to their lambda scope, don't collect them
            Ok(())
        }

        fn visit_empty_collection(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            Ok(())
        }

        fn visit_collection_variable(
            &mut self,
            index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            self.indices.insert(index);
            Ok(())
        }

        fn visit_generic_node(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            Ok(())
        }

        // Override the main visit method to handle multiset operations correctly
        fn visit(&mut self, expr: &ASTRepr<f64>) -> std::result::Result<Self::Output, Self::Error> {
            match expr {
                ASTRepr::Constant(value) => self.visit_constant(value),
                ASTRepr::Variable(index) => self.visit_variable(*index),
                ASTRepr::BoundVar(index) => self.visit_bound_var(*index),
                ASTRepr::Add(terms) => {
                    for term in terms.elements() {
                        self.visit(term)?;
                    }
                    Ok(())
                }
                ASTRepr::Sub(left, right) => {
                    self.visit(left)?;
                    self.visit(right)?;
                    Ok(())
                }
                ASTRepr::Mul(factors) => {
                    for factor in factors.elements() {
                        self.visit(factor)?;
                    }
                    Ok(())
                }
                ASTRepr::Div(left, right) => {
                    self.visit(left)?;
                    self.visit(right)?;
                    Ok(())
                }
                ASTRepr::Pow(base, exp) => {
                    self.visit(base)?;
                    self.visit(exp)?;
                    Ok(())
                }
                ASTRepr::Neg(inner) => self.visit(inner),
                ASTRepr::Sin(inner) => self.visit(inner),
                ASTRepr::Cos(inner) => self.visit(inner),
                ASTRepr::Ln(inner) => self.visit(inner),
                ASTRepr::Exp(inner) => self.visit(inner),
                ASTRepr::Sqrt(inner) => self.visit(inner),
                ASTRepr::Sum(collection) => {
                    // Visit the collection to collect variables
                    match collection.as_ref() {
                        Collection::Empty => Ok(()),
                        Collection::Singleton(expr) => self.visit(expr),
                        Collection::Range { start, end } => {
                            self.visit(start)?;
                            self.visit(end)
                        }
                        Collection::Variable(index) => self.visit_collection_variable(*index),
                        Collection::Filter {
                            collection: inner_collection,
                            predicate,
                        } => {
                            // Recursively visit the inner collection and predicate
                            self.visit(&ASTRepr::Sum(inner_collection.clone()))?;
                            self.visit(predicate)
                        }
                        Collection::Map {
                            lambda,
                            collection: inner_collection,
                        } => {
                            self.visit(&lambda.body)?;
                            self.visit(&ASTRepr::Sum(inner_collection.clone()))
                        }
                        Collection::DataArray(_) => Ok(()),
                    }
                }
                ASTRepr::Lambda(lambda) => self.visit(&lambda.body),
                ASTRepr::Let(_, expr, body) => {
                    self.visit(expr)?;
                    self.visit(body)
                }
            }
        }
    }

    /// Collect all variable indices used in an expression
    #[must_use]
    pub fn collect_variable_indices(expr: &ASTRepr<f64>) -> HashSet<usize> {
        let mut visitor = VariableCollector {
            indices: HashSet::new(),
        };
        let _ = visitor.visit(expr); // Ignore errors for this utility
        visitor.indices
    }

    /// Visitor to compute expression depth
    struct DepthCalculator {
        max_depth: usize,
        current_depth: usize,
    }

    impl ASTVisitor<f64> for DepthCalculator {
        type Output = usize;
        type Error = ();

        fn visit_constant(
            &mut self,
            _value: &f64,
        ) -> std::result::Result<Self::Output, Self::Error> {
            self.current_depth += 1;
            self.max_depth = self.max_depth.max(self.current_depth);
            self.current_depth -= 1;
            Ok(1)
        }

        fn visit_variable(
            &mut self,
            _index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            self.current_depth += 1;
            self.max_depth = self.max_depth.max(self.current_depth);
            self.current_depth -= 1;
            Ok(1)
        }

        fn visit_bound_var(
            &mut self,
            _index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            self.current_depth += 1;
            self.max_depth = self.max_depth.max(self.current_depth);
            self.current_depth -= 1;
            Ok(1)
        }

        fn visit_empty_collection(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            self.current_depth += 1;
            self.max_depth = self.max_depth.max(self.current_depth);
            self.current_depth -= 1;
            Ok(1)
        }

        fn visit_collection_variable(
            &mut self,
            _index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            self.current_depth += 1;
            self.max_depth = self.max_depth.max(self.current_depth);
            self.current_depth -= 1;
            Ok(1)
        }

        fn visit_generic_node(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            self.current_depth += 1;
            self.max_depth = self.max_depth.max(self.current_depth);
            self.current_depth -= 1;
            Ok(self.current_depth + 1)
        }
    }

    /// Compute the maximum depth of an expression
    #[must_use]
    pub fn compute_expression_depth(expr: &ASTRepr<f64>) -> usize {
        let mut visitor = DepthCalculator {
            max_depth: 0,
            current_depth: 0,
        };
        let _ = visitor.visit(expr); // Ignore errors for this utility
        visitor.max_depth
    }

    /// Visitor to check for Sub or Div operations
    struct SubDivChecker {
        found: bool,
    }

    impl ASTVisitor<f64> for SubDivChecker {
        type Output = bool;
        type Error = ();

        fn visit_constant(
            &mut self,
            _value: &f64,
        ) -> std::result::Result<Self::Output, Self::Error> {
            Ok(false)
        }

        fn visit_variable(
            &mut self,
            _index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            Ok(false)
        }

        fn visit_bound_var(
            &mut self,
            _index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            Ok(false)
        }

        fn visit_empty_collection(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            Ok(false)
        }

        fn visit_collection_variable(
            &mut self,
            _index: usize,
        ) -> std::result::Result<Self::Output, Self::Error> {
            Ok(false)
        }

        fn visit_sub_node(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            self.found = true;
            Ok(true)
        }

        fn visit_div_node(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            self.found = true;
            Ok(true)
        }

        fn visit_generic_node(&mut self) -> std::result::Result<Self::Output, Self::Error> {
            Ok(self.found)
        }
    }

    /// Check if an expression contains Sub or Div operations
    #[must_use]
    pub fn contains_sub_or_div(expr: &ASTRepr<f64>) -> bool {
        let mut visitor = SubDivChecker { found: false };
        let _ = visitor.visit(expr); // Ignore errors for this utility
        visitor.found
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
            expr = expr + ctx.constant(f64::from(i));
        }

        // Verify it evaluates correctly
        let result = ctx.eval(&expr, hlist![0.0]);
        let expected = f64::from((1..=20).sum::<i32>()); // Sum of 1 to 20
        assert!(
            (result - expected).abs() < 1e-12,
            "Deep nesting evaluation incorrect: {result} vs {expected}"
        );

        // Verify AST structure
        let ast = ctx.to_ast(&expr);
        let depth = ast_utils::compute_expression_depth(&ast);
        assert!(depth >= 20, "Deep expression should have appropriate depth");
    }
}
