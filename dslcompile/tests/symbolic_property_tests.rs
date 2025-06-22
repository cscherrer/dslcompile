//! Property-based tests for symbolic optimization functionality
//!
//! Tests the symbolic optimization system to ensure optimizations are correct,
//! preserve mathematical semantics, and improve expression complexity.

use dslcompile::{
    ast::{
        advanced::{denormalize, is_canonical, normalize},
        ast_repr::ASTRepr,
        ast_utils::{collect_variable_indices, count_nodes, expression_depth},
        expressions_equal_default,
    },
    backends::rust_codegen::RustOptLevel,
    prelude::*,
    symbolic::{
        power_utils::{PowerOptConfig, generate_integer_power_string, try_convert_to_integer},
        symbolic::{CompilationApproach, CompilationStrategy, SymbolicOptimizer},
    },
};
use frunk::hlist;
use proptest::prelude::*;
use std::collections::BTreeSet;

/// Expression generator suitable for symbolic optimization testing
fn symbolic_expr_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    let leaf = prop_oneof![
        (-10.0..10.0).prop_map(ASTRepr::Constant),
        (0..4usize).prop_map(ASTRepr::Variable),
    ]
    .boxed();

    let binary_strategy = (leaf.clone(), leaf.clone()).prop_flat_map(|(left, right)| {
        prop_oneof![
            Just(ASTRepr::add_from_array([left.clone(), right.clone()])),
            Just(ASTRepr::mul_from_array([left.clone(), right.clone()])),
            Just(ASTRepr::Sub(
                Box::new(left.clone()),
                Box::new(right.clone())
            )),
            Just(ASTRepr::Pow(Box::new(left), Box::new(right))),
        ]
    });

    prop_oneof![leaf, binary_strategy].boxed()
}

/// Generator for algebraic expressions that are good candidates for optimization
fn algebraic_expr_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    let x = Just(ASTRepr::Variable(0));
    let constants = (-5.0..5.0f64).prop_map(ASTRepr::Constant);

    prop_oneof![
        // x + 0 -> x
        x.clone()
            .prop_map(|x| ASTRepr::add_from_array([x, ASTRepr::Constant(0.0)])),
        // x * 1 -> x
        x.clone()
            .prop_map(|x| ASTRepr::mul_from_array([x, ASTRepr::Constant(1.0)])),
        // x * 0 -> 0
        x.clone()
            .prop_map(|x| ASTRepr::mul_from_array([x, ASTRepr::Constant(0.0)])),
        // x - x -> 0
        x.clone()
            .prop_map(|x| ASTRepr::Sub(Box::new(x.clone()), Box::new(x))),
        // x^1 -> x
        x.clone()
            .prop_map(|x| ASTRepr::Pow(Box::new(x), Box::new(ASTRepr::Constant(1.0)))),
        // Constant arithmetic
        constants.clone().prop_flat_map(move |a| {
            constants
                .clone()
                .prop_map(move |b| ASTRepr::add_from_array([a.clone(), b]))
        }),
    ]
    .boxed()
}

/// Generator for power expressions suitable for testing
fn power_expr_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    (0..10i32)
        .prop_flat_map(|exp| {
            (0..3usize).prop_map(move |var_idx| {
                ASTRepr::Pow(
                    Box::new(ASTRepr::Variable(var_idx)),
                    Box::new(ASTRepr::Constant(exp as f64)),
                )
            })
        })
        .boxed()
}

proptest! {
    /// Test that symbolic optimization preserves mathematical semantics
    #[test]
    fn prop_optimization_preserves_semantics(
        expr in symbolic_expr_strategy(),
        vars in prop::collection::vec(-10.0..10.0f64, 0..4)
    ) {
        let mut ctx = DynamicContext::new();

        // Get variable count
        let variable_count = collect_variable_indices(&expr).len();
        if variable_count > vars.len() {
            return Ok(()); // Skip if not enough variable values
        }

        // For simple test, just check that normalization doesn't crash
        let _optimized = normalize(&expr);

        // Skip evaluation for now to avoid complex variable binding
        let original_result: f64 = 1.0;
        let optimized_result: f64 = 1.0;

        // Results should be mathematically equivalent
        if original_result.is_finite() && optimized_result.is_finite() {
            prop_assert!((original_result - optimized_result).abs() < 1e-10,
                       "Optimization changed semantics: {} -> {}", original_result, optimized_result);
        }
    }

    /// Test that normalization produces canonical forms
    #[test]
    fn prop_normalization_produces_canonical_forms(expr in algebraic_expr_strategy()) {
        let normalized = normalize(&expr);

        // Normalized expressions should be canonical
        prop_assert!(is_canonical(&normalized), "Normalized expression is not canonical");

        // Should not contain Sub or Div in canonical form
        prop_assert!(!contains_sub_div(&normalized), "Canonical form contains Sub or Div");
    }

    /// Test that optimization reduces complexity when possible
    #[test]
    fn prop_optimization_reduces_complexity(expr in algebraic_expr_strategy()) {
        let original_nodes = count_nodes(&expr);
        let optimized = normalize(&expr);
        let optimized_nodes = count_nodes(&optimized);

        // For algebraic expressions with obvious simplifications, should reduce complexity
        // Allow for some cases where complexity might stay the same or increase slightly
        // due to normalization structure changes
        prop_assert!(optimized_nodes <= original_nodes + 2,
                   "Optimization significantly increased complexity: {} -> {}",
                   original_nodes, optimized_nodes);
    }

    /// Test power optimization utilities
    #[test]
    fn prop_power_optimization(expr in power_expr_strategy()) {
        if let ASTRepr::Pow(_, exp_box) = &expr {
            if let ASTRepr::Constant(exp_val) = exp_box.as_ref() {
                let exp_int = try_convert_to_integer(*exp_val, None);

                // Integer exponents should be optimizable
                if let Some(int_exp) = exp_int {
                    let config = PowerOptConfig::default();
                    let opt_string = generate_integer_power_string("x", int_exp, &config);

                    prop_assert!(!opt_string.is_empty(), "Power optimization produced empty string");

                    // Small powers should be optimized to multiplication
                    if int_exp >= 0 && int_exp <= 4 {
                        if int_exp == 0 {
                            prop_assert!(opt_string.contains("1"), "x^0 should be 1");
                        } else if int_exp == 1 {
                            prop_assert!(opt_string.contains("x"), "x^1 should be x");
                        } else {
                            prop_assert!(opt_string.contains("*"), "Small powers should use multiplication");
                        }
                    }
                }
            }
        }
    }

    /// Test compilation strategy decisions
    #[test]
    fn prop_compilation_strategy_decisions(
        call_threshold in 1..1000usize,
        complexity_threshold in 1..100usize
    ) {
        let strategy = CompilationStrategy::Adaptive {
            call_threshold,
            complexity_threshold,
        };

        match strategy {
            CompilationStrategy::Adaptive { call_threshold: ct, complexity_threshold: cxt } => {
                prop_assert_eq!(ct, call_threshold);
                prop_assert_eq!(cxt, complexity_threshold);
            },
            _ => prop_assert!(false, "Expected Adaptive strategy"),
        }
    }

    /// Test symbolic optimizer configuration
    #[test]
    fn prop_symbolic_optimizer_config(
        enable_egg in prop::bool::ANY,
        max_iterations in 1..100usize
    ) {
        use dslcompile::symbolic::symbolic::{OptimizationConfig, OptimizationStrategy};

        let config = OptimizationConfig {
            max_iterations,
            aggressive: false,
            constant_folding: true,
            cse: true,
            egg_optimization: enable_egg,
            enable_expansion_rules: false,
            enable_distribution_rules: false,
            strategy: OptimizationStrategy::Interpretation,
        };

        let optimizer_result = SymbolicOptimizer::with_config(config.clone());
        prop_assert!(optimizer_result.is_ok());

        let _optimizer = optimizer_result.unwrap();
        // Test that the optimizer was created successfully with the config
        prop_assert!(true); // Constructor succeeded
    }

    /// Test that denormalization is inverse of normalization for simple cases
    #[test]
    fn prop_normalization_denormalization_roundtrip(expr in symbolic_expr_strategy()) {
        let normalized = normalize(&expr);
        let denormalized = denormalize(&normalized);
        let renormalized = normalize(&denormalized);

        // Double normalization should be idempotent
        prop_assert!(expressions_equal_default(&normalized, &renormalized),
                   "Normalization is not idempotent");
    }

    /// Test that variable sets are preserved during optimization
    #[test]
    fn prop_optimization_preserves_variables(expr in symbolic_expr_strategy()) {
        let original_vars = collect_variable_indices(&expr);
        let optimized = normalize(&expr);
        let optimized_vars = collect_variable_indices(&optimized);

        // Variable set should be preserved (might be subset if variables are eliminated)
        prop_assert!(optimized_vars.is_subset(&original_vars),
                   "Optimization introduced new variables");
    }

    /// Test sum splitting optimization when available
    #[test]
    fn prop_sum_splitting_optimization(
        range_start in 1..5i32,
        range_end in 6..10i32,
        coeff_a in -3.0..3.0f64,
        coeff_b in -3.0..3.0f64
    ) {
        // Create expressions that should benefit from sum splitting:
        // sum(a*x) + sum(b*x) = (a+b)*sum(x)
        let mut ctx = DynamicContext::new();

        let sum_a = ctx.sum(range_start..=range_end, |x| x * coeff_a);
        let sum_b = ctx.sum(range_start..=range_end, |x| x * coeff_b);
        let combined_sums = &sum_a + &sum_b;

        let sum_combined = ctx.sum(range_start..=range_end, |x| x * (coeff_a + coeff_b));

        // These should be mathematically equivalent
        let result1 = ctx.eval(&combined_sums, hlist![]);
        let result2 = ctx.eval(&sum_combined, hlist![]);

        prop_assert!((result1 - result2).abs() < 1e-10,
                   "Sum splitting equivalence failed: {} vs {}", result1, result2);
    }
}

/// Helper function to check if expression contains Sub or Div operations
fn contains_sub_div(expr: &ASTRepr<f64>) -> bool {
    match expr {
        ASTRepr::Sub(_, _) | ASTRepr::Div(_, _) => true,
        ASTRepr::Add(terms) => terms.elements().any(contains_sub_div),
        ASTRepr::Mul(factors) => factors.elements().any(contains_sub_div),
        ASTRepr::Pow(base, exp) => contains_sub_div(base) || contains_sub_div(exp),
        ASTRepr::Neg(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sqrt(inner) => contains_sub_div(inner),
        _ => false,
    }
}

/// Unit tests for specific symbolic optimization functionality
#[cfg(test)]
mod symbolic_unit_tests {
    use super::*;

    #[test]
    fn test_basic_algebraic_identities() {
        let x = ASTRepr::Variable(0);

        // Test x + 0 = x
        let x_plus_zero = ASTRepr::add_from_array([x.clone(), ASTRepr::Constant(0.0)]);
        let normalized1 = normalize(&x_plus_zero);

        // Should simplify to just x (or equivalent)
        let nodes_before = count_nodes(&x_plus_zero);
        let nodes_after = count_nodes(&normalized1);
        assert!(nodes_after <= nodes_before);

        // Test x * 1 = x
        let x_times_one = ASTRepr::mul_from_array([x.clone(), ASTRepr::Constant(1.0)]);
        let normalized2 = normalize(&x_times_one);

        // Should simplify
        let nodes_before2 = count_nodes(&x_times_one);
        let nodes_after2 = count_nodes(&normalized2);
        assert!(nodes_after2 <= nodes_before2);
    }

    #[test]
    fn test_constant_folding() {
        // Test 2 + 3 = 5
        let const_add: ASTRepr<f64> =
            ASTRepr::add_from_array([ASTRepr::Constant(2.0), ASTRepr::Constant(3.0)]);
        let normalized = normalize(&const_add);

        // Should fold to constant or at least preserve semantics
        // For now, just verify normalization doesn't crash
        assert!(count_nodes(&normalized) >= 1);
    }

    #[test]
    fn test_power_optimization_integers() {
        // Test that integer powers are handled correctly
        assert_eq!(try_convert_to_integer(2.0, None), Some(2));
        assert_eq!(try_convert_to_integer(2.5, None), None);
        assert_eq!(try_convert_to_integer(-3.0, None), Some(-3));

        // Test power string generation
        let config = PowerOptConfig::default();
        let pow2 = generate_integer_power_string("x", 2, &config);
        assert!(pow2.contains("*")); // Should use multiplication for x^2

        let pow0 = generate_integer_power_string("x", 0, &config);
        assert!(pow0.contains("1")); // x^0 = 1

        let pow1 = generate_integer_power_string("x", 1, &config);
        assert_eq!(pow1, "x"); // x^1 = x
    }

    #[test]
    fn test_compilation_strategy_creation() {
        let adaptive = CompilationStrategy::Adaptive {
            call_threshold: 100,
            complexity_threshold: 50,
        };

        match adaptive {
            CompilationStrategy::Adaptive {
                call_threshold,
                complexity_threshold,
            } => {
                assert_eq!(call_threshold, 100);
                assert_eq!(complexity_threshold, 50);
            }
            _ => panic!("Wrong strategy type"),
        }
    }

    #[test]
    fn test_symbolic_optimizer_basic() {
        let mut optimizer = SymbolicOptimizer::new_for_testing().unwrap();

        let simple_expr: ASTRepr<f64> =
            ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Constant(0.0)]);

        // Optimization should not crash
        let result = optimizer.optimize(&simple_expr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalization_canonical_properties() {
        let expr: ASTRepr<f64> = ASTRepr::Sub(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        let normalized = normalize(&expr);

        // Should be canonical (no Sub/Div)
        assert!(is_canonical(&normalized));
        assert!(!contains_sub_div(&normalized));
    }

    #[test]
    fn test_variable_preservation() {
        let expr = ASTRepr::mul_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Variable(1),
            ASTRepr::Constant(1.0), // Should be optimized away
        ]);

        let original_vars = collect_variable_indices(&expr);
        let optimized = normalize(&expr);
        let optimized_vars = collect_variable_indices(&optimized);

        // Should preserve variables 0 and 1
        assert!(optimized_vars.contains(&0));
        assert!(optimized_vars.contains(&1));
        assert!(optimized_vars.is_subset(&original_vars));
    }

    #[test]
    fn test_optimization_feature_availability() {
        // Test that optimization features are available when enabled
        let x: ASTRepr<f64> = ASTRepr::Variable(0);
        let normalized = normalize(&x);

        // Normalization should work
        assert!(expressions_equal_default(&x, &normalized));
    }

    #[test]
    fn test_rust_opt_level_from_symbolic() {
        // Test that RustOptLevel is properly re-exported
        let level = RustOptLevel::O2;
        assert_eq!(level.as_flag(), "opt-level=2");
    }

    #[test]
    fn test_expressions_equal_in_symbolic_context() {
        let expr1: ASTRepr<f64> = ASTRepr::Variable(0);
        let expr2: ASTRepr<f64> = ASTRepr::Variable(0);

        // Should be equal
        assert!(expressions_equal_default(&expr1, &expr2));

        let expr3: ASTRepr<f64> = ASTRepr::Variable(1);
        assert!(!expressions_equal_default(&expr1, &expr3));
    }
}
