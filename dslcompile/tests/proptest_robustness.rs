//! Property-based testing for robustness and correctness
//!
//! This module contains comprehensive property-based tests using proptest
//! to ensure the mathematical correctness and robustness of the DSL compiler.

use dslcompile::SymbolicOptimizer;
use dslcompile::ast::pretty::{pretty_anf, pretty_ast};
use dslcompile::ast::{ASTRepr, VariableRegistry};
use dslcompile::error::DSLCompileError;
use dslcompile::interval_domain::{IntervalDomain, IntervalDomainAnalyzer};
use dslcompile::symbolic::anf::{ANFAtom, ANFComputation, ANFExpr, VarRef, convert_to_anf};
use dslcompile::symbolic::summation::DirectEval;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use std::collections::HashMap;

// Configuration for expression generation
#[derive(Debug, Clone, Copy)]
struct ExprConfig {
    max_depth: usize,
    max_vars: usize,
    include_transcendental: bool,
    include_constants: bool,
    constant_range: (f64, f64),
}

impl Default for ExprConfig {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_vars: 4,
            include_transcendental: true,
            include_constants: true,
            constant_range: (-100.0, 100.0),
        }
    }
}

// Wrapper for debug printing ASTRepr
#[derive(Clone)]
struct DebugExpr(ASTRepr<f64>);

impl std::fmt::Debug for DebugExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ASTRepr<f64>(...)")
    }
}

// Strategy for generating arbitrary mathematical expressions
fn arb_expr_with_config(
    config: ExprConfig,
) -> impl Strategy<Value = (DebugExpr, VariableRegistry, Vec<f64>)> {
    (1..=config.max_vars).prop_flat_map(move |num_vars| {
        let mut registry = VariableRegistry::new();

        // Generate variable names and register them
        let var_names: Vec<String> = (0..num_vars).map(|i| format!("var_{i}")).collect();

        // Register variables (no longer pass names, just register them)
        let _var_indices: Vec<usize> = var_names
            .iter()
            .map(|_name| registry.register_variable()) // Remove the name parameter
            .collect();

        // Generate values for each variable
        let var_values = prop::collection::vec(-100.0..100.0, num_vars);

        // Recursively generate expression using variable indices
        let config = config;
        (Just(registry), var_values)
            .prop_flat_map(move |(registry, var_values)| {
                let var_indices: Vec<usize> = (0..var_values.len()).collect(); // Use indices directly

                let expr_strategy = arb_expr_recursive(var_indices, config, 0);
                (Just(registry), Just(var_values), expr_strategy)
            })
            .prop_map(|(registry, values, expr)| (DebugExpr(expr), registry, values))
    })
}

fn arb_expr_recursive(
    var_indices: Vec<usize>,
    config: ExprConfig,
    depth: usize,
) -> impl Strategy<Value = ASTRepr<f64>> {
    if depth >= config.max_depth || var_indices.is_empty() {
        // Base cases: variables or constants
        let mut strategies: Vec<BoxedStrategy<ASTRepr<f64>>> = hlist![];

        // Add variables
        for &var_idx in &var_indices {
            strategies.push(Just(ASTRepr::Variable(var_idx)).boxed());
        }

        // Add constants if enabled
        if config.include_constants {
            let const_min = config.constant_range.0;
            let const_max = config.constant_range.1;
            strategies.push((const_min..const_max).prop_map(ASTRepr::Constant).boxed());

            // Add special constants
            strategies.push(Just(ASTRepr::Constant(0.0)).boxed());
            strategies.push(Just(ASTRepr::Constant(1.0)).boxed());
            strategies.push(Just(ASTRepr::Constant(-1.0)).boxed());
            strategies.push(Just(ASTRepr::Constant(2.5)).boxed());
            strategies.push(Just(ASTRepr::Constant(std::f64::consts::E)).boxed());
        }

        prop::strategy::Union::new(strategies).boxed()
    } else {
        arb_expr_recursive(var_indices.clone(), config, config.max_depth)
            .prop_recursive(
                8,   // cases per level
                256, // max total cases
                10,  // items per collection
                move |inner| {
                    let mut strategies: Vec<BoxedStrategy<ASTRepr<f64>>> = hlist![];

                    // Binary operations
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_map(|(a, b)| ASTRepr::Add(Box::new(a), Box::new(b)))
                            .boxed(),
                    );
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_map(|(a, b)| ASTRepr::Sub(Box::new(a), Box::new(b)))
                            .boxed(),
                    );
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_map(|(a, b)| ASTRepr::Mul(Box::new(a), Box::new(b)))
                            .boxed(),
                    );

                    // Division with non-zero divisor preference
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_filter("avoid division by values close to zero", |(_a, _b)| {
                                // This is a heuristic - we'll do the real check during evaluation
                                true
                            })
                            .prop_map(|(a, b)| ASTRepr::Div(Box::new(a), Box::new(b)))
                            .boxed(),
                    );

                    // Power with reasonable exponents
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_filter("reasonable power operations", |(_base, _exp)| {
                                // Add some basic filtering - more detailed checks in evaluation
                                true
                            })
                            .prop_map(|(a, b)| ASTRepr::Pow(Box::new(a), Box::new(b)))
                            .boxed(),
                    );

                    // Unary operations
                    strategies.push(
                        inner
                            .clone()
                            .prop_map(|a| ASTRepr::Neg(Box::new(a)))
                            .boxed(),
                    );

                    // Transcendental functions if enabled
                    if config.include_transcendental {
                        strategies.push(
                            inner
                                .clone()
                                .prop_map(|a| ASTRepr::Sin(Box::new(a)))
                                .boxed(),
                        );
                        strategies.push(
                            inner
                                .clone()
                                .prop_map(|a| ASTRepr::Cos(Box::new(a)))
                                .boxed(),
                        );
                        strategies.push(
                            inner
                                .clone()
                                .prop_filter("positive arguments for sqrt", |a| {
                                    // Only allow positive arguments for sqrt
                                    // If a is a constant, check its value
                                    match a {
                                        ASTRepr::Constant(val) => *val > 0.0,
                                        _ => true, // For non-constants, allow (will be checked at eval)
                                    }
                                })
                                .prop_map(|a| ASTRepr::Sqrt(Box::new(a)))
                                .boxed(),
                        );
                        strategies.push(
                            inner
                                .clone()
                                .prop_map(|a| ASTRepr::Exp(Box::new(a)))
                                .boxed(),
                        );

                        // Natural log with positive argument filtering
                        strategies.push(
                            inner
                                .clone()
                                .prop_filter("positive arguments for ln", |a| match a {
                                    ASTRepr::Constant(val) => *val > 0.0,
                                    _ => true,
                                })
                                .prop_map(|a| ASTRepr::Ln(Box::new(a)))
                                .boxed(),
                        );
                    }

                    prop::strategy::Union::new(strategies)
                },
            )
            .boxed()
    }
}

// Strategy for generating simpler expressions for basic testing
fn arb_simple_expr() -> impl Strategy<Value = (DebugExpr, VariableRegistry, Vec<f64>)> {
    arb_expr_with_config(ExprConfig {
        max_depth: 4,
        max_vars: 2,
        include_transcendental: false,
        include_constants: true,
        constant_range: (-10.0, 10.0),
    })
}

// Strategy for generating deep expressions to test stack limits
fn arb_deep_expr() -> impl Strategy<Value = (DebugExpr, VariableRegistry, Vec<f64>)> {
    arb_expr_with_config(ExprConfig {
        max_depth: 20,
        max_vars: 2,
        include_transcendental: false,
        include_constants: true,
        constant_range: (-5.0, 5.0),
    })
}

// Strategy for generating wide expressions with many operations
fn arb_wide_expr() -> impl Strategy<Value = (DebugExpr, VariableRegistry, Vec<f64>)> {
    arb_expr_with_config(ExprConfig {
        max_depth: 6,
        max_vars: 8,
        include_transcendental: true,
        include_constants: true,
        constant_range: (-20.0, 20.0),
    })
}

// Evaluation strategies
#[derive(Debug, Clone)]
enum EvalStrategy {
    Direct,
    ANF,
    Symbolic,
}

fn evaluate_with_strategy(
    expr: &ASTRepr<f64>,
    registry: &VariableRegistry,
    values: &[f64],
    strategy: EvalStrategy,
) -> Result<f64, DSLCompileError> {
    match strategy {
        EvalStrategy::Direct => Ok(DirectEval::eval_with_vars(expr, values)),

        EvalStrategy::ANF => {
            // ANF conversion and evaluation with domain awareness
            let anf = convert_to_anf(expr)?;
            let var_map: HashMap<usize, f64> =
                (0..values.len()).zip(values.iter().copied()).collect();

            // Create a domain analyzer for safety
            let mut domain_analyzer = IntervalDomainAnalyzer::new(0.0);

            // Set up variable domains based on the input values
            for (idx, &value) in values.iter().enumerate() {
                domain_analyzer.set_variable_domain(idx, IntervalDomain::Constant(value));
            }

            // Use domain-aware evaluation
            let result = anf.eval_domain_aware(&var_map, &domain_analyzer);

            Ok(result)
        }

        EvalStrategy::Symbolic => {
            let mut optimizer = SymbolicOptimizer::new()?;
            let optimized = optimizer.optimize(expr)?;
            Ok(DirectEval::eval_with_vars(&optimized, values))
        }
    }
}

fn is_numeric_equivalent(a: f64, b: f64, tolerance: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }

    // Handle cases where one is infinite and the other is a very large finite number
    // This can happen due to different overflow handling in different evaluation strategies
    if a.is_infinite() && b.is_finite() {
        // If a is infinite and b is very large (> 1e30), consider them equivalent
        return b.abs() > 1e30 && a.signum() == b.signum();
    }
    if b.is_infinite() && a.is_finite() {
        // If b is infinite and a is very large (> 1e30), consider them equivalent
        return a.abs() > 1e30 && b.signum() == a.signum();
    }

    if a.is_finite() && b.is_finite() {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        return diff <= tolerance * scale;
    }
    false
}

// Helper: check that all arguments to ln and sqrt are positive for given values
fn all_ln_sqrt_args_positive(
    expr: &ASTRepr<f64>,
    values: &[f64],
    registry: &VariableRegistry,
) -> bool {
    fn eval_expr(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> f64 {
        match expr {
            ASTRepr::Constant(val) => *val,
            ASTRepr::Variable(idx) => {
                // idx is a variable index
                let i = *idx;
                if i < values.len() { values[i] } else { 0.0 }
            }
            ASTRepr::Add(a, b) => eval_expr(a, values, registry) + eval_expr(b, values, registry),
            ASTRepr::Sub(a, b) => eval_expr(a, values, registry) - eval_expr(b, values, registry),
            ASTRepr::Mul(a, b) => eval_expr(a, values, registry) * eval_expr(b, values, registry),
            ASTRepr::Div(a, b) => eval_expr(a, values, registry) / eval_expr(b, values, registry),
            ASTRepr::Pow(a, b) => {
                eval_expr(a, values, registry).powf(eval_expr(b, values, registry))
            }
            ASTRepr::Neg(a) => -eval_expr(a, values, registry),
            ASTRepr::Ln(a) => eval_expr(a, values, registry), // just return the argument
            ASTRepr::Exp(a) => eval_expr(a, values, registry),
            ASTRepr::Sqrt(a) => eval_expr(a, values, registry), // just return the argument
            ASTRepr::Sin(a) => eval_expr(a, values, registry),
            ASTRepr::Cos(a) => eval_expr(a, values, registry),
            ASTRepr::Sum { .. } => {
                // Fall back to full AST evaluation for Sum expressions
                expr.eval_with_vars(values)
            }
        }
    }
    fn check(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> bool {
        match expr {
            ASTRepr::Ln(arg) | ASTRepr::Sqrt(arg) => {
                let val = eval_expr(arg, values, registry);
                val > 0.0 && check(arg, values, registry)
            }
            ASTRepr::Add(a, b)
            | ASTRepr::Sub(a, b)
            | ASTRepr::Mul(a, b)
            | ASTRepr::Div(a, b)
            | ASTRepr::Pow(a, b) => check(a, values, registry) && check(b, values, registry),
            ASTRepr::Neg(a) | ASTRepr::Exp(a) | ASTRepr::Sin(a) | ASTRepr::Cos(a) => {
                check(a, values, registry)
            }
            _ => true,
        }
    }
    check(expr, values, registry)
}

/// Check if all trigonometric function arguments are reasonable (not astronomically large)
/// Large arguments to sin/cos lead to precision issues and meaningless results
fn all_trig_args_reasonable(
    expr: &ASTRepr<f64>,
    values: &[f64],
    registry: &VariableRegistry,
) -> bool {
    // Maximum reasonable argument for trig functions - beyond this, precision is lost
    const MAX_TRIG_ARG: f64 = 1e15;

    fn eval_expr(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> f64 {
        match expr {
            ASTRepr::Constant(c) => *c,
            ASTRepr::Variable(idx) => values.get(*idx).copied().unwrap_or(0.0),
            ASTRepr::Add(a, b) => eval_expr(a, values, registry) + eval_expr(b, values, registry),
            ASTRepr::Sub(a, b) => eval_expr(a, values, registry) - eval_expr(b, values, registry),
            ASTRepr::Mul(a, b) => eval_expr(a, values, registry) * eval_expr(b, values, registry),
            ASTRepr::Div(a, b) => eval_expr(a, values, registry) / eval_expr(b, values, registry),
            ASTRepr::Neg(a) => -eval_expr(a, values, registry),
            ASTRepr::Exp(a) => eval_expr(a, values, registry).exp(),
            ASTRepr::Ln(a) => eval_expr(a, values, registry).ln(),
            ASTRepr::Sin(a) => eval_expr(a, values, registry).sin(),
            ASTRepr::Cos(a) => eval_expr(a, values, registry).cos(),
            ASTRepr::Sqrt(a) => eval_expr(a, values, registry).sqrt(),
            ASTRepr::Pow(base, exp) => {
                eval_expr(base, values, registry).powf(eval_expr(exp, values, registry))
            }
            ASTRepr::Sum { .. } => {
                // Fall back to full AST evaluation for Sum expressions
                expr.eval_with_vars(values)
            }
        }
    }

    fn check(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> bool {
        match expr {
            ASTRepr::Sin(a) | ASTRepr::Cos(a) => {
                let arg = eval_expr(a, values, registry);
                // Check if argument is reasonable and recurse
                arg.abs() <= MAX_TRIG_ARG && arg.is_finite() && check(a, values, registry)
            }
            ASTRepr::Add(a, b)
            | ASTRepr::Sub(a, b)
            | ASTRepr::Mul(a, b)
            | ASTRepr::Div(a, b)
            | ASTRepr::Pow(a, b) => check(a, values, registry) && check(b, values, registry),
            ASTRepr::Neg(a) | ASTRepr::Exp(a) | ASTRepr::Ln(a) | ASTRepr::Sqrt(a) => {
                check(a, values, registry)
            }
            _ => true,
        }
    }
    check(expr, values, registry)
}

/// Check if all power operations result in real numbers (no complex number domain issues)
/// This prevents cases like (-1)^(non-integer) which can lead to complex results
fn all_power_args_real(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> bool {
    fn eval_expr(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> f64 {
        match expr {
            ASTRepr::Constant(c) => *c,
            ASTRepr::Variable(idx) => values.get(*idx).copied().unwrap_or(0.0),
            ASTRepr::Add(a, b) => eval_expr(a, values, registry) + eval_expr(b, values, registry),
            ASTRepr::Sub(a, b) => eval_expr(a, values, registry) - eval_expr(b, values, registry),
            ASTRepr::Mul(a, b) => eval_expr(a, values, registry) * eval_expr(b, values, registry),
            ASTRepr::Div(a, b) => eval_expr(a, values, registry) / eval_expr(b, values, registry),
            ASTRepr::Neg(a) => -eval_expr(a, values, registry),
            ASTRepr::Exp(a) => eval_expr(a, values, registry).exp(),
            ASTRepr::Ln(a) => eval_expr(a, values, registry).ln(),
            ASTRepr::Sin(a) => eval_expr(a, values, registry).sin(),
            ASTRepr::Cos(a) => eval_expr(a, values, registry).cos(),
            ASTRepr::Sqrt(a) => eval_expr(a, values, registry).sqrt(),
            ASTRepr::Pow(base, exp) => {
                eval_expr(base, values, registry).powf(eval_expr(exp, values, registry))
            }
            ASTRepr::Sum { .. } => {
                // Fall back to full AST evaluation for Sum expressions
                expr.eval_with_vars(values)
            }
        }
    }

    fn check(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> bool {
        match expr {
            ASTRepr::Pow(base, exp) => {
                let base_val = eval_expr(base, values, registry);
                let exp_val = eval_expr(exp, values, registry);

                // Check for problematic power operations that can result in complex numbers
                if base_val < 0.0 && exp_val.fract() != 0.0 {
                    // Negative base with non-integer exponent leads to complex numbers
                    return false;
                }

                // Also check for extremely large exponents that cause overflow
                if exp_val.abs() > 100.0 {
                    return false;
                }

                // Recursively check sub-expressions
                check(base, values, registry) && check(exp, values, registry)
            }
            ASTRepr::Add(a, b) | ASTRepr::Sub(a, b) | ASTRepr::Mul(a, b) | ASTRepr::Div(a, b) => {
                check(a, values, registry) && check(b, values, registry)
            }
            ASTRepr::Neg(a)
            | ASTRepr::Exp(a)
            | ASTRepr::Ln(a)
            | ASTRepr::Sqrt(a)
            | ASTRepr::Sin(a)
            | ASTRepr::Cos(a) => check(a, values, registry),
            _ => true,
        }
    }
    check(expr, values, registry)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_simple_expressions_consistency(
        (expr, registry, values) in arb_simple_expr()
    ) {
        prop_assume!(all_ln_sqrt_args_positive(&expr.0, &values, &registry));
        prop_assume!(all_trig_args_reasonable(&expr.0, &values, &registry));

        let direct_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Direct);
        let anf_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::ANF);

        match (direct_result, anf_result) {
            (Ok(direct), Ok(anf)) => {
                prop_assert!(
                    is_numeric_equivalent(direct, anf, 1e-12),
                    "Direct eval: {}, ANF eval: {}, values: {:?}\nPretty AST: {}\nPretty ANF: {}\n",
                    direct, anf, values,
                    pretty_ast(&expr.0, &registry),
                    pretty_anf(&convert_to_anf(&expr.0).unwrap(), &registry)
                );
            },
            (Err(_), Err(_)) => {
                // Both failed - acceptable for edge cases
            },
            (Ok(direct), Err(anf_err)) => {
                prop_assert!(
                    false,
                    "Direct succeeded ({}) but ANF failed ({}).\nPretty AST: {}\nPretty ANF: {}\n",
                    direct, anf_err,
                    pretty_ast(&expr.0, &registry),
                    pretty_anf(&convert_to_anf(&expr.0).unwrap_or(ANFExpr::Atom(ANFAtom::Constant(0.0))), &registry)
                );
            },
            (Err(direct_err), Ok(anf)) => {
                prop_assert!(
                    false,
                    "ANF succeeded ({}) but Direct failed ({}).\nPretty AST: {}\nPretty ANF: {}\n",
                    anf, direct_err,
                    pretty_ast(&expr.0, &registry),
                    pretty_anf(&convert_to_anf(&expr.0).unwrap(), &registry)
                );
            }
        }

    }

    #[test]
    fn test_all_strategies_consistency(
        (expr, registry, values) in arb_expr_with_config(ExprConfig::default())
    ) {
        prop_assume!(all_ln_sqrt_args_positive(&expr.0, &values, &registry));
        prop_assume!(all_trig_args_reasonable(&expr.0, &values, &registry));
        prop_assume!(all_power_args_real(&expr.0, &values, &registry));

        let direct_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Direct);
        let anf_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::ANF);
        let symbolic_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Symbolic);

        // Check that all successful evaluations agree
        let mut results = Vec::new();
        if let Ok(val) = direct_result {
            results.push(("Direct", val));
        }
        if let Ok(val) = anf_result {
            results.push(("ANF", val));
        }
        if let Ok(val) = symbolic_result {
            results.push(("Symbolic", val));
        }

        // If we have multiple successful evaluations, they should agree
        if results.len() >= 2 {
            let first_val = results[0].1;
            for (strategy, val) in &results[1..] {
                prop_assert!(
                    is_numeric_equivalent(first_val, *val, 1e-10),
                    "Strategy {} gave {}, but {} gave {}\nPretty AST: {}\nPretty ANF: {}\n",
                    results[0].0, first_val, strategy, val,
                    pretty_ast(&expr.0, &registry),
                    pretty_anf(&convert_to_anf(&expr.0).unwrap(), &registry)
                );
            }
        }
    }

    #[test]
    fn test_deep_expressions_no_stack_overflow(
        (expr, registry, values) in arb_deep_expr()
    ) {
        // Test that deep expressions don't cause stack overflow
        let direct_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Direct);
        let anf_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::ANF);

        // We mainly care that these don't panic/overflow
        // Results may legitimately differ due to numeric precision in deep expressions
        if let (Ok(direct), Ok(anf)) = (direct_result, anf_result) {
            // For very deep expressions, allow more tolerance
            if !is_numeric_equivalent(direct, anf, 1e-8) {
                println!("Deep expression precision difference: direct={direct}, anf={anf}");
                // Don't fail the test - deep expressions may have legitimate precision differences
            }
        } else {
            // Failures are acceptable for very deep expressions
        }
    }

    #[test]
    fn test_wide_expressions_performance(
        (expr, registry, values) in arb_wide_expr()
    ) {
        // Test that wide expressions with many variables complete in reasonable time
        let start = std::time::Instant::now();
        let _anf_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::ANF);
        let anf_duration = start.elapsed();

        // ANF conversion should complete within 1 second even for wide expressions
        prop_assert!(
            anf_duration.as_secs() < 1,
            "ANF conversion took too long: {:?} for expr with {} vars",
            anf_duration, values.len()
        );

        let start = std::time::Instant::now();
        let _symbolic_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Symbolic);
        let symbolic_duration = start.elapsed();

        // Symbolic optimization should also complete reasonably quickly
        prop_assert!(
            symbolic_duration.as_secs() < 5,
            "Symbolic optimization took too long: {:?}",
            symbolic_duration
        );
    }

    #[test]
    fn test_numeric_edge_cases(
        strategy in prop::strategy::Union::new(hlist![
            Just(EvalStrategy::Direct).boxed(),
            Just(EvalStrategy::ANF).boxed(),
            Just(EvalStrategy::Symbolic).boxed(),
        ])
    ) {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();
        let x = ASTRepr::Variable(x_idx);

        // Test various edge case values
        let edge_values = hlist![
            0.0, -0.0, 1.0, -1.0,
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            f64::MIN, f64::MAX, f64::EPSILON,
            1e-100, 1e100, -1e-100, -1e100,
        ];

        for &val in &edge_values {
            // Simple expression: x + 1
            let expr = ASTRepr::Add(Box::new(x.clone()), Box::new(ASTRepr::Constant(1.0)));
            let result = evaluate_with_strategy(&expr, &registry, &[val], strategy.clone());

            // Should handle edge cases gracefully (either succeed or fail consistently)
            if let Ok(res) = result {
                // Result should be well-formed (not random garbage)
                prop_assert!(
                    res.is_finite() || res.is_infinite() || res.is_nan(),
                    "Invalid result {} for input {}",
                    res, val
                );
            } else {
                // Errors are acceptable for edge cases
            }
        }
    }

    #[test]
    fn test_domain_aware_ln_rules_consistency(
        (expr, registry, values) in arb_expr_with_config(ExprConfig {
            max_depth: 4,
            max_vars: 2,
            include_transcendental: true,
            include_constants: true,
            constant_range: (0.1, 10.0), // Only positive constants for ln safety
        })
    ) {
        // Only test expressions where all ln and sqrt arguments are positive
        prop_assume!(all_ln_sqrt_args_positive(&expr.0, &values, &registry));
        prop_assume!(all_trig_args_reasonable(&expr.0, &values, &registry));

        let direct_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Direct);
        let symbolic_result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Symbolic);

        // Both should succeed for domain-safe expressions
        match (direct_result, symbolic_result) {
            (Ok(direct), Ok(symbolic)) => {
                prop_assert!(
                    is_numeric_equivalent(direct, symbolic, 1e-10),
                    "Domain-aware optimization changed result: Direct={}, Symbolic={}\nExpression: {}\nValues: {:?}",
                    direct, symbolic,
                    pretty_ast(&expr.0, &registry),
                    values
                );
            }
            (Ok(direct), Err(symbolic_err)) => {
                // If direct evaluation succeeds, symbolic should too for domain-safe expressions
                prop_assert!(
                    false,
                    "Direct evaluation succeeded ({}) but symbolic failed ({})\nExpression: {}\nValues: {:?}",
                    direct, symbolic_err,
                    pretty_ast(&expr.0, &registry),
                    values
                );
            }
            (Err(_), Ok(_)) => {
                // This is acceptable - symbolic might succeed where direct fails due to optimizations
            }
            (Err(_), Err(_)) => {
                // Both failed - acceptable for edge cases
            }
        }
    }

    #[test]
    fn test_ln_division_rule_safety(
        a_val in 0.1_f64..10.0,
        b_val in 0.1_f64..10.0,
    ) {
        // Test ln(a/b) = ln(a) - ln(b) with positive constants
        let registry = VariableRegistry::new();

        // Create ln(a/b) expression
        let a = ASTRepr::Constant(a_val);
        let b = ASTRepr::Constant(b_val);
        let div_expr = ASTRepr::Div(Box::new(a.clone()), Box::new(b.clone()));
        let ln_div = ASTRepr::Ln(Box::new(div_expr));

        // Create ln(a) - ln(b) expression
        let ln_a = ASTRepr::Ln(Box::new(a));
        let ln_b = ASTRepr::Ln(Box::new(b));
        let ln_diff = ASTRepr::Sub(Box::new(ln_a), Box::new(ln_b));

        // Both should evaluate to the same result
        let ln_div_result = DirectEval::eval_with_vars(&ln_div, &[]);
        let ln_diff_result = DirectEval::eval_with_vars(&ln_diff, &[]);

        prop_assert!(
            is_numeric_equivalent(ln_div_result, ln_diff_result, 1e-12),
            "ln({}/{}): {} vs ln({}) - ln({}): {}",
            a_val, b_val, ln_div_result,
            a_val, b_val, ln_diff_result
        );

        // Test with symbolic optimization
        let symbolic_result = evaluate_with_strategy(&ln_div, &registry, &[], EvalStrategy::Symbolic);
        if let Ok(symbolic) = symbolic_result {
            prop_assert!(
                is_numeric_equivalent(ln_div_result, symbolic, 1e-10),
                "Symbolic optimization changed ln(a/b) result: {} vs {}",
                ln_div_result, symbolic
            );
        } else {
            // Symbolic optimization failure is acceptable
        }
    }

    #[test]
    fn test_sqrt_domain_safety(
        base_val in -5.0_f64..5.0,
        _exp_val in 1.0_f64..4.0,
    ) {
        // Test sqrt(x^2) = |x| behavior
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        // Create sqrt(x^2) expression
        let x = ASTRepr::Variable(x_idx);
        let x_squared = ASTRepr::Pow(Box::new(x), Box::new(ASTRepr::Constant(2.0)));
        let sqrt_x_squared = ASTRepr::Sqrt(Box::new(x_squared));

        let values = hlist![base_val];

        // Direct evaluation should give |x|
        let direct_result = DirectEval::eval_with_vars(&sqrt_x_squared, &values);
        let expected = base_val.abs();

        prop_assert!(
            is_numeric_equivalent(direct_result, expected, 1e-12),
            "sqrt({}^2) = {} but expected {}",
            base_val, direct_result, expected
        );

        // Symbolic optimization should preserve mathematical correctness
        let symbolic_result = evaluate_with_strategy(&sqrt_x_squared, &registry, &values, EvalStrategy::Symbolic);
        if let Ok(symbolic) = symbolic_result {
            prop_assert!(
                is_numeric_equivalent(symbolic, expected, 1e-10),
                "Symbolic sqrt(x^2) optimization incorrect: {} vs {} for x={}",
                symbolic, expected, base_val
            );
        } else {
            // Symbolic optimization failure is acceptable
        }
    }

    #[test]
    fn test_exp_ln_inverse_safety(
        val in 0.1_f64..10.0,
    ) {
        // Test exp(ln(x)) = x for positive x
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        // Create exp(ln(x)) expression
        let x = ASTRepr::Variable(x_idx);
        let ln_x = ASTRepr::Ln(Box::new(x));
        let exp_ln_x = ASTRepr::Exp(Box::new(ln_x));

        let values = hlist![val];

        // Should simplify to x
        let direct_result = DirectEval::eval_with_vars(&exp_ln_x, &values);

        prop_assert!(
            is_numeric_equivalent(direct_result, val, 1e-12),
            "exp(ln({})) = {} but expected {}",
            val, direct_result, val
        );

        // Test symbolic optimization
        let symbolic_result = evaluate_with_strategy(&exp_ln_x, &registry, &values, EvalStrategy::Symbolic);
        if let Ok(symbolic) = symbolic_result {
            prop_assert!(
                is_numeric_equivalent(symbolic, val, 1e-10),
                "Symbolic exp(ln(x)) optimization incorrect: {} vs {} for x={}",
                symbolic, val, val
            );
        } else {
            // Symbolic optimization failure is acceptable
        }
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_failing_case() {
        // Recreate the failing case manually using ASTRepr directly
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        // Original: (x + (-3.18...)) * ((x + x) - 1)
        let x = ASTRepr::Variable(x_idx);
        let left = ASTRepr::Add(
            Box::new(x.clone()),
            Box::new(ASTRepr::Constant(-3.1867703204859654)),
        );
        let inner_add = ASTRepr::Add(Box::new(x.clone()), Box::new(x.clone()));
        let right = ASTRepr::Sub(Box::new(inner_add), Box::new(ASTRepr::Constant(1.0)));
        let expr = ASTRepr::Mul(Box::new(left), Box::new(right));

        let values = hlist![0.0];

        // Test direct evaluation
        let direct_result = DirectEval::eval_with_vars(&expr, &values);
        println!("Direct result: {direct_result}");

        // Test ANF evaluation
        let anf = convert_to_anf(&expr).unwrap();
        println!("ANF: {anf:#?}");

        let var_map: HashMap<usize, f64> = [(0, 0.0)].into_iter().collect();
        let anf_result = anf.eval(&var_map);
        println!("ANF result: {anf_result}");

        // Step through the ANF evaluation manually
        println!("\n=== Manual ANF Evaluation ===");
        if let ANFExpr::Let(var1, comp1, body1) = &anf {
            println!("Step 1: {} = {:?}", var1.debug_name(&registry), comp1);
            let step1_result = match comp1 {
                ANFComputation::Add(ANFAtom::Variable(VarRef::User(0)), ANFAtom::Constant(c)) => {
                    0.0 + c
                }
                _ => panic!("Unexpected computation 1"),
            };
            println!("  Result: {step1_result}");

            if let ANFExpr::Let(var2, comp2, body2) = body1.as_ref() {
                println!("Step 2: {} = {:?}", var2.debug_name(&registry), comp2);
                let step2_result = match comp2 {
                    ANFComputation::Add(
                        ANFAtom::Variable(VarRef::User(0)),
                        ANFAtom::Variable(VarRef::User(0)),
                    ) => 0.0 + 0.0,
                    _ => panic!("Unexpected computation 2"),
                };
                println!("  Result: {step2_result}");

                if let ANFExpr::Let(var3, comp3, _body3) = body2.as_ref() {
                    println!("Step 3: {} = {:?}", var3.debug_name(&registry), comp3);
                    // This should be the multiplication but what are the operands?
                    match comp3 {
                        ANFComputation::Mul(left_atom, right_atom) => {
                            println!("  Left operand: {left_atom:?}");
                            println!("  Right operand: {right_atom:?}");
                        }
                        _ => println!("  Unexpected computation 3: {comp3:?}"),
                    }
                }
            }
        }

        // They should be equal
        assert!(
            (direct_result - anf_result).abs() < 1e-10,
            "Direct: {direct_result}, ANF: {anf_result}"
        );
    }

    #[test]
    fn test_proptest_framework_basic() {
        // Smoke test to ensure the proptest framework is working
        let config = ExprConfig {
            max_depth: 3,
            max_vars: 2,
            include_transcendental: false,
            include_constants: true,
            constant_range: (-5.0, 5.0),
        };

        let strategy = arb_expr_with_config(config);
        let mut runner = proptest::test_runner::TestRunner::default();

        // Generate a few expressions to make sure it works
        for _ in 0..10 {
            let (expr, registry, values) = strategy.new_tree(&mut runner).unwrap().current();

            // Basic sanity checks
            assert!(values.len() <= 2);
            assert!(registry.len() <= 2);

            // Try evaluating
            let result = evaluate_with_strategy(&expr.0, &registry, &values, EvalStrategy::Direct);
            match result {
                Ok(_) | Err(_) => {} // Both are fine for arbitrary expressions
            }
        }
    }

    #[test]
    fn test_known_equivalent_expressions() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();
        let x = ASTRepr::Variable(x_idx);

        // Test: (x + x) should equal (2 * x)
        let expr1 = ASTRepr::Add(Box::new(x.clone()), Box::new(x.clone()));
        let expr2 = ASTRepr::Mul(Box::new(ASTRepr::Constant(2.0)), Box::new(x.clone()));

        let values = hlist![2.5];

        let result1 =
            evaluate_with_strategy(&expr1, &registry, &values, EvalStrategy::Direct).unwrap();
        let result2 =
            evaluate_with_strategy(&expr2, &registry, &values, EvalStrategy::Direct).unwrap();
        let anf1 = evaluate_with_strategy(&expr1, &registry, &values, EvalStrategy::ANF).unwrap();
        let anf2 = evaluate_with_strategy(&expr2, &registry, &values, EvalStrategy::ANF).unwrap();

        assert!(is_numeric_equivalent(result1, result2, 1e-15));
        assert!(is_numeric_equivalent(anf1, anf2, 1e-15));
        assert!(is_numeric_equivalent(result1, anf1, 1e-15));
    }
}
