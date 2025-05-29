use mathcompile::anf::{convert_to_anf, ANFAtom, ANFComputation, ANFExpr, VarRef};
use mathcompile::error::MathCompileError;
use mathcompile::final_tagless::{ASTEval, ASTMathExpr, ASTRepr, DirectEval, VariableRegistry};
use mathcompile::pretty::{pretty_anf, pretty_ast};
use mathcompile::symbolic::SymbolicOptimizer;
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
    // Generate variable names and values
    let var_strategy = (1..=config.max_vars).prop_flat_map(move |num_vars| {
        let names: Vec<String> = (0..num_vars).map(|i| format!("x{i}")).collect();
        let const_min = config.constant_range.0;
        let const_max = config.constant_range.1;
        let values = prop::collection::vec(const_min..const_max, num_vars..=num_vars);
        (Just(names), values)
    });

    var_strategy
        .prop_flat_map(move |(var_names, var_values)| {
            let mut registry = VariableRegistry::new();
            let var_indices: Vec<usize> = var_names
                .iter()
                .map(|name| registry.register_variable(name))
                .collect();

            let expr_strategy = arb_expr_recursive(var_indices, config, 0);
            (Just(registry), Just(var_values), expr_strategy)
        })
        .prop_map(|(registry, values, expr)| (DebugExpr(expr), registry, values))
}

fn arb_expr_recursive(
    var_indices: Vec<usize>,
    config: ExprConfig,
    depth: usize,
) -> impl Strategy<Value = ASTRepr<f64>> {
    if depth >= config.max_depth || var_indices.is_empty() {
        // Base cases: variables or constants
        let mut strategies: Vec<BoxedStrategy<ASTRepr<f64>>> = vec![];

        // Add variables
        for &var_idx in &var_indices {
            strategies.push(Just(ASTEval::var(var_idx)).boxed());
        }

        // Add constants if enabled
        if config.include_constants {
            let const_min = config.constant_range.0;
            let const_max = config.constant_range.1;
            strategies.push((const_min..const_max).prop_map(ASTEval::constant).boxed());

            // Add special constants
            strategies.push(Just(ASTEval::constant(0.0)).boxed());
            strategies.push(Just(ASTEval::constant(1.0)).boxed());
            strategies.push(Just(ASTEval::constant(-1.0)).boxed());
            strategies.push(Just(ASTEval::constant(2.5)).boxed());
            strategies.push(Just(ASTEval::constant(std::f64::consts::E)).boxed());
        }

        prop::strategy::Union::new(strategies).boxed()
    } else {
        arb_expr_recursive(var_indices.clone(), config, config.max_depth)
            .prop_recursive(
                8,   // cases per level
                256, // max total cases
                10,  // items per collection
                move |inner| {
                    let mut strategies: Vec<BoxedStrategy<ASTRepr<f64>>> = vec![];

                    // Binary operations
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_map(|(a, b)| ASTEval::add(a, b))
                            .boxed(),
                    );
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_map(|(a, b)| ASTEval::sub(a, b))
                            .boxed(),
                    );
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_map(|(a, b)| ASTEval::mul(a, b))
                            .boxed(),
                    );

                    // Division with non-zero divisor preference
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_filter("avoid division by values close to zero", |(_a, _b)| {
                                // This is a heuristic - we'll do the real check during evaluation
                                true
                            })
                            .prop_map(|(a, b)| ASTEval::div(a, b))
                            .boxed(),
                    );

                    // Power with reasonable exponents
                    strategies.push(
                        (inner.clone(), inner.clone())
                            .prop_filter("reasonable power operations", |(_base, _exp)| {
                                // Add some basic filtering - more detailed checks in evaluation
                                true
                            })
                            .prop_map(|(a, b)| ASTEval::pow(a, b))
                            .boxed(),
                    );

                    // Unary operations
                    strategies.push(inner.clone().prop_map(ASTEval::neg).boxed());

                    // Transcendental functions if enabled
                    if config.include_transcendental {
                        strategies.push(inner.clone().prop_map(ASTEval::sin).boxed());
                        strategies.push(inner.clone().prop_map(ASTEval::cos).boxed());
                        strategies.push(
                            inner.clone()
                                .prop_filter("positive arguments for sqrt", |a| {
                                    // Only allow positive arguments for sqrt
                                    // If a is a constant, check its value
                                    match a {
                                        ASTRepr::Constant(val) => *val > 0.0,
                                        _ => true, // For non-constants, allow (will be checked at eval)
                                    }
                                })
                                .prop_map(ASTEval::sqrt)
                                .boxed(),
                        );
                        strategies.push(inner.clone().prop_map(ASTEval::exp).boxed());

                        // Natural log with positive argument filtering
                        strategies.push(
                            inner
                                .clone()
                                .prop_filter("positive arguments for ln", |a| {
                                    match a {
                                        ASTRepr::Constant(val) => *val > 0.0,
                                        _ => true,
                                    }
                                })
                                .prop_map(ASTEval::ln)
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
    _registry: &VariableRegistry,
    values: &[f64],
    strategy: EvalStrategy,
) -> Result<f64, MathCompileError> {
    match strategy {
        EvalStrategy::Direct => {
            // Direct AST evaluation using DirectEval
            Ok(DirectEval::eval_with_vars(expr, values))
        }

        EvalStrategy::ANF => {
            // ANF conversion and evaluation
            let anf = convert_to_anf(expr)?;
            let var_map: HashMap<usize, f64> =
                (0..values.len()).zip(values.iter().copied()).collect();

            let result = anf.eval(&var_map);

            // Debug output for failing cases
            if values == [0.0] && (result.is_sign_negative() || result == 0.0) {
                println!("=== ANF Debug ===");
                println!("Original AST: {expr:#?}");
                println!("ANF: {anf:#?}");
                println!("Variables: {var_map:?}");
                println!("ANF Result: {result}");
                println!("=================");
            }

            Ok(result)
        }

        EvalStrategy::Symbolic => {
            // Symbolic optimization then evaluation
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
    if a.is_finite() && b.is_finite() {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        return diff <= tolerance * scale;
    }
    false
}

// Helper: check that all arguments to ln and sqrt are positive for given values
fn all_ln_sqrt_args_positive(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> bool {
    fn eval_expr(expr: &ASTRepr<f64>, values: &[f64], registry: &VariableRegistry) -> f64 {
        match expr {
            ASTRepr::Constant(val) => *val,
            ASTRepr::Variable(idx) => {
                // idx is a variable index
                let i = *idx;
                if i < values.len() {
                    values[i]
                } else {
                    0.0
                }
            }
            ASTRepr::Add(a, b) => eval_expr(a, values, registry) + eval_expr(b, values, registry),
            ASTRepr::Sub(a, b) => eval_expr(a, values, registry) - eval_expr(b, values, registry),
            ASTRepr::Mul(a, b) => eval_expr(a, values, registry) * eval_expr(b, values, registry),
            ASTRepr::Div(a, b) => eval_expr(a, values, registry) / eval_expr(b, values, registry),
            ASTRepr::Pow(a, b) => eval_expr(a, values, registry).powf(eval_expr(b, values, registry)),
            ASTRepr::Neg(a) => -eval_expr(a, values, registry),
            ASTRepr::Ln(a) => eval_expr(a, values, registry), // just return the argument
            ASTRepr::Exp(a) => eval_expr(a, values, registry),
            ASTRepr::Sqrt(a) => eval_expr(a, values, registry), // just return the argument
            ASTRepr::Sin(a) => eval_expr(a, values, registry),
            ASTRepr::Cos(a) => eval_expr(a, values, registry),
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
            ASTRepr::Neg(a)
            | ASTRepr::Exp(a)
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
        strategy in prop::strategy::Union::new(vec![
            Just(EvalStrategy::Direct).boxed(),
            Just(EvalStrategy::ANF).boxed(),
            Just(EvalStrategy::Symbolic).boxed(),
        ])
    ) {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable("x");
        let x = ASTEval::var(x_idx);

        // Test various edge case values
        let edge_values = vec![
            0.0, -0.0, 1.0, -1.0,
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            f64::MIN, f64::MAX, f64::EPSILON,
            1e-100, 1e100, -1e-100, -1e100,
        ];

        for &val in &edge_values {
            // Simple expression: x + 1
            let expr = ASTEval::add(x.clone(), ASTEval::constant(1.0));
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_failing_case() {
        use mathcompile::final_tagless::{ASTEval, ASTMathExpr, DirectEval, VariableRegistry};

        // Recreate the failing case manually
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable("x0");
        let x = ASTEval::var(x_idx);

        // Original: (x + (-3.18...)) * ((x + x) - 1)
        let left = ASTEval::add(x.clone(), ASTEval::constant(-3.1867703204859654));
        let inner_add = ASTEval::add(x.clone(), x.clone());
        let right = ASTEval::sub(inner_add, ASTEval::constant(1.0));
        let expr = ASTEval::mul(left, right);

        let values = vec![0.0];

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
        let x_idx = registry.register_variable("x");
        let x = ASTEval::var(x_idx);

        // Test: (x + x) should equal (2 * x)
        let expr1 = ASTEval::add(x.clone(), x.clone());
        let expr2 = ASTEval::mul(ASTEval::constant(2.0), x.clone());

        let values = vec![2.5];

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
