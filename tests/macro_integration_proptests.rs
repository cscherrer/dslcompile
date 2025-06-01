#![cfg(feature = "proptest_macro_integration")]

// This test file is currently disabled because it's incompatible with our procedural macro design.
// The procedural macro expects literal expressions, not variables passed at runtime.
// TODO: Redesign these tests to work with the new procedural macro approach.

use mathcompile::compile_time::{constant, optimize_compile_time, var};
use mathcompile::final_tagless::{ASTEval, DirectEval};
use proptest::prelude::*;

/// Test the actual procedural macro by generating expressions and comparing with reference
/// This tests the full end-to-end pipeline: syntax → macro → optimization → code generation

/// Strategy for generating macro-compatible expressions
#[derive(Debug, Clone)]
enum MacroExpr {
    Var(usize),
    Constant(f64),
    Add(Box<MacroExpr>, Box<MacroExpr>),
    Mul(Box<MacroExpr>, Box<MacroExpr>),
    Sub(Box<MacroExpr>, Box<MacroExpr>),
    Sin(Box<MacroExpr>),
    Cos(Box<MacroExpr>),
    Exp(Box<MacroExpr>),
    Ln(Box<MacroExpr>),
}

impl MacroExpr {
    /// Evaluate using reference implementation (DirectEval)
    fn eval_reference(&self, values: &[f64]) -> f64 {
        match self {
            MacroExpr::Var(idx) => values.get(*idx).copied().unwrap_or(0.0),
            MacroExpr::Constant(c) => *c,
            MacroExpr::Add(left, right) => {
                left.eval_reference(values) + right.eval_reference(values)
            }
            MacroExpr::Mul(left, right) => {
                left.eval_reference(values) * right.eval_reference(values)
            }
            MacroExpr::Sub(left, right) => {
                left.eval_reference(values) - right.eval_reference(values)
            }
            MacroExpr::Sin(inner) => inner.eval_reference(values).sin(),
            MacroExpr::Cos(inner) => inner.eval_reference(values).cos(),
            MacroExpr::Exp(inner) => inner.eval_reference(values).exp(),
            MacroExpr::Ln(inner) => inner.eval_reference(values).ln(),
        }
    }

    /// Check if expression has valid arguments (no negative ln, etc.)
    fn has_valid_args(&self, values: &[f64]) -> bool {
        match self {
            MacroExpr::Var(_) | MacroExpr::Constant(_) => true,
            MacroExpr::Add(l, r) | MacroExpr::Mul(l, r) | MacroExpr::Sub(l, r) => {
                l.has_valid_args(values) && r.has_valid_args(values)
            }
            MacroExpr::Sin(inner) | MacroExpr::Cos(inner) | MacroExpr::Exp(inner) => {
                inner.has_valid_args(values)
            }
            MacroExpr::Ln(inner) => {
                inner.has_valid_args(values) && inner.eval_reference(values) > 0.0
            }
        }
    }
}

/// Strategy for generating macro expressions
fn arb_macro_expr(max_depth: usize, var_count: usize) -> impl Strategy<Value = MacroExpr> {
    let leaf = prop_oneof![
        (0..var_count).prop_map(MacroExpr::Var),
        (-10.0_f64..10.0_f64).prop_map(MacroExpr::Constant),
        Just(MacroExpr::Constant(0.0)),
        Just(MacroExpr::Constant(1.0)),
        Just(MacroExpr::Constant(-1.0)),
    ];

    leaf.prop_recursive(max_depth, 32, 4, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| MacroExpr::Add(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| MacroExpr::Mul(Box::new(l), Box::new(r))),
            (inner.clone(), inner.clone())
                .prop_map(|(l, r)| MacroExpr::Sub(Box::new(l), Box::new(r))),
            inner.clone().prop_map(|e| MacroExpr::Sin(Box::new(e))),
            inner.clone().prop_map(|e| MacroExpr::Cos(Box::new(e))),
            inner.clone().prop_map(|e| MacroExpr::Exp(Box::new(e))),
            // Only allow ln for positive expressions
            inner
                .clone()
                .prop_filter("positive for ln", |e| match e {
                    MacroExpr::Constant(c) => *c > 0.0,
                    MacroExpr::Exp(_) => true, // exp is always positive
                    _ => true,                 // Will be checked at runtime
                })
                .prop_map(|e| MacroExpr::Ln(Box::new(e))),
        ]
    })
}

/// Test specific patterns that should optimize well
fn test_optimization_patterns() -> Vec<(MacroExpr, &'static str)> {
    vec![
        // ln(exp(x)) → x
        (
            MacroExpr::Ln(Box::new(MacroExpr::Exp(Box::new(MacroExpr::Var(0))))),
            "ln(exp(x))",
        ),
        // exp(ln(x)) → x (for positive x)
        (
            MacroExpr::Exp(Box::new(MacroExpr::Ln(Box::new(MacroExpr::Var(0))))),
            "exp(ln(x))",
        ),
        // x + 0 → x
        (
            MacroExpr::Add(
                Box::new(MacroExpr::Var(0)),
                Box::new(MacroExpr::Constant(0.0)),
            ),
            "x + 0",
        ),
        // 0 + x → x
        (
            MacroExpr::Add(
                Box::new(MacroExpr::Constant(0.0)),
                Box::new(MacroExpr::Var(0)),
            ),
            "0 + x",
        ),
        // x * 1 → x
        (
            MacroExpr::Mul(
                Box::new(MacroExpr::Var(0)),
                Box::new(MacroExpr::Constant(1.0)),
            ),
            "x * 1",
        ),
        // 1 * x → x
        (
            MacroExpr::Mul(
                Box::new(MacroExpr::Constant(1.0)),
                Box::new(MacroExpr::Var(0)),
            ),
            "1 * x",
        ),
        // x * 0 → 0
        (
            MacroExpr::Mul(
                Box::new(MacroExpr::Var(0)),
                Box::new(MacroExpr::Constant(0.0)),
            ),
            "x * 0",
        ),
        // 0 * x → 0
        (
            MacroExpr::Mul(
                Box::new(MacroExpr::Constant(0.0)),
                Box::new(MacroExpr::Var(0)),
            ),
            "0 * x",
        ),
    ]
}

/// Macro to test 1-variable expressions
macro_rules! test_1var_expr {
    ($expr:expr, $x:expr) => {{
        let x = $x;
        optimize_compile_time!($expr, [x])
    }};
}

/// Macro to test 2-variable expressions
macro_rules! test_2var_expr {
    ($expr:expr, $x:expr, $y:expr) => {{
        let x = $x;
        let y = $y;
        optimize_compile_time!($expr, [x, y])
    }};
}

/// Macro to test 3-variable expressions
macro_rules! test_3var_expr {
    ($expr:expr, $x:expr, $y:expr, $z:expr) => {{
        let x = $x;
        let y = $y;
        let z = $z;
        optimize_compile_time!($expr, [x, y, z])
    }};
}

/// Test macro expressions by evaluating them and comparing with reference
fn test_macro_expression(expr: &MacroExpr, values: &[f64]) -> Result<(), String> {
    // Skip expressions with invalid arguments
    if !expr.has_valid_args(values) {
        return Ok(());
    }

    let reference_result = expr.eval_reference(values);

    // Test based on variable count and expression structure
    let macro_result = match (expr, values.len()) {
        (_, 1) => {
            let x = values[0];
            match expr {
                MacroExpr::Var(0) => test_1var_expr!(var::<0>(), x),
                MacroExpr::Constant(c) => test_1var_expr!(constant(*c), x),
                MacroExpr::Add(l, r)
                    if matches!(
                        (l.as_ref(), r.as_ref()),
                        (MacroExpr::Var(0), MacroExpr::Constant(c))
                    ) =>
                {
                    if let MacroExpr::Constant(c) = r.as_ref() {
                        test_1var_expr!(var::<0>().add(constant(*c)), x)
                    } else {
                        return Ok(());
                    }
                }
                MacroExpr::Add(l, r)
                    if matches!(
                        (l.as_ref(), r.as_ref()),
                        (MacroExpr::Constant(c), MacroExpr::Var(0))
                    ) =>
                {
                    if let MacroExpr::Constant(c) = l.as_ref() {
                        test_1var_expr!(constant(*c).add(var::<0>()), x)
                    } else {
                        return Ok(());
                    }
                }
                MacroExpr::Mul(l, r)
                    if matches!(
                        (l.as_ref(), r.as_ref()),
                        (MacroExpr::Var(0), MacroExpr::Constant(c))
                    ) =>
                {
                    if let MacroExpr::Constant(c) = r.as_ref() {
                        test_1var_expr!(var::<0>().mul(constant(*c)), x)
                    } else {
                        return Ok(());
                    }
                }
                MacroExpr::Mul(l, r)
                    if matches!(
                        (l.as_ref(), r.as_ref()),
                        (MacroExpr::Constant(c), MacroExpr::Var(0))
                    ) =>
                {
                    if let MacroExpr::Constant(c) = l.as_ref() {
                        test_1var_expr!(constant(*c).mul(var::<0>()), x)
                    } else {
                        return Ok(());
                    }
                }
                MacroExpr::Sin(inner) if matches!(inner.as_ref(), MacroExpr::Var(0)) => {
                    test_1var_expr!(var::<0>().sin(), x)
                }
                MacroExpr::Cos(inner) if matches!(inner.as_ref(), MacroExpr::Var(0)) => {
                    test_1var_expr!(var::<0>().cos(), x)
                }
                MacroExpr::Exp(inner) if matches!(inner.as_ref(), MacroExpr::Var(0)) => {
                    test_1var_expr!(var::<0>().exp(), x)
                }
                MacroExpr::Ln(inner) if matches!(inner.as_ref(), MacroExpr::Var(0)) => {
                    test_1var_expr!(var::<0>().ln(), x)
                }
                MacroExpr::Ln(inner)
                    if matches!(inner.as_ref(), MacroExpr::Exp(e))
                        && matches!(e.as_ref(), MacroExpr::Var(0)) =>
                {
                    test_1var_expr!(var::<0>().exp().ln(), x)
                }
                MacroExpr::Exp(inner)
                    if matches!(inner.as_ref(), MacroExpr::Ln(e))
                        && matches!(e.as_ref(), MacroExpr::Var(0)) =>
                {
                    test_1var_expr!(var::<0>().ln().exp(), x)
                }
                _ => return Ok(()), // Skip complex expressions for now
            }
        }
        (_, 2) => {
            let x = values[0];
            let y = values[1];
            match expr {
                MacroExpr::Add(l, r)
                    if matches!(
                        (l.as_ref(), r.as_ref()),
                        (MacroExpr::Var(0), MacroExpr::Var(1))
                    ) =>
                {
                    test_2var_expr!(var::<0>().add(var::<1>()), x, y)
                }
                MacroExpr::Mul(l, r)
                    if matches!(
                        (l.as_ref(), r.as_ref()),
                        (MacroExpr::Var(0), MacroExpr::Var(1))
                    ) =>
                {
                    test_2var_expr!(var::<0>().mul(var::<1>()), x, y)
                }
                MacroExpr::Sub(l, r)
                    if matches!(
                        (l.as_ref(), r.as_ref()),
                        (MacroExpr::Var(0), MacroExpr::Var(1))
                    ) =>
                {
                    test_2var_expr!(var::<0>().sub(var::<1>()), x, y)
                }
                _ => return Ok(()), // Skip other patterns for now
            }
        }
        _ => return Ok(()), // Skip 3+ variables for now
    };

    // Check if results match
    if !is_numerically_equivalent(reference_result, macro_result, 1e-10) {
        return Err(format!(
            "Macro result differs from reference: {} vs {} (diff: {})",
            macro_result,
            reference_result,
            (macro_result - reference_result).abs()
        ));
    }

    Ok(())
}

/// Check numerical equivalence with tolerance
fn is_numerically_equivalent(a: f64, b: f64, tolerance: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_finite() && b.is_finite() {
        return (a - b).abs() <= tolerance || (a - b).abs() <= tolerance * a.abs().max(b.abs());
    }
    false
}

// ============================================================================
// PROPTEST DEFINITIONS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Test 1-variable expressions
    #[test]
    fn prop_1var_macro_expressions(
        expr in arb_macro_expr(3, 1),
        x in -10.0_f64..10.0_f64
    ) {
        let values = vec![x.abs() + 0.1]; // Ensure positive for ln
        test_macro_expression(&expr, &values)?;
    }

    /// Test 2-variable expressions
    #[test]
    fn prop_2var_macro_expressions(
        expr in arb_macro_expr(2, 2),
        x in -5.0_f64..5.0_f64,
        y in -5.0_f64..5.0_f64
    ) {
        let values = vec![x, y];
        test_macro_expression(&expr, &values)?;
    }

    /// Test simple arithmetic expressions
    #[test]
    fn prop_simple_arithmetic(
        x in -10.0_f64..10.0_f64,
        y in -10.0_f64..10.0_f64,
        c in -5.0_f64..5.0_f64
    ) {
        // Test x + y
        let result1 = test_2var_expr!(var::<0>().add(var::<1>()), x, y);
        let expected1 = x + y;
        prop_assert!((result1 - expected1).abs() < 1e-10);

        // Test x * y
        let result2 = test_2var_expr!(var::<0>().mul(var::<1>()), x, y);
        let expected2 = x * y;
        prop_assert!((result2 - expected2).abs() < 1e-10);

        // Test x + c
        let result3 = test_1var_expr!(var::<0>().add(constant(c)), x);
        let expected3 = x + c;
        prop_assert!((result3 - expected3).abs() < 1e-10);
    }

    /// Test transcendental functions
    #[test]
    fn prop_transcendental_functions(
        x in 0.1_f64..10.0_f64 // Positive for ln safety
    ) {
        // Test sin(x)
        let result1 = test_1var_expr!(var::<0>().sin(), x);
        let expected1 = x.sin();
        prop_assert!((result1 - expected1).abs() < 1e-10);

        // Test cos(x)
        let result2 = test_1var_expr!(var::<0>().cos(), x);
        let expected2 = x.cos();
        prop_assert!((result2 - expected2).abs() < 1e-10);

        // Test exp(x)
        let result3 = test_1var_expr!(var::<0>().exp(), x);
        let expected3 = x.exp();
        prop_assert!((result3 - expected3).abs() < 1e-10);

        // Test ln(x)
        let result4 = test_1var_expr!(var::<0>().ln(), x);
        let expected4 = x.ln();
        prop_assert!((result4 - expected4).abs() < 1e-10);
    }
}

// ============================================================================
// MANUAL TESTS FOR SPECIFIC OPTIMIZATION PATTERNS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_optimization_patterns() {
        let patterns = test_optimization_patterns();

        for (expr, name) in patterns {
            println!("Testing optimization pattern: {}", name);

            // Use appropriate test values
            let values = match name {
                "exp(ln(x))" | "ln(exp(x))" => vec![2.5], // Positive for ln
                _ => vec![3.14, 2.71, 1.41],              // General values
            };

            // Test with first value only for 1-var expressions
            let test_values = vec![values[0]];
            test_macro_expression(&expr, &test_values).unwrap();
        }
    }

    #[test]
    fn test_complex_optimization_example() {
        // Test: ln(exp(x)) + y * 1 + 0 * z should optimize to x + y
        let x = 1.5_f64;
        let y = 2.5_f64;
        let z = 999.0_f64;

        let result = test_3var_expr!(
            var::<0>()
                .exp()
                .ln()
                .add(var::<1>().mul(constant(1.0)))
                .add(constant(0.0).mul(var::<2>())),
            x,
            y,
            z
        );

        let expected = x + y; // Should optimize to this
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_identity_optimizations() {
        let x = 3.14_f64;

        // x + 0 → x
        let result1 = test_1var_expr!(var::<0>().add(constant(0.0)), x);
        assert!((result1 - x).abs() < 1e-10);

        // x * 1 → x
        let result2 = test_1var_expr!(var::<0>().mul(constant(1.0)), x);
        assert!((result2 - x).abs() < 1e-10);

        // x * 0 → 0
        let result3 = test_1var_expr!(var::<0>().mul(constant(0.0)), x);
        assert!(result3.abs() < 1e-10);
    }

    #[test]
    fn test_inverse_function_optimizations() {
        let x = 2.5_f64;

        // ln(exp(x)) → x
        let result1 = test_1var_expr!(var::<0>().exp().ln(), x);
        assert!((result1 - x).abs() < 1e-10);

        // exp(ln(x)) → x (for positive x)
        let result2 = test_1var_expr!(var::<0>().ln().exp(), x);
        assert!((result2 - x).abs() < 1e-10);
    }
}
