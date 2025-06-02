#![cfg(feature = "proptest_macro_integration")]

// This test file is currently disabled because it has syntax errors and is incompatible
// with our procedural macro design.
// TODO: Fix the proptest syntax and redesign to work with the new procedural macro approach.

use dslcompile::ast::ASTRepr;
use dslcompile::compile_time::optimized::{ToAst, equality_saturation, eval_ast};
use dslcompile::compile_time::{constant, optimize_compile_time, var};
use proptest::prelude::*;

/// Test the procedural macro with patterns that are known to work
/// This focuses on correctness testing for the implemented optimization patterns

/// Check if two floating point numbers are numerically equivalent
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

/// Test simple arithmetic operations with literal constants
fn test_simple_arithmetic_literals(x: f64, y: f64) -> Result<(), TestCaseError> {
    // Test x + y
    let result1 = optimize_compile_time!(var::<0>().add(var::<1>()), [x, y]);
    let expected1 = x + y;
    prop_assert!(
        is_numerically_equivalent(result1, expected1, 1e-10),
        "x + y failed: {} vs {}",
        result1,
        expected1
    );

    // Test x * y
    let result2 = optimize_compile_time!(var::<0>().mul(var::<1>()), [x, y]);
    let expected2 = x * y;
    prop_assert!(
        is_numerically_equivalent(result2, expected2, 1e-10),
        "x * y failed: {} vs {}",
        result2,
        expected2
    );

    // Test x + 0 (literal constant)
    let result3 = optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);
    let expected3 = x + 0.0;
    prop_assert!(
        is_numerically_equivalent(result3, expected3, 1e-10),
        "x + 0 failed: {} vs {}",
        result3,
        expected3
    );

    // Test x * 1 (literal constant)
    let result4 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [x]);
    let expected4 = x * 1.0;
    prop_assert!(
        is_numerically_equivalent(result4, expected4, 1e-10),
        "x * 1 failed: {} vs {}",
        result4,
        expected4
    );

    Ok(())
}

/// Test transcendental functions
fn test_transcendental_functions(x: f64) -> Result<(), TestCaseError> {
    // Test sin(x)
    let result1 = optimize_compile_time!(var::<0>().sin(), [x]);
    let expected1 = x.sin();
    prop_assert!(
        is_numerically_equivalent(result1, expected1, 1e-10),
        "sin(x) failed: {} vs {}",
        result1,
        expected1
    );

    // Test cos(x)
    let result2 = optimize_compile_time!(var::<0>().cos(), [x]);
    let expected2 = x.cos();
    prop_assert!(
        is_numerically_equivalent(result2, expected2, 1e-10),
        "cos(x) failed: {} vs {}",
        result2,
        expected2
    );

    // Test exp(x)
    let result3 = optimize_compile_time!(var::<0>().exp(), [x]);
    let expected3 = x.exp();
    prop_assert!(
        is_numerically_equivalent(result3, expected3, 1e-10),
        "exp(x) failed: {} vs {}",
        result3,
        expected3
    );

    // Test ln(x) for positive x
    if x > 0.0 {
        let result4 = optimize_compile_time!(var::<0>().ln(), [x]);
        let expected4 = x.ln();
        prop_assert!(
            is_numerically_equivalent(result4, expected4, 1e-10),
            "ln(x) failed: {} vs {}",
            result4,
            expected4
        );
    }

    Ok(())
}

/// Test identity optimizations with literal constants
fn test_identity_optimizations(x: f64) -> Result<(), TestCaseError> {
    // Test x + 0 → x
    let result1 = optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);
    prop_assert!(
        is_numerically_equivalent(result1, x, 1e-10),
        "x + 0 optimization failed: {} vs {}",
        result1,
        x
    );

    // Test 0 + x → x
    let result2 = optimize_compile_time!(constant(0.0).add(var::<0>()), [x]);
    prop_assert!(
        is_numerically_equivalent(result2, x, 1e-10),
        "0 + x optimization failed: {} vs {}",
        result2,
        x
    );

    // Test x * 1 → x
    let result3 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [x]);
    prop_assert!(
        is_numerically_equivalent(result3, x, 1e-10),
        "x * 1 optimization failed: {} vs {}",
        result3,
        x
    );

    // Test 1 * x → x
    let result4 = optimize_compile_time!(constant(1.0).mul(var::<0>()), [x]);
    prop_assert!(
        is_numerically_equivalent(result4, x, 1e-10),
        "1 * x optimization failed: {} vs {}",
        result4,
        x
    );

    // Test x * 0 → 0
    let result5 = optimize_compile_time!(var::<0>().mul(constant(0.0)), [x]);
    prop_assert!(
        is_numerically_equivalent(result5, 0.0, 1e-10),
        "x * 0 optimization failed: {} vs 0",
        result5
    );

    // Test 0 * x → 0
    let result6 = optimize_compile_time!(constant(0.0).mul(var::<0>()), [x]);
    prop_assert!(
        is_numerically_equivalent(result6, 0.0, 1e-10),
        "0 * x optimization failed: {} vs 0",
        result6
    );

    Ok(())
}

/// Test inverse function optimizations
fn test_inverse_optimizations(x: f64) -> Result<(), TestCaseError> {
    // Test ln(exp(x)) → x
    let result1 = optimize_compile_time!(var::<0>().exp().ln(), [x]);
    prop_assert!(
        is_numerically_equivalent(result1, x, 1e-10),
        "ln(exp(x)) optimization failed: {} vs {}",
        result1,
        x
    );

    // Test exp(ln(x)) → x for positive x
    if x > 0.0 {
        let result2 = optimize_compile_time!(var::<0>().ln().exp(), [x]);
        prop_assert!(
            is_numerically_equivalent(result2, x, 1e-10),
            "exp(ln(x)) optimization failed: {} vs {}",
            result2,
            x
        );
    }

    Ok(())
}

/// Test complex optimization patterns
fn test_complex_optimizations(x: f64, y: f64, z: f64) -> Result<(), TestCaseError> {
    // Test ln(exp(x)) + y * 1 + 0 * z → x + y
    let result = optimize_compile_time!(
        var::<0>()
            .exp()
            .ln()
            .add(var::<1>().mul(constant(1.0)))
            .add(constant(0.0).mul(var::<2>())),
        [x, y, z]
    );
    let expected = x + y;
    prop_assert!(
        is_numerically_equivalent(result, expected, 1e-10),
        "Complex optimization failed: {} vs {}",
        result,
        expected
    );

    Ok(())
}

// ============================================================================
// PROPTEST DEFINITIONS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Test simple arithmetic operations
    #[test]
    fn prop_simple_arithmetic(
        x in -10.0_f64..10.0_f64,
        y in -10.0_f64..10.0_f64
    ) {
        test_simple_arithmetic_literals(x, y)?;
    }

    /// Test transcendental functions
    #[test]
    fn prop_transcendental_functions(
        x in 0.1_f64..10.0_f64 // Positive for ln safety
    ) {
        test_transcendental_functions(x)?;
    }

    /// Test identity optimizations
    #[test]
    fn prop_identity_optimizations(
        x in -10.0_f64..10.0_f64
    ) {
        test_identity_optimizations(x)?;
    }

    /// Test inverse function optimizations
    #[test]
    fn prop_inverse_optimizations(
        x in 0.1_f64..10.0_f64 // Positive for ln safety
    ) {
        test_inverse_optimizations(x)?;
    }

    /// Test complex optimization patterns
    #[test]
    fn prop_complex_optimizations(
        x in -5.0_f64..5.0_f64,
        y in -5.0_f64..5.0_f64,
        z in -5.0_f64..5.0_f64
    ) {
        test_complex_optimizations(x, y, z)?;
    }

    /// Test with edge cases
    #[test]
    fn prop_edge_cases(
        x in prop_oneof![
            Just(0.0),
            Just(1.0),
            Just(-1.0),
            Just(std::f64::consts::E),
            Just(std::f64::consts::PI),
            Just(f64::MIN_POSITIVE),
            Just(1e-10),
            Just(1e10)
        ]
    ) {
        // Test basic operations with edge cases
        let result1 = optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);
        prop_assert!((result1 - x).abs() < 1e-10);

        let result2 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [x]);
        prop_assert!((result2 - x).abs() < 1e-10);

        if x > 0.0 {
            let result3 = optimize_compile_time!(var::<0>().exp().ln(), [x]);
            prop_assert!((result3 - x).abs() < 1e-10);
        }
    }

    /// Test constants
    #[test]
    fn prop_constants() {
        // Test specific constant values
        let result1 = optimize_compile_time!(constant(42.0), []);
        prop_assert!((result1 - 42.0).abs() < 1e-10);

        let result2 = optimize_compile_time!(constant(-3.14), []);
        prop_assert!((result2 - (-3.14)).abs() < 1e-10);

        let result3 = optimize_compile_time!(constant(0.0), []);
        prop_assert!(result3.abs() < 1e-10);
    }

    /// Test variable access
    #[test]
    fn prop_variables(
        x in -100.0_f64..100.0_f64,
        y in -100.0_f64..100.0_f64,
        z in -100.0_f64..100.0_f64
    ) {
        let result1 = optimize_compile_time!(var::<0>(), [x, y, z]);
        prop_assert!((result1 - x).abs() < 1e-10);

        let result2 = optimize_compile_time!(var::<1>(), [x, y, z]);
        prop_assert!((result2 - y).abs() < 1e-10);

        let result3 = optimize_compile_time!(var::<2>(), [x, y, z]);
        prop_assert!((result3 - z).abs() < 1e-10);
    }
}

// ============================================================================
// MANUAL TESTS FOR SPECIFIC PATTERNS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_optimization_patterns() {
        // ln(exp(x)) → x
        let x = 2.5_f64;
        let result1 = optimize_compile_time!(var::<0>().exp().ln(), [x]);
        assert!((result1 - x).abs() < 1e-10);

        // exp(ln(x)) → x
        let result2 = optimize_compile_time!(var::<0>().ln().exp(), [x]);
        assert!((result2 - x).abs() < 1e-10);

        // x + 0 → x
        let result3 = optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);
        assert!((result3 - x).abs() < 1e-10);

        // x * 1 → x
        let result4 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [x]);
        assert!((result4 - x).abs() < 1e-10);

        // x * 0 → 0
        let result5 = optimize_compile_time!(var::<0>().mul(constant(0.0)), [x]);
        assert!(result5.abs() < 1e-10);
    }

    #[test]
    fn test_complex_optimization_example() {
        // Test: ln(exp(x)) + y * 1 + 0 * z should optimize to x + y
        let x = 1.5_f64;
        let y = 2.5_f64;
        let z = 999.0_f64;

        let result = optimize_compile_time!(
            var::<0>()
                .exp()
                .ln()
                .add(var::<1>().mul(constant(1.0)))
                .add(constant(0.0).mul(var::<2>())),
            [x, y, z]
        );

        let expected = x + y; // Should optimize to this
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mathematical_correctness() {
        // Test that optimizations preserve mathematical semantics
        let test_values = vec![0.1, 1.0, 2.5, std::f64::consts::E, std::f64::consts::PI];

        for &x in &test_values {
            // ln(exp(x)) should equal x
            let optimized = optimize_compile_time!(var::<0>().exp().ln(), [x]);
            let manual = x;
            assert!(
                (optimized - manual).abs() < 1e-10,
                "ln(exp({})) = {} vs {}",
                x,
                optimized,
                manual
            );

            // exp(ln(x)) should equal x
            let optimized2 = optimize_compile_time!(var::<0>().ln().exp(), [x]);
            let manual2 = x;
            assert!(
                (optimized2 - manual2).abs() < 1e-10,
                "exp(ln({})) = {} vs {}",
                x,
                optimized2,
                manual2
            );
        }
    }

    #[test]
    fn test_optimization_vs_manual() {
        // Compare optimized expressions with manual calculations
        let x = 3.14_f64;
        let y = 2.71_f64;

        // Test: sin(x) + cos(y)
        let optimized = optimize_compile_time!(var::<0>().sin().add(var::<1>().cos()), [x, y]);
        let manual = x.sin() + y.cos();
        assert!((optimized - manual).abs() < 1e-10);

        // Test: exp(x) * ln(y)
        let optimized2 = optimize_compile_time!(var::<0>().exp().mul(var::<1>().ln()), [x, y]);
        let manual2 = x.exp() * y.ln();
        assert!((optimized2 - manual2).abs() < 1e-10);
    }

    #[test]
    fn test_zero_cost_abstraction() {
        // Verify that the macro generates efficient code
        // This is more of a conceptual test - the real verification is in benchmarks

        let x = 1.0_f64;
        let y = 2.0_f64;

        // Simple addition should be as fast as manual
        let result = optimize_compile_time!(var::<0>().add(var::<1>()), [x, y]);
        assert_eq!(result, x + y);

        // Identity optimization should be as fast as just the variable
        let result2 = optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);
        assert_eq!(result2, x);

        // Complex optimization should be as fast as the simplified form
        let result3 = optimize_compile_time!(
            var::<0>().exp().ln().add(var::<1>().mul(constant(1.0))),
            [x, y]
        );
        assert_eq!(result3, x + y);
    }

    #[test]
    fn test_specific_constant_values() {
        // Test various constant values work correctly
        let constants = vec![0.0, 1.0, -1.0, 2.5, -3.14, 42.0, 1e-10, 1e10];

        for &c in &constants {
            let result = optimize_compile_time!(constant(c), []);
            assert!(
                (result - c).abs() < 1e-10,
                "Constant {} failed: got {}",
                c,
                result
            );
        }
    }

    #[test]
    fn test_optimization_correctness_comprehensive() {
        // Comprehensive test of all optimization patterns
        let x = 2.718_f64;
        let y = 3.141_f64;

        // Identity optimizations
        assert_eq!(
            optimize_compile_time!(var::<0>().add(constant(0.0)), [x]),
            x
        );
        assert_eq!(
            optimize_compile_time!(constant(0.0).add(var::<0>()), [x]),
            x
        );
        assert_eq!(
            optimize_compile_time!(var::<0>().mul(constant(1.0)), [x]),
            x
        );
        assert_eq!(
            optimize_compile_time!(constant(1.0).mul(var::<0>()), [x]),
            x
        );
        assert_eq!(
            optimize_compile_time!(var::<0>().mul(constant(0.0)), [x]),
            0.0
        );
        assert_eq!(
            optimize_compile_time!(constant(0.0).mul(var::<0>()), [x]),
            0.0
        );

        // Inverse function optimizations
        let ln_exp_result = optimize_compile_time!(var::<0>().exp().ln(), [x]);
        assert!((ln_exp_result - x).abs() < 1e-10);

        let exp_ln_result = optimize_compile_time!(var::<0>().ln().exp(), [x]);
        assert!((exp_ln_result - x).abs() < 1e-10);

        // Basic arithmetic
        let add_result = optimize_compile_time!(var::<0>().add(var::<1>()), [x, y]);
        assert!((add_result - (x + y)).abs() < 1e-10);

        let mul_result = optimize_compile_time!(var::<0>().mul(var::<1>()), [x, y]);
        assert!((mul_result - (x * y)).abs() < 1e-10);
    }
}
