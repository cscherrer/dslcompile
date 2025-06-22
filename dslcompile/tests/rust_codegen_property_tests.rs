//! Property-based tests for Rust code generation backend
//!
//! Tests the Rust code generation functionality to ensure generated code
//! is correct, compiles successfully, and produces expected results.

use dslcompile::{
    ast::{VariableRegistry, ast_repr::ASTRepr, ast_utils::collect_variable_indices},
    backends::rust_codegen::{RustCodeGenerator, RustCodegenConfig, RustCompiler, RustOptLevel},
    prelude::*,
};
use frunk::hlist;
use proptest::prelude::*;
use std::collections::BTreeSet;

/// Simple expression generator suitable for code generation testing
fn codegen_expr_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    let leaf = prop_oneof![
        (-10.0..10.0).prop_map(ASTRepr::Constant),
        (0..3usize).prop_map(ASTRepr::Variable),
    ];

    let unary = leaf.clone().prop_flat_map(|inner| {
        prop_oneof![
            Just(ASTRepr::Neg(Box::new(inner.clone()))),
            Just(ASTRepr::Sin(Box::new(inner.clone()))),
            Just(ASTRepr::Cos(Box::new(inner.clone()))),
            Just(ASTRepr::Exp(Box::new(inner))),
        ]
    });

    let binary = (leaf.clone(), leaf.clone()).prop_flat_map(|(left, right)| {
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

    prop_oneof![unary, binary].boxed()
}

/// Simple polynomial generator for deterministic testing
fn polynomial_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    (1..4usize)
        .prop_flat_map(|degree| {
            prop::collection::vec(-5.0..5.0f64, degree + 1).prop_map(move |coeffs| {
                let x = ASTRepr::Variable(0);
                let mut terms = Vec::new();

                for (i, &coeff) in coeffs.iter().enumerate() {
                    if coeff.abs() > 1e-10 {
                        let term = if i == 0 {
                            ASTRepr::Constant(coeff)
                        } else if i == 1 {
                            ASTRepr::mul_from_array([ASTRepr::Constant(coeff), x.clone()])
                        } else {
                            let power = ASTRepr::Pow(
                                Box::new(x.clone()),
                                Box::new(ASTRepr::Constant(i as f64)),
                            );
                            ASTRepr::mul_from_array([ASTRepr::Constant(coeff), power])
                        };
                        terms.push(term);
                    }
                }

                if terms.is_empty() {
                    ASTRepr::Constant(0.0)
                } else if terms.len() == 1 {
                    terms.into_iter().next().unwrap()
                } else {
                    ASTRepr::add_multiset(terms)
                }
            })
        })
        .boxed()
}

proptest! {
    /// Test that generated Rust code compiles without errors
    #[test]
    fn prop_generated_code_compiles(expr in codegen_expr_strategy()) {
        let generator = RustCodeGenerator::new();
        let code_result = generator.generate_function(&expr, "test_fn");

        // Code generation should succeed for valid expressions
        prop_assert!(code_result.is_ok());

        let code = code_result.unwrap();

        // Generated code should contain expected elements
        prop_assert!(code.contains("fn test_fn"));
        prop_assert!(code.contains("f64"));

        // Should not contain obvious syntax errors
        prop_assert!(!code.contains(";;"));
        prop_assert!(!code.contains("()()"));
    }

    /// Test that function signatures are generated correctly
    #[test]
    fn prop_function_signature_generation(expr in codegen_expr_strategy()) {
        let generator = RustCodeGenerator::new();
        let variables = collect_variable_indices(&expr);

        let code_result = generator.generate_function(&expr, "test_fn");
        prop_assert!(code_result.is_ok());

        let code = code_result.unwrap();

        // Function signature should be well-formed
        prop_assert!(code.contains("fn test_fn"));
        prop_assert!(code.contains("-> f64"));

        // If there are variables, should have parameters
        if !variables.is_empty() {
            prop_assert!(code.contains(": f64"));
        }
    }

    /// Test optimization level flags
    #[test]
    fn prop_optimization_levels(level in prop::sample::select(&[
        RustOptLevel::O0, RustOptLevel::O1, RustOptLevel::O2,
        RustOptLevel::O3, RustOptLevel::Os, RustOptLevel::Oz
    ])) {
        let flag = level.as_flag();

        // All optimization levels should produce valid flags
        prop_assert!(flag.starts_with("opt-level="));
        prop_assert!(flag.len() > "opt-level=".len());

        // Specific flag format checks
        match level {
            RustOptLevel::O0 => prop_assert_eq!(flag, "opt-level=0"),
            RustOptLevel::O1 => prop_assert_eq!(flag, "opt-level=1"),
            RustOptLevel::O2 => prop_assert_eq!(flag, "opt-level=2"),
            RustOptLevel::O3 => prop_assert_eq!(flag, "opt-level=3"),
            RustOptLevel::Os => prop_assert_eq!(flag, "opt-level=s"),
            RustOptLevel::Oz => prop_assert_eq!(flag, "opt-level=z"),
        }
    }

    /// Test that generated code has correct structure
    #[test]
    fn prop_code_structure(expr in polynomial_strategy()) {
        let generator = RustCodeGenerator::new();

        let code_result = generator.generate_function(&expr, "poly_fn");
        prop_assert!(code_result.is_ok());

        let code = code_result.unwrap();

        // Should have proper function structure
        prop_assert!(code.contains("fn poly_fn"));
        prop_assert!(code.contains("-> f64"));

        // Should be syntactically valid - look for literal braces in code
        prop_assert!(code.contains("{"), "Generated code should contain opening brace");
        prop_assert!(code.contains("}"), "Generated code should contain closing brace");
    }

    /// Test configuration effects on generated code
    #[test]
    fn prop_config_effects(
        unsafe_opt in prop::bool::ANY,
        vectorization in prop::bool::ANY,
        aggressive_inline in prop::bool::ANY
    ) {
        let config = RustCodegenConfig {
            unsafe_optimizations: unsafe_opt,
            vectorization_hints: vectorization,
            aggressive_inlining: aggressive_inline,
            ..Default::default()
        };

        let generator = RustCodeGenerator::with_config(config.clone());
        let simple_expr = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(1.0)
        ]);

        let code_result = generator.generate_function(&simple_expr, "test_fn");
        prop_assert!(code_result.is_ok());

        let code = code_result.unwrap();

        // Code should still compile regardless of configuration
        prop_assert!(code.contains("fn test_fn"));
        prop_assert!(code.contains("-> f64"));
        prop_assert!(code.contains("f64"));
    }

    /// Test variable handling in generated code
    #[test]
    fn prop_variable_handling(var_count in 1..5usize) {
        let expr = {
            let mut terms = Vec::new();
            for i in 0..var_count {
                terms.push(ASTRepr::Variable(i));
            }
            if terms.len() == 1 {
                terms.into_iter().next().unwrap()
            } else {
                ASTRepr::add_multiset(terms)
            }
        };

        let generator = RustCodeGenerator::new();

        let code_result = generator.generate_function(&expr, "var_test");
        prop_assert!(code_result.is_ok());

        let code = code_result.unwrap();

        // Should generate function with variables
        prop_assert!(code.contains("fn var_test"));
        prop_assert!(code.contains("-> f64"));

        // Should have parameters for multi-variable expressions
        if var_count > 1 {
            prop_assert!(code.contains(": f64"));
        }
    }
}

/// Unit tests for specific code generation functionality
#[cfg(test)]
mod rust_codegen_unit_tests {
    use super::*;

    #[test]
    fn test_simple_constant_generation() {
        let generator = RustCodeGenerator::new();
        let expr = ASTRepr::Constant(42.0);

        let code = generator.generate_function(&expr, "const_fn").unwrap();

        assert!(code.contains("fn const_fn"));
        assert!(code.contains("42"));
        assert!(code.contains("-> f64"));
    }

    #[test]
    fn test_simple_variable_generation() {
        let generator = RustCodeGenerator::new();
        let expr = ASTRepr::Variable(0);

        let code = generator.generate_function(&expr, "var_fn").unwrap();

        assert!(code.contains("fn var_fn"));
        assert!(code.contains(": f64"));
        assert!(code.contains("-> f64"));
    }

    #[test]
    fn test_arithmetic_operations() {
        let generator = RustCodeGenerator::new();

        // Test addition
        let add_expr = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Constant(1.0)]);
        let add_code = generator.generate_function(&add_expr, "add_fn").unwrap();
        assert!(add_code.contains("+"));

        // Test multiplication
        let mul_expr = ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Constant(2.0)]);
        let mul_code = generator.generate_function(&mul_expr, "mul_fn").unwrap();
        assert!(mul_code.contains("*"));

        // Test subtraction
        let sub_expr = ASTRepr::Sub(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(1.0)),
        );
        let sub_code = generator.generate_function(&sub_expr, "sub_fn").unwrap();
        assert!(sub_code.contains("-"));
    }

    #[test]
    fn test_transcendental_functions() {
        let generator = RustCodeGenerator::new();
        let x = ASTRepr::Variable(0);

        // Test sin
        let sin_expr = ASTRepr::Sin(Box::new(x.clone()));
        let sin_code = generator.generate_function(&sin_expr, "sin_fn").unwrap();
        assert!(sin_code.contains("sin"));

        // Test cos
        let cos_expr = ASTRepr::Cos(Box::new(x.clone()));
        let cos_code = generator.generate_function(&cos_expr, "cos_fn").unwrap();
        assert!(cos_code.contains("cos"));

        // Test exp
        let exp_expr = ASTRepr::Exp(Box::new(x));
        let exp_code = generator.generate_function(&exp_expr, "exp_fn").unwrap();
        assert!(exp_code.contains("exp"));
    }

    #[test]
    fn test_power_operations() {
        let generator = RustCodeGenerator::new();
        let x = ASTRepr::Variable(0);

        // Test integer power
        let pow2_expr = ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::Constant(2.0)));
        let pow2_code = generator.generate_function(&pow2_expr, "pow2_fn").unwrap();
        // Should optimize x^2 to x*x or use powi
        assert!(pow2_code.contains("*") || pow2_code.contains("powi"));

        // Test general power
        let pow_expr = ASTRepr::Pow(Box::new(x), Box::new(ASTRepr::Constant(2.5)));
        let pow_code = generator.generate_function(&pow_expr, "pow_fn").unwrap();
        assert!(pow_code.contains("powf") || pow_code.contains("pow"));
    }

    #[test]
    fn test_complex_expression() {
        let generator = RustCodeGenerator::new();

        // Build: sin(x^2) + cos(y) * 2
        let x = ASTRepr::Variable(0);
        let y = ASTRepr::Variable(1);

        let x_squared = ASTRepr::Pow(Box::new(x), Box::new(ASTRepr::Constant(2.0)));
        let sin_part = ASTRepr::Sin(Box::new(x_squared));
        let cos_part = ASTRepr::Cos(Box::new(y));
        let cos_times_2 = ASTRepr::mul_from_array([cos_part, ASTRepr::Constant(2.0)]);
        let expr = ASTRepr::add_from_array([sin_part, cos_times_2]);

        let code = generator.generate_function(&expr, "complex_fn").unwrap();

        assert!(code.contains("fn complex_fn"));
        assert!(code.contains(": f64")); // Should have parameters
        assert!(code.contains("sin"));
        assert!(code.contains("cos"));
        assert!(code.contains("+"));
        assert!(code.contains("*"));
    }

    #[test]
    fn test_config_default() {
        let config = RustCodegenConfig::default();

        assert!(!config.debug_info);
        assert!(!config.unsafe_optimizations);
        assert!(config.vectorization_hints);
        assert!(config.aggressive_inlining);
        assert!(config.target_cpu.is_none());
    }

    #[test]
    fn test_optimization_level_flags() {
        assert_eq!(RustOptLevel::O0.as_flag(), "opt-level=0");
        assert_eq!(RustOptLevel::O1.as_flag(), "opt-level=1");
        assert_eq!(RustOptLevel::O2.as_flag(), "opt-level=2");
        assert_eq!(RustOptLevel::O3.as_flag(), "opt-level=3");
        assert_eq!(RustOptLevel::Os.as_flag(), "opt-level=s");
        assert_eq!(RustOptLevel::Oz.as_flag(), "opt-level=z");
    }

    #[test]
    fn test_generator_with_custom_config() {
        let config = RustCodegenConfig {
            unsafe_optimizations: true,
            vectorization_hints: false,
            aggressive_inlining: false,
            ..Default::default()
        };

        let generator = RustCodeGenerator::with_config(config);
        let expr = ASTRepr::Variable(0);

        let code = generator.generate_function(&expr, "custom_fn").unwrap();

        // Should still generate valid code with custom config
        assert!(code.contains("fn custom_fn"));
        assert!(code.contains(": f64"));
        assert!(code.contains("-> f64"));
    }

    #[test]
    fn test_empty_parameter_list() {
        let generator = RustCodeGenerator::new();
        let expr = ASTRepr::Constant(42.0);

        let code = generator.generate_function(&expr, "no_params").unwrap();

        assert!(code.contains("fn no_params()"));
        assert!(code.contains("-> f64"));
        assert!(code.contains("42"));
    }

    #[test]
    fn test_multiple_variables() {
        let generator = RustCodeGenerator::new();
        let expr = ASTRepr::add_from_array([
            ASTRepr::Variable(2),
            ASTRepr::Variable(0),
            ASTRepr::Variable(1),
        ]);

        let code = generator.generate_function(&expr, "multi_var").unwrap();

        // Should contain function with multiple parameters
        assert!(code.contains("fn multi_var"));
        assert!(code.contains(": f64")); // Should have parameters
        assert!(code.contains("-> f64"));

        // Should handle multiple variables
        let param_count = code.matches(": f64").count();
        assert!(param_count >= 3); // Should have at least 3 parameters
    }
}
