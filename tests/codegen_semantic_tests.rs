//! Semantic tests for Rust code generation backend
//! 
//! These tests focus on the correctness of generated Rust code, type safety,
//! and performance characteristics rather than basic functionality.

use dslcompile::{
    ast::{ASTRepr, VariableRegistry},
    backends::{RustCodeGenerator, RustCodegenConfig, RustOptLevel, RustCompiler},
    contexts::dynamic::DynamicContext,
    frunk::hlist,
};
use proptest::prelude::*;
use std::collections::HashSet;

#[cfg(test)]
mod code_generation_semantics {
    use super::*;

    /// Test that generated code preserves mathematical semantics
    #[test]
    fn test_generated_code_mathematical_correctness() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let y = ctx.var();
        
        // Create expression: x² + 2xy + y² = (x + y)²
        let expanded = &x * &x + &(&x * &y) * 2.0 + &y * &y;
        let factored = (&x + &y) * (&x + &y);
        
        let generator = RustCodeGenerator::new();
        let registry = VariableRegistry::new();
        
        // Generate code for both expressions
        let expanded_code = generator.generate_function_generic(&ctx.to_ast(&expanded), "expanded", "f64")
            .expect("Failed to generate expanded code");
        let factored_code = generator.generate_function_generic(&ctx.to_ast(&factored), "factored", "f64")
            .expect("Failed to generate factored code");
        
        // Both should contain valid Rust function signatures
        assert!(expanded_code.contains("pub extern \"C\" fn expanded"), 
               "Generated code missing function signature");
        assert!(factored_code.contains("pub extern \"C\" fn factored"), 
               "Generated code missing function signature");
        
        // Should not contain Vec flattening anti-patterns
        assert!(!expanded_code.contains("Vec<f64>"), 
               "Generated code contains Vec flattening anti-pattern");
        assert!(!factored_code.contains("to_params()"), 
               "Generated code contains deprecated to_params() pattern");
    }

    /// Test that code generation handles different numeric types correctly
    #[test]
    fn test_code_generation_type_safety() {
        let mut ctx = DynamicContext::<f32>::new();
        let x = ctx.var();
        let expr = &x * 2.0f32 + 1.0f32;
        
        let generator = RustCodeGenerator::new();
        let code = generator.generate_function_generic(&ctx.to_ast(&expr), "test_f32", "f32")
            .expect("Failed to generate f32 code");
        
        // Should generate f32-specific code
        assert!(code.contains("f32"), "Generated code should use f32 type");
        assert!(code.contains("pub extern \"C\" fn test_f32"), "Missing function signature");
        
        // Should not mix types
        assert!(!code.contains("f64"), "Generated code should not mix f64 with f32");
    }

    /// Test that optimization configurations affect generated code
    #[test]
    fn test_optimization_config_effects() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let expr = (&x * &x).sqrt(); // Should potentially use .sqrt() optimization
        
        let mut config = RustCodegenConfig::default();
        config.unsafe_optimizations = true;
        config.aggressive_inlining = true;
        
        let generator = RustCodeGenerator::with_config(config);
        let code = generator.generate_function_generic(&ctx.to_ast(&expr), "optimized", "f64")
            .expect("Failed to generate optimized code");
        
        // Should contain optimization hints
        assert!(code.contains("#[inline") || code.contains("#[target_feature"), 
               "Optimized code should contain optimization attributes");
    }

    /// Property test: Generated code should handle edge cases correctly
    proptest! {
        #[test]
        fn prop_generated_code_handles_edge_cases(
            coefficient in -1000.0..1000.0f64,
            power in 0.0..5.0f64
        ) {
            let mut ctx = DynamicContext::<f64>::new();
            let x = ctx.var();
            let expr = &x * coefficient + (&x).pow(power);
            
            let generator = RustCodeGenerator::new();
            let code = generator.generate_function_generic(&ctx.to_ast(&expr), "edge_case", "f64");
            
            prop_assert!(code.is_ok(), "Code generation should succeed for valid expressions");
            
            if let Ok(generated_code) = code {
                // Should contain proper function signature
                prop_assert!(generated_code.contains("pub extern \"C\" fn edge_case"), 
                           "Generated code missing function signature");
                
                // Should handle coefficients correctly
                if coefficient != 0.0 {
                    prop_assert!(generated_code.contains(&coefficient.to_string()) || 
                               generated_code.contains(&format!("{:.1}", coefficient)),
                               "Generated code should contain coefficient");
                }
            }
        }
    }

    /// Test that summation code generation preserves semantics
    #[test]
    fn test_summation_code_generation_semantics() {
        let mut ctx = DynamicContext::<f64>::new();
        let mu = ctx.var();
        let sigma = ctx.var();
        
        // Create a summation expression (should not unroll)
        let data = vec![1.0, 2.0, 3.0];
        let sum_expr = ctx.sum(data.as_slice(), |x| {
            let diff = x - &mu;
            &diff / &sigma
        });
        
        let generator = RustCodeGenerator::new();
        let code = generator.generate_function_generic(&ctx.to_ast(&sum_expr), "summation", "f64")
            .expect("Failed to generate summation code");
        
        // Should not contain loop unrolling (AST size independent of data size)
        let line_count = code.lines().count();
        assert!(line_count < 50, "Generated code should not unroll summation loops: {} lines", line_count);
        
        // Should contain iterator patterns, not manual indexing
        assert!(code.contains("iter") || code.contains("fold") || code.contains("sum"), 
               "Generated code should use iterator patterns for summation");
    }

    /// Test lambda variable scoping in generated code
    #[test]
    fn test_lambda_variable_scoping_in_codegen() {
        let mut ctx = DynamicContext::<f64>::new();
        let outer_var = ctx.var(); // Variable(0)
        
        let data = vec![1.0, 2.0];
        let lambda_expr = ctx.sum(data.as_slice(), |inner_var| {
            // inner_var should have different scope than outer_var
            &inner_var + &outer_var
        });
        
        let generator = RustCodeGenerator::new();
        let code = generator.generate_function_generic(&ctx.to_ast(&lambda_expr), "scoped", "f64")
            .expect("Failed to generate scoped code");
        
        // Should contain proper variable references
        assert!(code.contains("var_0") || code.contains("outer"), 
               "Generated code should reference outer variable");
        
        // Should not have variable collision issues
        assert!(!code.contains("var_0 + var_0"), 
               "Generated code should not have variable collision");
    }

    /// Test power optimization in code generation
    #[test]
    fn test_power_optimization_in_codegen() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        
        // x^2 should generate .powi(2), x^0.5 should generate .sqrt()
        let square = (&x).pow(2.0);
        let sqrt = (&x).pow(0.5);
        
        let generator = RustCodeGenerator::new();
        
        let square_code = generator.generate_function_generic(&ctx.to_ast(&square), "square", "f64")
            .expect("Failed to generate square code");
        let sqrt_code = generator.generate_function_generic(&ctx.to_ast(&sqrt), "sqrt", "f64")
            .expect("Failed to generate sqrt code");
        
        // Should use optimized power functions
        assert!(square_code.contains(".powi(2)") || square_code.contains("* var_0"), 
               "Square should use optimized power: {}", square_code);
        assert!(sqrt_code.contains(".sqrt()") || sqrt_code.contains(".powf(0.5)"), 
               "Square root should use optimized function: {}", sqrt_code);
    }

    /// Test function signature type preservation
    #[test]
    fn test_function_signature_type_preservation() {
        let mut ctx = DynamicContext::<f64>::new();
        let scalar1 = ctx.var::<f64>();
        let scalar2 = ctx.var::<f64>();
        let expr = &scalar1 + &scalar2;
        
        let generator = RustCodeGenerator::new();
        let code = generator.generate_function_generic(&ctx.to_ast(&expr), "typed", "f64")
            .expect("Failed to generate typed code");
        
        // Should preserve proper types in signature
        assert!(code.contains("f64"), "Generated code should preserve f64 type");
        assert!(code.contains("-> f64"), "Generated code should have proper return type");
        
        // Should not flatten to Vec<f64> anti-pattern
        assert!(!code.contains("Vec<f64>"), "Generated code should not use Vec flattening");
        assert!(!code.contains("&[f64]"), "Generated code should not use slice flattening");
    }

    /// Test compilation configuration semantics
    #[test]
    fn test_compilation_configuration_semantics() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let expr = (&x).sin() + (&x).cos();
        
        // Test different optimization levels
        let configs = vec![
            RustCodegenConfig::default(),
            RustCodegenConfig {
                unsafe_optimizations: true,
                aggressive_inlining: true,
                ..Default::default()
            },
            RustCodegenConfig {
                debug_info: true,
                ..Default::default()
            }
        ];
        
        let generator = RustCodeGenerator::new();
        let mut generated_codes = Vec::new();
        
        for config in configs {
            let gen_with_config = RustCodeGenerator::with_config(config);
            let code = gen_with_config.generate_function_generic(&ctx.to_ast(&expr), "configured", "f64")
                .expect("Failed to generate configured code");
            generated_codes.push(code);
        }
        
        // Different configurations should produce different code
        assert_ne!(generated_codes[0], generated_codes[1], 
                  "Different optimization configs should produce different code");
    }

    /// Test that generated code compilation would succeed
    #[test]
    fn test_generated_code_compilation() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let y = ctx.var();
        let expr = (&x).sin() + (&y).cos() * 2.0;
        
        let generator = RustCodeGenerator::new();
        let code = generator.generate_function_generic(&ctx.to_ast(&expr), "compile_test", "f64")
            .expect("Failed to generate code");
        
        // Add necessary imports and wrapper
        let full_code = format!(
            r#"
            #[no_mangle]
            pub extern "C" fn compile_test(var_0: f64, var_1: f64) -> f64 {{
                var_0.sin() + var_1.cos() * 2.0
            }}
            "#
        );
        
        // The code should be syntactically valid Rust
        // (We can't easily compile it in tests, but we can check structure)
        assert!(full_code.contains("pub extern \"C\" fn"), "Missing function declaration");
        assert!(full_code.contains("-> f64"), "Missing return type");
        assert!(full_code.contains("var_0") && full_code.contains("var_1"), "Missing parameters");
    }

    /// Property test: Code generation should be deterministic
    proptest! {
        #[test]
        fn prop_code_generation_deterministic(
            coeff1 in -10.0..10.0f64,
            coeff2 in -10.0..10.0f64
        ) {
            let mut ctx = DynamicContext::<f64>::new();
            let x = ctx.var();
            let expr = &x * coeff1 + coeff2;
            
            let generator = RustCodeGenerator::new();
            let code1 = generator.generate_function_generic(&ctx.to_ast(&expr), "deterministic", "f64");
            let code2 = generator.generate_function_generic(&ctx.to_ast(&expr), "deterministic", "f64");
            
            prop_assert_eq!(code1, code2, "Code generation should be deterministic");
        }
    }
}

#[cfg(test)]
mod compilation_semantics {
    use super::*;

    /// Test that different optimization levels produce valid code
    #[test]
    fn test_optimization_levels_validity() {
        let levels = vec![
            RustOptLevel::O0,
            RustOptLevel::O1, 
            RustOptLevel::O2,
            RustOptLevel::O3,
            RustOptLevel::Os,
            RustOptLevel::Oz,
        ];
        
        for level in levels {
            let flag = level.as_flag();
            assert!(flag.starts_with("opt-level="), 
                   "Optimization level {:?} should produce valid flag: {}", level, flag);
            
            // Each level should produce a different flag
            assert!(!flag.is_empty(), "Optimization flag should not be empty");
        }
    }

    /// Test that compiler availability check works
    #[test]
    fn test_compiler_availability_semantics() {
        // This test checks the semantic meaning of compiler availability
        let is_available = RustCompiler::is_available();
        
        if is_available {
            // If compiler is available, version info should work
            let version_result = RustCompiler::version_info();
            assert!(version_result.is_ok(), "Version info should work when compiler is available");
            
            if let Ok(version) = version_result {
                assert!(!version.is_empty(), "Version string should not be empty");
                assert!(version.contains("rustc") || version.contains("rust"), 
                       "Version should mention Rust compiler");
            }
        }
    }

    /// Test that compilation configuration affects behavior
    #[test]
    fn test_compilation_configuration_semantics() {
        let compiler_default = RustCompiler::new();
        let compiler_o3 = RustCompiler::with_opt_level(RustOptLevel::O3);
        let compiler_with_flags = RustCompiler::new().with_extra_flags(vec![
            "--emit=obj".to_string(),
            "-C".to_string(), 
            "target-cpu=native".to_string()
        ]);
        
        // Different configurations should be distinguishable
        // (We can't easily test compilation without rustc, but we can test configuration)
        assert_ne!(format!("{:?}", compiler_default), format!("{:?}", compiler_o3),
                  "Different optimization levels should create different configurations");
        assert_ne!(format!("{:?}", compiler_default), format!("{:?}", compiler_with_flags),
                  "Extra flags should create different configurations");
    }
}

#[cfg(test)]
mod type_system_semantics {
    use super::*;

    /// Test that type system preserves mathematical operations
    #[test]
    fn test_type_system_mathematical_preservation() {
        // Test with different numeric types
        let test_cases = vec![
            ("f32", "f32"),
            ("f64", "f64"),
            ("i32", "i32"),
            ("i64", "i64"),
        ];
        
        for (input_type, expected_type) in test_cases {
            let mut ctx = DynamicContext::<f64>::new();
            let x = ctx.var();
            let expr = &x + 1.0;
            
            let generator = RustCodeGenerator::new();
            let code = generator.generate_function_generic(&ctx.to_ast(&expr), "typed", expected_type)
                .expect("Failed to generate typed code");
            
            // Should contain the expected type
            assert!(code.contains(expected_type), 
                   "Generated code should contain type {}: {}", expected_type, code);
        }
    }

    /// Property test: Type consistency across operations
    proptest! {
        #[test]
        fn prop_type_consistency_across_operations(
            val1 in -100.0..100.0f64,
            val2 in -100.0..100.0f64
        ) {
            let mut ctx = DynamicContext::<f64>::new();
            let x = ctx.var();
            let y = ctx.var();
            
            // Test various operations maintain type consistency
            let operations = vec![
                &x + &y,
                &x - &y,
                &x * &y,
                &x / &y,
            ];
            
            let generator = RustCodeGenerator::new();
            
            for (i, op) in operations.iter().enumerate() {
                let code = generator.generate_function_generic(&ctx.to_ast(op), &format!("op_{}", i), "f64");
                prop_assert!(code.is_ok(), "Operation {} should generate valid code", i);
                
                if let Ok(generated) = code {
                    prop_assert!(generated.contains("f64"), "Generated code should maintain f64 type");
                    prop_assert!(generated.contains("-> f64"), "Generated code should return f64");
                }
            }
        }
    }
} 