//! Property-based tests for AST utilities
//!
//! Tests AST manipulation functions to ensure mathematical correctness,
//! proper variable handling, and structural consistency.

use dslcompile::{
    ast::{
        ast_repr::ASTRepr,
        ast_utils::{
            collect_variable_indices, count_nodes, expression_depth, expressions_equal_default,
            extract_constant, extract_variable_index, is_constant, is_variable,
            generate_variable_names, count_operations_visitor,
            summation_aware_cost_visitor, transform_expression,
        },
        VariableRegistry,
    },
};
use proptest::prelude::*;
use std::collections::BTreeSet;

/// Simple expression generator for focused testing
fn simple_expr_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    let leaf = prop_oneof![
        (0.0..100.0).prop_map(ASTRepr::Constant),
        (0..5usize).prop_map(ASTRepr::Variable),
    ];
    
    let simple_ops = leaf.clone().prop_flat_map(move |left| {
        leaf.clone().prop_map(move |right| {
            ASTRepr::add_from_array([left.clone(), right])
        })
    });
    
    prop_oneof![simple_ops].boxed()
}

/// Complex expression generator for stress testing
fn complex_expr_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    let leaf = prop_oneof![
        (-100.0..100.0).prop_map(ASTRepr::Constant),
        (0..4usize).prop_map(ASTRepr::Variable),
    ];
    
    let unary = leaf.clone().prop_map(|inner| {
        ASTRepr::Sin(Box::new(inner))
    });
    
    prop_oneof![leaf, unary].boxed()
}

proptest! {
    /// Test that expression equality works correctly
    #[test]
    fn prop_expression_equality(expr in simple_expr_strategy()) {
        // Expression should be equal to itself
        prop_assert!(expressions_equal_default(&expr, &expr));
        
        // Deep clone should be equal to original
        let cloned = expr.clone();
        prop_assert!(expressions_equal_default(&expr, &cloned));
    }

    /// Test variable collection properties
    #[test]
    fn prop_variable_collection(expr in simple_expr_strategy()) {
        let variables = collect_variable_indices(&expr);
        
        // Should not be empty if expression contains variables
        match &expr {
            ASTRepr::Variable(_) => {
                prop_assert!(!variables.is_empty(), "Variable expression should have variables");
            },
            ASTRepr::Constant(_) => {
                prop_assert!(variables.is_empty(), "Constant expression should have no variables");
            },
            _ => {
                // Complex expressions may or may not have variables
            }
        }
        
        // All collected indices should be reasonable
        for &var_idx in &variables {
            prop_assert!(var_idx < 1000, "Variable index should be reasonable");
        }
    }

    /// Test expression depth calculation
    #[test]
    fn prop_expression_depth(expr in simple_expr_strategy()) {
        let depth = expression_depth(&expr);
        
        // Depth should be reasonable
        prop_assert!(depth >= 1, "Expression depth should be at least 1");
        prop_assert!(depth <= 10, "Expression depth should be reasonable");
        
        // Leaf expressions should have depth 1
        match &expr {
            ASTRepr::Variable(_) | ASTRepr::Constant(_) => {
                prop_assert_eq!(depth, 1, "Leaf expressions should have depth 1");
            },
            _ => {
                prop_assert!(depth >= 1, "Complex expressions should have depth >= 1");
            }
        }
    }

    /// Test node counting
    #[test]
    fn prop_node_counting(expr in simple_expr_strategy()) {
        let node_count = count_nodes(&expr);
        
        // Should have at least 1 node
        prop_assert!(node_count >= 1, "Expression should have at least 1 node");
        prop_assert!(node_count <= 100, "Node count should be reasonable");
        
        // Leaf expressions should have exactly 1 node
        match &expr {
            ASTRepr::Variable(_) | ASTRepr::Constant(_) => {
                prop_assert_eq!(node_count, 1, "Leaf expressions should have 1 node");
            },
            _ => {
                prop_assert!(node_count >= 1, "Complex expressions should have >= 1 node");
            }
        }
    }

    /// Test transformation functionality
    #[test]
    fn prop_transformation_properties(expr in simple_expr_strategy()) {
        // Identity transformation should preserve expression
        let identity = |_: &ASTRepr<f64>| None;
        let result = transform_expression(&expr, &identity);
        
        prop_assert!(expressions_equal_default(&expr, &result), 
                   "Identity transformation should preserve expression");
    }

    /// Test cost computation
    #[test]
    fn prop_cost_computation(expr in simple_expr_strategy()) {
        let cost = summation_aware_cost_visitor(&expr);
        
        // Cost should be non-negative
        prop_assert!(cost >= 0, "Cost should be non-negative");
        
        // Variables should have low cost
        match &expr {
            ASTRepr::Variable(_) => {
                prop_assert!(cost <= 1, "Variable should have low cost");
            },
            ASTRepr::Constant(_) => {
                prop_assert!(cost <= 1, "Constant should have low cost");
            },
            _ => {
                prop_assert!(cost >= 0, "Complex expressions should have non-negative cost");
            }
        }
    }

    /// Test operation counting
    #[test]
    fn prop_operation_counting(expr in simple_expr_strategy()) {
        let op_count = count_operations_visitor(&expr);
        
        // Operation count should be non-negative
        prop_assert!(op_count >= 0, "Operation count should be non-negative");
        
        // Leaf expressions should have 0 operations
        match &expr {
            ASTRepr::Variable(_) | ASTRepr::Constant(_) => {
                prop_assert_eq!(op_count, 0, "Leaf expressions should have 0 operations");
            },
            _ => {
                prop_assert!(op_count >= 0, "Complex expressions should have >= 0 operations");
            }
        }
    }

    /// Test variable depth property
    #[test]
    fn prop_variable_depth_consistency(var_idx in 0..10usize) {
        prop_assert_eq!(expression_depth(&ASTRepr::<f64>::Variable(var_idx)), 1);
    }

    /// Test that constant extraction works
    #[test]
    fn prop_constant_extraction(value in -100.0..100.0f64) {
        let const_expr: ASTRepr<f64> = ASTRepr::Constant(value);
        let extracted = extract_constant(&const_expr);
        
        prop_assert_eq!(extracted, Some(value));
        prop_assert!(is_constant(&const_expr));
        prop_assert!(!is_variable(&const_expr));
    }

    /// Test that variable extraction works
    #[test]
    fn prop_variable_extraction(var_idx in 0..20usize) {
        let var_expr: ASTRepr<f64> = ASTRepr::Variable(var_idx);
        let extracted = extract_variable_index(&var_expr);
        
        prop_assert_eq!(extracted, Some(var_idx));
        prop_assert!(is_variable(&var_expr));
        prop_assert!(!is_constant(&var_expr));
    }

    /// Test edge cases that should not crash
    #[test]
    fn prop_edge_cases_handling(var_idx in 0..10usize) {
        // Test with very simple expressions
        let var_expr: ASTRepr<f64> = ASTRepr::Variable(var_idx);
        let const_expr: ASTRepr<f64> = ASTRepr::Constant(0.0);
        
        // Should not crash on simple cases
        let vars = collect_variable_indices(&var_expr);
        prop_assert!(vars.contains(&var_idx));
        
        let const_vars = collect_variable_indices(&const_expr);
        prop_assert!(const_vars.is_empty());
        
        // Depth calculation should work
        let depth = expression_depth(&var_expr);
        prop_assert_eq!(depth, 1);
    }

    /// Test basic complexity measures
    #[test]
    fn prop_complexity_consistency(expr in simple_expr_strategy()) {
        let depth = expression_depth(&expr);
        let nodes = count_nodes(&expr);
        let ops = count_operations_visitor(&expr);
        
        // Basic sanity checks
        prop_assert!(depth >= 1);
        prop_assert!(nodes >= 1);
        prop_assert!(ops >= 0);
        
        // Node count should be at least as large as operation count
        prop_assert!(nodes >= ops as usize);
    }
}

/// Unit tests for specific ast_utils functionality
#[cfg(test)]
mod ast_utils_unit_tests {
    use super::*;

    #[test]
    fn test_variable_collection_complex() {
        // Build expression: x₀ + x₁ * x₂ - x₀
        let expr: ASTRepr<f64> = ASTRepr::Sub(
            Box::new(ASTRepr::add_from_array([
                ASTRepr::Variable(0),
                ASTRepr::mul_from_array([ASTRepr::Variable(1), ASTRepr::Variable(2)])
            ])),
            Box::new(ASTRepr::Variable(0))
        );
        
        let vars = collect_variable_indices(&expr);
        assert_eq!(vars, [0, 1, 2].iter().cloned().collect::<BTreeSet<_>>());
    }

    #[test]
    fn test_operation_counting() {
        let simple_expr: ASTRepr<f64> = ASTRepr::Variable(0);
        let complex_expr: ASTRepr<f64> = ASTRepr::Sin(Box::new(
            ASTRepr::add_from_array([
                ASTRepr::Variable(0),
                ASTRepr::Constant(1.0)
            ])
        ));
        
        let simple_ops = count_operations_visitor(&simple_expr);
        let complex_ops = count_operations_visitor(&complex_expr);
        
        assert_eq!(simple_ops, 0); // Variables are not operations
        assert!(complex_ops > simple_ops); // Complex expression has more operations
    }

    #[test]
    fn test_transform_identity() {
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(1.0)
        ]);
        
        // Identity transformation
        let identity = |_: &ASTRepr<f64>| None;
        let result = transform_expression(&expr, &identity);
        
        // Should be identical 
        assert!(expressions_equal_default(&expr, &result));
    }

    #[test]
    fn test_extract_functions() {
        let const_expr: ASTRepr<f64> = ASTRepr::Constant(42.0);
        let var_expr: ASTRepr<f64> = ASTRepr::Variable(3);
        
        // Test constant extraction
        assert_eq!(extract_constant(&const_expr), Some(42.0));
        assert_eq!(extract_constant(&var_expr), None);
        
        // Test variable extraction
        assert_eq!(extract_variable_index(&var_expr), Some(3));
        assert_eq!(extract_variable_index(&const_expr), None);
        
        // Test type predicates
        assert!(is_constant(&const_expr));
        assert!(!is_constant(&var_expr));
        assert!(is_variable(&var_expr));
        assert!(!is_variable(&const_expr));
    }

    #[test]
    fn test_special_value_detection() {
        let zero_expr: ASTRepr<f64> = ASTRepr::Constant(0.0);
        let one_expr: ASTRepr<f64> = ASTRepr::Constant(1.0);
        
        // Test constant values
        assert!(is_constant(&zero_expr));
        assert!(is_constant(&one_expr));
        
        assert_eq!(extract_constant(&zero_expr), Some(0.0));
        assert_eq!(extract_constant(&one_expr), Some(1.0));
    }

    #[test]
    fn test_variable_name_generation() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();
        let y_idx = registry.register_variable();
        
        let indices = [x_idx, y_idx].iter().cloned().collect::<BTreeSet<_>>();
        let names = generate_variable_names(&indices, &registry);
        
        // Should generate some names for the variables
        assert_eq!(names.len(), 2);
        assert!(!names[0].is_empty());
        assert!(!names[1].is_empty());
        assert_ne!(names[0], names[1]); // Should be different names
    }

    #[test]
    fn test_traversal_functionality() {
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Sin(Box::new(ASTRepr::Variable(1)))
        ]);
        
        let variables = collect_variable_indices(&expr);
        let depth = expression_depth(&expr);
        let nodes = count_nodes(&expr);
        
        // Should contain both variables
        assert!(variables.contains(&0));
        assert!(variables.contains(&1));
        assert_eq!(variables.len(), 2);
        
        // Should have reasonable depth and node count
        assert!(depth >= 2); // At least 2 for nested structure
        assert!(nodes >= 3); // At least 3 nodes
    }

    #[test]
    fn test_cost_computation() {
        let simple_expr: ASTRepr<f64> = ASTRepr::Variable(0);
        let complex_expr: ASTRepr<f64> = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        
        let simple_cost = summation_aware_cost_visitor(&simple_expr);
        let complex_cost = summation_aware_cost_visitor(&complex_expr);
        
        // Variable has zero cost, sin is expensive
        assert_eq!(simple_cost, 0);
        assert!(complex_cost > simple_cost);
    }
}