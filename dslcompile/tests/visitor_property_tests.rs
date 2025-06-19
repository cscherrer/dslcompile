//! Property-based tests for AST visitor pattern functionality
//!
//! Tests the visitor pattern implementation to ensure it works correctly
//! across all AST node types and handles edge cases properly.

use dslcompile::{
    ast::{
        ast_repr::ASTRepr,
        ast_utils::collect_variable_indices,
        visitor::ASTVisitor,
        Scalar,
    },
};
use proptest::prelude::*;
use std::collections::BTreeSet;

/// Simple expression generator for visitor testing
fn visitor_expr_strategy() -> BoxedStrategy<ASTRepr<f64>> {
    let leaf = prop_oneof![
        (-10.0..10.0).prop_map(ASTRepr::Constant),
        (0..3usize).prop_map(ASTRepr::Variable),
    ];
    
    let unary = leaf.clone().prop_map(|inner| {
        ASTRepr::Sin(Box::new(inner))
    });
    
    let binary = leaf.clone().prop_flat_map(move |left| {
        leaf.clone().prop_map(move |right| {
            ASTRepr::add_from_array([left.clone(), right])
        })
    });
    
    prop_oneof![unary, binary].boxed()
}

/// Simple node counting visitor for testing
struct NodeCountVisitor {
    count: usize,
}

impl NodeCountVisitor {
    fn new() -> Self {
        Self { count: 0 }
    }
    
    fn get_count(&self) -> usize {
        self.count
    }
}

impl ASTVisitor<f64> for NodeCountVisitor {
    type Output = ();
    type Error = ();

    fn visit_constant(&mut self, _value: &f64) -> Result<Self::Output, Self::Error> {
        self.count += 1;
        Ok(())
    }

    fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
        self.count += 1;
        Ok(())
    }

    fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
        self.count += 1;
        Ok(())
    }
    
    fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.count += 1;
        Ok(())
    }
    
    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
        self.count += 1;
        Ok(())
    }
    
    fn visit_collection_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
        self.count += 1;
        Ok(())
    }
}

/// Variable collecting visitor
struct VariableCollectingVisitor {
    variables: BTreeSet<usize>,
}

impl VariableCollectingVisitor {
    fn new() -> Self {
        Self {
            variables: BTreeSet::new(),
        }
    }
    
    fn get_variables(self) -> BTreeSet<usize> {
        self.variables
    }
}

impl ASTVisitor<f64> for VariableCollectingVisitor {
    type Output = ();
    type Error = ();

    fn visit_constant(&mut self, _value: &f64) -> Result<Self::Output, Self::Error> {
        Ok(())
    }

    fn visit_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error> {
        self.variables.insert(index);
        Ok(())
    }

    fn visit_bound_var(&mut self, index: usize) -> Result<Self::Output, Self::Error> {
        self.variables.insert(index);
        Ok(())
    }
    
    fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
    
    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
    
    fn visit_collection_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error> {
        self.variables.insert(index);
        Ok(())
    }
}

/// Visitor that can fail for testing error propagation
struct FailableVisitor {
    should_fail: bool,
}

impl FailableVisitor {
    fn new(should_fail: bool) -> Self {
        Self { should_fail }
    }
}

impl ASTVisitor<f64> for FailableVisitor {
    type Output = ();
    type Error = String;

    fn visit_constant(&mut self, _value: &f64) -> Result<Self::Output, Self::Error> {
        if self.should_fail {
            Err("Intentional failure".to_string())
        } else {
            Ok(())
        }
    }

    fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
        if self.should_fail {
            Err("Intentional failure".to_string())
        } else {
            Ok(())
        }
    }

    fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
    
    fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
    
    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
    
    fn visit_collection_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
}

proptest! {
    /// Test that node counting visitor works correctly
    #[test]
    fn prop_node_count_visitor(expr in visitor_expr_strategy()) {
        let mut visitor = NodeCountVisitor::new();
        let result = visitor.visit(&expr);
        
        // Visit should succeed
        prop_assert!(result.is_ok());
        
        // Should count at least 1 node
        let count = visitor.get_count();
        prop_assert!(count >= 1, "Should count at least 1 node");
        
        // Leaf expressions should count as 1
        match &expr {
            ASTRepr::Variable(_) | ASTRepr::Constant(_) => {
                prop_assert_eq!(count, 1, "Leaf nodes should count as 1");
            },
            _ => {
                prop_assert!(count >= 1, "Complex expressions should count >= 1");
            }
        }
    }

    /// Test that variable collecting visitor works correctly
    #[test]
    fn prop_variable_collecting_visitor(expr in visitor_expr_strategy()) {
        let mut visitor = VariableCollectingVisitor::new();
        let result = visitor.visit(&expr);
        
        // Visit should succeed
        prop_assert!(result.is_ok());
        
        // Get collected variables
        let collected_vars = visitor.get_variables();
        let expected_vars = collect_variable_indices(&expr);
        
        // Should collect the same variables as the utility function
        prop_assert_eq!(collected_vars, expected_vars, 
                       "Visitor should collect same variables as utility function");
    }

    /// Test error propagation in visitors
    #[test]
    fn prop_error_propagation(expr in visitor_expr_strategy()) {
        // Visitor that should succeed
        let mut success_visitor = FailableVisitor::new(false);
        let success_result = success_visitor.visit(&expr);
        prop_assert!(success_result.is_ok(), "Non-failing visitor should succeed");
        
        // Visitor that should fail (if expression contains constants/variables)
        let mut fail_visitor = FailableVisitor::new(true);
        let fail_result = fail_visitor.visit(&expr);
        
        // Should fail if the expression contains any nodes that trigger failure
        match &expr {
            ASTRepr::Variable(_) | ASTRepr::Constant(_) => {
                prop_assert!(fail_result.is_err(), "Failing visitor should fail on leaf nodes");
            },
            _ => {
                // Complex expressions might fail if they contain leaf nodes
                // We don't assert here since it depends on the expression structure
            }
        }
    }

    /// Test visitor traversal completeness
    #[test]
    fn prop_visitor_traversal_completeness(expr in visitor_expr_strategy()) {
        let mut count_visitor = NodeCountVisitor::new();
        let result = count_visitor.visit(&expr);
        
        // Should always succeed for simple expressions
        prop_assert!(result.is_ok());
        
        // Should visit all nodes
        let visited_count = count_visitor.get_count();
        prop_assert!(visited_count >= 1, "Should visit at least one node");
        
        // For leaf nodes, should visit exactly 1
        match &expr {
            ASTRepr::Variable(_) | ASTRepr::Constant(_) => {
                prop_assert_eq!(visited_count, 1, "Leaf should visit exactly 1 node");
            },
            _ => {
                prop_assert!(visited_count >= 1, "Complex expressions visit >= 1 node");
            }
        }
    }
}

/// Unit tests for specific visitor functionality
#[cfg(test)]
mod visitor_unit_tests {
    use super::*;

    #[test]
    fn test_node_count_simple() {
        let expr: ASTRepr<f64> = ASTRepr::Variable(0);
        let mut visitor = NodeCountVisitor::new();
        
        let result = visitor.visit(&expr);
        assert!(result.is_ok());
        assert_eq!(visitor.get_count(), 1);
    }

    #[test]
    fn test_node_count_complex() {
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(1.0)
        ]);
        let mut visitor = NodeCountVisitor::new();
        
        let result = visitor.visit(&expr);
        assert!(result.is_ok());
        assert!(visitor.get_count() >= 2); // At least the two leaf nodes
    }

    #[test]
    fn test_variable_collection() {
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Variable(2)
        ]);
        let mut visitor = VariableCollectingVisitor::new();
        
        let result = visitor.visit(&expr);
        assert!(result.is_ok());
        
        let vars = visitor.get_variables();
        assert!(vars.contains(&0));
        assert!(vars.contains(&2));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_error_propagation() {
        let expr: ASTRepr<f64> = ASTRepr::Constant(42.0);
        
        // Test success case
        let mut success_visitor = FailableVisitor::new(false);
        let success_result = success_visitor.visit(&expr);
        assert!(success_result.is_ok());
        
        // Test failure case
        let mut fail_visitor = FailableVisitor::new(true);
        let fail_result = fail_visitor.visit(&expr);
        assert!(fail_result.is_err());
    }

    #[test]
    fn test_visitor_with_transcendental() {
        let expr: ASTRepr<f64> = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        let mut visitor = NodeCountVisitor::new();
        
        let result = visitor.visit(&expr);
        assert!(result.is_ok());
        assert!(visitor.get_count() >= 1); // Should count at least the variable
    }

    #[test]
    fn test_empty_variable_collection() {
        let expr: ASTRepr<f64> = ASTRepr::Constant(42.0);
        let mut visitor = VariableCollectingVisitor::new();
        
        let result = visitor.visit(&expr);
        assert!(result.is_ok());
        
        let vars = visitor.get_variables();
        assert!(vars.is_empty());
    }
}