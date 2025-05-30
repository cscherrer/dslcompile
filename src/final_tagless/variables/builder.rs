//! Expression Builder with Variable Management
//!
//! This module provides a clean API for building expressions with named variables
//! while using efficient indices internally.

use super::registry::VariableRegistry;
use crate::ast::ASTRepr;

/// Expression builder that maintains its own variable registry
/// This provides a clean API for building expressions with named variables
/// while using efficient indices internally.
#[derive(Debug, Clone)]
pub struct ExpressionBuilder {
    registry: VariableRegistry,
}

impl ExpressionBuilder {
    /// Create a new expression builder with an empty variable registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: VariableRegistry::new(),
        }
    }

    /// Register a variable and return its index
    pub fn register_variable(&mut self, name: &str) -> usize {
        self.registry.register_variable(name)
    }

    /// Create a variable expression by name (registers automatically)
    pub fn var(&mut self, name: &str) -> ASTRepr<f64> {
        let index = self.register_variable(name);
        ASTRepr::Variable(index)
    }

    /// Create a variable expression by index (for performance)
    #[must_use]
    pub fn var_by_index(&self, index: usize) -> ASTRepr<f64> {
        ASTRepr::Variable(index)
    }

    /// Create a constant expression
    #[must_use]
    pub fn constant(&self, value: f64) -> ASTRepr<f64> {
        ASTRepr::Constant(value)
    }

    /// Get the variable registry (for evaluation)
    #[must_use]
    pub fn registry(&self) -> &VariableRegistry {
        &self.registry
    }

    /// Get a mutable reference to the variable registry
    pub fn registry_mut(&mut self) -> &mut VariableRegistry {
        &mut self.registry
    }

    /// Evaluate an expression with named variables
    #[must_use]
    pub fn eval_with_named_vars(&self, expr: &ASTRepr<f64>, named_vars: &[(String, f64)]) -> f64 {
        let var_array = self.registry.create_variable_map(named_vars);
        expr.eval_with_vars(&var_array)
    }

    /// Evaluate an expression with indexed variables (most efficient)
    #[must_use]
    pub fn eval_with_vars(&self, expr: &ASTRepr<f64>, variables: &[f64]) -> f64 {
        expr.eval_with_vars(variables)
    }

    /// Get the number of registered variables
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.registry.len()
    }

    /// Get all variable names in registration order
    #[must_use]
    pub fn variable_names(&self) -> &[String] {
        self.registry.get_all_names()
    }

    /// Get the index of a variable by name
    #[must_use]
    pub fn get_variable_index(&self, name: &str) -> Option<usize> {
        self.registry.get_index(name)
    }

    /// Get the name of a variable by index
    #[must_use]
    pub fn get_variable_name(&self, index: usize) -> Option<&str> {
        self.registry.get_name(index)
    }
}

impl Default for ExpressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_builder_basic() {
        let mut builder = ExpressionBuilder::new();

        // Test variable creation
        let x = builder.var("x");
        let y = builder.var("y");

        // Verify they have different indices
        assert_ne!(x.variable_index(), y.variable_index());
        assert_eq!(x.variable_index(), Some(0));
        assert_eq!(y.variable_index(), Some(1));

        // Test constant creation
        let const_5 = builder.constant(5.0);
        match const_5 {
            ASTRepr::Constant(val) => assert_eq!(val, 5.0),
            _ => panic!("Expected constant"),
        }
    }

    #[test]
    fn test_expression_builder_evaluation() {
        let mut builder = ExpressionBuilder::new();

        // Create expression: x + y
        let x = builder.var("x");
        let y = builder.var("y");
        let expr = ASTRepr::Add(Box::new(x), Box::new(y));

        // Test evaluation with named variables
        let named_vars = vec![("x".to_string(), 3.0), ("y".to_string(), 4.0)];
        let result = builder.eval_with_named_vars(&expr, &named_vars);
        assert_eq!(result, 7.0);

        // Test evaluation with indexed variables
        let result2 = builder.eval_with_vars(&expr, &[3.0, 4.0]);
        assert_eq!(result2, 7.0);
    }

    #[test]
    fn test_expression_builder_variable_management() {
        let mut builder = ExpressionBuilder::new();

        // Register variables and check indices
        let x_idx = builder.register_variable("x");
        let y_idx = builder.register_variable("y");
        let x_idx_again = builder.register_variable("x"); // Should return same index

        assert_eq!(x_idx, 0);
        assert_eq!(y_idx, 1);
        assert_eq!(x_idx_again, x_idx);

        // Test lookups
        assert_eq!(builder.get_variable_index("x"), Some(x_idx));
        assert_eq!(builder.get_variable_index("y"), Some(y_idx));
        assert_eq!(builder.get_variable_index("z"), None);

        assert_eq!(builder.get_variable_name(x_idx), Some("x"));
        assert_eq!(builder.get_variable_name(y_idx), Some("y"));
        assert_eq!(builder.get_variable_name(99), None);

        // Test variable count
        assert_eq!(builder.num_variables(), 2);

        // Test variable names
        let names = builder.variable_names();
        assert_eq!(names, &["x", "y"]);
    }

    #[test]
    fn test_expression_builder_complex_expression() {
        let mut builder = ExpressionBuilder::new();

        // Build expression: 2*x + 3*y + 1
        let x = builder.var("x");
        let y = builder.var("y");
        let two = builder.constant(2.0);
        let three = builder.constant(3.0);
        let one = builder.constant(1.0);

        let two_x = ASTRepr::Mul(Box::new(two), Box::new(x));
        let three_y = ASTRepr::Mul(Box::new(three), Box::new(y));
        let sum = ASTRepr::Add(Box::new(two_x), Box::new(three_y));
        let expr = ASTRepr::Add(Box::new(sum), Box::new(one));

        // Test evaluation
        let named_vars = vec![("x".to_string(), 2.0), ("y".to_string(), 3.0)];
        let result = builder.eval_with_named_vars(&expr, &named_vars);
        assert_eq!(result, 14.0); // 2*2 + 3*3 + 1 = 4 + 9 + 1 = 14
    }

    #[test]
    fn test_expression_builder_var_by_index() {
        let mut builder = ExpressionBuilder::new();

        // Register some variables first
        let x_idx = builder.register_variable("x");
        let y_idx = builder.register_variable("y");

        // Create variables by index
        let x_by_idx = builder.var_by_index(x_idx);
        let y_by_idx = builder.var_by_index(y_idx);

        assert_eq!(x_by_idx.variable_index(), Some(x_idx));
        assert_eq!(y_by_idx.variable_index(), Some(y_idx));

        // Test evaluation
        let expr = ASTRepr::Add(Box::new(x_by_idx), Box::new(y_by_idx));
        let result = builder.eval_with_vars(&expr, &[5.0, 7.0]);
        assert_eq!(result, 12.0);
    }

    #[test]
    fn test_expression_builder_registry_access() {
        let mut builder = ExpressionBuilder::new();

        // Add some variables
        builder.var("x");
        builder.var("y");

        // Test registry access
        let registry = builder.registry();
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.get_index("x"), Some(0));
        assert_eq!(registry.get_index("y"), Some(1));

        // Test mutable registry access
        let registry_mut = builder.registry_mut();
        let z_idx = registry_mut.register_variable("z");
        assert_eq!(z_idx, 2);
        assert_eq!(builder.num_variables(), 3);
    }
}
