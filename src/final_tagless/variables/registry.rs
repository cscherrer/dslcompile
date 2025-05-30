//! Variable Registry for Mathematical Expressions
//!
//! This module provides variable management for mapping between variable names and indices.
//! This allows user-facing string-based variable access while using efficient
//! indices internally for performance-critical operations.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Global variable registry for mapping between variable names and indices
/// This allows user-facing string-based variable access while using efficient
/// indices internally for performance-critical operations.
#[derive(Debug, Clone)]
pub struct VariableRegistry {
    /// Mapping from variable names to indices
    name_to_index: HashMap<String, usize>,
    /// Mapping from indices to variable names
    index_to_name: Vec<String>,
}

impl VariableRegistry {
    /// Create a new empty variable registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            name_to_index: HashMap::new(),
            index_to_name: Vec::new(),
        }
    }

    /// Register a variable name and return its index
    /// If the variable already exists, returns its existing index
    pub fn register_variable(&mut self, name: &str) -> usize {
        if let Some(&index) = self.name_to_index.get(name) {
            index
        } else {
            let index = self.index_to_name.len();
            self.name_to_index.insert(name.to_string(), index);
            self.index_to_name.push(name.to_string());
            index
        }
    }

    /// Get the index for a variable name
    #[must_use]
    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Get the name for a variable index
    #[must_use]
    pub fn get_name(&self, index: usize) -> Option<&str> {
        self.index_to_name
            .get(index)
            .map(std::string::String::as_str)
    }

    /// Get all registered variable names
    #[must_use]
    pub fn get_all_names(&self) -> &[String] {
        &self.index_to_name
    }

    /// Get the number of registered variables
    #[must_use]
    pub fn len(&self) -> usize {
        self.index_to_name.len()
    }

    /// Check if the registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index_to_name.is_empty()
    }

    /// Clear all registered variables
    pub fn clear(&mut self) {
        self.name_to_index.clear();
        self.index_to_name.clear();
    }

    /// Create a variable mapping for evaluation
    /// Maps variable names to their values for use with `eval_with_vars`
    #[must_use]
    pub fn create_variable_map(&self, values: &[(String, f64)]) -> Vec<f64> {
        let mut result = vec![0.0; self.len()];
        for (name, value) in values {
            if let Some(index) = self.get_index(name) {
                result[index] = *value;
            }
        }
        result
    }

    /// Create a variable mapping from a slice of values in name order
    /// Assumes values are provided in the same order as variable registration
    #[must_use]
    pub fn create_ordered_variable_map(&self, values: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.len()];
        for (i, &value) in values.iter().enumerate() {
            if i < result.len() {
                result[i] = value;
            }
        }
        result
    }
}

impl Default for VariableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe global variable registry
static GLOBAL_REGISTRY: std::sync::LazyLock<Arc<RwLock<VariableRegistry>>> =
    std::sync::LazyLock::new(|| Arc::new(RwLock::new(VariableRegistry::new())));

/// Get a reference to the global variable registry
pub fn global_registry() -> Arc<RwLock<VariableRegistry>> {
    GLOBAL_REGISTRY.clone()
}

/// Convenience function to register a variable globally and get its index
#[must_use]
pub fn register_variable(name: &str) -> usize {
    let registry = global_registry();
    let mut guard = registry.write().unwrap();
    guard.register_variable(name)
}

/// Convenience function to get a variable index from the global registry
#[must_use]
pub fn get_variable_index(name: &str) -> Option<usize> {
    let registry = global_registry();
    let guard = registry.read().unwrap();
    guard.get_index(name)
}

/// Convenience function to get a variable name from the global registry
#[must_use]
pub fn get_variable_name(index: usize) -> Option<String> {
    let registry = global_registry();
    let guard = registry.read().unwrap();
    guard.get_name(index).map(std::string::ToString::to_string)
}

/// Convenience function to create a variable map for evaluation
#[must_use]
pub fn create_variable_map(values: &[(String, f64)]) -> Vec<f64> {
    let registry = global_registry();
    let guard = registry.read().unwrap();
    guard.create_variable_map(values)
}

/// Clear the global variable registry (useful for testing)
pub fn clear_global_registry() {
    let registry = global_registry();
    let mut guard = registry.write().unwrap();
    guard.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_registry() {
        let mut registry = VariableRegistry::new();

        // Test variable registration
        let x_index = registry.register_variable("x");
        let y_index = registry.register_variable("y");
        let x_index_again = registry.register_variable("x"); // Should return same index

        // Check that indices are different for different variables
        assert_ne!(x_index, y_index);
        // Check that same variable returns same index
        assert_eq!(x_index_again, x_index);

        // Test lookups
        assert_eq!(registry.get_index("x"), Some(x_index));
        assert_eq!(registry.get_index("y"), Some(y_index));
        assert_eq!(registry.get_index("z"), None);

        assert_eq!(registry.get_name(x_index), Some("x"));
        assert_eq!(registry.get_name(y_index), Some("y"));
        // Test an index that shouldn't exist
        let max_index = std::cmp::max(x_index, y_index);
        assert_eq!(registry.get_name(max_index + 10), None);
    }

    #[test]
    fn test_variable_registry_performance() {
        let mut registry = VariableRegistry::new();

        // Get the starting state (should be empty for new registry)
        let start_count = registry.len();
        assert_eq!(start_count, 0); // Should start empty

        // Register many variables to test performance
        let mut indices = Vec::new();
        for i in 0..1000 {
            let var_name = format!("perf_test_var_{i}");
            let index = registry.register_variable(&var_name);
            indices.push(index);
            assert_eq!(index, i); // Should get sequential indices starting from 0
        }

        // Test lookups are fast
        for i in 0..1000 {
            let var_name = format!("perf_test_var_{i}");
            let found_index = registry.get_index(&var_name);
            assert_eq!(found_index, Some(i));

            let found_name = registry.get_name(i);
            assert_eq!(found_name, Some(var_name.as_str()));
        }

        // Test that we have exactly 1000 variables registered
        let final_count = registry.len();
        assert_eq!(final_count, 1000);
    }

    #[test]
    fn test_variable_map_creation() {
        let mut registry = VariableRegistry::new();

        // Register some variables
        let x_idx = registry.register_variable("x");
        let y_idx = registry.register_variable("y");
        let z_idx = registry.register_variable("z");

        // Create a variable map
        let named_vars = vec![
            ("x".to_string(), 1.0),
            ("y".to_string(), 2.0),
            ("z".to_string(), 3.0),
        ];
        let var_map = registry.create_variable_map(&named_vars);

        assert_eq!(var_map[x_idx], 1.0);
        assert_eq!(var_map[y_idx], 2.0);
        assert_eq!(var_map[z_idx], 3.0);
    }

    #[test]
    fn test_ordered_variable_map() {
        let mut registry = VariableRegistry::new();

        // Register variables in order
        registry.register_variable("x");
        registry.register_variable("y");
        registry.register_variable("z");

        // Create ordered map
        let values = [1.0, 2.0, 3.0];
        let var_map = registry.create_ordered_variable_map(&values);

        assert_eq!(var_map, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = VariableRegistry::new();

        // Add some variables
        registry.register_variable("x");
        registry.register_variable("y");
        assert_eq!(registry.len(), 2);

        // Clear and verify
        registry.clear();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
        assert_eq!(registry.get_index("x"), None);
    }

    #[test]
    fn test_global_registry_functions() {
        // Clear any existing state
        clear_global_registry();

        // Test global functions
        let x_idx = register_variable("global_x");
        let y_idx = register_variable("global_y");

        assert_ne!(x_idx, y_idx);
        assert_eq!(get_variable_index("global_x"), Some(x_idx));
        assert_eq!(get_variable_index("global_y"), Some(y_idx));
        assert_eq!(get_variable_name(x_idx), Some("global_x".to_string()));
        assert_eq!(get_variable_name(y_idx), Some("global_y".to_string()));

        // Test variable map creation
        let named_vars = vec![
            ("global_x".to_string(), 10.0),
            ("global_y".to_string(), 20.0),
        ];
        let var_map = create_variable_map(&named_vars);
        assert_eq!(var_map[x_idx], 10.0);
        assert_eq!(var_map[y_idx], 20.0);

        // Clean up
        clear_global_registry();
    }
}
