//! Index-Only Variable Registry for Mathematical Expressions
//!
//! This module provides high-performance variable management using pure index-based
//! tracking with type information. No string storage or lookup overhead.

use crate::ast::Scalar;
use std::any::TypeId;
use std::marker::PhantomData;

/// Type category information for variables
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeCategory {
    Float(TypeId),
    Int(TypeId),
    UInt(TypeId),
    Custom(TypeId, String),
}

impl TypeCategory {
    /// Create `TypeCategory` from a Rust type
    #[must_use]
    pub fn from_type<T: Scalar + 'static>() -> Self {
        let type_id = TypeId::of::<T>();

        // Check if it's a float type
        if Self::is_float_type::<T>() {
            TypeCategory::Float(type_id)
        }
        // Check if it's an int type
        else if Self::is_int_type::<T>() {
            TypeCategory::Int(type_id)
        }
        // Check if it's a uint type
        else if Self::is_uint_type::<T>() {
            TypeCategory::UInt(type_id)
        } else {
            // Default to custom
            TypeCategory::Custom(type_id, std::any::type_name::<T>().to_string())
        }
    }

    /// Check if a type implements `FloatType`
    fn is_float_type<T: 'static>() -> bool {
        // Use TypeId comparison for known float types
        let type_id = TypeId::of::<T>();
        type_id == TypeId::of::<f32>() || type_id == TypeId::of::<f64>()
    }

    /// Check if a type implements `IntType`
    fn is_int_type<T: 'static>() -> bool {
        let type_id = TypeId::of::<T>();
        type_id == TypeId::of::<i32>() || type_id == TypeId::of::<i64>()
    }

    /// Check if a type implements `UIntType`
    fn is_uint_type<T: 'static>() -> bool {
        let type_id = TypeId::of::<T>();
        type_id == TypeId::of::<u32>() || type_id == TypeId::of::<u64>()
    }

    /// Get the string representation of this type category
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            TypeCategory::Float(_) => "Float",
            TypeCategory::Int(_) => "Int",
            TypeCategory::UInt(_) => "UInt",
            TypeCategory::Custom(_, name) => name,
        }
    }

    /// Get type ID for this category
    #[must_use]
    pub fn type_id(&self) -> TypeId {
        match self {
            TypeCategory::Float(id) => *id,
            TypeCategory::Int(id) => *id,
            TypeCategory::UInt(id) => *id,
            TypeCategory::Custom(id, _) => *id,
        }
    }
}

/// A typed variable reference that carries type information at compile time
/// Static with type-level scoping for predictable variable indexing
#[derive(Debug, Clone)]
pub struct TypedVar<T> {
    /// Unique variable ID - predictable and stable across contexts
    id: usize,
    /// Registry index - runtime position in the variable registry
    index: usize,
    _phantom: PhantomData<T>,
}

impl<T> TypedVar<T> {
    /// Create a new typed variable with both ID and index
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self {
            id: index, // For now, ID equals index for backward compatibility
            index,
            _phantom: PhantomData,
        }
    }

    /// Create a new typed variable with explicit ID and index
    /// This enables type-level scoping where ID is predictable but index may vary
    #[must_use]
    pub fn with_id(id: usize, index: usize) -> Self {
        Self {
            id,
            index,
            _phantom: PhantomData,
        }
    }

    /// Get the unique variable ID (type-level scoping)
    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the variable index (runtime registry position)
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the variable name (generated from ID for consistency)
    #[must_use]
    pub fn name(&self) -> String {
        format!("var_{}", self.id)
    }
}

/// High-performance index-only variable registry with type tracking
#[derive(Debug, Clone)]
pub struct VariableRegistry {
    /// Type information for each variable by index
    index_to_type: Vec<TypeCategory>,
}

impl VariableRegistry {
    /// Create a new variable registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            index_to_type: Vec::new(),
        }
    }

    /// Create a registry with a specified number of variables (all f64)
    /// This is useful for compatibility with expressions that use indexed variables
    #[must_use]
    pub fn with_capacity(num_vars: usize) -> Self {
        let mut registry = Self::new();
        for _ in 0..num_vars {
            registry.register_variable();
        }
        registry
    }

    /// Create a registry that can handle variables up to the specified maximum index
    /// This automatically registers variables 0, 1, 2, ..., `max_index`
    #[must_use]
    pub fn for_max_index(max_index: usize) -> Self {
        Self::with_capacity(max_index + 1)
    }

    /// Create a registry that can handle all variables used in an expression
    /// This analyzes the expression and registers the appropriate number of variables
    #[must_use]
    pub fn for_expression<T>(expr: &crate::ast::ASTRepr<T>) -> Self {
        let max_index = Self::find_max_variable_index(expr);
        match max_index {
            Some(max) => Self::for_max_index(max),
            None => Self::new(), // No variables in expression
        }
    }

    /// Find the maximum variable index used in an expression
    fn find_max_variable_index<T>(expr: &crate::ast::ASTRepr<T>) -> Option<usize> {
        match expr {
            crate::ast::ASTRepr::Variable(index) => Some(*index),
            crate::ast::ASTRepr::Constant(_) => None,
            crate::ast::ASTRepr::Add(left, right)
            | crate::ast::ASTRepr::Sub(left, right)
            | crate::ast::ASTRepr::Mul(left, right)
            | crate::ast::ASTRepr::Div(left, right)
            | crate::ast::ASTRepr::Pow(left, right) => {
                let left_max = Self::find_max_variable_index(left);
                let right_max = Self::find_max_variable_index(right);
                match (left_max, right_max) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            crate::ast::ASTRepr::Neg(inner)
            | crate::ast::ASTRepr::Sqrt(inner)
            | crate::ast::ASTRepr::Sin(inner)
            | crate::ast::ASTRepr::Cos(inner)
            | crate::ast::ASTRepr::Exp(inner)
            | crate::ast::ASTRepr::Ln(inner) => Self::find_max_variable_index(inner),
            crate::ast::ASTRepr::Sum(_collection) => {
                // TODO: Handle Collection format for max variable index finding
                None // Placeholder until Collection variable analysis is implemented
            }
        }
    }

    /// Register a typed variable and return a `TypedVar`
    pub fn register_typed_variable<T: Scalar + 'static>(&mut self) -> TypedVar<T> {
        let type_info = TypeCategory::from_type::<T>();
        let index = self.index_to_type.len();
        self.index_to_type.push(type_info);
        TypedVar::new(index)
    }

    /// Register a variable with explicit type category
    pub fn register_variable_with_type(&mut self, type_category: TypeCategory) -> usize {
        let index = self.index_to_type.len();
        self.index_to_type.push(type_category);
        index
    }

    /// Register an untyped variable (defaults to f64)
    pub fn register_variable(&mut self) -> usize {
        let typed_var = self.register_typed_variable::<f64>();
        typed_var.index()
    }

    /// Get the type information for a variable by index
    #[must_use]
    pub fn get_type_by_index(&self, index: usize) -> Option<&TypeCategory> {
        self.index_to_type.get(index)
    }

    /// Get all variables of a specific type
    #[must_use]
    pub fn get_variables_of_type(&self, target_type: &TypeCategory) -> Vec<usize> {
        self.index_to_type
            .iter()
            .enumerate()
            .filter(|(_, var_type)| *var_type == target_type)
            .map(|(index, _)| index)
            .collect()
    }

    /// Get the number of registered variables
    #[must_use]
    pub fn len(&self) -> usize {
        self.index_to_type.len()
    }

    /// Check if the registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index_to_type.is_empty()
    }

    /// Clear all registered variables
    pub fn clear(&mut self) {
        self.index_to_type.clear();
    }

    /// Create a variable mapping for evaluation from indexed values
    #[must_use]
    pub fn create_variable_map<T: Copy + Default>(&self, values: &[T]) -> Vec<T> {
        let mut result = vec![T::default(); self.len()];
        for (i, &value) in values.iter().enumerate() {
            if i < result.len() {
                result[i] = value;
            }
        }
        result
    }

    /// Generate a debug name for a variable by index
    #[must_use]
    pub fn debug_name(&self, index: usize) -> String {
        format!("var_{index}")
    }

    /// Check if two variables have compatible types
    #[must_use]
    pub fn are_types_compatible(&self, index1: usize, index2: usize) -> bool {
        use std::any::TypeId;

        let type1 = self.get_type_by_index(index1);
        let type2 = self.get_type_by_index(index2);

        match (type1, type2) {
            (Some(TypeCategory::Float(id1)), Some(TypeCategory::Float(id2))) => {
                // f32 can promote to f64
                *id1 == *id2 || (*id1 == TypeId::of::<f32>() && *id2 == TypeId::of::<f64>())
            }
            (Some(TypeCategory::Int(id1)), Some(TypeCategory::Float(id2))) => {
                // i32 can promote to f64
                *id1 == TypeId::of::<i32>() && *id2 == TypeId::of::<f64>()
            }
            (Some(TypeCategory::Float(id1)), Some(TypeCategory::Int(id2))) => {
                // i32 can promote to f64
                *id1 == TypeId::of::<f64>() && *id2 == TypeId::of::<i32>()
            }
            (Some(cat1), Some(cat2)) => cat1 == cat2,
            _ => false,
        }
    }
}

impl Default for VariableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_category_creation() {
        assert_eq!(
            TypeCategory::from_type::<f64>(),
            TypeCategory::Float(TypeId::of::<f64>())
        );
        assert_eq!(
            TypeCategory::from_type::<f32>(),
            TypeCategory::Float(TypeId::of::<f32>())
        );
        assert_eq!(
            TypeCategory::from_type::<i32>(),
            TypeCategory::Int(TypeId::of::<i32>())
        );
        assert_eq!(
            TypeCategory::from_type::<i64>(),
            TypeCategory::Int(TypeId::of::<i64>())
        );
    }

    #[test]
    fn test_typed_variable_registration() {
        let mut registry = VariableRegistry::new();

        // Register typed variables
        let x: TypedVar<f64> = registry.register_typed_variable();
        let y: TypedVar<f32> = registry.register_typed_variable();

        assert_eq!(x.name(), "var_0");
        assert_eq!(y.name(), "var_1");
        assert_ne!(x.index(), y.index());

        // Check type information
        assert_eq!(
            registry.get_type_by_index(x.index()),
            Some(&TypeCategory::Float(TypeId::of::<f64>()))
        );
        assert_eq!(
            registry.get_type_by_index(y.index()),
            Some(&TypeCategory::Float(TypeId::of::<f32>()))
        );
    }

    #[test]
    fn test_untyped_variable_registration() {
        let mut registry = VariableRegistry::new();

        // Register untyped variable (should default to f64)
        let x_index = registry.register_variable();

        // Should be registered as f64
        assert_eq!(
            registry.get_type_by_index(x_index),
            Some(&TypeCategory::Float(TypeId::of::<f64>()))
        );
    }

    #[test]
    fn test_type_compatibility() {
        let mut registry = VariableRegistry::new();

        let x: TypedVar<f64> = registry.register_typed_variable();
        let y: TypedVar<f32> = registry.register_typed_variable();
        let z: TypedVar<i32> = registry.register_typed_variable();

        // f32 and f64 should be compatible (f32 can promote to f64)
        assert!(registry.are_types_compatible(x.index(), y.index()));

        // i32 and f64 should be compatible (i32 can promote to f64)
        assert!(registry.are_types_compatible(x.index(), z.index()));

        // f32 and i32 should not be directly compatible
        assert!(!registry.are_types_compatible(y.index(), z.index()));
    }

    #[test]
    fn test_variables_by_type() {
        let mut registry = VariableRegistry::new();

        let x1: TypedVar<f64> = registry.register_typed_variable();
        let x2: TypedVar<f64> = registry.register_typed_variable();
        let y1: TypedVar<f32> = registry.register_typed_variable();

        let f64_vars = registry.get_variables_of_type(&TypeCategory::Float(TypeId::of::<f64>()));
        let f32_vars = registry.get_variables_of_type(&TypeCategory::Float(TypeId::of::<f32>()));

        assert_eq!(f64_vars.len(), 2);
        assert!(f64_vars.contains(&x1.index()));
        assert!(f64_vars.contains(&x2.index()));

        assert_eq!(f32_vars.len(), 1);
        assert!(f32_vars.contains(&y1.index()));
    }

    #[test]
    fn test_variable_map_creation() {
        let registry = VariableRegistry::new();

        // Test with f64 values
        let values = [1.0, 2.0, 3.0];
        let var_map = registry.create_variable_map(&values);

        // Should create a vector of the same length as registry (0 in this case)
        assert_eq!(var_map.len(), 0);

        // Test with registered variables
        let mut registry = VariableRegistry::new();
        let _x = registry.register_variable();
        let _y = registry.register_variable();
        let _z = registry.register_variable();

        let var_map = registry.create_variable_map(&values);
        assert_eq!(var_map.len(), 3);
        assert_eq!(var_map[0], 1.0);
        assert_eq!(var_map[1], 2.0);
        assert_eq!(var_map[2], 3.0);
    }
}
