//! Typed Variable Registry for Mathematical Expressions
//!
//! This module provides enhanced variable management with compile-time type safety.
//! It extends the existing variable registry with type information while maintaining
//! full backward compatibility.

use super::registry::VariableRegistry;
use crate::final_tagless::traits::NumericType;
use std::any::TypeId;
use std::collections::HashMap;
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
    pub fn from_type<T: NumericType + 'static>() -> Self {
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

    /// Check if this type can be automatically promoted to another type
    #[must_use]
    pub fn can_promote_to(&self, other: &TypeCategory) -> bool {
        match (self, other) {
            // Same category and type
            (a, b) if a == b => true,

            // Integer to Float promotions - only to f64 to avoid precision loss
            (TypeCategory::Int(_), TypeCategory::Float(to_id)) => *to_id == TypeId::of::<f64>(),
            (TypeCategory::UInt(_), TypeCategory::Float(to_id)) => *to_id == TypeId::of::<f64>(),

            // Float size promotions (f32 -> f64)
            (TypeCategory::Float(from_id), TypeCategory::Float(to_id)) => {
                // f32 can promote to f64
                *from_id == TypeId::of::<f32>() && *to_id == TypeId::of::<f64>()
            }

            // No other automatic promotions
            _ => false,
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
#[derive(Debug, Clone)]
pub struct TypedVar<T> {
    index: usize,
    name: String,
    _phantom: PhantomData<T>,
}

impl<T> TypedVar<T> {
    /// Create a new typed variable
    #[must_use]
    pub fn new(index: usize, name: String) -> Self {
        Self {
            index,
            name,
            _phantom: PhantomData,
        }
    }

    /// Get the variable index
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the variable name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Variable information combining index and type
#[derive(Debug, Clone)]
struct VariableInfo {
    index: usize,
    type_info: TypeCategory,
}

/// Enhanced variable registry that tracks types while maintaining backward compatibility
#[derive(Debug, Clone)]
pub struct TypedVariableRegistry {
    /// The underlying untyped registry for backward compatibility
    base_registry: VariableRegistry,
    /// Type information for each variable
    variable_types: HashMap<String, TypeCategory>,
    /// Reverse mapping from index to type info
    index_to_type: Vec<Option<TypeCategory>>,
}

impl TypedVariableRegistry {
    /// Create a new typed variable registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_registry: VariableRegistry::new(),
            variable_types: HashMap::new(),
            index_to_type: Vec::new(),
        }
    }

    /// Register a typed variable and return a `TypedVar`
    pub fn register_typed_variable<T: NumericType + 'static>(&mut self, name: &str) -> TypedVar<T> {
        let type_info = TypeCategory::from_type::<T>();

        // Check if variable already exists with compatible type
        if let Some(existing_type) = self.variable_types.get(name) {
            if existing_type == &type_info {
                // Same type, return existing
                let index = self.base_registry.get_index(name).unwrap();
                return TypedVar::new(index, name.to_string());
            }
            // Different type - this is a type error
            panic!(
                "Variable '{}' already registered with type '{}', cannot re-register as '{}'",
                name,
                existing_type.as_str(),
                type_info.as_str()
            );
        }

        // Register new variable
        let index = self.base_registry.register_variable(name);
        self.variable_types
            .insert(name.to_string(), type_info.clone());

        // Extend index_to_type if needed
        while self.index_to_type.len() <= index {
            self.index_to_type.push(None);
        }
        self.index_to_type[index] = Some(type_info);

        TypedVar::new(index, name.to_string())
    }

    /// Register an untyped variable (backward compatibility) - defaults to f64
    pub fn register_variable(&mut self, name: &str) -> usize {
        // If already registered as typed, return existing index
        if let Some(index) = self.base_registry.get_index(name) {
            return index;
        }

        // Register as f64 by default
        let typed_var = self.register_typed_variable::<f64>(name);
        typed_var.index()
    }

    /// Get the type information for a variable by name
    #[must_use]
    pub fn get_variable_type(&self, name: &str) -> Option<&TypeCategory> {
        self.variable_types.get(name)
    }

    /// Get the type information for a variable by index
    #[must_use]
    pub fn get_type_by_index(&self, index: usize) -> Option<&TypeCategory> {
        self.index_to_type.get(index).and_then(|opt| opt.as_ref())
    }

    /// Check if two variables have compatible types for operations
    #[must_use]
    pub fn are_types_compatible(&self, name1: &str, name2: &str) -> bool {
        match (self.get_variable_type(name1), self.get_variable_type(name2)) {
            (Some(type1), Some(type2)) => {
                type1 == type2 || type1.can_promote_to(type2) || type2.can_promote_to(type1)
            }
            _ => false,
        }
    }

    /// Get all variables of a specific type
    #[must_use]
    pub fn get_variables_of_type(&self, target_type: &TypeCategory) -> Vec<String> {
        self.variable_types
            .iter()
            .filter(|(_, var_type)| *var_type == target_type)
            .map(|(name, _)| name.clone())
            .collect()
    }

    // Delegate to base registry for backward compatibility

    /// Get the index for a variable name
    #[must_use]
    pub fn get_index(&self, name: &str) -> Option<usize> {
        self.base_registry.get_index(name)
    }

    /// Get the name for a variable index
    #[must_use]
    pub fn get_name(&self, index: usize) -> Option<&str> {
        self.base_registry.get_name(index)
    }

    /// Get all registered variable names
    #[must_use]
    pub fn get_all_names(&self) -> &[String] {
        self.base_registry.get_all_names()
    }

    /// Get the number of registered variables
    #[must_use]
    pub fn len(&self) -> usize {
        self.base_registry.len()
    }

    /// Check if the registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.base_registry.is_empty()
    }

    /// Clear all registered variables
    pub fn clear(&mut self) {
        self.base_registry.clear();
        self.variable_types.clear();
        self.index_to_type.clear();
    }

    /// Create a variable mapping for evaluation (backward compatibility)
    #[must_use]
    pub fn create_variable_map(&self, values: &[(String, f64)]) -> Vec<f64> {
        self.base_registry.create_variable_map(values)
    }

    /// Get the underlying untyped registry
    #[must_use]
    pub fn base_registry(&self) -> &VariableRegistry {
        &self.base_registry
    }
}

impl Default for TypedVariableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_info_creation() {
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
    fn test_type_promotion() {
        let f32_type = TypeCategory::Float(TypeId::of::<f32>());
        let f64_type = TypeCategory::Float(TypeId::of::<f64>());
        let i32_type = TypeCategory::Int(TypeId::of::<i32>());

        assert!(f32_type.can_promote_to(&f64_type));
        assert!(i32_type.can_promote_to(&f64_type));
        assert!(!f64_type.can_promote_to(&f32_type));
    }

    #[test]
    fn test_typed_variable_registration() {
        let mut registry = TypedVariableRegistry::new();

        // Register typed variables
        let x: TypedVar<f64> = registry.register_typed_variable("x");
        let y: TypedVar<f32> = registry.register_typed_variable("y");

        assert_eq!(x.name(), "x");
        assert_eq!(y.name(), "y");
        assert_ne!(x.index(), y.index());

        // Check type information
        assert_eq!(
            registry.get_variable_type("x"),
            Some(&TypeCategory::Float(TypeId::of::<f64>()))
        );
        assert_eq!(
            registry.get_variable_type("y"),
            Some(&TypeCategory::Float(TypeId::of::<f32>()))
        );
    }

    #[test]
    fn test_backward_compatibility() {
        let mut registry = TypedVariableRegistry::new();

        // Register untyped variable (should default to f64)
        let x_index = registry.register_variable("x");

        // Should be registered as f64
        assert_eq!(
            registry.get_variable_type("x"),
            Some(&TypeCategory::Float(TypeId::of::<f64>()))
        );
        assert_eq!(registry.get_index("x"), Some(x_index));
    }

    #[test]
    #[should_panic(expected = "already registered with type")]
    fn test_type_conflict() {
        let mut registry = TypedVariableRegistry::new();

        // Register as f64
        let _x_f64: TypedVar<f64> = registry.register_typed_variable("x");

        // Try to register same name as f32 - should panic
        let _x_f32: TypedVar<f32> = registry.register_typed_variable("x");
    }

    #[test]
    fn test_type_compatibility() {
        let mut registry = TypedVariableRegistry::new();

        let _x: TypedVar<f64> = registry.register_typed_variable("x");
        let _y: TypedVar<f32> = registry.register_typed_variable("y");
        let _z: TypedVar<i32> = registry.register_typed_variable("z");

        // f32 and f64 should be compatible (f32 can promote to f64)
        assert!(registry.are_types_compatible("x", "y"));

        // i32 and f64 should be compatible (i32 can promote to f64)
        assert!(registry.are_types_compatible("x", "z"));

        // f32 and i32 should not be directly compatible
        assert!(!registry.are_types_compatible("y", "z"));
    }

    #[test]
    fn test_variables_by_type() {
        let mut registry = TypedVariableRegistry::new();

        let _x1: TypedVar<f64> = registry.register_typed_variable("x1");
        let _x2: TypedVar<f64> = registry.register_typed_variable("x2");
        let _y1: TypedVar<f32> = registry.register_typed_variable("y1");

        let f64_vars = registry.get_variables_of_type(&TypeCategory::Float(TypeId::of::<f64>()));
        let f32_vars = registry.get_variables_of_type(&TypeCategory::Float(TypeId::of::<f32>()));

        assert_eq!(f64_vars.len(), 2);
        assert!(f64_vars.contains(&"x1".to_string()));
        assert!(f64_vars.contains(&"x2".to_string()));

        assert_eq!(f32_vars.len(), 1);
        assert!(f32_vars.contains(&"y1".to_string()));
    }
}
