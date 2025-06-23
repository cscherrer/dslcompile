//! Type System Infrastructure for DSLCompile
//!
//! This module contains the core type system traits and implementations that enable
//! type-safe code generation and evaluation across different numeric types.
//!
//! ## Key Components
//!
//! - `DslType`: Core trait for scalar types that can participate in DSL expressions
//! - `DataType`: Extended trait for non-scalar types like Vec<f64> that can be used in HLists
//! - Concrete implementations for standard Rust numeric types

use crate::ast::{ExpressionType, Scalar};

// ============================================================================
// CORE TYPE SYSTEM TRAITS
// ============================================================================

/// Core trait for scalar types that can participate in DSL expressions
///
/// This trait defines the interface for types that can be used as scalars in
/// mathematical expressions, including code generation capabilities.
pub trait DslType: Scalar + ExpressionType {
    /// The native Rust type this DSL type maps to
    type Native: Copy + std::fmt::Debug + std::fmt::Display;

    /// Type identifier for code generation
    const TYPE_NAME: &'static str;

    /// Generate Rust code for addition operation
    #[must_use]
    fn codegen_add() -> &'static str {
        "+"
    }

    /// Generate Rust code for multiplication operation  
    #[must_use]
    fn codegen_mul() -> &'static str {
        "*"
    }

    /// Generate Rust code for subtraction operation
    #[must_use]
    fn codegen_sub() -> &'static str {
        "-"
    }

    /// Generate Rust code for division operation
    #[must_use]
    fn codegen_div() -> &'static str {
        "/"
    }

    /// Generate Rust code for a literal value
    fn codegen_literal(value: Self::Native) -> String;

    /// Convert to evaluation value (for runtime interpretation)
    /// Generic version - returns same type for type safety
    fn to_eval_value(value: Self::Native) -> Self::Native {
        value
    }
}

/// Extended trait for data types that can participate in evaluation but aren't scalar
///
/// This enables Vec<f64>, matrices, and other non-scalar types in `HLists` while
/// maintaining type safety and providing evaluation capabilities.
pub trait DataType: Clone + std::fmt::Debug + 'static {
    /// Type identifier for signatures
    const TYPE_NAME: &'static str;

    /// Convert to evaluation data for runtime interpretation
    /// Returns the data as a vector that can be used in data summation
    fn to_eval_data(&self) -> Vec<f64>;
}

// ============================================================================
// CONCRETE IMPLEMENTATIONS FOR STANDARD TYPES
// ============================================================================

impl DslType for f64 {
    type Native = f64;
    const TYPE_NAME: &'static str = "f64";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_f64")
    }

    // Uses default implementation which returns same type
}

impl DslType for f32 {
    type Native = f32;
    const TYPE_NAME: &'static str = "f32";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_f32")
    }

    // Uses default implementation which returns same type
}

impl DslType for i32 {
    type Native = i32;
    const TYPE_NAME: &'static str = "i32";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_i32")
    }

    // Uses default implementation which returns same type
}

impl DslType for i64 {
    type Native = i64;
    const TYPE_NAME: &'static str = "i64";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_i64")
    }

    // Uses default implementation which returns same type
}

impl DslType for usize {
    type Native = usize;
    const TYPE_NAME: &'static str = "usize";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_usize")
    }

    // Uses default implementation which returns same type
}

impl DslType for u32 {
    type Native = u32;
    const TYPE_NAME: &'static str = "u32";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_u32")
    }
}

impl DslType for u64 {
    type Native = u64;
    const TYPE_NAME: &'static str = "u64";

    fn codegen_literal(value: Self::Native) -> String {
        format!("{value}_u64")
    }
}

// ============================================================================
// DATA TYPE IMPLEMENTATIONS
// ============================================================================

impl DataType for Vec<f64> {
    const TYPE_NAME: &'static str = "Vec<f64>";

    fn to_eval_data(&self) -> Vec<f64> {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsl_type_implementations() {
        // Test f64
        assert_eq!(f64::TYPE_NAME, "f64");
        assert_eq!(f64::codegen_literal(3.14), "3.14_f64");
        assert_eq!(f64::to_eval_value(2.5), 2.5);

        // Test i32
        assert_eq!(i32::TYPE_NAME, "i32");
        assert_eq!(i32::codegen_literal(42), "42_i32");
        assert_eq!(i32::to_eval_value(10), 10);

        // Test operators
        assert_eq!(f64::codegen_add(), "+");
        assert_eq!(f64::codegen_mul(), "*");
        assert_eq!(f64::codegen_sub(), "-");
        assert_eq!(f64::codegen_div(), "/");
    }

    #[test]
    fn test_data_type_implementations() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(Vec::<f64>::TYPE_NAME, "Vec<f64>");
        assert_eq!(data.to_eval_data(), vec![1.0, 2.0, 3.0]);
    }
}
