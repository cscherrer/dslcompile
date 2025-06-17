//! Scalar Trait Definitions and Implementations
//!
//! This module contains additional scalar traits beyond the basic type system,
//! including code generation traits and float operation traits with their
//! concrete implementations for standard Rust numeric types.
//!
//! ## Key Components
//!
//! - `CodegenScalar`: Extended trait for code generation capabilities
//! - `ScalarFloat`: Float-specific mathematical operations  
//! - Concrete implementations for f64, f32, i32, i64, usize
//! - Mathematical function implementations

/// Rust-idiomatic scalar trait without 'static constraints or auto-promotion
/// Extended Scalar trait for code generation  
pub trait CodegenScalar: crate::ast::Scalar {
    /// Type identifier for code generation
    const TYPE_NAME: &'static str;

    /// Generate Rust code for a literal value
    fn codegen_literal(value: Self) -> String;
}

/// Float operations for scalar types that support them
pub trait ScalarFloat: crate::ast::Scalar + num_traits::Float {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn pow(self, exp: Self) -> Self;
}

// ============================================================================
// SCALAR IMPLEMENTATIONS (Code generation support)
// ============================================================================

impl CodegenScalar for f64 {
    const TYPE_NAME: &'static str = "f64";

    fn codegen_literal(value: Self) -> String {
        format!("{value}")
    }
}

impl ScalarFloat for f64 {
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn pow(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl CodegenScalar for f32 {
    const TYPE_NAME: &'static str = "f32";

    fn codegen_literal(value: Self) -> String {
        format!("{value}f32")
    }
}

impl ScalarFloat for f32 {
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn pow(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl CodegenScalar for i32 {
    const TYPE_NAME: &'static str = "i32";

    fn codegen_literal(value: Self) -> String {
        format!("{value}i32")
    }
}

impl CodegenScalar for i64 {
    const TYPE_NAME: &'static str = "i64";

    fn codegen_literal(value: Self) -> String {
        format!("{value}i64")
    }
}

impl CodegenScalar for usize {
    const TYPE_NAME: &'static str = "usize";

    fn codegen_literal(value: Self) -> String {
        format!("{value}usize")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_scalar_implementations() {
        // Test f64
        assert_eq!(f64::TYPE_NAME, "f64");
        assert_eq!(f64::codegen_literal(3.14), "3.14");

        // Test i32
        assert_eq!(i32::TYPE_NAME, "i32");
        assert_eq!(i32::codegen_literal(42), "42i32");

        // Test f32
        assert_eq!(f32::TYPE_NAME, "f32");
        assert_eq!(f32::codegen_literal(2.5), "2.5f32");
    }

    #[test]
    fn test_scalar_float_operations() {
        use std::f64::consts::PI;

        let x = PI / 2.0;
        assert!((x.sin() - 1.0).abs() < 1e-10);
        assert!((x.cos() - 0.0).abs() < 1e-10);

        let y = 1.0_f64;
        assert!((y.exp() - std::f64::consts::E).abs() < 1e-10);
        assert!((y.exp().ln() - 1.0).abs() < 1e-10);

        let z = 4.0_f64;
        assert!((z.sqrt() - 2.0).abs() < 1e-10);
        assert!((z.pow(0.5) - 2.0).abs() < 1e-10);
    }
}
