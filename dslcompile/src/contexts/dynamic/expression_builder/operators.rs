//! Operator Overloading for DSLCompile Expression Types
//!
//! This module contains all arithmetic operator implementations for VariableExpr and TypedBuilderExpr,
//! including operations between same types, cross-type operations, reference operations, and scalar operations.
//!
//! ## Key Components
//!
//! - **VariableExpr Operations**: Automatic conversion to TypedBuilderExpr
//! - **TypedBuilderExpr Operations**: Direct AST manipulation
//! - **Reference Operations**: Efficient operations with borrowed values
//! - **Scalar Operations**: Operations between expressions and scalar values
//! - **Cross-Type Operations**: Type-safe operations between different numeric types

use crate::{
    ast::{Scalar, ast_repr::ASTRepr},
    contexts::dynamic::expression_builder::{TypedBuilderExpr, VariableExpr},
};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// OPERATOR OVERLOADING FOR VariableExpr - AUTOMATIC CONVERSION
// ============================================================================

// Arithmetic operations for VariableExpr - automatically convert to TypedBuilderExpr
impl<T> Add for VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.into_expr() + rhs.into_expr()
    }
}

impl<T> Add<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() + rhs.clone().into_expr()
    }
}

impl<T> Add<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() + rhs.into_expr()
    }
}

impl<T> Add<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr() + rhs.clone().into_expr()
    }
}

impl<T> Mul for VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.into_expr() * rhs.into_expr()
    }
}

impl<T> Mul<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() * rhs.clone().into_expr()
    }
}

impl<T> Mul<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() * rhs.into_expr()
    }
}

impl<T> Mul<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr() * rhs.clone().into_expr()
    }
}

impl<T> Sub for VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.into_expr() - rhs.into_expr()
    }
}

impl<T> Sub<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() - rhs.clone().into_expr()
    }
}

impl<T> Sub<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr() - rhs.into_expr()
    }
}

impl<T> Sub<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr() - rhs.clone().into_expr()
    }
}

impl<T> Neg for VariableExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        -self.into_expr()
    }
}

impl<T> Neg for &VariableExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        -self.clone().into_expr()
    }
}

// ============================================================================
// SCALAR OPERATIONS FOR VariableExpr - SPECIFIC IMPLEMENTATIONS
// ============================================================================

impl Add<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: f64) -> Self::Output {
        self.into_expr() + rhs
    }
}

impl Add<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: VariableExpr<f64>) -> Self::Output {
        self + rhs.into_expr()
    }
}

impl Mul<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: f64) -> Self::Output {
        self.into_expr() * rhs
    }
}

impl Mul<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: VariableExpr<f64>) -> Self::Output {
        self * rhs.into_expr()
    }
}

impl Sub<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn sub(self, rhs: f64) -> Self::Output {
        self.into_expr() - rhs
    }
}

impl Sub<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn sub(self, rhs: VariableExpr<f64>) -> Self::Output {
        self - rhs.into_expr()
    }
}

impl Div<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn div(self, rhs: f64) -> Self::Output {
        self.into_expr() / rhs
    }
}

impl Div<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn div(self, rhs: VariableExpr<f64>) -> Self::Output {
        self / rhs.into_expr()
    }
}

// ============================================================================
// CROSS-TYPE OPERATIONS FOR VariableExpr
// ============================================================================
// Note: Cross-type operations removed by design - use explicit conversions instead
// This follows Rust's philosophy of explicit type conversions
// Example: x_f64 + y_f32.into() or x_f64 + TypedBuilderExpr::<f64>::from(y_f32)

// ============================================================================
// SAME-TYPE ARITHMETIC OPERATIONS FOR TypedBuilderExpr
// ============================================================================

impl<T> Add for TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast + rhs.ast, self.registry)
    }
}

impl<T> Mul for TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast * rhs.ast, self.registry)
    }
}

impl<T> Sub for TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast - rhs.ast, self.registry)
    }
}

impl<T> Div for TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast / rhs.ast, self.registry)
    }
}

impl<T> Neg for TypedBuilderExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-self.ast, self.registry)
    }
}

// ============================================================================
// SCALAR OPERATIONS FOR TypedBuilderExpr - MACRO GENERATED
// ============================================================================

/// Macro to generate scalar operations for TypedBuilderExpr
///
/// This generates all combinations of:
/// - TypedBuilderExpr<T> op scalar
/// - scalar op TypedBuilderExpr<T>  
/// - &TypedBuilderExpr<T> op scalar
/// - scalar op &TypedBuilderExpr<T>
///
/// For operations: Add, Sub, Mul, Div
macro_rules! impl_scalar_ops {
    ($expr_type:ty, $scalar:ty) => {
        // TypedBuilderExpr<T> + scalar
        impl Add<$scalar> for $expr_type {
            type Output = $expr_type;

            fn add(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(self.ast + ASTRepr::Constant(rhs), self.registry)
            }
        }

        // scalar + TypedBuilderExpr<T>
        impl Add<$expr_type> for $scalar {
            type Output = $expr_type;

            fn add(self, rhs: $expr_type) -> Self::Output {
                TypedBuilderExpr::new(ASTRepr::Constant(self) + rhs.ast, rhs.registry)
            }
        }

        // &TypedBuilderExpr<T> + scalar
        impl Add<$scalar> for &$expr_type {
            type Output = $expr_type;

            fn add(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(
                    self.ast.clone() + ASTRepr::Constant(rhs),
                    self.registry.clone(),
                )
            }
        }

        // scalar + &TypedBuilderExpr<T>
        impl Add<&$expr_type> for $scalar {
            type Output = $expr_type;

            fn add(self, rhs: &$expr_type) -> Self::Output {
                TypedBuilderExpr::new(
                    ASTRepr::Constant(self) + rhs.ast.clone(),
                    rhs.registry.clone(),
                )
            }
        }

        // TypedBuilderExpr<T> - scalar
        impl Sub<$scalar> for $expr_type {
            type Output = $expr_type;

            fn sub(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(self.ast - ASTRepr::Constant(rhs), self.registry)
            }
        }

        // scalar - TypedBuilderExpr<T>
        impl Sub<$expr_type> for $scalar {
            type Output = $expr_type;

            fn sub(self, rhs: $expr_type) -> Self::Output {
                TypedBuilderExpr::new(ASTRepr::Constant(self) - rhs.ast, rhs.registry)
            }
        }

        // &TypedBuilderExpr<T> - scalar
        impl Sub<$scalar> for &$expr_type {
            type Output = $expr_type;

            fn sub(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(
                    self.ast.clone() - ASTRepr::Constant(rhs),
                    self.registry.clone(),
                )
            }
        }

        // scalar - &TypedBuilderExpr<T>
        impl Sub<&$expr_type> for $scalar {
            type Output = $expr_type;

            fn sub(self, rhs: &$expr_type) -> Self::Output {
                TypedBuilderExpr::new(
                    ASTRepr::Constant(self) - rhs.ast.clone(),
                    rhs.registry.clone(),
                )
            }
        }

        // TypedBuilderExpr<T> * scalar
        impl Mul<$scalar> for $expr_type {
            type Output = $expr_type;

            fn mul(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(self.ast * ASTRepr::Constant(rhs), self.registry)
            }
        }

        // scalar * TypedBuilderExpr<T>
        impl Mul<$expr_type> for $scalar {
            type Output = $expr_type;

            fn mul(self, rhs: $expr_type) -> Self::Output {
                TypedBuilderExpr::new(ASTRepr::Constant(self) * rhs.ast, rhs.registry)
            }
        }

        // &TypedBuilderExpr<T> * scalar
        impl Mul<$scalar> for &$expr_type {
            type Output = $expr_type;

            fn mul(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(
                    self.ast.clone() * ASTRepr::Constant(rhs),
                    self.registry.clone(),
                )
            }
        }

        // scalar * &TypedBuilderExpr<T>
        impl Mul<&$expr_type> for $scalar {
            type Output = $expr_type;

            fn mul(self, rhs: &$expr_type) -> Self::Output {
                TypedBuilderExpr::new(
                    ASTRepr::Constant(self) * rhs.ast.clone(),
                    rhs.registry.clone(),
                )
            }
        }

        // TypedBuilderExpr<T> / scalar
        impl Div<$scalar> for $expr_type {
            type Output = $expr_type;

            fn div(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(self.ast / ASTRepr::Constant(rhs), self.registry)
            }
        }

        // scalar / TypedBuilderExpr<T>
        impl Div<$expr_type> for $scalar {
            type Output = $expr_type;

            fn div(self, rhs: $expr_type) -> Self::Output {
                TypedBuilderExpr::new(
                    ASTRepr::Div(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast)),
                    rhs.registry,
                )
            }
        }

        // &TypedBuilderExpr<T> / scalar
        impl Div<$scalar> for &$expr_type {
            type Output = $expr_type;

            fn div(self, rhs: $scalar) -> Self::Output {
                TypedBuilderExpr::new(
                    self.ast.clone() / ASTRepr::Constant(rhs),
                    self.registry.clone(),
                )
            }
        }

        // scalar / &TypedBuilderExpr<T>
        impl Div<&$expr_type> for $scalar {
            type Output = $expr_type;

            fn div(self, rhs: &$expr_type) -> Self::Output {
                TypedBuilderExpr::new(
                    ASTRepr::Div(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast.clone())),
                    rhs.registry.clone(),
                )
            }
        }
    };
}

// Generate scalar operations for the most common types
impl_scalar_ops!(TypedBuilderExpr<f64>, f64);
impl_scalar_ops!(TypedBuilderExpr<f32>, f32);
impl_scalar_ops!(TypedBuilderExpr<i32>, i32);
impl_scalar_ops!(TypedBuilderExpr<i64>, i64);
impl_scalar_ops!(TypedBuilderExpr<u32>, u32);
impl_scalar_ops!(TypedBuilderExpr<u64>, u64);
impl_scalar_ops!(TypedBuilderExpr<usize>, usize);

// ============================================================================
// REFERENCE OPERATIONS FOR TypedBuilderExpr
// ============================================================================

impl<T> Add<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + &rhs.ast, self.registry.clone())
    }
}

impl<T> Add<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + rhs.ast, self.registry.clone())
    }
}

impl<T> Add<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast + &rhs.ast, self.registry)
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * &rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast * &rhs.ast, self.registry)
    }
}

impl<T> Sub<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - &rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast - &rhs.ast, self.registry)
    }
}

impl<T> Div<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / &rhs.ast, self.registry.clone())
    }
}

impl<T> Div<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / rhs.ast, self.registry.clone())
    }
}

impl<T> Div<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast / &rhs.ast, self.registry)
    }
}

// Note: Scalar operations with references removed - handled by generic implementations
// Users can rely on From<f64> for TypedBuilderExpr<f64> and generic operators

// Add missing negation operator for references
impl<T> Neg for &TypedBuilderExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-&self.ast, self.registry.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contexts::dynamic::expression_builder::DynamicContext;

    #[test]
    fn test_variable_expr_arithmetic() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let y = ctx.var();

        // Test basic arithmetic operations
        let sum: TypedBuilderExpr<f64> = x.clone() + y.clone();
        let product: TypedBuilderExpr<f64> = x.clone() * y.clone();
        let difference: TypedBuilderExpr<f64> = x.clone() - y.clone();
        let negation: TypedBuilderExpr<f64> = -x.clone();

        // These should all be TypedBuilderExpr instances
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(negation.as_ast(), ASTRepr::Neg(_)));
    }

    #[test]
    fn test_variable_expr_scalar_operations() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();

        // Test scalar operations
        let sum: TypedBuilderExpr<f64> = x.clone() + 5.0;
        let product: TypedBuilderExpr<f64> = x.clone() * 2.0;
        let difference: TypedBuilderExpr<f64> = x.clone() - 1.0;
        let quotient: TypedBuilderExpr<f64> = x.clone() / 3.0;

        // Verify AST structure
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
    }

    #[test]
    fn test_typed_builder_expr_arithmetic() {
        let mut ctx = DynamicContext::<f64>::new();
        let x: TypedBuilderExpr<f64> = ctx.var().into_expr();
        let y: TypedBuilderExpr<f64> = ctx.var().into_expr();

        // Test arithmetic operations
        let sum: TypedBuilderExpr<f64> = x.clone() + y.clone();
        let product: TypedBuilderExpr<f64> = x.clone() * y.clone();
        let difference: TypedBuilderExpr<f64> = x.clone() - y.clone();
        let quotient: TypedBuilderExpr<f64> = x.clone() / y.clone();
        let negation: TypedBuilderExpr<f64> = -x.clone();

        // Verify AST structure
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
        assert!(matches!(negation.as_ast(), ASTRepr::Neg(_)));
    }

    #[test]
    fn test_reference_operations() {
        let mut ctx = DynamicContext::<f64>::new();
        let x: TypedBuilderExpr<f64> = ctx.var().into_expr();
        let y: TypedBuilderExpr<f64> = ctx.var().into_expr();

        // Test reference operations
        let sum = &x + &y;
        let difference = &x - &y;
        let quotient = &x / &y;

        // Verify AST structure
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
    }

    #[test]
    fn test_scalar_commutative_operations() {
        let mut ctx = DynamicContext::<f64>::new();
        let x: TypedBuilderExpr<f64> = ctx.var().into_expr();

        // Test commutative operations
        let sum1: TypedBuilderExpr<f64> = x.clone() + 5.0;
        let sum2: TypedBuilderExpr<f64> = 5.0 + x.clone();
        let product1: TypedBuilderExpr<f64> = x.clone() * 2.0;
        let product2: TypedBuilderExpr<f64> = 2.0 * x.clone();

        // Both should create valid AST structures
        assert!(matches!(sum1.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(sum2.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product1.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(product2.as_ast(), ASTRepr::Mul(_, _)));
    }
}
