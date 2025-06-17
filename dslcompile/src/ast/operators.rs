//! Operator Overloading for `ASTRepr`<T>
//!
//! This module provides natural mathematical syntax for building AST expressions
//! through operator overloading. It now uses a unified AsRef-based approach
//! that supports all reference patterns with fewer implementations per operator.

use crate::ast::{ASTRepr, Scalar};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// Unified Operators for ASTRepr
// ============================================================================

/// Addition operator for owned `ASTRepr`
impl<T> Add for ASTRepr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = ASTRepr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        ASTRepr::add_binary(self, rhs)
    }
}

/// Addition operator for references using `AsRef` pattern
impl<T, R> Add<R> for &ASTRepr<T>
where
    T: Scalar + Add<Output = T>,
    R: AsRef<ASTRepr<T>>,
{
    type Output = ASTRepr<T>;

    fn add(self, rhs: R) -> Self::Output {
        ASTRepr::add_binary(self.clone(), rhs.as_ref().clone())
    }
}

/// Addition operator for mixed types (owned + reference)
impl<T> Add<&ASTRepr<T>> for ASTRepr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = ASTRepr<T>;

    fn add(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::add_binary(self, rhs.clone())
    }
}

/// Subtraction operator for owned `ASTRepr`
impl<T> Sub for ASTRepr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = ASTRepr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        ASTRepr::Sub(Box::new(self), Box::new(rhs))
    }
}

/// Subtraction operator for references using `AsRef` pattern
impl<T, R> Sub<R> for &ASTRepr<T>
where
    T: Scalar + Sub<Output = T>,
    R: AsRef<ASTRepr<T>>,
{
    type Output = ASTRepr<T>;

    fn sub(self, rhs: R) -> Self::Output {
        ASTRepr::Sub(Box::new(self.clone()), Box::new(rhs.as_ref().clone()))
    }
}

/// Subtraction operator for mixed types (owned + reference)
impl<T> Sub<&ASTRepr<T>> for ASTRepr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = ASTRepr<T>;

    fn sub(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Sub(Box::new(self), Box::new(rhs.clone()))
    }
}

/// Multiplication operator for owned `ASTRepr`
impl<T> Mul for ASTRepr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = ASTRepr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        ASTRepr::mul_binary(self, rhs)
    }
}

/// Multiplication operator for references using `AsRef` pattern
impl<T, R> Mul<R> for &ASTRepr<T>
where
    T: Scalar + Mul<Output = T>,
    R: AsRef<ASTRepr<T>>,
{
    type Output = ASTRepr<T>;

    fn mul(self, rhs: R) -> Self::Output {
        ASTRepr::mul_binary(self.clone(), rhs.as_ref().clone())
    }
}

/// Multiplication operator for mixed types (owned + reference)
impl<T> Mul<&ASTRepr<T>> for ASTRepr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = ASTRepr<T>;

    fn mul(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::mul_binary(self, rhs.clone())
    }
}

/// Division operator for owned `ASTRepr`
impl<T> Div for ASTRepr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = ASTRepr<T>;

    fn div(self, rhs: Self) -> Self::Output {
        ASTRepr::Div(Box::new(self), Box::new(rhs))
    }
}

/// Division operator for references using `AsRef` pattern
impl<T, R> Div<R> for &ASTRepr<T>
where
    T: Scalar + Div<Output = T>,
    R: AsRef<ASTRepr<T>>,
{
    type Output = ASTRepr<T>;

    fn div(self, rhs: R) -> Self::Output {
        ASTRepr::Div(Box::new(self.clone()), Box::new(rhs.as_ref().clone()))
    }
}

/// Division operator for mixed types (owned + reference)
impl<T> Div<&ASTRepr<T>> for ASTRepr<T>
where
    T: Scalar + Div<Output = T>,
{
    type Output = ASTRepr<T>;

    fn div(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Div(Box::new(self), Box::new(rhs.clone()))
    }
}

/// Negation operator for owned `ASTRepr`
impl<T> Neg for ASTRepr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = ASTRepr<T>;

    fn neg(self) -> Self::Output {
        ASTRepr::Neg(Box::new(self))
    }
}

/// Negation operator for references
impl<T> Neg for &ASTRepr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = ASTRepr<T>;

    fn neg(self) -> Self::Output {
        ASTRepr::Neg(Box::new(self.clone()))
    }
}

// ============================================================================
// AsRef Implementations for ASTRepr
// ============================================================================

impl<T> AsRef<ASTRepr<T>> for ASTRepr<T> {
    fn as_ref(&self) -> &ASTRepr<T> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_operator_overloading() {
        // Test with f64
        let x_f64 = ASTRepr::<f64>::Variable(0);
        let y_f64 = ASTRepr::<f64>::Variable(1);
        let const_f64 = ASTRepr::<f64>::Constant(2.5);

        // All reference patterns should work
        let expr1 = &x_f64 + &y_f64; // both borrowed
        let expr2 = x_f64.clone() + &y_f64; // mixed: owned + borrowed
        let expr3 = &x_f64 + y_f64.clone(); // mixed: borrowed + owned
        let expr4 = x_f64.clone() + y_f64.clone(); // both owned

        // Verify structure
        assert_eq!(expr1.count_operations(), 1);
        assert_eq!(expr2.count_operations(), 1);
        assert_eq!(expr3.count_operations(), 1);
        assert_eq!(expr4.count_operations(), 1);

        // Test complex expression: 2.5 * x + y
        let expr_complex = &const_f64 * &x_f64 + &y_f64;
        assert_eq!(expr_complex.count_operations(), 2);
    }

    #[test]
    fn test_unified_operators_all_types() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let two = ASTRepr::<f64>::Constant(2.0);

        // Test all operators work with unified implementation
        let add_expr = &x + &y;
        let sub_expr = &x - &y;
        let mul_expr = &x * &two;
        let div_expr = &x / &two;
        let neg_expr = -&x;

        // Verify correct AST structure
        match add_expr {
            ASTRepr::Add(_) => {}
            _ => panic!("Expected Add"),
        }
        match sub_expr {
            ASTRepr::Sub(_, _) => {}
            _ => panic!("Expected Sub"),
        }
        match mul_expr {
            ASTRepr::Mul(_) => {}
            _ => panic!("Expected Mul"),
        }
        match div_expr {
            ASTRepr::Div(_, _) => {}
            _ => panic!("Expected Div"),
        }
        match neg_expr {
            ASTRepr::Neg(_) => {}
            _ => panic!("Expected Neg"),
        }
    }

    #[test]
    fn test_complex_expression_building() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let two = ASTRepr::<f64>::Constant(2.0);
        let three = ASTRepr::<f64>::Constant(3.0);

        // Build: 2*x + 3*y using unified operators
        let expr = &two * &x + &three * &y;
        assert_eq!(expr.count_operations(), 3); // two muls, one add

        // Test with negation: -(2*x + 3*y)
        let neg_expr = -(&two * &x + &three * &y);
        assert_eq!(neg_expr.count_operations(), 4); // two muls, one add, one neg
    }

    #[test]
    fn test_generic_numeric_types() {
        // Test with f32
        let x_f32 = ASTRepr::<f32>::Variable(0);
        let y_f32 = ASTRepr::<f32>::Variable(1);
        let const_f32 = ASTRepr::<f32>::Constant(2.5_f32);

        let expr_f32 = &x_f32 + &y_f32 * &const_f32;
        assert_eq!(expr_f32.count_operations(), 2);

        // Test with i32 (if it implements Scalar)
        let x_i32 = ASTRepr::<i32>::Variable(0);
        let const_i32 = ASTRepr::<i32>::Constant(42);

        let expr_i32 = &x_i32 + &const_i32;
        assert_eq!(expr_i32.count_operations(), 1);
    }
}
