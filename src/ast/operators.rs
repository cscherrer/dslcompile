//! Operator Overloading for `ASTRepr`<T>
//!
//! This module provides natural mathematical syntax for building AST expressions
//! through operator overloading. It supports both owned and borrowed operations
//! for maximum flexibility.

use super::ast_repr::ASTRepr;
use crate::final_tagless::traits::NumericType;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// Addition Operators
// ============================================================================

/// Addition operator overloading for `ASTRepr<T>`
impl<T> Add for ASTRepr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = ASTRepr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        ASTRepr::Add(Box::new(self), Box::new(rhs))
    }
}

/// Addition with references
impl<T> Add<&ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = ASTRepr<T>;

    fn add(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Add(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

/// Addition with mixed references
impl<T> Add<ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = ASTRepr<T>;

    fn add(self, rhs: ASTRepr<T>) -> Self::Output {
        ASTRepr::Add(Box::new(self.clone()), Box::new(rhs))
    }
}

impl<T> Add<&ASTRepr<T>> for ASTRepr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = ASTRepr<T>;

    fn add(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Add(Box::new(self), Box::new(rhs.clone()))
    }
}

// ============================================================================
// Subtraction Operators
// ============================================================================

/// Subtraction operator overloading for `ASTRepr<T>`
impl<T> Sub for ASTRepr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = ASTRepr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        ASTRepr::Sub(Box::new(self), Box::new(rhs))
    }
}

/// Subtraction with references
impl<T> Sub<&ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = ASTRepr<T>;

    fn sub(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Sub(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

/// Subtraction with mixed references
impl<T> Sub<ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = ASTRepr<T>;

    fn sub(self, rhs: ASTRepr<T>) -> Self::Output {
        ASTRepr::Sub(Box::new(self.clone()), Box::new(rhs))
    }
}

impl<T> Sub<&ASTRepr<T>> for ASTRepr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = ASTRepr<T>;

    fn sub(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Sub(Box::new(self), Box::new(rhs.clone()))
    }
}

// ============================================================================
// Multiplication Operators
// ============================================================================

/// Multiplication operator overloading for `ASTRepr<T>`
impl<T> Mul for ASTRepr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = ASTRepr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        ASTRepr::Mul(Box::new(self), Box::new(rhs))
    }
}

/// Multiplication with references
impl<T> Mul<&ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = ASTRepr<T>;

    fn mul(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Mul(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

/// Multiplication with mixed references
impl<T> Mul<ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Mul<Output = T> + num_traits::Float,
{
    type Output = ASTRepr<T>;

    fn mul(self, rhs: ASTRepr<T>) -> Self::Output {
        ASTRepr::Mul(Box::new(self.clone()), Box::new(rhs))
    }
}

impl<T> Mul<&ASTRepr<T>> for ASTRepr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = ASTRepr<T>;

    fn mul(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Mul(Box::new(self), Box::new(rhs.clone()))
    }
}

// ============================================================================
// Division Operators
// ============================================================================

/// Division operator overloading for `ASTRepr<T>`
impl<T> Div for ASTRepr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = ASTRepr<T>;

    fn div(self, rhs: Self) -> Self::Output {
        ASTRepr::Div(Box::new(self), Box::new(rhs))
    }
}

/// Division with references
impl<T> Div<&ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = ASTRepr<T>;

    fn div(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Div(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

/// Division with mixed references
impl<T> Div<ASTRepr<T>> for &ASTRepr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = ASTRepr<T>;

    fn div(self, rhs: ASTRepr<T>) -> Self::Output {
        ASTRepr::Div(Box::new(self.clone()), Box::new(rhs))
    }
}

impl<T> Div<&ASTRepr<T>> for ASTRepr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = ASTRepr<T>;

    fn div(self, rhs: &ASTRepr<T>) -> Self::Output {
        ASTRepr::Div(Box::new(self), Box::new(rhs.clone()))
    }
}

// ============================================================================
// Negation Operators
// ============================================================================

/// Negation operator overloading for `ASTRepr<T>`
impl<T> Neg for ASTRepr<T>
where
    T: NumericType + Neg<Output = T>,
{
    type Output = ASTRepr<T>;

    fn neg(self) -> Self::Output {
        ASTRepr::Neg(Box::new(self))
    }
}

/// Negation with references
impl<T> Neg for &ASTRepr<T>
where
    T: NumericType + Neg<Output = T>,
{
    type Output = ASTRepr<T>;

    fn neg(self) -> Self::Output {
        ASTRepr::Neg(Box::new(self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_operator_overloading() {
        // Test with f64
        let x_f64 = ASTRepr::<f64>::Variable(0);
        let y_f64 = ASTRepr::<f64>::Variable(1);
        let const_f64 = ASTRepr::<f64>::Constant(2.5);

        let expr_f64 = &x_f64 + &y_f64 * &const_f64;
        assert_eq!(expr_f64.count_operations(), 2); // one add, one mul

        // Test with f32
        let x_f32 = ASTRepr::<f32>::Variable(0);
        let y_f32 = ASTRepr::<f32>::Variable(1);
        let const_f32 = ASTRepr::<f32>::Constant(2.5_f32);

        let expr_f32 = &x_f32 + &y_f32 * &const_f32;
        assert_eq!(expr_f32.count_operations(), 2); // one add, one mul

        // Test negation
        let neg_f64 = -&x_f64;
        let neg_f32 = -&x_f32;

        match neg_f64 {
            ASTRepr::Neg(_) => {}
            _ => panic!("Expected negation"),
        }

        match neg_f32 {
            ASTRepr::Neg(_) => {}
            _ => panic!("Expected negation"),
        }

        // Test transcendental functions (require Float trait)
        let sin_f64 = x_f64.sin();
        let exp_f32 = x_f32.exp();

        match sin_f64 {
            ASTRepr::Trig(trig_cat) => match &trig_cat.function {
                crate::ast::function_categories::TrigFunction::Sin(_) => {}
                crate::ast::function_categories::TrigFunction::Cos(_) => {}
                _ => {}
            },
            _ => panic!("Expected sine"),
        }

        match exp_f32 {
            ASTRepr::Exp(_) => {}
            _ => panic!("Expected exponential"),
        }
    }

    #[test]
    fn test_mixed_reference_operations() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);

        // Test owned + reference
        let expr1 = x.clone() + &y;
        assert_eq!(expr1.count_operations(), 1);

        // Test reference + owned
        let expr2 = &x + y.clone();
        assert_eq!(expr2.count_operations(), 1);

        // Test reference + reference
        let expr3 = &x + &y;
        assert_eq!(expr3.count_operations(), 1);

        // Test owned + owned
        let expr4 = x + y;
        assert_eq!(expr4.count_operations(), 1);
    }

    #[test]
    fn test_complex_expression_building() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let two = ASTRepr::<f64>::Constant(2.0);
        let three = ASTRepr::<f64>::Constant(3.0);

        // Build: 2*x + 3*y using operator overloading
        let expr = &two * &x + &three * &y;
        assert_eq!(expr.count_operations(), 3); // two muls, one add

        // Test with negation: -(2*x + 3*y)
        let neg_expr = -(&two * &x + &three * &y);
        assert_eq!(neg_expr.count_operations(), 4); // two muls, one add
    }

    #[test]
    fn test_division_and_subtraction() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let two = ASTRepr::<f64>::Constant(2.0);

        // Test division: x / 2
        let div_expr = &x / &two;
        assert_eq!(div_expr.count_operations(), 1);

        // Test subtraction: x - y
        let sub_expr = &x - &y;
        assert_eq!(sub_expr.count_operations(), 1);

        // Test complex: (x - y) / 2
        let complex_expr = (&x - &y) / &two;
        assert_eq!(complex_expr.count_operations(), 2); // one sub, one div
    }
}
