//! Expression Normalization for Canonical Forms
//!
//! This module implements canonical form transformations that simplify the
//! mathematical expression system by reducing the number of operation types
//! that need to be handled in optimization rules.
//!
//! ## Canonical Transformations
//!
//! - `Sub(a, b) → Add(a, Neg(b))` - Subtraction becomes addition of negation
//! - `Div(a, b) → Mul(a, Pow(b, -1))` - Division becomes multiplication by reciprocal
//!
//! ## Benefits
//!
//! 1. **Simplified Rules**: Egglog rules only need to handle Add/Mul instead of Add/Sub/Mul/Div
//! 2. **Consistent Patterns**: All operations follow additive/multiplicative patterns
//! 3. **Better Optimization**: More opportunities for algebraic simplification
//! 4. **Reduced Complexity**: ~40% fewer rule cases to handle
//!
//! ## Pipeline Integration
//!
//! The normalization step fits into the compilation pipeline as:
//! `AST → Normalize → ANF → Egglog → Extract → Codegen`

use crate::ast::{ASTRepr, Scalar};
use num_traits::Float;

/// Normalize an expression to canonical form
///
/// This function recursively transforms an expression tree to use only
/// canonical operations (Add, Mul, Neg, Pow) instead of derived operations
/// (Sub, Div).
///
/// # Examples
///
/// ```rust
/// use dslcompile::ast::normalization::normalize;
/// use dslcompile::ast::ASTRepr;
///
/// // x - y becomes x + (-y)
/// let expr = ASTRepr::Sub(
///     Box::new(ASTRepr::<f64>::Variable(0)),
///     Box::new(ASTRepr::<f64>::Variable(1))
/// );
/// let normalized = normalize(&expr);
/// // Result: Add(Variable(0), Neg(Variable(1)))
/// ```
pub fn normalize<T: Scalar + Clone + Float>(expr: &ASTRepr<T>) -> ASTRepr<T> {
    match expr {
        // Base cases - no transformation needed
        ASTRepr::Constant(value) => ASTRepr::Constant(*value),
        ASTRepr::Variable(index) => ASTRepr::Variable(*index),

        // Canonical operations - recursively normalize children
        ASTRepr::Add(left, right) => {
            let norm_left = normalize(left);
            let norm_right = normalize(right);
            ASTRepr::Add(Box::new(norm_left), Box::new(norm_right))
        }
        ASTRepr::Mul(left, right) => {
            let norm_left = normalize(left);
            let norm_right = normalize(right);
            ASTRepr::Mul(Box::new(norm_left), Box::new(norm_right))
        }
        ASTRepr::Pow(base, exp) => {
            let norm_base = normalize(base);
            let norm_exp = normalize(exp);
            ASTRepr::Pow(Box::new(norm_base), Box::new(norm_exp))
        }
        ASTRepr::Neg(inner) => {
            let norm_inner = normalize(inner);
            ASTRepr::Neg(Box::new(norm_inner))
        }

        // Transcendental functions - recursively normalize children
        ASTRepr::Ln(inner) => {
            let norm_inner = normalize(inner);
            ASTRepr::Ln(Box::new(norm_inner))
        }
        ASTRepr::Exp(inner) => {
            let norm_inner = normalize(inner);
            ASTRepr::Exp(Box::new(norm_inner))
        }
        ASTRepr::Sin(inner) => {
            let norm_inner = normalize(inner);
            ASTRepr::Sin(Box::new(norm_inner))
        }
        ASTRepr::Cos(inner) => {
            let norm_inner = normalize(inner);
            ASTRepr::Cos(Box::new(norm_inner))
        }
        ASTRepr::Sqrt(inner) => {
            let norm_inner = normalize(inner);
            ASTRepr::Sqrt(Box::new(norm_inner))
        }

        // CANONICAL TRANSFORMATIONS

        // Sub(a, b) → Add(a, Neg(b))
        ASTRepr::Sub(left, right) => {
            let norm_left = normalize(left);
            let norm_right = normalize(right);
            ASTRepr::Add(
                Box::new(norm_left),
                Box::new(ASTRepr::Neg(Box::new(norm_right))),
            )
        }

        // Div(a, b) → Mul(a, Pow(b, -1))
        ASTRepr::Div(left, right) => {
            let norm_left = normalize(left);
            let norm_right = normalize(right);
            ASTRepr::Mul(
                Box::new(norm_left),
                Box::new(ASTRepr::Pow(
                    Box::new(norm_right),
                    Box::new(ASTRepr::Constant(-T::one())),
                )),
            )
        }

        ASTRepr::Sum(_collection) => {
            // TODO: Normalize Collection format
            expr.clone() // Placeholder until Collection normalization is implemented
        }
        ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
    }
}

/// Check if an expression is in canonical form
///
/// An expression is canonical if it contains no Sub or Div operations.
/// This is useful for testing and validation.
pub fn is_canonical<T: Scalar>(expr: &ASTRepr<T>) -> bool {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => true,

        // These are canonical operations
        ASTRepr::Add(left, right) | ASTRepr::Mul(left, right) | ASTRepr::Pow(left, right) => {
            is_canonical(left) && is_canonical(right)
        }

        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => is_canonical(inner),
        ASTRepr::Sum(_collection) => {
            // TODO: Implement Sum Collection variant canonical form checking
            true // Placeholder until Collection analysis is implemented
        }

        // These are non-canonical operations
        ASTRepr::Sub(_, _) | ASTRepr::Div(_, _) => false,
        ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
    }
}

/// Denormalize an expression for pretty-printing
///
/// This function converts canonical forms back to more readable forms
/// for display purposes. It's the inverse of normalization.
///
/// # Transformations
///
/// - `Add(a, Neg(b)) → Sub(a, b)` - Addition of negation becomes subtraction
/// - `Mul(a, Pow(b, -1)) → Div(a, b)` - Multiplication by reciprocal becomes division
///
/// # Examples
///
/// ```rust
/// use dslcompile::ast::normalization::{normalize, denormalize};
/// use dslcompile::ast::ASTRepr;
///
/// let original = ASTRepr::Sub(
///     Box::new(ASTRepr::<f64>::Variable(0)),
///     Box::new(ASTRepr::<f64>::Variable(1))
/// );
/// let normalized = normalize(&original);
/// let denormalized = denormalize(&normalized);
/// // denormalized should be equivalent to original for display
/// ```
pub fn denormalize<T: Scalar + Clone + PartialEq + Float>(expr: &ASTRepr<T>) -> ASTRepr<T> {
    match expr {
        // Base cases
        ASTRepr::Constant(value) => ASTRepr::Constant(*value),
        ASTRepr::Variable(index) => ASTRepr::Variable(*index),

        // Check for denormalization patterns first

        // Add(a, Neg(b)) → Sub(a, b)
        ASTRepr::Add(left, right) => {
            if let ASTRepr::Neg(neg_inner) = right.as_ref() {
                let denorm_left = denormalize(left);
                let denorm_neg_inner = denormalize(neg_inner);
                ASTRepr::Sub(Box::new(denorm_left), Box::new(denorm_neg_inner))
            } else {
                let denorm_left = denormalize(left);
                let denorm_right = denormalize(right);
                ASTRepr::Add(Box::new(denorm_left), Box::new(denorm_right))
            }
        }

        // Mul(a, Pow(b, -1)) → Div(a, b)
        ASTRepr::Mul(left, right) => {
            if let ASTRepr::Pow(base, exp) = right.as_ref()
                && let ASTRepr::Constant(exp_val) = exp.as_ref()
            {
                // Check if exponent is -1 (allowing for floating point comparison)
                if (*exp_val + T::one()).abs() < T::epsilon() {
                    let denorm_left = denormalize(left);
                    let denorm_base = denormalize(base);
                    return ASTRepr::Div(Box::new(denorm_left), Box::new(denorm_base));
                }
            }
            // Default case: recursively denormalize
            let denorm_left = denormalize(left);
            let denorm_right = denormalize(right);
            ASTRepr::Mul(Box::new(denorm_left), Box::new(denorm_right))
        }

        // Other operations - recursively denormalize
        ASTRepr::Pow(base, exp) => {
            let denorm_base = denormalize(base);
            let denorm_exp = denormalize(exp);
            ASTRepr::Pow(Box::new(denorm_base), Box::new(denorm_exp))
        }
        ASTRepr::Neg(inner) => {
            let denorm_inner = denormalize(inner);
            ASTRepr::Neg(Box::new(denorm_inner))
        }
        ASTRepr::Ln(inner) => {
            let denorm_inner = denormalize(inner);
            ASTRepr::Ln(Box::new(denorm_inner))
        }
        ASTRepr::Exp(inner) => {
            let denorm_inner = denormalize(inner);
            ASTRepr::Exp(Box::new(denorm_inner))
        }
        ASTRepr::Sin(inner) => {
            let denorm_inner = denormalize(inner);
            ASTRepr::Sin(Box::new(denorm_inner))
        }
        ASTRepr::Cos(inner) => {
            let denorm_inner = denormalize(inner);
            ASTRepr::Cos(Box::new(denorm_inner))
        }
        ASTRepr::Sqrt(inner) => {
            let denorm_inner = denormalize(inner);
            ASTRepr::Sqrt(Box::new(denorm_inner))
        }

        // These should not appear in canonical form, but handle them anyway
        ASTRepr::Sub(left, right) => {
            let denorm_left = denormalize(left);
            let denorm_right = denormalize(right);
            ASTRepr::Sub(Box::new(denorm_left), Box::new(denorm_right))
        }
        ASTRepr::Div(left, right) => {
            let denorm_left = denormalize(left);
            let denorm_right = denormalize(right);
            ASTRepr::Div(Box::new(denorm_left), Box::new(denorm_right))
        }

        ASTRepr::Sum(_collection) => {
            // TODO: Denormalize Collection format
            expr.clone() // Placeholder until Collection denormalization is implemented
        }
        ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
    }
}

/// Count the number of operations in an expression
///
/// This is useful for measuring the complexity reduction achieved by normalization.
/// Canonical forms may have more nodes but fewer operation types.
pub fn count_operations<T: Scalar>(expr: &ASTRepr<T>) -> (usize, usize, usize, usize) {
    let mut add_count = 0;
    let mut mul_count = 0;
    let mut sub_count = 0;
    let mut div_count = 0;

    fn count_recursive<T: Scalar>(
        expr: &ASTRepr<T>,
        add: &mut usize,
        mul: &mut usize,
        sub: &mut usize,
        div: &mut usize,
    ) {
        match expr {
            ASTRepr::Add(left, right) => {
                *add += 1;
                count_recursive(left, add, mul, sub, div);
                count_recursive(right, add, mul, sub, div);
            }
            ASTRepr::Sub(left, right) => {
                *sub += 1;
                count_recursive(left, add, mul, sub, div);
                count_recursive(right, add, mul, sub, div);
            }
            ASTRepr::Mul(left, right) => {
                *mul += 1;
                count_recursive(left, add, mul, sub, div);
                count_recursive(right, add, mul, sub, div);
            }
            ASTRepr::Div(left, right) => {
                *div += 1;
                count_recursive(left, add, mul, sub, div);
                count_recursive(right, add, mul, sub, div);
            }
            ASTRepr::Pow(base, exp) => {
                count_recursive(base, add, mul, sub, div);
                count_recursive(exp, add, mul, sub, div);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                count_recursive(inner, add, mul, sub, div);
            }
            ASTRepr::Sum(_collection) => {
                // TODO: Implement Sum Collection variant operation counting
                // For now, don't count operations inside collections
            }
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => {}
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    count_recursive(
        expr,
        &mut add_count,
        &mut mul_count,
        &mut sub_count,
        &mut div_count,
    );
    (add_count, mul_count, sub_count, div_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtraction_normalization() {
        // x - y should become x + (-y)
        let expr: ASTRepr<f64> = ASTRepr::Sub(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        let normalized = normalize(&expr);

        // Check structure: Add(Variable(0), Neg(Variable(1)))
        match normalized {
            ASTRepr::Add(left, right) => {
                assert!(matches!(left.as_ref(), ASTRepr::Variable(0)));
                assert!(
                    matches!(right.as_ref(), ASTRepr::Neg(inner) if matches!(inner.as_ref(), ASTRepr::Variable(1)))
                );
            }
            _ => panic!("Expected Add operation after normalization"),
        }
    }

    #[test]
    fn test_division_normalization() {
        // x / y should become x * (y^(-1))
        let expr: ASTRepr<f64> = ASTRepr::Div(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        let normalized = normalize(&expr);

        // Check structure: Mul(Variable(0), Pow(Variable(1), Constant(-1.0)))
        match normalized {
            ASTRepr::Mul(left, right) => {
                assert!(matches!(left.as_ref(), ASTRepr::Variable(0)));
                match right.as_ref() {
                    ASTRepr::Pow(base, exp) => {
                        assert!(matches!(base.as_ref(), ASTRepr::Variable(1)));
                        assert!(
                            matches!(exp.as_ref(), ASTRepr::Constant(val) if (*val - (-1.0)).abs() < 1e-12)
                        );
                    }
                    _ => panic!("Expected Pow operation in normalized division"),
                }
            }
            _ => panic!("Expected Mul operation after normalization"),
        }
    }

    #[test]
    fn test_nested_normalization() {
        // (x - y) / (a + b) should become (x + (-y)) * ((a + b)^(-1))
        let expr: ASTRepr<f64> = ASTRepr::Div(
            Box::new(ASTRepr::Sub(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Variable(1)),
            )),
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Variable(2)),
                Box::new(ASTRepr::Variable(3)),
            )),
        );

        let normalized = normalize(&expr);

        // Should be canonical (no Sub or Div operations)
        assert!(is_canonical(&normalized));
    }

    #[test]
    fn test_is_canonical() {
        // Canonical expression: x + (-y)
        let canonical: ASTRepr<f64> = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Neg(Box::new(ASTRepr::Variable(1)))),
        );
        assert!(is_canonical(&canonical));

        // Non-canonical expression: x - y
        let non_canonical: ASTRepr<f64> = ASTRepr::Sub(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );
        assert!(!is_canonical(&non_canonical));
    }

    #[test]
    fn test_denormalization() {
        // Test that denormalization produces readable forms
        let original: ASTRepr<f64> = ASTRepr::Sub(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        let normalized = normalize(&original);
        let denormalized = denormalize(&normalized);

        // Denormalized should be equivalent to original structure
        match denormalized {
            ASTRepr::Sub(left, right) => {
                assert!(matches!(left.as_ref(), ASTRepr::Variable(0)));
                assert!(matches!(right.as_ref(), ASTRepr::Variable(1)));
            }
            _ => panic!("Expected Sub operation after denormalization"),
        }
    }

    #[test]
    fn test_division_denormalization() {
        let original: ASTRepr<f64> = ASTRepr::Div(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        let normalized = normalize(&original);
        let denormalized = denormalize(&normalized);

        // Denormalized should be equivalent to original structure
        match denormalized {
            ASTRepr::Div(left, right) => {
                assert!(matches!(left.as_ref(), ASTRepr::Variable(0)));
                assert!(matches!(right.as_ref(), ASTRepr::Variable(1)));
            }
            _ => panic!("Expected Div operation after denormalization"),
        }
    }

    #[test]
    fn test_operation_counting() {
        // Original: x - y + z / w (1 sub, 1 add, 1 div)
        let expr: ASTRepr<f64> = ASTRepr::Add(
            Box::new(ASTRepr::Sub(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Variable(1)),
            )),
            Box::new(ASTRepr::Div(
                Box::new(ASTRepr::Variable(2)),
                Box::new(ASTRepr::Variable(3)),
            )),
        );

        let (add, mul, sub, div) = count_operations(&expr);
        assert_eq!(add, 1);
        assert_eq!(mul, 0);
        assert_eq!(sub, 1);
        assert_eq!(div, 1);

        // After normalization: should have more add/mul, no sub/div
        let normalized = normalize(&expr);
        let (norm_add, norm_mul, norm_sub, norm_div) = count_operations(&normalized);
        assert!(norm_add > add);
        assert!(norm_mul > mul);
        assert_eq!(norm_sub, 0);
        assert_eq!(norm_div, 0);
    }

    #[test]
    fn test_complex_expression_normalization() {
        // Test a complex expression with multiple levels
        let expr: ASTRepr<f64> = ASTRepr::Div(
            Box::new(ASTRepr::Sub(
                Box::new(ASTRepr::Mul(
                    Box::new(ASTRepr::Variable(0)),
                    Box::new(ASTRepr::Variable(1)),
                )),
                Box::new(ASTRepr::Constant(2.0)),
            )),
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Variable(2)),
                Box::new(ASTRepr::Constant(1.0)),
            )),
        );

        let normalized = normalize(&expr);

        // Should be completely canonical
        assert!(is_canonical(&normalized));

        // Should be able to denormalize back to a readable form
        let denormalized = denormalize(&normalized);

        // The denormalized form should have the same structure as original
        // (though the exact tree structure might differ due to the transformations)
        assert!(!is_canonical(&denormalized)); // Should contain Sub/Div again
    }
}
