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
//! `AST → Normalize → Egglog → Extract → Codegen`

use crate::ast::{ASTRepr, Scalar, StackBasedMutVisitor, StackBasedVisitor};
use num_traits::Float;

/// Stack-based normalizer that transforms expressions to canonical form
struct Normalizer<T: Scalar + Clone + Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Clone + Float> Normalizer<T> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Clone + Float> StackBasedMutVisitor<T> for Normalizer<T> {
    type Error = ();

    fn transform_node(&mut self, expr: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        // Apply canonical transformations - much more concise than before!
        match expr {
            // Sub(a, b) → Add(a, Neg(b))
            ASTRepr::Sub(left, right) => Ok(ASTRepr::Add(left, Box::new(ASTRepr::Neg(right)))),
            // Div(a, b) → Mul(a, Pow(b, -1))
            ASTRepr::Div(left, right) => {
                let neg_one = ASTRepr::Constant(-T::one());
                let reciprocal = ASTRepr::Pow(right, Box::new(neg_one));
                Ok(ASTRepr::Mul(left, Box::new(reciprocal)))
            }
            // All other expressions pass through unchanged
            _ => Ok(expr),
        }
    }
}

/// Normalize an expression to canonical form
///
/// This function transforms an expression tree to use only canonical operations
/// (Add, Mul, Neg, Pow) instead of derived operations (Sub, Div).
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
    let mut normalizer = Normalizer::new();
    normalizer
        .transform(expr.clone())
        .unwrap_or_else(|()| expr.clone())
}

/// Stack-based canonical checker
struct CanonicalChecker {
    is_canonical: bool,
}

impl CanonicalChecker {
    fn new() -> Self {
        Self { is_canonical: true }
    }
}

impl<T: Scalar + Clone> crate::ast::StackBasedVisitor<T> for CanonicalChecker {
    type Output = ();
    type Error = ();

    fn visit_node(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        // Check for non-canonical operations
        match expr {
            ASTRepr::Sub(_, _) | ASTRepr::Div(_, _) => {
                self.is_canonical = false;
            }
            _ => {} // All other operations are canonical or handled automatically
        }
        Ok(())
    }

    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
}

/// Check if an expression is in canonical form
///
/// Returns true if the expression contains only canonical operations
/// (Add, Mul, Neg, Pow, transcendental functions) and no derived operations (Sub, Div).
pub fn is_canonical<T: Scalar + Clone>(expr: &ASTRepr<T>) -> bool {
    let mut checker = CanonicalChecker::new();
    let _ = checker.traverse(expr.clone()).unwrap_or_default();
    checker.is_canonical
}

/// Stack-based operation counter
struct OperationCounter {
    add: usize,
    mul: usize,
    sub: usize,
    div: usize,
}

impl OperationCounter {
    fn new() -> Self {
        Self {
            add: 0,
            mul: 0,
            sub: 0,
            div: 0,
        }
    }
}

impl<T: Scalar + Clone> crate::ast::StackBasedVisitor<T> for OperationCounter {
    type Output = ();
    type Error = ();

    fn visit_node(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        // Count operations - single place for all logic!
        match expr {
            ASTRepr::Add(_, _) => self.add += 1,
            ASTRepr::Mul(_, _) => self.mul += 1,
            ASTRepr::Sub(_, _) => self.sub += 1,
            ASTRepr::Div(_, _) => self.div += 1,
            _ => {} // All other cases handled automatically by traversal
        }
        Ok(())
    }

    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
}

/// Count arithmetic operations in an expression
///
/// Returns a tuple of (`add_count`, `mul_count`, `sub_count`, `div_count`).
/// This is useful for complexity analysis and optimization decisions.
pub fn count_operations<T: Scalar + Clone>(expr: &ASTRepr<T>) -> (usize, usize, usize, usize) {
    let mut counter = OperationCounter::new();
    let _ = counter.traverse(expr.clone()).unwrap_or_default();
    (counter.add, counter.mul, counter.sub, counter.div)
}

/// Stack-based denormalizer for pretty printing
struct Denormalizer<T: Scalar + Clone + Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Clone + Float> Denormalizer<T> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Clone + Float> StackBasedMutVisitor<T> for Denormalizer<T> {
    type Error = ();

    fn transform_node(&mut self, expr: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        // Convert canonical forms back to readable forms
        match expr {
            // Add(a, Neg(b)) → Sub(a, b)
            ASTRepr::Add(left, right) => {
                if let ASTRepr::Neg(neg_inner) = *right {
                    Ok(ASTRepr::Sub(left, neg_inner))
                } else {
                    Ok(ASTRepr::Add(left, right))
                }
            }
            // Mul(a, Pow(b, -1)) → Div(a, b)
            ASTRepr::Mul(left, right) => {
                if let ASTRepr::Pow(base, exp) = right.as_ref()
                    && let ASTRepr::Constant(exp_val) = exp.as_ref()
                {
                    // Check if exponent is -1
                    if (*exp_val + T::one()).abs() < T::epsilon() {
                        return Ok(ASTRepr::Div(left, base.clone()));
                    }
                }
                Ok(ASTRepr::Mul(left, right))
            }
            _ => Ok(expr),
        }
    }
}

/// Denormalize an expression for pretty-printing
///
/// This function converts canonical forms back to more readable forms:
/// - `Add(a, Neg(b)) → Sub(a, b)`
/// - `Mul(a, Pow(b, -1)) → Div(a, b)`
pub fn denormalize<T: Scalar + Clone + Float>(expr: &ASTRepr<T>) -> ASTRepr<T> {
    let mut denormalizer = Denormalizer::new();
    denormalizer
        .transform(expr.clone())
        .unwrap_or_else(|()| expr.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_subtraction() {
        let expr = ASTRepr::Sub(
            Box::new(ASTRepr::<f64>::Variable(0)),
            Box::new(ASTRepr::<f64>::Variable(1)),
        );
        let normalized = normalize(&expr);

        // Should become Add(Variable(0), Neg(Variable(1)))
        match normalized {
            ASTRepr::Add(left, right) => {
                assert!(matches!(left.as_ref(), ASTRepr::Variable(0)));
                assert!(matches!(right.as_ref(), ASTRepr::Neg(_)));
            }
            _ => panic!("Expected Add with Neg"),
        }
    }

    #[test]
    fn test_normalize_division() {
        let expr = ASTRepr::Div(
            Box::new(ASTRepr::<f64>::Variable(0)),
            Box::new(ASTRepr::<f64>::Variable(1)),
        );
        let normalized = normalize(&expr);

        // Should become Mul(Variable(0), Pow(Variable(1), -1))
        match normalized {
            ASTRepr::Mul(left, right) => {
                assert!(matches!(left.as_ref(), ASTRepr::Variable(0)));
                assert!(matches!(right.as_ref(), ASTRepr::Pow(_, _)));
            }
            _ => panic!("Expected Mul with Pow"),
        }
    }

    #[test]
    fn test_is_canonical() {
        let canonical = ASTRepr::Add(
            Box::new(ASTRepr::<f64>::Variable(0)),
            Box::new(ASTRepr::<f64>::Variable(1)),
        );
        assert!(is_canonical(&canonical));

        let non_canonical = ASTRepr::Sub(
            Box::new(ASTRepr::<f64>::Variable(0)),
            Box::new(ASTRepr::<f64>::Variable(1)),
        );
        assert!(!is_canonical(&non_canonical));
    }

    #[test]
    fn test_count_operations() {
        // (x + y) * (a - b)
        let expr = ASTRepr::Mul(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::<f64>::Variable(0)),
                Box::new(ASTRepr::<f64>::Variable(1)),
            )),
            Box::new(ASTRepr::Sub(
                Box::new(ASTRepr::<f64>::Variable(2)),
                Box::new(ASTRepr::<f64>::Variable(3)),
            )),
        );

        let (add, mul, sub, div) = count_operations(&expr);
        assert_eq!(add, 1);
        assert_eq!(mul, 1);
        assert_eq!(sub, 1);
        assert_eq!(div, 0);
    }

    #[test]
    fn test_denormalize() {
        // Create Add(x, Neg(y))
        let canonical = ASTRepr::Add(
            Box::new(ASTRepr::<f64>::Variable(0)),
            Box::new(ASTRepr::Neg(Box::new(ASTRepr::<f64>::Variable(1)))),
        );

        let denormalized = denormalize(&canonical);

        // Should become Sub(x, y)
        match denormalized {
            ASTRepr::Sub(left, right) => {
                assert!(matches!(left.as_ref(), ASTRepr::Variable(0)));
                assert!(matches!(right.as_ref(), ASTRepr::Variable(1)));
            }
            _ => panic!("Expected Sub"),
        }
    }
}
