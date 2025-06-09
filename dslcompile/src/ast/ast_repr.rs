//! AST Representation for Mathematical Expressions
//!
//! This module defines the `ASTRepr` enum which represents mathematical expressions
//! as an abstract syntax tree. This representation is used for JIT compilation,
//! symbolic optimization, and other analysis tasks.

use crate::ast::Scalar;
use num_traits::{Float, FromPrimitive};

/// Collection types for compositional summation operations
#[derive(Debug, Clone, PartialEq)]
pub enum Collection<T> {
    /// Empty collection
    Empty,
    /// Single element collection
    Singleton(Box<ASTRepr<T>>),
    /// Mathematical range [start, end] (inclusive)
    Range {
        start: Box<ASTRepr<T>>,
        end: Box<ASTRepr<T>>,
    },
    /// Set union
    Union {
        left: Box<Collection<T>>,
        right: Box<Collection<T>>,
    },
    /// Set intersection
    Intersection {
        left: Box<Collection<T>>,
        right: Box<Collection<T>>,
    },
    /// Data array for runtime binding (referenced by variable index)
    DataArray(usize),
    /// Filtered collection with predicate
    Filter {
        collection: Box<Collection<T>>,
        predicate: Box<ASTRepr<T>>,
    },
    /// Map function over collection (Iterator pattern)
    Map {
        lambda: Box<Lambda<T>>,
        collection: Box<Collection<T>>,
    },
}

/// Lambda expressions for mapping functions
#[derive(Debug, Clone, PartialEq)]
pub enum Lambda<T> {
    /// Lambda expression: lambda var_index -> body
    /// Uses variable index for automatic scope management
    Lambda {
        var_index: usize,
        body: Box<ASTRepr<T>>,
    },
    /// Identity function: lambda x -> x
    Identity,
    /// Constant function: lambda x -> c
    Constant(Box<ASTRepr<T>>),
    /// Function composition: f ∘ g
    Compose {
        f: Box<Lambda<T>>,
        g: Box<Lambda<T>>,
    },
}

/// JIT compilation representation for mathematical expressions
///
/// This enum represents mathematical expressions in a form suitable for JIT compilation
/// using Cranelift. Each variant corresponds to a mathematical operation that can be
/// compiled to native machine code.
///
/// # Performance Note
///
/// Variables are referenced by index for optimal performance with `DirectEval`,
/// using vector indexing instead of string lookups:
///
/// ```rust
/// use dslcompile::ast::ASTRepr;
/// // Efficient: uses vector indexing
/// let expr = ASTRepr::Add(
///     Box::new(ASTRepr::Variable(0)), // x
///     Box::new(ASTRepr::Variable(1)), // y
/// );
/// let result = expr.eval_with_vars(&[2.0, 3.0]);
/// assert_eq!(result, 5.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ASTRepr<T> {
    /// Constants: 42.0, π, etc.
    Constant(T),
    /// Variables: x, y, z (referenced by index for performance)
    Variable(usize),
    /// Binary operations
    Add(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    Sub(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    Mul(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    Div(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    Pow(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    /// Unary operations
    Neg(Box<ASTRepr<T>>),
    Ln(Box<ASTRepr<T>>),
    Exp(Box<ASTRepr<T>>),
    Sin(Box<ASTRepr<T>>),
    Cos(Box<ASTRepr<T>>),
    Sqrt(Box<ASTRepr<T>>),
    /// Compositional summation using iterator abstraction
    ///
    /// Creates expressions like:
    /// - Simple: `Sum(Range(1, n))` → sum over range with identity
    /// - Mapped: `Sum(Map(f, Range(1, n)))` → sum f(i) for i in 1..n
    /// - Data: `Sum(DataArray(0))` → sum over data array
    /// - Complex: `Sum(Map(f, Union(A, B)))` → sum f(x) for x in A∪B
    Sum(Box<Collection<T>>),
}

impl<T> ASTRepr<T> {
    /// Count the total number of operations in the expression tree
    pub fn count_operations(&self) -> usize {
        match self {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => 1 + left.count_operations() + right.count_operations(),
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => 1 + inner.count_operations(),
            ASTRepr::Sum(collection) => 1 + collection.count_operations(),
        }
    }

    /// Get the variable index if this is a variable, otherwise None
    pub fn variable_index(&self) -> Option<usize> {
        match self {
            ASTRepr::Variable(index) => Some(*index),
            _ => None,
        }
    }

    /// Count summation operations specifically
    pub fn count_summations(&self) -> usize {
        match self {
            ASTRepr::Sum(_) => 1 + self.count_summations_recursive(),
            _ => self.count_summations_recursive(),
        }
    }

    /// Recursively count summations in subexpressions
    fn count_summations_recursive(&self) -> usize {
        match self {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => left.count_summations() + right.count_summations(),
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => inner.count_summations(),
            ASTRepr::Sum(collection) => collection.count_summations(),
        }
    }
}

impl<T> Collection<T> {
    /// Count operations in the collection
    pub fn count_operations(&self) -> usize {
        match self {
            Collection::Empty => 0,
            Collection::Singleton(expr) => expr.count_operations(),
            Collection::Range { start, end } => start.count_operations() + end.count_operations(),
            Collection::Union { left, right } | Collection::Intersection { left, right } => {
                1 + left.count_operations() + right.count_operations()
            }
            Collection::DataArray(_) => 0,
            Collection::Filter {
                collection,
                predicate,
            } => 1 + collection.count_operations() + predicate.count_operations(),
            Collection::Map { lambda, collection } => {
                1 + lambda.count_operations() + collection.count_operations()
            }
        }
    }

    /// Count summations in the collection
    pub fn count_summations(&self) -> usize {
        match self {
            Collection::Empty | Collection::DataArray(_) => 0,
            Collection::Singleton(expr) => expr.count_summations(),
            Collection::Range { start, end } => start.count_summations() + end.count_summations(),
            Collection::Union { left, right } | Collection::Intersection { left, right } => {
                left.count_summations() + right.count_summations()
            }
            Collection::Filter {
                collection,
                predicate,
            } => collection.count_summations() + predicate.count_summations(),
            Collection::Map { lambda, collection } => {
                lambda.count_summations() + collection.count_summations()
            }
        }
    }
}

impl<T> Lambda<T> {
    /// Count operations in the lambda
    pub fn count_operations(&self) -> usize {
        match self {
            Lambda::Identity => 0,
            Lambda::Constant(expr) => expr.count_operations(),
            Lambda::Lambda { body, .. } => body.count_operations(),
            Lambda::Compose { f, g } => 1 + f.count_operations() + g.count_operations(),
        }
    }

    /// Count summations in the lambda
    pub fn count_summations(&self) -> usize {
        match self {
            Lambda::Identity => 0,
            Lambda::Constant(expr) => expr.count_summations(),
            Lambda::Lambda { body, .. } => body.count_summations(),
            Lambda::Compose { f, g } => f.count_summations() + g.count_summations(),
        }
    }
}

/// Additional convenience methods for `ASTRepr<T>` with generic types
impl<T> ASTRepr<T>
where
    T: Scalar,
{
    /// Power operation with natural syntax
    #[must_use]
    pub fn pow(self, exp: ASTRepr<T>) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Pow(Box::new(self), Box::new(exp))
    }

    /// Power operation with reference
    #[must_use]
    pub fn pow_ref(&self, exp: &ASTRepr<T>) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Pow(Box::new(self.clone()), Box::new(exp.clone()))
    }

    /// Natural logarithm
    #[must_use]
    pub fn ln(self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Ln(Box::new(self))
    }

    /// Natural logarithm with reference
    #[must_use]
    pub fn ln_ref(&self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Ln(Box::new(self.clone()))
    }

    /// Exponential function
    #[must_use]
    pub fn exp(self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Exp(Box::new(self))
    }

    /// Exponential function with reference
    #[must_use]
    pub fn exp_ref(&self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Exp(Box::new(self.clone()))
    }

    /// Square root (implemented as x^0.5 for unified power handling)
    #[must_use]
    pub fn sqrt(self) -> ASTRepr<T>
    where
        T: Float + FromPrimitive,
    {
        let half = T::from_f64(0.5).unwrap_or_else(|| {
            panic!("Type T must support conversion from f64 for sqrt operation")
        });
        ASTRepr::Pow(Box::new(self), Box::new(ASTRepr::Constant(half)))
    }

    /// Square root with reference (implemented as x^0.5 for unified power handling)
    #[must_use]
    pub fn sqrt_ref(&self) -> ASTRepr<T>
    where
        T: Float + FromPrimitive,
    {
        let half = T::from_f64(0.5).unwrap_or_else(|| {
            panic!("Type T must support conversion from f64 for sqrt operation")
        });
        ASTRepr::Pow(Box::new(self.clone()), Box::new(ASTRepr::Constant(half)))
    }

    /// Sine function
    #[must_use]
    pub fn sin(self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Sin(Box::new(self))
    }

    /// Sine function with reference
    #[must_use]
    pub fn sin_ref(&self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Sin(Box::new(self.clone()))
    }

    /// Cosine function
    #[must_use]
    pub fn cos(self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Cos(Box::new(self))
    }

    /// Cosine function with reference
    #[must_use]
    pub fn cos_ref(&self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Cos(Box::new(self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_repr_basic_operations() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let const_2 = ASTRepr::<f64>::Constant(2.0);

        // Test addition
        let add_expr = ASTRepr::Add(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(add_expr.count_operations(), 1);

        // Test multiplication
        let mul_expr = ASTRepr::Mul(Box::new(x.clone()), Box::new(const_2.clone()));
        assert_eq!(mul_expr.count_operations(), 1);

        // Test complex expression: (x + y) * 2
        let complex_expr = ASTRepr::Mul(Box::new(add_expr), Box::new(const_2));
        assert_eq!(complex_expr.count_operations(), 2); // one add, one mul
    }

    #[test]
    fn test_variable_index_access() {
        let expr: ASTRepr<f64> = ASTRepr::Variable(5);
        assert_eq!(expr.variable_index(), Some(5));

        let expr: ASTRepr<f64> = ASTRepr::Constant(42.0);
        assert_eq!(expr.variable_index(), None);
    }

    #[test]
    fn test_transcendental_functions() {
        let x = ASTRepr::<f64>::Variable(0);

        // Test sine
        let sin_expr = x.clone().sin();
        match sin_expr {
            ASTRepr::Sin(_) => {}
            _ => panic!("Expected sine expression"),
        }

        // Test exponential
        let exp_expr = x.clone().exp();
        match exp_expr {
            ASTRepr::Exp(_) => {}
            _ => panic!("Expected exponential expression"),
        }

        // Test natural logarithm
        let ln_expr = x.ln();
        match ln_expr {
            ASTRepr::Ln(_) => {}
            _ => panic!("Expected natural logarithm expression"),
        }
    }

    #[test]
    fn test_convenience_methods() {
        let x = ASTRepr::<f64>::Variable(0);
        let two = ASTRepr::<f64>::Constant(2.0);

        // Test power with reference
        let pow_expr = x.pow_ref(&two);
        match pow_expr {
            ASTRepr::Pow(_, _) => {}
            _ => panic!("Expected power expression"),
        }

        // Test sqrt with reference (now creates Pow(x, 0.5))
        let sqrt_expr = x.sqrt_ref();
        match sqrt_expr {
            ASTRepr::Pow(base, exp) => {
                // Verify it's x^0.5
                match (base.as_ref(), exp.as_ref()) {
                    (ASTRepr::Variable(0), ASTRepr::Constant(val))
                        if (*val - 0.5).abs() < 1e-15 => {}
                    _ => panic!("Expected Pow(x, 0.5) for sqrt"),
                }
            }
            _ => panic!("Expected power expression (x^0.5) for sqrt"),
        }
    }
}
