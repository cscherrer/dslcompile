//! AST Representation for Mathematical Expressions
//!
//! This module defines the `ASTRepr` enum which represents mathematical expressions
//! as an abstract syntax tree. This representation is used for JIT compilation,
//! symbolic optimization, and other analysis tasks.

use crate::ast::NumericType;
use num_traits::Float;

/// Range types for symbolic summation
#[derive(Debug, Clone, PartialEq)]
pub enum SumRange<T> {
    /// Mathematical range: 1..=n, start..=end
    /// Generates: (start..=end).map(|i| `body).sum()`
    Mathematical {
        start: Box<ASTRepr<T>>,
        end: Box<ASTRepr<T>>,
    },
    /// Data parameter: references a data array variable
    /// Generates: data.iter().map(|&x| `body).sum()` or `data.iter().sum()`
    DataParameter {
        data_var: usize, // Variable index pointing to data array
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
/// use dslcompile::symbolic::summation::DirectEval;
///
/// // Efficient: uses vector indexing
/// let expr = ASTRepr::Add(
///     Box::new(ASTRepr::Variable(0)), // x
///     Box::new(ASTRepr::Variable(1)), // y
/// );
/// let result = DirectEval::eval_with_vars(&expr, &[2.0, 3.0]);
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
    /// Symbolic summation that generates runtime iteration code
    ///
    /// Creates expressions like:
    /// - Mathematical: `(1..=n).map(|i| body_expr).sum()`
    /// - Data: `data.iter().map(|&x| body_expr).sum()`
    /// - Optimized: `data.iter().sum()` when body is identity
    Sum {
        /// Range specification (mathematical or data parameter)
        range: SumRange<T>,
        /// Body expression using iterator variable
        body: Box<ASTRepr<T>>,
        /// Iterator variable index for substitution
        iter_var: usize,
    },
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
            ASTRepr::Sum { range, body, .. } => {
                1 + body.count_operations()
                    + match range {
                        SumRange::Mathematical { start, end } => {
                            start.count_operations() + end.count_operations()
                        }
                        SumRange::DataParameter { .. } => 0,
                    }
            }
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
    pub fn count_summation_operations(&self) -> usize {
        match self {
            ASTRepr::Sum { body, range, .. } => {
                1 + body.count_summation_operations()
                    + match range {
                        SumRange::Mathematical { start, end } => {
                            start.count_summation_operations() + end.count_summation_operations()
                        }
                        SumRange::DataParameter { .. } => 0,
                    }
            }
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                left.count_summation_operations() + right.count_summation_operations()
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => inner.count_summation_operations(),
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
        }
    }
}

/// Additional convenience methods for `ASTRepr<T>` with generic types
impl<T> ASTRepr<T>
where
    T: NumericType,
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

    /// Square root
    #[must_use]
    pub fn sqrt(self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Sqrt(Box::new(self))
    }

    /// Square root with reference
    #[must_use]
    pub fn sqrt_ref(&self) -> ASTRepr<T>
    where
        T: Float,
    {
        ASTRepr::Sqrt(Box::new(self.clone()))
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

        // Test sqrt with reference
        let sqrt_expr = x.sqrt_ref();
        match sqrt_expr {
            ASTRepr::Sqrt(_) => {}
            _ => panic!("Expected square root expression"),
        }
    }
}
