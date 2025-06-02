//! AST Evaluation Interpreter
//!
//! This interpreter builds AST representations that can later be compiled
//! to native machine code for high-performance evaluation.

use crate::ast::ASTRepr;
use crate::final_tagless::traits::{ASTMathExpr, MathExpr, NumericType, StatisticalExpr};
use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// JIT evaluation interpreter that builds an intermediate representation
/// suitable for compilation with Cranelift or Rust codegen
///
/// This interpreter constructs a `ASTRepr` tree that can later be compiled
/// to native machine code for high-performance evaluation.
pub struct ASTEval;

impl ASTEval {
    /// Create a constant from a numeric value
    #[must_use]
    pub fn constant<T: NumericType>(value: T) -> ASTRepr<T> {
        ASTRepr::Constant(value)
    }

    /// Create a variable by index (recommended approach)
    #[must_use]
    pub fn var<T: NumericType>(index: usize) -> ASTRepr<T> {
        ASTRepr::Variable(index)
    }

    /// Addition operation
    #[must_use]
    pub fn add<T: NumericType>(left: ASTRepr<T>, right: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Add(Box::new(left), Box::new(right))
    }

    /// Subtraction operation
    #[must_use]
    pub fn sub<T: NumericType>(left: ASTRepr<T>, right: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Sub(Box::new(left), Box::new(right))
    }

    /// Multiplication operation
    #[must_use]
    pub fn mul<T: NumericType>(left: ASTRepr<T>, right: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Mul(Box::new(left), Box::new(right))
    }

    /// Division operation
    #[must_use]
    pub fn div<T: NumericType>(left: ASTRepr<T>, right: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Div(Box::new(left), Box::new(right))
    }

    /// Power operation
    #[must_use]
    pub fn pow<T: NumericType>(base: ASTRepr<T>, exp: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Pow(Box::new(base), Box::new(exp))
    }

    /// Negation operation
    #[must_use]
    pub fn neg<T: NumericType>(inner: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Neg(Box::new(inner))
    }

    /// Natural logarithm
    #[must_use]
    pub fn ln<T: Float + NumericType>(inner: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Ln(Box::new(inner))
    }

    /// Exponential function
    #[must_use]
    pub fn exp<T: Float + NumericType>(inner: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Exp(Box::new(inner))
    }

    /// Sine function
    #[must_use]
    pub fn sin<T: Float + NumericType>(inner: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Sin(Box::new(inner))
    }

    /// Cosine function
    #[must_use]
    pub fn cos<T: Float + NumericType>(inner: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Cos(Box::new(inner))
    }

    /// Square root function
    #[must_use]
    pub fn sqrt<T: Float + NumericType>(inner: ASTRepr<T>) -> ASTRepr<T> {
        ASTRepr::Sqrt(Box::new(inner))
    }
}

impl ASTMathExpr for ASTEval {
    type Repr = ASTRepr<f64>;

    fn constant(value: f64) -> Self::Repr {
        ASTRepr::Constant(value)
    }

    fn var(index: usize) -> Self::Repr {
        ASTRepr::Variable(index)
    }

    fn add(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        match (&left, &right) {
            (ASTRepr::Constant(l), ASTRepr::Constant(r)) => ASTRepr::Constant(l + r),
            (ASTRepr::Constant(l), r) if *l == 0.0 => r.clone(),
            (l, ASTRepr::Constant(r)) if *r == 0.0 => l.clone(),
            _ => ASTRepr::Add(Box::new(left), Box::new(right)),
        }
    }

    fn sub(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        match (&left, &right) {
            (ASTRepr::Constant(l), ASTRepr::Constant(r)) => ASTRepr::Constant(l - r),
            (ASTRepr::Constant(l), r) if *l == 0.0 => ASTRepr::Neg(Box::new(r.clone())),
            (l, ASTRepr::Constant(r)) if *r == 0.0 => l.clone(),
            _ => ASTRepr::Sub(Box::new(left), Box::new(right)),
        }
    }

    fn mul(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        match (&left, &right) {
            (ASTRepr::Constant(l), ASTRepr::Constant(r)) => ASTRepr::Constant(l * r),
            (ASTRepr::Constant(l), r) if *l == 1.0 => r.clone(),
            (l, ASTRepr::Constant(r)) if *r == 1.0 => l.clone(),
            (ASTRepr::Constant(l), r) if *l == -1.0 => ASTRepr::Neg(Box::new(r.clone())),
            (l, ASTRepr::Constant(r)) if *r == -1.0 => ASTRepr::Neg(Box::new(l.clone())),
            // TODO: Add zero&inf cases? Indeterminacy
            _ => ASTRepr::Mul(Box::new(left), Box::new(right)),
        }
    }

    fn div(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        match (&left, &right) {
            (ASTRepr::Constant(l), ASTRepr::Constant(r)) => ASTRepr::Constant(l / r),
            (ASTRepr::Constant(l), r) if *l == 1.0 => r.clone(),
            (l, ASTRepr::Constant(r)) if *r == 1.0 => l.clone(),
            _ => ASTRepr::Div(Box::new(left), Box::new(right)),
        }
    }

    fn pow(base: Self::Repr, exp: Self::Repr) -> Self::Repr {
        match (&base, &exp) {
            (ASTRepr::Constant(l), ASTRepr::Constant(r)) => {
                // Use domain analysis to determine if constant folding is safe
                let result = l.powf(*r);
                if result.is_finite() && !result.is_nan() {
                    ASTRepr::Constant(result)
                } else {
                    // Don't fold - preserve the expression for runtime evaluation
                    ASTRepr::Pow(Box::new(base), Box::new(exp))
                }
            }
            (ASTRepr::Constant(l), _) if *l == 1.0 => ASTRepr::Constant(1.0), // 1^x = 1
            (_, ASTRepr::Constant(r)) if *r == 0.0 => ASTRepr::Constant(1.0), // x^0 = 1 (including 0^0, which Rust returns 1.0)
            (ASTRepr::Constant(l), _) if *l == 0.0 => ASTRepr::Constant(0.0), // 0^x = 0 for x > 0, but see below
            _ => ASTRepr::Pow(Box::new(base), Box::new(exp)),
        }
    }

    fn neg(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(-x),
            _ => ASTRepr::Neg(Box::new(expr)),
        }
    }

    fn ln(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.ln()),
            ASTRepr::Exp(e) => *e,
            _ => ASTRepr::Ln(Box::new(expr)),
        }
    }

    fn exp(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.exp()),
            _ => ASTRepr::Exp(Box::new(expr)),
        }
    }

    fn sqrt(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.sqrt()),
            _ => ASTRepr::Sqrt(Box::new(expr)),
        }
    }

    fn sin(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.sin()),
            _ => ASTRepr::Sin(Box::new(expr)),
        }
    }

    fn cos(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.cos()),
            _ => ASTRepr::Cos(Box::new(expr)),
        }
    }
}

/// For compatibility with the main `MathExpr` trait, we provide a limited implementation
/// that works only with f64 types
impl MathExpr for ASTEval {
    type Repr<T> = ASTRepr<T>;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        ASTRepr::Constant(value)
    }

    fn var<T: NumericType>(index: usize) -> Self::Repr<T> {
        ASTRepr::Variable(index)
    }

    fn add<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Add<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        // This is a placeholder implementation for the generic trait
        // In practice, you would use the specific f64 version
        unimplemented!("Use ASTMathExpr for concrete implementations")
    }

    fn sub<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr for concrete implementations")
    }

    fn mul<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr for concrete implementations")
    }

    fn div<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr for concrete implementations")
    }

    fn pow<T: NumericType + Float>(base: Self::Repr<T>, exp: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Pow(Box::new(base), Box::new(exp))
    }

    fn neg<T: NumericType + Neg<Output = T>>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Neg(Box::new(expr))
    }

    fn ln<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Ln(Box::new(expr))
    }

    fn exp<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Exp(Box::new(expr))
    }

    fn sqrt<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Sqrt(Box::new(expr))
    }

    fn sin<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Sin(Box::new(expr))
    }

    fn cos<T: NumericType + Float>(expr: Self::Repr<T>) -> Self::Repr<T> {
        ASTRepr::Cos(Box::new(expr))
    }
}

impl StatisticalExpr for ASTEval {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::traits::ASTMathExpr;

    #[test]
    fn test_ast_eval_basic_construction() {
        // Test basic AST construction using ASTMathExpr
        let x = <ASTEval as ASTMathExpr>::var(0);
        let y = <ASTEval as ASTMathExpr>::var(1);
        let two = <ASTEval as ASTMathExpr>::constant(2.0);

        let expr = <ASTEval as ASTMathExpr>::add(<ASTEval as ASTMathExpr>::mul(x, two), y);

        // Verify the structure
        match expr {
            ASTRepr::Add(left, right) => {
                match (left.as_ref(), right.as_ref()) {
                    (ASTRepr::Mul(_, _), ASTRepr::Variable(1)) => {
                        // Correct structure: (x * 2) + y
                    }
                    _ => panic!("Unexpected AST structure"),
                }
            }
            _ => panic!("Expected addition at root"),
        }
    }

    #[test]
    fn test_ast_eval_transcendental_functions() {
        // Test transcendental function construction
        let x = <ASTEval as ASTMathExpr>::var(0);

        let sin_x = <ASTEval as ASTMathExpr>::sin(x.clone());
        let exp_x = <ASTEval as ASTMathExpr>::exp(x.clone());
        let ln_x = <ASTEval as ASTMathExpr>::ln(x);

        match sin_x {
            ASTRepr::Sin(_) => {}
            _ => panic!("Expected sine function"),
        }

        match exp_x {
            ASTRepr::Exp(_) => {}
            _ => panic!("Expected exponential function"),
        }

        match ln_x {
            ASTRepr::Ln(_) => {}
            _ => panic!("Expected natural logarithm function"),
        }
    }

    #[test]
    fn test_ast_eval_complex_expression() {
        // Test building a complex expression: sin(x^2) + exp(y)
        let x = <ASTEval as ASTMathExpr>::var(0);
        let y = <ASTEval as ASTMathExpr>::var(1);
        let two = <ASTEval as ASTMathExpr>::constant(2.0);

        let x_squared = <ASTEval as ASTMathExpr>::pow(x, two);
        let sin_x_squared = <ASTEval as ASTMathExpr>::sin(x_squared);
        let exp_y = <ASTEval as ASTMathExpr>::exp(y);
        let result = <ASTEval as ASTMathExpr>::add(sin_x_squared, exp_y);

        // Verify the operation count
        assert_eq!(result.count_operations(), 4); // pow, sin, exp, add
    }

    #[test]
    fn test_ast_eval_variable_creation() {
        // Test variable creation methods using index-based approach
        let var_by_index = <ASTEval as ASTMathExpr>::var(5);
        assert_eq!(var_by_index.variable_index(), Some(5));

        // Test additional variable indices
        let var_0 = <ASTEval as ASTMathExpr>::var(0);
        assert_eq!(var_0.variable_index(), Some(0));
        
        let var_10 = <ASTEval as ASTMathExpr>::var(10);
        assert_eq!(var_10.variable_index(), Some(10));
    }

    #[test]
    fn test_ast_eval_with_evaluation() {
        // Test that AST expressions can be evaluated
        let x = <ASTEval as ASTMathExpr>::var(0);
        let y = <ASTEval as ASTMathExpr>::var(1);
        let expr = <ASTEval as ASTMathExpr>::add(x, y);

        // Use the evaluation methods from the AST
        let result = expr.eval_with_vars(&[3.0, 4.0]);
        assert_eq!(result, 7.0);
    }
}
