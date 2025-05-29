//! AST Evaluation Interpreter
//!
//! This interpreter builds AST representations that can later be compiled
//! to native machine code for high-performance evaluation.

use crate::final_tagless::traits::{ASTMathExpr, ASTMathExprf64, MathExpr, NumericType, StatisticalExpr};
use crate::ast::ASTRepr;
use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// JIT evaluation interpreter that builds an intermediate representation
/// suitable for compilation with Cranelift or Rust codegen
///
/// This interpreter constructs a `ASTRepr` tree that can later be compiled
/// to native machine code for high-performance evaluation.
pub struct ASTEval;

impl ASTEval {
    /// Create a variable reference for JIT compilation using an index (efficient)
    #[must_use]
    pub fn var<T: NumericType>(index: usize) -> ASTRepr<T> {
        ASTRepr::Variable(index)
    }

    /// Convenience method for creating variables by name (for backward compatibility)
    /// Note: This no longer registers variables - use `ExpressionBuilder` for proper variable management
    #[must_use]
    pub fn var_by_name(_name: &str) -> ASTRepr<f64> {
        // Default to variable index 0 for backward compatibility
        ASTRepr::Variable(0)
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
        ASTRepr::Add(Box::new(left), Box::new(right))
    }

    fn sub(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Sub(Box::new(left), Box::new(right))
    }

    fn mul(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Mul(Box::new(left), Box::new(right))
    }

    fn div(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Div(Box::new(left), Box::new(right))
    }

    fn pow(base: Self::Repr, exp: Self::Repr) -> Self::Repr {
        ASTRepr::Pow(Box::new(base), Box::new(exp))
    }

    fn neg(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Neg(Box::new(expr))
    }

    fn ln(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Ln(Box::new(expr))
    }

    fn exp(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Exp(Box::new(expr))
    }

    fn sqrt(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Sqrt(Box::new(expr))
    }

    fn sin(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Sin(Box::new(expr))
    }

    fn cos(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Cos(Box::new(expr))
    }
}

impl ASTMathExprf64 for ASTEval {
    type Repr = ASTRepr<f64>;

    fn constant(value: f64) -> Self::Repr {
        ASTRepr::Constant(value)
    }

    fn var(index: usize) -> Self::Repr {
        ASTRepr::Variable(index)
    }

    fn add(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Add(Box::new(left), Box::new(right))
    }

    fn sub(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Sub(Box::new(left), Box::new(right))
    }

    fn mul(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Mul(Box::new(left), Box::new(right))
    }

    fn div(left: Self::Repr, right: Self::Repr) -> Self::Repr {
        ASTRepr::Div(Box::new(left), Box::new(right))
    }

    fn pow(base: Self::Repr, exp: Self::Repr) -> Self::Repr {
        ASTRepr::Pow(Box::new(base), Box::new(exp))
    }

    fn neg(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Neg(Box::new(expr))
    }

    fn ln(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Ln(Box::new(expr))
    }

    fn exp(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Exp(Box::new(expr))
    }

    fn sqrt(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Sqrt(Box::new(expr))
    }

    fn sin(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Sin(Box::new(expr))
    }

    fn cos(expr: Self::Repr) -> Self::Repr {
        ASTRepr::Cos(Box::new(expr))
    }
}

/// For compatibility with the main `MathExpr` trait, we provide a limited implementation
/// that works only with f64 types
impl MathExpr for ASTEval {
    type Repr<T> = ASTRepr<T>;

    fn constant<T: NumericType>(value: T) -> Self::Repr<T> {
        ASTRepr::Constant(value)
    }

    fn var<T: NumericType>(_name: &str) -> Self::Repr<T> {
        // Default to variable index 0 for compatibility
        ASTRepr::Variable(0)
    }

    fn var_by_index<T: NumericType>(index: usize) -> Self::Repr<T> {
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
        unimplemented!("Use ASTMathExpr or ASTMathExprf64 for concrete implementations")
    }

    fn sub<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Sub<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr or ASTMathExprf64 for concrete implementations")
    }

    fn mul<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Mul<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr or ASTMathExprf64 for concrete implementations")
    }

    fn div<L, R, Output>(_left: Self::Repr<L>, _right: Self::Repr<R>) -> Self::Repr<Output>
    where
        L: NumericType + Div<R, Output = Output>,
        R: NumericType,
        Output: NumericType,
    {
        unimplemented!("Use ASTMathExpr or ASTMathExprf64 for concrete implementations")
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
        // Test variable creation methods
        let var_by_index = ASTEval::var::<f64>(5);
        assert_eq!(var_by_index.variable_index(), Some(5));

        let var_by_name = ASTEval::var_by_name("test");
        assert_eq!(var_by_name.variable_index(), Some(0)); // Default to index 0
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

    #[test]
    fn test_ast_eval_f64_specialization() {
        // Test the f64-specific trait implementation
        use crate::final_tagless::traits::ASTMathExprf64;
        
        let x = <ASTEval as ASTMathExprf64>::var(0);
        let const_val = <ASTEval as ASTMathExprf64>::constant(3.14);
        let expr = <ASTEval as ASTMathExprf64>::mul(x, const_val);

        match expr {
            ASTRepr::Mul(left, right) => {
                match (left.as_ref(), right.as_ref()) {
                    (ASTRepr::Variable(0), ASTRepr::Constant(val)) => {
                        assert!((val - 3.14).abs() < 1e-10);
                    }
                    _ => panic!("Unexpected structure"),
                }
            }
            _ => panic!("Expected multiplication"),
        }
    }
} 