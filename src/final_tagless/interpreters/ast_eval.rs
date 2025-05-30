//! AST Evaluation Interpreter
//!
//! This interpreter builds AST representations that can later be compiled
//! to native machine code for high-performance evaluation.

use crate::final_tagless::ASTRepr;
use crate::final_tagless::traits::{ASTMathExpr, NumericType};
use num_traits::Float;

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

    /// Convenience method for creating variables by name (uses global registry)
    #[must_use]
    pub fn var_by_name(name: &str) -> ASTRepr<f64> {
        // Register the variable in the global registry and return its index
        let index = crate::final_tagless::variables::register_variable(name);
        ASTRepr::Variable(index)
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
            (ASTRepr::Constant(l), ASTRepr::Constant(r)) => ASTRepr::Constant(l.powf(*r)),
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
            #[cfg(feature = "logexp")]
            _ => ASTRepr::Log(Box::new(expr)),
            #[cfg(not(feature = "logexp"))]
            _ => panic!("ln requires logexp feature"),
        }
    }

    fn exp(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.exp()),
            #[cfg(feature = "logexp")]
            _ => ASTRepr::Exp(Box::new(expr)),
            #[cfg(not(feature = "logexp"))]
            _ => panic!("exp requires logexp feature"),
        }
    }

    fn sqrt(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.sqrt()),
            _ => {
                let half = ASTRepr::Constant(0.5);
                ASTRepr::Pow(Box::new(expr), Box::new(half))
            }
        }
    }

    fn sin(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.sin()),
            _ => ASTRepr::Trig(Box::new(
                crate::ast::function_categories::TrigCategory::sin(expr),
            )),
        }
    }

    fn cos(expr: Self::Repr) -> Self::Repr {
        match expr {
            ASTRepr::Constant(x) => ASTRepr::Constant(x.cos()),
            _ => ASTRepr::Trig(Box::new(
                crate::ast::function_categories::TrigCategory::cos(expr),
            )),
        }
    }
}

/// TODO: Fix trait bounds issue - for now, use ASTMathExpr instead
// impl MathExpr for ASTEval {
//     type Repr<T> = ASTRepr<T>;
//     // ... implementation would go here when trait bounds are resolved
// }

// impl StatisticalExpr for ASTEval {} // Requires MathExpr

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
            ASTRepr::Trig(_) => {}
            _ => panic!("Expected trig function"),
        }

        #[cfg(feature = "logexp")]
        match exp_x {
            ASTRepr::Exp(_) => {}
            _ => panic!("Expected exponential function"),
        }

        #[cfg(feature = "logexp")]
        match ln_x {
            ASTRepr::Log(_) => {}
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
}
