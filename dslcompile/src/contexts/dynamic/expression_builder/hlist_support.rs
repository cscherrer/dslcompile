//! HList Integration Support for DSLCompile
//!
//! This module provides zero-cost heterogeneous operations using frunk HLists.
//! It enables type-safe, compile-time optimized parameter passing and evaluation
//! without runtime type erasure or Vec flattening.
//!
//! ## Key Components
//!
//! - `IntoVarHList`: Convert values into typed variable expressions
//! - `IntoConcreteSignature`: Generate function signatures from HList types  
//! - `HListEval`: Zero-cost evaluation with HList storage
//! - `FunctionSignature`: Code generation support

use crate::{
    ast::{Scalar, ast_repr::ASTRepr},
    contexts::dynamic::expression_builder::{
        DynamicContext, TypedBuilderExpr, type_system::DslType,
    },
};
use frunk::{HCons, HNil};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

// ============================================================================
// CORE HLIST INTEGRATION TRAITS
// ============================================================================

/// Trait for converting values into typed variable expressions using HLists
///
/// This enables zero-cost conversion from heterogeneous data into typed
/// expression variables while preserving type information at compile time.
pub trait IntoVarHList {
    type Output;
    fn into_vars(self, ctx: &DynamicContext) -> Self::Output;
}

/// Trait for converting HLists into concrete function signatures
///
/// This enables automatic generation of function signatures for code generation
/// based on the types present in an HList structure.
pub trait IntoConcreteSignature {
    fn concrete_signature() -> FunctionSignature;
}

/// Zero-cost HList evaluation trait - no flattening to Vec
///
/// This trait provides efficient evaluation of expressions using HList storage,
/// avoiding the performance overhead of Vec flattening while maintaining type safety.
pub trait HListEval<T: Scalar> {
    /// Evaluate AST with zero-cost HList storage
    fn eval_expr(&self, ast: &ASTRepr<T>) -> T;

    /// Get variable value by index with zero runtime dispatch
    fn get_var(&self, index: usize) -> T;

    /// Apply a lambda function to arguments from this HList
    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<T>, args: &[T]) -> T;
}

// ============================================================================
// FUNCTION SIGNATURE SUPPORT
// ============================================================================

/// Function signature for code generation
///
/// Represents the signature of a generated function including parameter types
/// and return type, used for generating efficient compiled code.
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    params: Vec<String>,
    return_type: String,
}

impl FunctionSignature {
    pub fn new(param_types: Vec<&str>) -> Self {
        Self {
            params: param_types
                .iter()
                .enumerate()
                .map(|(i, t)| format!("x{i}: {t}"))
                .collect(),
            return_type: "f64".to_string(), // Default return type
        }
    }

    pub fn parameters(&self) -> String {
        self.params.join(", ")
    }

    pub fn return_type(&self) -> &str {
        &self.return_type
    }

    pub fn function_name(&self) -> String {
        // Generate a unique function name based on signature
        let mut hasher = DefaultHasher::new();
        self.params.hash(&mut hasher);
        self.return_type.hash(&mut hasher);
        format!("expr_{:x}", hasher.finish())
    }
}

// ============================================================================
// HLIST EVALUATION IMPLEMENTATIONS
// ============================================================================

// Base case: HNil - no values stored (generic implementation)
impl<T> HListEval<T> for HNil
where
    T: Scalar + Copy + num_traits::Float + num_traits::FromPrimitive,
{
    fn eval_expr(&self, ast: &ASTRepr<T>) -> T {
        // Can only evaluate constant expressions with no variables
        match ast {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(_) => panic!("Cannot evaluate variable with empty HList"),
            ASTRepr::Add(left, right) => self.eval_expr(left) + self.eval_expr(right),
            ASTRepr::Sub(left, right) => self.eval_expr(left) - self.eval_expr(right),
            ASTRepr::Mul(left, right) => self.eval_expr(left) * self.eval_expr(right),
            ASTRepr::Div(left, right) => self.eval_expr(left) / self.eval_expr(right),
            ASTRepr::Pow(base, exp) => self.eval_expr(base).powf(self.eval_expr(exp)),
            ASTRepr::Neg(inner) => -self.eval_expr(inner),
            ASTRepr::Ln(inner) => self.eval_expr(inner).ln(),
            ASTRepr::Exp(inner) => self.eval_expr(inner).exp(),
            ASTRepr::Sin(inner) => self.eval_expr(inner).sin(),
            ASTRepr::Cos(inner) => self.eval_expr(inner).cos(),
            ASTRepr::Sqrt(inner) => self.eval_expr(inner).sqrt(),
            ASTRepr::Sum(_collection) => {
                // TODO: Implement collection evaluation
                T::from_f64(0.0).unwrap_or_else(|| panic!("Cannot convert 0.0 to target type"))
            }
            ASTRepr::Lambda(lambda) => {
                // Lambda expressions can be evaluated if they have no variables (constant lambdas)
                if lambda.var_indices.is_empty() {
                    self.eval_expr(&lambda.body)
                } else {
                    panic!(
                        "Cannot evaluate lambda expression with unbound variables without arguments"
                    )
                }
            }
            ASTRepr::BoundVar(index) => {
                // BoundVar behaves like Variable for HList evaluation
                self.get_var(*index)
            }
            ASTRepr::Let(_, expr, body) => {
                // Let expressions: evaluate expr then body (simplified version)
                // TODO: Proper Let evaluation would require variable substitution
                let _expr_val = self.eval_expr(expr);
                self.eval_expr(body)
            }
        }
    }

    fn get_var(&self, _index: usize) -> T {
        panic!("Variable index out of bounds in HNil")
    }

    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<T>, _args: &[T]) -> T {
        // For HNil, we can only evaluate constant lambdas
        if lambda.var_indices.is_empty() {
            self.eval_expr(&lambda.body)
        } else {
            panic!("Cannot apply lambda with variables using empty HList")
        }
    }
}

// Generic implementation for any Scalar type at head position
impl<T, Tail> HListEval<T> for HCons<T, Tail>
where
    T: Scalar + Copy + num_traits::Float + num_traits::FromPrimitive,
    Tail: HListEval<T>,
{
    fn eval_expr(&self, ast: &ASTRepr<T>) -> T {
        match ast {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => self.get_var(*index),
            ASTRepr::Add(left, right) => self.eval_expr(left) + self.eval_expr(right),
            ASTRepr::Sub(left, right) => self.eval_expr(left) - self.eval_expr(right),
            ASTRepr::Mul(left, right) => self.eval_expr(left) * self.eval_expr(right),
            ASTRepr::Div(left, right) => self.eval_expr(left) / self.eval_expr(right),
            ASTRepr::Pow(base, exp) => self.eval_expr(base).powf(self.eval_expr(exp)),
            ASTRepr::Neg(inner) => -self.eval_expr(inner),
            ASTRepr::Ln(inner) => self.eval_expr(inner).ln(),
            ASTRepr::Exp(inner) => self.eval_expr(inner).exp(),
            ASTRepr::Sin(inner) => self.eval_expr(inner).sin(),
            ASTRepr::Cos(inner) => self.eval_expr(inner).cos(),
            ASTRepr::Sqrt(inner) => self.eval_expr(inner).sqrt(),
            ASTRepr::Sum(_collection) => {
                // TODO: Implement collection evaluation with HList storage
                T::from_f64(0.0).unwrap_or_else(|| panic!("Cannot convert 0.0 to target type"))
            }
            ASTRepr::Lambda(lambda) => {
                // Lambda expressions can be evaluated if they have no variables (constant lambdas)
                if lambda.var_indices.is_empty() {
                    self.eval_expr(&lambda.body)
                } else {
                    panic!(
                        "Cannot evaluate lambda expression with unbound variables without arguments"
                    )
                }
            }
            ASTRepr::BoundVar(index) => {
                // BoundVar behaves like Variable for HList evaluation
                self.get_var(*index)
            }
            ASTRepr::Let(_, expr, body) => {
                // Let expressions: evaluate expr then body (simplified version)
                // TODO: Proper Let evaluation would require variable substitution
                let _expr_val = self.eval_expr(expr);
                self.eval_expr(body)
            }
        }
    }

    fn get_var(&self, index: usize) -> T {
        match index {
            0 => self.head,
            n => self.tail.get_var(n - 1),
        }
    }

    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<T>, args: &[T]) -> T {
        // Create a substitution context by binding lambda variables to arguments
        if lambda.var_indices.len() > args.len() {
            panic!(
                "Not enough arguments for lambda application: expected {}, got {}",
                lambda.var_indices.len(),
                args.len()
            );
        }

        // Use helper function for variable substitution evaluation
        eval_lambda_with_substitution(self, &lambda.body, &lambda.var_indices, args)
    }
}

/// Helper function to evaluate lambda body with variable substitution
fn eval_lambda_with_substitution<T, H>(
    hlist: &H,
    body: &ASTRepr<T>,
    var_indices: &[usize],
    args: &[T],
) -> T
where
    T: Scalar + Copy + num_traits::Float + num_traits::FromPrimitive,
    H: HListEval<T>,
{
    match body {
        ASTRepr::Variable(index) => {
            // Check if this variable is bound by the lambda
            if let Some(pos) = var_indices.iter().position(|&v| v == *index) {
                args[pos] // Use the argument value
            } else {
                hlist.get_var(*index) // Use HList variable
            }
        }
        ASTRepr::Constant(value) => *value,
        ASTRepr::Add(left, right) => {
            let left_val = eval_lambda_with_substitution(hlist, left, var_indices, args);
            let right_val = eval_lambda_with_substitution(hlist, right, var_indices, args);
            left_val + right_val
        }
        ASTRepr::Sub(left, right) => {
            let left_val = eval_lambda_with_substitution(hlist, left, var_indices, args);
            let right_val = eval_lambda_with_substitution(hlist, right, var_indices, args);
            left_val - right_val
        }
        ASTRepr::Mul(left, right) => {
            let left_val = eval_lambda_with_substitution(hlist, left, var_indices, args);
            let right_val = eval_lambda_with_substitution(hlist, right, var_indices, args);
            left_val * right_val
        }
        ASTRepr::Div(left, right) => {
            let left_val = eval_lambda_with_substitution(hlist, left, var_indices, args);
            let right_val = eval_lambda_with_substitution(hlist, right, var_indices, args);
            left_val / right_val
        }
        ASTRepr::Pow(base, exp) => {
            let base_val = eval_lambda_with_substitution(hlist, base, var_indices, args);
            let exp_val = eval_lambda_with_substitution(hlist, exp, var_indices, args);
            base_val.powf(exp_val)
        }
        ASTRepr::Neg(inner) => {
            let inner_val = eval_lambda_with_substitution(hlist, inner, var_indices, args);
            -inner_val
        }
        ASTRepr::Ln(inner) => {
            let inner_val = eval_lambda_with_substitution(hlist, inner, var_indices, args);
            inner_val.ln()
        }
        ASTRepr::Exp(inner) => {
            let inner_val = eval_lambda_with_substitution(hlist, inner, var_indices, args);
            inner_val.exp()
        }
        ASTRepr::Sin(inner) => {
            let inner_val = eval_lambda_with_substitution(hlist, inner, var_indices, args);
            inner_val.sin()
        }
        ASTRepr::Cos(inner) => {
            let inner_val = eval_lambda_with_substitution(hlist, inner, var_indices, args);
            inner_val.cos()
        }
        ASTRepr::Sqrt(inner) => {
            let inner_val = eval_lambda_with_substitution(hlist, inner, var_indices, args);
            inner_val.sqrt()
        }
        ASTRepr::Lambda(nested_lambda) => {
            // Nested lambda: apply with remaining arguments
            if args.len() >= nested_lambda.var_indices.len() {
                let (lambda_args, remaining) = args.split_at(nested_lambda.var_indices.len());
                let result = hlist.apply_lambda(nested_lambda, lambda_args);
                if remaining.is_empty() {
                    result
                } else {
                    // TODO: Handle remaining arguments for currying
                    result
                }
            } else {
                panic!("Not enough arguments for nested lambda")
            }
        }
        _ => {
            // For other cases, fall back to standard evaluation
            hlist.eval_expr(body)
        }
    }
}

// ============================================================================
// HLIST BASE CASES
// ============================================================================

impl IntoVarHList for HNil {
    type Output = HNil;
    fn into_vars(self, _ctx: &DynamicContext) -> Self::Output {
        HNil
    }
}

impl IntoConcreteSignature for HNil {
    fn concrete_signature() -> FunctionSignature {
        FunctionSignature::new(vec![])
    }
}

// ============================================================================
// HLIST RECURSIVE CASES
// ============================================================================

impl<T, Tail> IntoVarHList for HCons<T, Tail>
where
    T: DslType + crate::ast::Scalar,
    Tail: IntoVarHList,
{
    type Output = HCons<TypedBuilderExpr<T>, Tail::Output>;

    fn into_vars(self, ctx: &DynamicContext) -> Self::Output {
        // Create a typed context for this type using new_explicit
        let mut ctx_typed: DynamicContext<T, 0> = DynamicContext::new_explicit();
        let head_expr = ctx_typed.var();
        let tail_vars = self.tail.into_vars(ctx);
        HCons {
            head: head_expr,
            tail: tail_vars,
        }
    }
}

impl<T, Tail> IntoConcreteSignature for HCons<T, Tail>
where
    T: DslType,
    Tail: IntoConcreteSignature,
{
    fn concrete_signature() -> FunctionSignature {
        let mut sig = Tail::concrete_signature();
        sig.params.insert(0, format!("x0: {}", T::TYPE_NAME));
        // Update parameter indices
        for (i, param) in sig.params.iter_mut().enumerate().skip(1) {
            *param = param.replace(&format!("x{}", i - 1), &format!("x{i}"));
        }
        sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frunk::hlist;

    #[test]
    fn test_function_signature_generation() {
        let sig = FunctionSignature::new(vec!["f64", "i32", "f64"]);
        assert_eq!(sig.parameters(), "x0: f64, x1: i32, x2: f64");
        assert_eq!(sig.return_type(), "f64");
        assert!(sig.function_name().starts_with("expr_"));
    }

    #[test]
    fn test_hlist_eval_constants() {
        let hlist = hlist![2.0, 3.0];
        let ast = ASTRepr::Add(
            Box::new(ASTRepr::Constant(1.0)),
            Box::new(ASTRepr::Constant(4.0)),
        );
        assert_eq!(hlist.eval_expr(&ast), 5.0);
    }

    #[test]
    fn test_hlist_eval_variables() {
        let hlist = hlist![2.0, 3.0];
        let ast = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );
        assert_eq!(hlist.eval_expr(&ast), 5.0);
    }

    #[test]
    fn test_hlist_get_var() {
        let hlist = hlist![10.0, 20.0, 30.0];
        assert_eq!(hlist.get_var(0), 10.0);
        assert_eq!(hlist.get_var(1), 20.0);
        assert_eq!(hlist.get_var(2), 30.0);
    }

    #[test]
    #[should_panic(expected = "Variable index out of bounds in HNil")]
    fn test_hnil_get_var_panics() {
        let hlist = HNil;
        let _result: f64 = hlist.get_var(0);
    }

    #[test]
    fn test_lambda_constant_evaluation() {
        use crate::ast::ast_repr::Lambda;

        let hlist = hlist![2.0, 3.0];

        // Constant lambda: λ().42
        let constant_lambda = Lambda::new(vec![], Box::new(ASTRepr::Constant(42.0)));
        let result = hlist.apply_lambda(&constant_lambda, &[]);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_lambda_single_argument() {
        use crate::ast::ast_repr::Lambda;

        let hlist = hlist![10.0, 20.0];

        // Identity lambda: λx.x
        let identity_lambda = Lambda::single(0, Box::new(ASTRepr::Variable(0)));
        let result = hlist.apply_lambda(&identity_lambda, &[5.0]);
        assert_eq!(result, 5.0);

        // Doubling lambda: λx.x*2
        let double_lambda = Lambda::single(
            0,
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(2.0)),
            )),
        );
        let result = hlist.apply_lambda(&double_lambda, &[7.0]);
        assert_eq!(result, 14.0);
    }

    #[test]
    fn test_lambda_multiple_arguments() {
        use crate::ast::ast_repr::Lambda;

        let hlist = hlist![100.0];

        // Addition lambda: λ(x,y).x+y
        let add_lambda = Lambda::new(
            vec![0, 1],
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Variable(1)),
            )),
        );
        let result = hlist.apply_lambda(&add_lambda, &[3.0, 4.0]);
        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_lambda_with_hlist_variables() {
        use crate::ast::ast_repr::Lambda;

        let hlist = hlist![10.0, 20.0];

        // Lambda that uses both lambda argument and HList variable: λx.x + hlist[1]
        let mixed_lambda = Lambda::single(
            0,
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Variable(0)), // Lambda argument
                Box::new(ASTRepr::Variable(1)), // HList variable
            )),
        );
        let result = hlist.apply_lambda(&mixed_lambda, &[5.0]);
        assert_eq!(result, 25.0); // 5.0 + 20.0
    }

    #[test]
    #[should_panic(expected = "Not enough arguments for lambda application")]
    fn test_lambda_insufficient_arguments() {
        use crate::ast::ast_repr::Lambda;

        let hlist = hlist![1.0];
        let lambda = Lambda::new(vec![0, 1], Box::new(ASTRepr::Variable(0)));
        let _result = hlist.apply_lambda(&lambda, &[1.0]); // Only 1 arg, needs 2
    }
}
