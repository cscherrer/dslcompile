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
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    fn get_var(&self, _index: usize) -> T {
        panic!("Variable index out of bounds in HNil")
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
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    fn get_var(&self, index: usize) -> T {
        match index {
            0 => self.head,
            n => self.tail.get_var(n - 1),
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
        // Create a typed context for this type
        let mut ctx_typed: DynamicContext<T> = DynamicContext::new();
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
        hlist.get_var(0);
    }
}
