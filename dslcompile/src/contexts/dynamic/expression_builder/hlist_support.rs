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
    ast::{ExpressionType, Scalar, ast_repr::ASTRepr},
    contexts::dynamic::expression_builder::{DynamicContext, DynamicExpr, type_system::DslType},
};
use frunk::{HCons, HNil};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

// ============================================================================
// CORE HLIST INTEGRATION TRAITS
// ============================================================================

/// Trait for converting values into typed variable expressions using `HLists`
///
/// This enables zero-cost conversion from heterogeneous data into typed
/// expression variables while preserving type information at compile time.
pub trait IntoVarHList {
    type Output;
    fn into_vars(self, ctx: &DynamicContext) -> Self::Output;
}

/// Trait for converting `HLists` into concrete function signatures
///
/// This enables automatic generation of function signatures for code generation
/// based on the types present in an `HList` structure.
pub trait IntoConcreteSignature {
    fn concrete_signature() -> FunctionSignature;
}

/// Zero-cost `HList` evaluation trait - no flattening to Vec
///
/// This trait provides efficient evaluation of expressions using `HList` storage,
/// avoiding the performance overhead of Vec flattening while maintaining type safety.
pub trait HListEval<T: Scalar + ExpressionType + PartialEq> {
    /// Evaluate AST with zero-cost `HList` storage
    fn eval_expr(&self, ast: &ASTRepr<T>) -> T;

    /// Get variable value by index with zero runtime dispatch
    fn get_var(&self, index: usize) -> T;

    /// Apply a lambda function to arguments from this `HList`
    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<T>, args: &[T]) -> T;

    /// Convert `HList` variables to Vec for external evaluation
    fn to_variables_vec(&self) -> Vec<T>;

    /// Get the number of variables available in this `HList`
    fn variable_count(&self) -> usize;

    /// Evaluate collection summation directly using `HList` heterogeneous capabilities
    fn eval_collection_sum(&self, collection: &crate::ast::ast_repr::Collection<T>) -> T
    where
        T: num_traits::Zero + num_traits::Float + num_traits::FromPrimitive,
    {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Variable(_index) => {
                // No variables available in empty HList - return 0 as placeholder
                T::zero()
            }
            Collection::Map {
                lambda,
                collection: inner_collection,
            } => {
                // For Map collections, delegate to specialized handling
                self.eval_map_collection(lambda, inner_collection)
            }
            Collection::Constant(data) => {
                // Sum directly over embedded data array
                data.iter().fold(T::zero(), |acc, x| acc + *x)
            }
            Collection::Range { start, end } => {
                // Handle mathematical ranges directly
                let start_val = self.eval_expr(start);
                let end_val = self.eval_expr(end);

                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    sum = sum + i_val;
                }
                sum
            }
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => {
                // Single element collection
                self.eval_expr(expr)
            }
            Collection::Filter { .. } => {
                // TODO: Implement filtered collection evaluation
                T::zero()
            }
        }
    }

    /// Evaluate Map collection by applying lambda to data elements
    /// Default implementation: treat as scalar
    fn eval_map_collection(
        &self,
        lambda: &crate::ast::ast_repr::Lambda<T>,
        collection: &crate::ast::ast_repr::Collection<T>,
    ) -> T
    where
        T: num_traits::Zero + num_traits::Float + num_traits::FromPrimitive,
    {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Variable(index) => {
                // Default: treat as scalar variable
                let scalar_val = self.get_var(*index);
                self.apply_lambda(lambda, &[scalar_val])
            }
            Collection::Range { start, end } => {
                // Apply lambda to each element in the range
                let start_val = self.eval_expr(start);
                let end_val = self.eval_expr(end);

                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    let lambda_result = self.apply_lambda(lambda, &[i_val]);
                    sum = sum + lambda_result;
                }
                sum
            }
            Collection::Constant(data) => {
                // Apply lambda to each element in the data array
                data.iter()
                    .map(|x| self.apply_lambda(lambda, &[*x]))
                    .fold(T::zero(), |acc, x| acc + x)
            }
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => {
                // Apply lambda to the single element
                let element_val = self.eval_expr(expr);
                self.apply_lambda(lambda, &[element_val])
            }
            _ => T::zero(),
        }
    }
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
    #[must_use]
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

    #[must_use]
    pub fn parameters(&self) -> String {
        self.params.join(", ")
    }

    #[must_use]
    pub fn return_type(&self) -> &str {
        &self.return_type
    }

    #[must_use]
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
    T: Scalar + ExpressionType + PartialEq + Copy + num_traits::Float + num_traits::FromPrimitive,
{
    fn eval_expr(&self, ast: &ASTRepr<T>) -> T {
        // Can only evaluate constant expressions with no variables
        match ast {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(_) => panic!("Cannot evaluate variable with empty HList"),
            ASTRepr::Add(terms) => {
                use num_traits::Zero;
                terms
                    .elements()
                    .map(|term| self.eval_expr(term))
                    .fold(T::zero(), |acc, x| acc + x)
            }
            ASTRepr::Sub(left, right) => self.eval_expr(left) - self.eval_expr(right),
            ASTRepr::Mul(factors) => {
                use num_traits::One;
                factors
                    .elements()
                    .map(|factor| self.eval_expr(factor))
                    .fold(T::one(), |acc, x| acc * x)
            }
            ASTRepr::Div(left, right) => self.eval_expr(left) / self.eval_expr(right),
            ASTRepr::Pow(base, exp) => self.eval_expr(base).powf(self.eval_expr(exp)),
            ASTRepr::Neg(inner) => -self.eval_expr(inner),
            ASTRepr::Ln(inner) => self.eval_expr(inner).ln(),
            ASTRepr::Exp(inner) => self.eval_expr(inner).exp(),
            ASTRepr::Sin(inner) => self.eval_expr(inner).sin(),
            ASTRepr::Cos(inner) => self.eval_expr(inner).cos(),
            ASTRepr::Sqrt(inner) => self.eval_expr(inner).sqrt(),
            ASTRepr::Sum(collection) => {
                // Use HList-specific collection evaluation instead of falling back to eval_with_vars
                self.eval_collection_sum(collection)
            }
            ASTRepr::Lambda(lambda) => {
                // Lambda expressions can be evaluated if they have no variables (constant lambdas)
                if lambda.var_indices.is_empty() {
                    self.eval_expr(&lambda.body)
                } else {
                    panic!("Cannot evaluate lambda without function application")
                }
            }
            ASTRepr::BoundVar(index) => {
                // BoundVar behaves like Variable for HList evaluation, but with custom error message
                assert!(
                    (*index < <Self as HListEval<T>>::variable_count(self)),
                    "BoundVar index {index} is out of bounds"
                );
                self.get_var(*index)
            }
            ASTRepr::Let(binding_var, expr, body) => {
                // Let expressions: evaluate expr and bind it to the variable
                let expr_val = self.eval_expr(expr);
                eval_with_substitution(self, body, *binding_var, expr_val)
            }
        }
    }

    fn get_var(&self, index: usize) -> T {
        panic!(
            "Variable index {index} out of bounds in HNil (no variables available). \
            This usually means the expression contains Variable nodes but is being evaluated \
            with an empty variable list. For embedded data expressions, this might indicate \
            incorrect lambda evaluation or missing bound variable substitution."
        )
    }

    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<T>, args: &[T]) -> T {
        // For HNil, we can evaluate lambdas if arguments are provided for all bound variables
        if lambda.var_indices.is_empty() {
            self.eval_expr(&lambda.body)
        } else if lambda.var_indices.len() == args.len() {
            // Use the lambda evaluation helper with variable substitution
            eval_lambda_with_substitution(self, &lambda.body, &lambda.var_indices, args)
        } else {
            panic!(
                "Cannot apply lambda: expected {} arguments, got {}",
                lambda.var_indices.len(),
                args.len()
            )
        }
    }

    fn to_variables_vec(&self) -> Vec<T> {
        // Empty HList produces empty vector
        Vec::new()
    }

    fn variable_count(&self) -> usize {
        0
    }
}

// Generic implementation for any Scalar type at head position
impl<T, Tail> HListEval<T> for HCons<T, Tail>
where
    T: Scalar + ExpressionType + PartialEq + Copy + num_traits::Float + num_traits::FromPrimitive,
    Tail: HListEval<T>,
{
    fn eval_expr(&self, ast: &ASTRepr<T>) -> T {
        match ast {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => self.get_var(*index),
            ASTRepr::Add(terms) => terms
                .elements()
                .map(|term| self.eval_expr(term))
                .fold(T::from(0.0).unwrap(), |acc, x| acc + x),
            ASTRepr::Sub(left, right) => self.eval_expr(left) - self.eval_expr(right),
            ASTRepr::Mul(factors) => factors
                .elements()
                .map(|factor| self.eval_expr(factor))
                .fold(T::from(1.0).unwrap(), |acc, x| acc * x),
            ASTRepr::Div(left, right) => self.eval_expr(left) / self.eval_expr(right),
            ASTRepr::Pow(base, exp) => self.eval_expr(base).powf(self.eval_expr(exp)),
            ASTRepr::Neg(inner) => -self.eval_expr(inner),
            ASTRepr::Ln(inner) => self.eval_expr(inner).ln(),
            ASTRepr::Exp(inner) => self.eval_expr(inner).exp(),
            ASTRepr::Sin(inner) => self.eval_expr(inner).sin(),
            ASTRepr::Cos(inner) => self.eval_expr(inner).cos(),
            ASTRepr::Sqrt(inner) => self.eval_expr(inner).sqrt(),
            ASTRepr::Sum(collection) => {
                // Use HList-specific collection evaluation instead of falling back to eval_with_vars
                self.eval_collection_sum(collection)
            }
            ASTRepr::Lambda(lambda) => {
                // Lambda expressions can be evaluated if they have no variables (constant lambdas)
                if lambda.var_indices.is_empty() {
                    self.eval_expr(&lambda.body)
                } else {
                    panic!("Cannot evaluate lambda without function application")
                }
            }
            ASTRepr::BoundVar(index) => {
                // BoundVar behaves like Variable for HList evaluation, but with custom error message
                assert!(
                    (*index < <Self as HListEval<T>>::variable_count(self)),
                    "BoundVar index {index} is out of bounds"
                );
                self.get_var(*index)
            }
            ASTRepr::Let(binding_var, expr, body) => {
                // Let expressions: evaluate expr and bind it to the variable
                let expr_val = self.eval_expr(expr);
                eval_with_substitution(self, body, *binding_var, expr_val)
            }
        }
    }

    fn get_var(&self, index: usize) -> T {
        match index {
            0 => self.head,
            n => {
                // Check if we'll go out of bounds before recursing
                // We need n-1 to be a valid index in the tail, so n-1 < tail.variable_count()
                // which means n <= tail.variable_count()
                // BUT since we already handled index 0, we need n-1 < tail.variable_count()
                assert!(
                    (n <= self.tail.variable_count()),
                    "Variable index {index} is out of bounds for evaluation"
                );
                self.tail.get_var(n - 1)
            }
        }
    }

    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<T>, args: &[T]) -> T {
        // Create a substitution context by binding lambda variables to arguments
        assert!(
            (lambda.var_indices.len() <= args.len()),
            "Not enough arguments for lambda application: expected {}, got {}",
            lambda.var_indices.len(),
            args.len()
        );

        // Use helper function for variable substitution evaluation
        eval_lambda_with_substitution(self, &lambda.body, &lambda.var_indices, args)
    }

    fn to_variables_vec(&self) -> Vec<T> {
        // Build vector by collecting head and tail variables
        let mut vars = vec![self.head];
        vars.extend(self.tail.to_variables_vec());
        vars
    }

    fn variable_count(&self) -> usize {
        1 + self.tail.variable_count()
    }

    fn eval_collection_sum(&self, collection: &crate::ast::ast_repr::Collection<T>) -> T
    where
        T: num_traits::Zero + num_traits::Float + num_traits::FromPrimitive,
    {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Variable(index) => {
                // For scalar elements, check if this index refers to our scalar
                if *index == 0 {
                    // This Collection::Variable refers to our scalar - treat as single-element collection
                    self.head
                } else {
                    // Delegate to tail with adjusted index
                    let adjusted_collection = Collection::Variable(index - 1);
                    self.tail.eval_collection_sum(&adjusted_collection)
                }
            }
            Collection::Map {
                lambda,
                collection: inner_collection,
            } => {
                // For Map collections, delegate to specialized handling
                self.eval_map_collection(lambda, inner_collection)
            }
            Collection::Range { start, end } => {
                // Handle mathematical ranges directly
                let start_val = self.eval_expr(start);
                let end_val = self.eval_expr(end);

                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    sum = sum + i_val;
                }
                sum
            }
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => {
                // Single element collection
                self.eval_expr(expr)
            }
            Collection::Constant(data) => {
                // Sum directly over embedded data array
                data.iter().fold(T::zero(), |acc, x| acc + *x)
            }
            Collection::Filter { .. } => {
                // TODO: Implement filtered collection evaluation
                T::zero()
            }
        }
    }
}

// Simple implementation for Vec<f64> in HList - treats it as non-variable data
// This allows heterogeneous HLists like HCons<Vec<f64>, HCons<f64, HNil>>
impl<Tail> HListEval<f64> for HCons<Vec<f64>, Tail>
where
    Tail: HListEval<f64>,
{
    fn eval_expr(&self, ast: &ASTRepr<f64>) -> f64 {
        match ast {
            ASTRepr::Sum(collection) => {
                // Use our specialized collection sum evaluation
                self.eval_collection_sum(collection)
            }
            _ => {
                // For other AST nodes, delegate to tail
                self.tail.eval_expr(ast)
            }
        }
    }

    fn get_var(&self, index: usize) -> f64 {
        // Vec<f64> doesn't provide scalar variables, delegate to tail
        self.tail.get_var(index)
    }

    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<f64>, args: &[f64]) -> f64 {
        // Delegate lambda evaluation to tail
        self.tail.apply_lambda(lambda, args)
    }

    fn to_variables_vec(&self) -> Vec<f64> {
        // Vec<f64> is data, not variables - just return tail's variables
        self.tail.to_variables_vec()
    }

    fn variable_count(&self) -> usize {
        // Vec<f64> doesn't count as variables - only count tail
        self.tail.variable_count()
    }

    fn eval_map_collection(
        &self,
        lambda: &crate::ast::ast_repr::Lambda<f64>,
        collection: &crate::ast::ast_repr::Collection<f64>,
    ) -> f64
    where
        f64: num_traits::Zero + num_traits::Float + num_traits::FromPrimitive,
    {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Variable(index) => {
                // For index 0, this refers to our Vec<f64> data
                if *index == 0 {
                    // Apply lambda to each element in our vector data
                    use num_traits::Zero;
                    self.head
                        .iter()
                        .map(|&x| self.tail.apply_lambda(lambda, &[x]))
                        .fold(f64::zero(), |acc, x| acc + x)
                } else {
                    // Delegate to tail with adjusted index
                    let adjusted_collection = Collection::Variable(index - 1);
                    self.tail.eval_map_collection(lambda, &adjusted_collection)
                }
            }
            _ => {
                // For other collection types, use default implementation
                // Delegate to the default trait implementation
                self.tail.eval_map_collection(lambda, collection)
            }
        }
    }

    fn eval_collection_sum(&self, collection: &crate::ast::ast_repr::Collection<f64>) -> f64
    where
        f64: num_traits::Zero + num_traits::Float + num_traits::FromPrimitive,
    {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Variable(index) => {
                // For index 0, this refers to our Vec<f64> data - just sum it directly
                if *index == 0 {
                    use num_traits::Zero;
                    self.head.iter().fold(f64::zero(), |acc, &x| acc + x)
                } else {
                    // Delegate to tail with adjusted index
                    let adjusted_collection = Collection::Variable(index - 1);
                    self.tail.eval_collection_sum(&adjusted_collection)
                }
            }
            Collection::Map { lambda, collection: inner_collection } => {
                // For Map collections, use our specialized handling
                self.eval_map_collection(lambda, inner_collection)
            }
            _ => {
                // For other collection types, use default behavior
                self.tail.eval_collection_sum(collection)
            }
        }
    }
}

/// Helper function for evaluating expressions with a single variable substitution (for Let bindings)
fn eval_with_substitution<T, H>(
    hlist: &H,
    expr: &ASTRepr<T>,
    substitute_var: usize,
    substitute_value: T,
) -> T
where
    T: Scalar + ExpressionType + PartialEq + Copy + num_traits::Float + num_traits::FromPrimitive,
    H: HListEval<T>,
{
    match expr {
        ASTRepr::Variable(index) => {
            if *index == substitute_var {
                substitute_value
            } else {
                hlist.get_var(*index)
            }
        }
        ASTRepr::BoundVar(index) => {
            // BoundVar should not be affected by Let substitution
            assert!(
                (*index < <H as HListEval<T>>::variable_count(hlist)),
                "BoundVar index {index} is out of bounds"
            );
            hlist.get_var(*index)
        }
        ASTRepr::Constant(value) => *value,
        ASTRepr::Add(terms) => terms
            .elements()
            .map(|term| eval_with_substitution(hlist, term, substitute_var, substitute_value))
            .fold(T::from(0.0).unwrap(), |acc, x| acc + x),
        ASTRepr::Sub(left, right) => {
            let left_val = eval_with_substitution(hlist, left, substitute_var, substitute_value);
            let right_val = eval_with_substitution(hlist, right, substitute_var, substitute_value);
            left_val - right_val
        }
        ASTRepr::Mul(factors) => factors
            .elements()
            .map(|factor| eval_with_substitution(hlist, factor, substitute_var, substitute_value))
            .fold(T::from(1.0).unwrap(), |acc, x| acc * x),
        ASTRepr::Div(left, right) => {
            let left_val = eval_with_substitution(hlist, left, substitute_var, substitute_value);
            let right_val = eval_with_substitution(hlist, right, substitute_var, substitute_value);
            left_val / right_val
        }
        ASTRepr::Pow(base, exp) => {
            let base_val = eval_with_substitution(hlist, base, substitute_var, substitute_value);
            let exp_val = eval_with_substitution(hlist, exp, substitute_var, substitute_value);
            base_val.powf(exp_val)
        }
        ASTRepr::Neg(inner) => {
            let inner_val = eval_with_substitution(hlist, inner, substitute_var, substitute_value);
            -inner_val
        }
        ASTRepr::Ln(inner) => {
            let inner_val = eval_with_substitution(hlist, inner, substitute_var, substitute_value);
            inner_val.ln()
        }
        ASTRepr::Exp(inner) => {
            let inner_val = eval_with_substitution(hlist, inner, substitute_var, substitute_value);
            inner_val.exp()
        }
        ASTRepr::Sin(inner) => {
            let inner_val = eval_with_substitution(hlist, inner, substitute_var, substitute_value);
            inner_val.sin()
        }
        ASTRepr::Cos(inner) => {
            let inner_val = eval_with_substitution(hlist, inner, substitute_var, substitute_value);
            inner_val.cos()
        }
        ASTRepr::Sqrt(inner) => {
            let inner_val = eval_with_substitution(hlist, inner, substitute_var, substitute_value);
            inner_val.sqrt()
        }
        ASTRepr::Let(nested_var, nested_expr, nested_body) => {
            // Nested Let: evaluate expr with current substitution, then apply new binding
            let nested_expr_val =
                eval_with_substitution(hlist, nested_expr, substitute_var, substitute_value);
            eval_with_substitution(hlist, nested_body, *nested_var, nested_expr_val)
        }
        // For other cases (Sum, Lambda), fall back to standard evaluation with current substitution
        _ => {
            // This is a simplified approach - we could implement full substitution for all cases
            hlist.eval_expr(expr)
        }
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
    T: Scalar + ExpressionType + PartialEq + Copy + num_traits::Float + num_traits::FromPrimitive,
    H: HListEval<T>,
{
    match body {
        ASTRepr::Variable(index) => {
            // Variables are never bound by lambdas - they're always external
            // But in the context of embedded data (HNil), there should be no variables
            if hlist.variable_count() == 0 {
                panic!(
                    "Found Variable({index}) in lambda body but no variables available in evaluation context. \
                    This suggests the lambda body incorrectly contains Variable nodes instead of BoundVar nodes. \
                    Lambda var_indices: {var_indices:?}, Args: {args:?}"
                );
            }
            hlist.get_var(*index) // Use HList variable
        }
        ASTRepr::BoundVar(index) => {
            // Check if this bound variable is provided in the arguments
            if let Some(pos) = var_indices.iter().position(|&v| v == *index) {
                args[pos] // Use the argument value
            } else {
                panic!("BoundVar({index}) not found in lambda var_indices {var_indices:?}")
            }
        }
        ASTRepr::Constant(value) => *value,
        ASTRepr::Add(terms) => terms
            .elements()
            .map(|term| eval_lambda_with_substitution(hlist, term, var_indices, args))
            .fold(T::from(0.0).unwrap(), |acc, x| acc + x),
        ASTRepr::Sub(left, right) => {
            let left_val = eval_lambda_with_substitution(hlist, left, var_indices, args);
            let right_val = eval_lambda_with_substitution(hlist, right, var_indices, args);
            left_val - right_val
        }
        ASTRepr::Mul(factors) => factors
            .elements()
            .map(|factor| eval_lambda_with_substitution(hlist, factor, var_indices, args))
            .fold(T::from(1.0).unwrap(), |acc, x| acc * x),
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
    T: DslType + crate::ast::Scalar + ExpressionType + PartialEq,
    Tail: IntoVarHList,
{
    type Output = HCons<DynamicExpr<T>, Tail::Output>;

    fn into_vars(self, ctx: &DynamicContext) -> Self::Output {
        // Create a typed context for this type using new_explicit
        let mut ctx_typed: DynamicContext<0> = DynamicContext::new_explicit();
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
        let ast = ASTRepr::add_from_array([ASTRepr::Constant(1.0), ASTRepr::Constant(4.0)]);
        assert_eq!(hlist.eval_expr(&ast), 5.0);
    }

    #[test]
    fn test_hlist_eval_variables() {
        let hlist = hlist![2.0, 3.0];
        let ast = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Variable(1)]);
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
    #[should_panic(expected = "Variable index 0 out of bounds in HNil")]
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
        let identity_lambda = Lambda::single(0, Box::new(ASTRepr::BoundVar(0)));
        let result = hlist.apply_lambda(&identity_lambda, &[5.0]);
        assert_eq!(result, 5.0);

        // Doubling lambda: λx.x*2
        let double_lambda = Lambda::single(
            0,
            Box::new(ASTRepr::mul_from_array([
                ASTRepr::BoundVar(0),
                ASTRepr::Constant(2.0),
            ])),
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
            Box::new(ASTRepr::add_from_array([
                ASTRepr::BoundVar(0),
                ASTRepr::BoundVar(1),
            ])),
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
            Box::new(ASTRepr::add_from_array([
                ASTRepr::BoundVar(0), // Lambda argument
                ASTRepr::Variable(1), // HList variable
            ])),
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
