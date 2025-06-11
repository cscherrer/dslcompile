//! AST Evaluation Utilities
//!
//! This module provides efficient evaluation methods for AST expressions,
//! including optimized variable handling and specialized evaluation functions.

use crate::ast::{
    Scalar,
    ast_repr::{ASTRepr, Collection, Lambda},
};
use num_traits::{Float, FromPrimitive,  Zero};

/// Optimized evaluation methods for AST expressions
impl<T> ASTRepr<T>
where
    T: Scalar + Float + Copy + FromPrimitive + Zero,
{
    /// Evaluate the expression with given variable values
    #[must_use]
    pub fn eval_with_vars(&self, variables: &[T]) -> T {
        match self {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => {
                if *index < variables.len() {
                    variables[*index]
                } else {
                    panic!(
                        "Variable index {index} is out of bounds for evaluation! \
                           Tried to access variable at index {index}, but only {} variables provided. \
                           Use a valid variable index or provide more variables.",
                        variables.len()
                    )
                }
            }
            ASTRepr::Add(left, right) => {
                left.eval_with_vars(variables) + right.eval_with_vars(variables)
            }
            ASTRepr::Sub(left, right) => {
                left.eval_with_vars(variables) - right.eval_with_vars(variables)
            }
            ASTRepr::Mul(left, right) => {
                left.eval_with_vars(variables) * right.eval_with_vars(variables)
            }
            ASTRepr::Div(left, right) => {
                left.eval_with_vars(variables) / right.eval_with_vars(variables)
            }
            ASTRepr::Pow(base, exp) => {
                let base_val = base.eval_with_vars(variables);
                let exp_val = exp.eval_with_vars(variables);
                base_val.powf(exp_val)
            }
            ASTRepr::Neg(expr) => -expr.eval_with_vars(variables),
            ASTRepr::Ln(expr) => expr.eval_with_vars(variables).ln(),
            ASTRepr::Exp(expr) => expr.eval_with_vars(variables).exp(),
            ASTRepr::Sin(expr) => expr.eval_with_vars(variables).sin(),
            ASTRepr::Cos(expr) => expr.eval_with_vars(variables).cos(),
            ASTRepr::Sqrt(expr) => expr.eval_with_vars(variables).sqrt(),
            ASTRepr::Sum(collection) => self.eval_collection_sum(collection, variables),
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    /// Evaluate a sum over a collection
    fn eval_collection_sum(&self, collection: &Collection<T>, variables: &[T]) -> T {
        match collection {
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => expr.eval_with_vars(variables),
            Collection::Range { start, end } => {
                // Evaluate range bounds
                let start_val = start.eval_with_vars(variables);
                let end_val = end.eval_with_vars(variables);

                // Convert to integers for iteration
                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                // Sum over the mathematical range with identity function
                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    sum = sum + i_val;
                }
                sum
            }
            Collection::DataArray(_data_var) => {
                // TODO: Data array evaluation requires runtime data binding
                // For now, return zero as placeholder
                T::zero()
            }
            Collection::Union { left, right } => {
                // Sum over union: Σ(A ∪ B) = Σ(A) + Σ(B) - Σ(A ∩ B)
                // For now, simplified to just Σ(A) + Σ(B)
                // TODO: Handle intersection properly to avoid double counting
                let left_sum = self.eval_collection_sum(left, variables);
                let right_sum = self.eval_collection_sum(right, variables);
                left_sum + right_sum
            }
            Collection::Intersection { left: _, right: _ } => {
                // TODO: Implement intersection evaluation
                T::zero()
            }
            Collection::Filter {
                collection: _,
                predicate: _,
            } => {
                // TODO: Implement filtered collection evaluation
                T::zero()
            }
            Collection::Map { lambda, collection } => {
                self.eval_mapped_collection(lambda, collection, variables)
            }
        }
    }

    /// Evaluate a mapped collection (lambda applied to each element)
    fn eval_mapped_collection(
        &self,
        lambda: &Lambda<T>,
        collection: &Collection<T>,
        variables: &[T],
    ) -> T {
        match collection {
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => {
                let element_val = expr.eval_with_vars(variables);
                self.eval_lambda(lambda, element_val, variables)
            }
            Collection::Range { start, end } => {
                // Evaluate range bounds
                let start_val = start.eval_with_vars(variables);
                let end_val = end.eval_with_vars(variables);

                // Convert to integers for iteration
                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                // Sum lambda(i) for i in range
                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    let lambda_result = self.eval_lambda(lambda, i_val, variables);
                    sum = sum + lambda_result;
                }
                sum
            }
            Collection::DataArray(_data_var) => {
                // TODO: Data array evaluation with lambda mapping
                T::zero()
            }
            Collection::Union { left, right } => {
                // Map over union: map(f, A ∪ B) = map(f, A) + map(f, B) - map(f, A ∩ B)
                // Simplified for now
                let left_sum = self.eval_mapped_collection(lambda, left, variables);
                let right_sum = self.eval_mapped_collection(lambda, right, variables);
                left_sum + right_sum
            }
            Collection::Intersection { left: _, right: _ } => {
                // TODO: Implement intersection with mapping
                T::zero()
            }
            Collection::Filter {
                collection: _,
                predicate: _,
            } => {
                // TODO: Implement filtered mapping
                T::zero()
            }
            Collection::Map {
                lambda: inner_lambda,
                collection: inner_collection,
            } => {
                // Composition: map(f, map(g, X)) = map(f∘g, X)
                let composed = Lambda::Compose {
                    f: Box::new(lambda.clone()),
                    g: Box::new(inner_lambda.as_ref().clone()),
                };
                self.eval_mapped_collection(&composed, inner_collection, variables)
            }
        }
    }

    /// Evaluate a lambda function applied to a value
    fn eval_lambda(&self, lambda: &Lambda<T>, value: T, variables: &[T]) -> T {
        match lambda {
            Lambda::Identity => value,
            Lambda::Constant(expr) => expr.eval_with_vars(variables),
            Lambda::Lambda { var_index, body } => {
                // Create new variable context with the lambda variable bound
                let mut lambda_vars = variables.to_vec();
                // Ensure we have enough space for the lambda variable
                while lambda_vars.len() <= *var_index {
                    lambda_vars.push(T::zero());
                }
                lambda_vars[*var_index] = value;
                body.eval_with_vars(&lambda_vars)
            }
            Lambda::Compose { f, g } => {
                // Function composition: (f ∘ g)(x) = f(g(x))
                let g_result = self.eval_lambda(g, value, variables);
                self.eval_lambda(f, g_result, variables)
            }
        }
    }

    /// Evaluate a two-variable expression with specific values
    #[must_use]
    pub fn eval_two_vars(&self, x: T, y: T) -> T {
        self.eval_with_vars(&[x, y])
    }

    /// Evaluate with a single variable value
    #[must_use]
    pub fn eval_one_var(&self, value: T) -> T {
        self.eval_with_vars(&[value])
    }

    /// Evaluate with no variables (constants only)
    #[must_use]
    pub fn eval_no_vars(&self) -> T {
        self.eval_with_vars(&[])
    }

    /// Evaluate expression with data arrays (for DataArray collections)
    #[must_use]
    pub(crate) fn eval_with_data(&self, params: &[T], data_arrays: &[Vec<T>]) -> T {
        match self {
            ASTRepr::Sum(collection) => {
                self.eval_collection_sum_with_data(collection, params, data_arrays)
            }
            _ => {
                // For non-sum expressions, use regular evaluation with params
                self.eval_with_vars(params)
            }
        }
    }

    /// Evaluate collection sum with data arrays
    fn eval_collection_sum_with_data(
        &self,
        collection: &Collection<T>,
        params: &[T],
        data_arrays: &[Vec<T>],
    ) -> T {
        match collection {
            Collection::DataArray(data_var) => {
                // Sum over data array with identity function
                if *data_var < data_arrays.len() {
                    data_arrays[*data_var]
                        .iter()
                        .fold(T::zero(), |acc, &x| acc + x)
                } else {
                    T::zero()
                }
            }
            Collection::Map { lambda, collection } => {
                self.eval_mapped_collection_with_data(lambda, collection, params, data_arrays)
            }
            Collection::Range { start, end } => {
                // Mathematical ranges don't need data arrays
                let start_val = start.eval_with_vars(params);
                let end_val = end.eval_with_vars(params);

                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    sum = sum + i_val;
                }
                sum
            }
            _ => {
                // For other collection types, use regular evaluation
                self.eval_collection_sum(collection, params)
            }
        }
    }

    /// Evaluate mapped collection with data arrays
    fn eval_mapped_collection_with_data(
        &self,
        lambda: &Lambda<T>,
        collection: &Collection<T>,
        params: &[T],
        data_arrays: &[Vec<T>],
    ) -> T {
        match collection {
            Collection::DataArray(data_var) => {
                // Map lambda over data array
                if *data_var < data_arrays.len() {
                    data_arrays[*data_var]
                        .iter()
                        .map(|&x| self.eval_lambda(lambda, x, params))
                        .fold(T::zero(), |acc, x| acc + x)
                } else {
                    T::zero()
                }
            }
            Collection::Range { start, end } => {
                // Map lambda over mathematical range
                let start_val = start.eval_with_vars(params);
                let end_val = end.eval_with_vars(params);

                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    let lambda_result = self.eval_lambda(lambda, i_val, params);
                    sum = sum + lambda_result;
                }
                sum
            }
            _ => {
                // For other collection types, use regular evaluation
                self.eval_mapped_collection(lambda, collection, params)
            }
        }
    }
}

/// Specialized evaluation methods for f64 expressions
impl ASTRepr<f64> {
    /// Fast evaluation without heap allocation for two variables
    #[must_use]
    pub fn eval_two_vars_fast(expr: &ASTRepr<f64>, x: f64, y: f64) -> f64 {
        match expr {
            ASTRepr::Constant(value) => *value,
            ASTRepr::Variable(index) => match *index {
                0 => x,
                1 => y,
                _ => panic!(
                    "Variable index {index} is out of bounds for two-variable evaluation! \
                    eval_two_vars_fast only supports Variable(0) and Variable(1). \
                    Use eval_with_vars() for expressions with more variables."
                ),
            },
            ASTRepr::Add(left, right) => {
                Self::eval_two_vars_fast(left, x, y) + Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Sub(left, right) => {
                Self::eval_two_vars_fast(left, x, y) - Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Mul(left, right) => {
                Self::eval_two_vars_fast(left, x, y) * Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Div(left, right) => {
                Self::eval_two_vars_fast(left, x, y) / Self::eval_two_vars_fast(right, x, y)
            }
            ASTRepr::Pow(base, exp) => {
                Self::eval_two_vars_fast(base, x, y).powf(Self::eval_two_vars_fast(exp, x, y))
            }
            ASTRepr::Neg(inner) => -Self::eval_two_vars_fast(inner, x, y),
            ASTRepr::Ln(inner) => Self::eval_two_vars_fast(inner, x, y).ln(),
            ASTRepr::Exp(inner) => Self::eval_two_vars_fast(inner, x, y).exp(),
            ASTRepr::Sin(inner) => Self::eval_two_vars_fast(inner, x, y).sin(),
            ASTRepr::Cos(inner) => Self::eval_two_vars_fast(inner, x, y).cos(),
            ASTRepr::Sqrt(inner) => Self::eval_two_vars_fast(inner, x, y).sqrt(),
            ASTRepr::Sum(_collection) => {
                // Fall back to general evaluation for Sum (needs variable arrays)
                expr.eval_with_vars(&[x, y])
            }
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficient_variable_indexing() {
        // Test efficient index-based variables
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = expr.eval_with_vars(&[2.0, 3.0]);
        assert_eq!(result, 5.0);

        // Test multiplication with index-based variables
        let expr = ASTRepr::Mul(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = expr.eval_with_vars(&[4.0, 5.0]);
        assert_eq!(result, 20.0);
    }

    #[test]
    #[should_panic(expected = "Variable index 10 is out of bounds")]
    fn test_out_of_bounds_variable_index() {
        // Test behavior when variable index is out of bounds - should panic
        let expr = ASTRepr::Variable(10); // Index 10, but only 2 variables provided
        let _result = expr.eval_with_vars(&[1.0, 2.0]); // Should panic!
    }

    #[test]
    fn test_two_variable_evaluation() {
        // Test two-variable evaluation: x + y
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)), // x
            Box::new(ASTRepr::Variable(1)), // y
        );
        let result = expr.eval_two_vars(3.0, 4.0);
        assert_eq!(result, 7.0);

        // Test more complex expression: x * y + 1
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)), // x
                Box::new(ASTRepr::Variable(1)), // y
            )),
            Box::new(ASTRepr::Constant(1.0)),
        );
        let result = expr.eval_two_vars(2.0, 3.0);
        assert_eq!(result, 7.0); // 2 * 3 + 1 = 7
    }

    #[test]
    fn test_transcendental_evaluation() {
        // Test sine evaluation
        let expr = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[0.0]);
        assert!((result - 0.0).abs() < 1e-10); // sin(0) = 0

        // Test exponential evaluation
        let expr = ASTRepr::Exp(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[0.0]);
        assert!((result - 1.0).abs() < 1e-10); // exp(0) = 1

        // Test natural logarithm evaluation
        let expr = ASTRepr::Ln(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[1.0]);
        assert!((result - 0.0).abs() < 1e-10); // ln(1) = 0
    }

    #[test]
    fn test_power_evaluation() {
        // Test power evaluation: x^2
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        );
        let result = expr.eval_with_vars(&[3.0]);
        assert_eq!(result, 9.0); // 3^2 = 9

        // Test fractional power: x^0.5 (square root)
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(0.5)),
        );
        let result = expr.eval_with_vars(&[4.0]);
        assert!((result - 2.0).abs() < 1e-10); // 4^0.5 = 2
    }

    #[test]
    #[should_panic(expected = "Variable index 2 is out of bounds for two-variable evaluation")]
    fn test_two_vars_fast_out_of_bounds() {
        // Test that eval_two_vars_fast panics for Variable(2) and higher
        let expr = ASTRepr::Variable(2); // Index 2, but only supports 0 and 1
        let _result = ASTRepr::eval_two_vars_fast(&expr, 1.0, 2.0); // Should panic!
    }
}
