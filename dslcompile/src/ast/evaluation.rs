//! AST Evaluation Utilities
//!
//! This module provides efficient evaluation methods for AST expressions,
//! including optimized variable handling and specialized evaluation functions.

use crate::ast::{
    Scalar,
    ast_repr::{ASTRepr, Collection, Lambda},
};
use num_traits::{Float, FromPrimitive, Zero};

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
            ASTRepr::BoundVar(index) => {
                if *index < variables.len() {
                    variables[*index]
                } else {
                    panic!(
                        "BoundVar index {index} is out of bounds for evaluation! \
                           Tried to access variable at index {index}, but only {} variables provided.",
                        variables.len()
                    )
                }
            }
            ASTRepr::Let(_, expr, body) => {
                let expr_val = expr.eval_with_vars(variables);
                // TODO: Proper Let evaluation would substitute the bound variable
                // For now, just evaluate the body with current variables
                body.eval_with_vars(variables)
            }
            ASTRepr::Lambda(lambda) => {
                // Lambda evaluation without arguments returns the lambda itself as a constant
                // This is a simplification - proper lambda evaluation would require function application
                // For now, if the lambda has no variables or is a constant lambda, evaluate the body
                if lambda.var_indices.is_empty() {
                    lambda.body.eval_with_vars(variables)
                } else {
                    // Cannot evaluate lambda without arguments - this is an error state
                    panic!("Cannot evaluate lambda without function application")
                }
            }
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
            Collection::Variable(_data_var) => {
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
            Collection::Variable(_data_var) => {
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
                // Apply the lambda over the inner mapped collection
                // This is less optimal than composition but simpler to maintain
                let inner_result =
                    self.eval_mapped_collection(inner_lambda, inner_collection, variables);
                self.eval_lambda(lambda, inner_result, variables)
            }
        }
    }

    /// Evaluate a lambda function applied to a value
    fn eval_lambda(&self, lambda: &Lambda<T>, value: T, variables: &[T]) -> T {
        // For single-value application, bind the first variable if available
        if lambda.var_indices.is_empty() {
            // Constant lambda - just evaluate the body
            lambda.body.eval_with_vars(variables)
        } else {
            // Bind the first variable to the input value
            let first_var = lambda.var_indices[0];
            let mut lambda_vars = variables.to_vec();

            // Ensure we have enough space for the lambda variable
            while lambda_vars.len() <= first_var {
                lambda_vars.push(T::zero());
            }
            lambda_vars[first_var] = value;
            lambda.body.eval_with_vars(&lambda_vars)
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
            Collection::Variable(data_var) => {
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
            Collection::Variable(data_var) => {
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
            ASTRepr::BoundVar(index) => match *index {
                0 => x,
                1 => y,
                _ => panic!(
                    "BoundVar index {index} is out of bounds for two-variable evaluation! \
                    eval_two_vars_fast only supports BoundVar(0) and BoundVar(1)."
                ),
            },
            ASTRepr::Let(_, expr_val, body) => {
                // For simplicity, evaluate without proper substitution for now
                let _expr_result = Self::eval_two_vars_fast(expr_val, x, y);
                Self::eval_two_vars_fast(body, x, y)
            }
            ASTRepr::Lambda(lambda) => {
                // Lambda evaluation in two-variable context
                if lambda.var_indices.is_empty() {
                    Self::eval_two_vars_fast(&lambda.body, x, y)
                } else {
                    panic!(
                        "Cannot evaluate lambda without function application in two-variable context"
                    )
                }
            }
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

    #[test]
    fn test_two_vars_fast_out_of_bounds_comprehensive() {
        let expr = ASTRepr::<f64>::Variable(2); // Index 2 is out of bounds for two variables
        let result = std::panic::catch_unwind(|| {
            ASTRepr::eval_two_vars_fast(&expr, 1.0, 2.0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_one_var() {
        // Test simple variable access
        let x = ASTRepr::<f64>::Variable(0);
        assert_eq!(x.eval_one_var(5.0), 5.0);

        // Test constant
        let const_expr = ASTRepr::<f64>::Constant(42.0);
        assert_eq!(const_expr.eval_one_var(5.0), 42.0);

        // Test arithmetic with one variable
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::<f64>::Variable(0)),
            Box::new(ASTRepr::<f64>::Constant(10.0)),
        );
        assert_eq!(expr.eval_one_var(5.0), 15.0);

        // Test transcendental functions
        let sin_expr = ASTRepr::Sin(Box::new(ASTRepr::<f64>::Variable(0)));
        assert!((sin_expr.eval_one_var(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_no_vars() {
        // Test constant expression
        let const_expr = ASTRepr::<f64>::Constant(3.14);
        assert_eq!(const_expr.eval_no_vars(), 3.14);

        // Test arithmetic with constants
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::<f64>::Constant(2.0)),
            Box::new(ASTRepr::<f64>::Constant(3.0)),
        );
        assert_eq!(expr.eval_no_vars(), 5.0);

        // Test transcendental functions with constants
        let sin_expr = ASTRepr::Sin(Box::new(ASTRepr::<f64>::Constant(0.0)));
        assert!((sin_expr.eval_no_vars() - 0.0).abs() < 1e-10);

        // Test complex constant expression
        let complex_expr = ASTRepr::Mul(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::<f64>::Constant(2.0)),
                Box::new(ASTRepr::<f64>::Constant(3.0)),
            )),
            Box::new(ASTRepr::<f64>::Constant(4.0)),
        );
        assert_eq!(complex_expr.eval_no_vars(), 20.0); // (2 + 3) * 4 = 20
    }

    #[test]
    fn test_division_evaluation() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let div_expr = ASTRepr::Div(Box::new(x), Box::new(y));

        assert_eq!(div_expr.eval_with_vars(&[10.0, 2.0]), 5.0);
        assert_eq!(div_expr.eval_with_vars(&[1.0, 4.0]), 0.25);
    }

    #[test]
    fn test_negation_evaluation() {
        let x = ASTRepr::<f64>::Variable(0);
        let neg_expr = ASTRepr::Neg(Box::new(x));

        assert_eq!(neg_expr.eval_with_vars(&[5.0]), -5.0);
        assert_eq!(neg_expr.eval_with_vars(&[-3.0]), 3.0);
        assert_eq!(neg_expr.eval_with_vars(&[0.0]), 0.0);
    }

    #[test]
    fn test_sqrt_evaluation() {
        let x = ASTRepr::<f64>::Variable(0);
        let sqrt_expr = ASTRepr::Sqrt(Box::new(x));

        assert_eq!(sqrt_expr.eval_with_vars(&[9.0]), 3.0);
        assert_eq!(sqrt_expr.eval_with_vars(&[16.0]), 4.0);
        assert!((sqrt_expr.eval_with_vars(&[2.0]) - 1.4142135623730951).abs() < 1e-10);
    }

    #[test]
    fn test_complex_nested_evaluation() {
        // Test (x + y) * (x - y) = x² - y²
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);

        let x_plus_y = ASTRepr::Add(Box::new(x.clone()), Box::new(y.clone()));
        let x_minus_y = ASTRepr::Sub(Box::new(x.clone()), Box::new(y.clone()));
        let expr = ASTRepr::Mul(Box::new(x_plus_y), Box::new(x_minus_y));

        let result = expr.eval_with_vars(&[5.0, 3.0]);
        let expected = 5.0 * 5.0 - 3.0 * 3.0; // 25 - 9 = 16
        assert_eq!(result, expected);
    }

    #[test]
    fn test_trigonometric_identities() {
        let x = ASTRepr::<f64>::Variable(0);

        // Test sin²(x) + cos²(x) = 1
        let sin_x = ASTRepr::Sin(Box::new(x.clone()));
        let cos_x = ASTRepr::Cos(Box::new(x.clone()));
        let sin_squared = ASTRepr::Pow(Box::new(sin_x), Box::new(ASTRepr::<f64>::Constant(2.0)));
        let cos_squared = ASTRepr::Pow(Box::new(cos_x), Box::new(ASTRepr::<f64>::Constant(2.0)));
        let identity = ASTRepr::Add(Box::new(sin_squared), Box::new(cos_squared));

        let result = identity.eval_with_vars(&[1.0]);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_logarithm_inverse() {
        let x = ASTRepr::<f64>::Variable(0);

        // Test exp(ln(x)) = x for positive x
        let ln_x = ASTRepr::Ln(Box::new(x.clone()));
        let exp_ln_x = ASTRepr::Exp(Box::new(ln_x));

        let test_values = [1.0, 2.0, 5.0, 10.0, 100.0];
        for &val in &test_values {
            let result = exp_ln_x.eval_with_vars(&[val]);
            assert!(
                (result - val).abs() < 1e-10,
                "exp(ln({})) = {} != {}",
                val,
                result,
                val
            );
        }

        // Test ln(exp(x)) = x
        let exp_x = ASTRepr::Exp(Box::new(x.clone()));
        let ln_exp_x = ASTRepr::Ln(Box::new(exp_x));

        let test_values = [0.0, 1.0, -1.0, 2.5, -3.0];
        for &val in &test_values {
            let result = ln_exp_x.eval_with_vars(&[val]);
            assert!(
                (result - val).abs() < 1e-10,
                "ln(exp({})) = {} != {}",
                val,
                result,
                val
            );
        }
    }

    #[test]
    fn test_power_special_cases() {
        let x = ASTRepr::<f64>::Variable(0);

        // Test x^0 = 1
        let x_pow_0 = ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::<f64>::Constant(0.0)));
        assert_eq!(x_pow_0.eval_with_vars(&[5.0]), 1.0);
        assert_eq!(x_pow_0.eval_with_vars(&[0.0]), 1.0);

        // Test x^1 = x
        let x_pow_1 = ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::<f64>::Constant(1.0)));
        assert_eq!(x_pow_1.eval_with_vars(&[7.0]), 7.0);

        // Test 0^x = 0 for positive x
        let zero_pow_x = ASTRepr::Pow(Box::new(ASTRepr::<f64>::Constant(0.0)), Box::new(x.clone()));
        assert_eq!(zero_pow_x.eval_with_vars(&[2.0]), 0.0);

        // Test 1^x = 1
        let one_pow_x = ASTRepr::Pow(Box::new(ASTRepr::<f64>::Constant(1.0)), Box::new(x.clone()));
        assert_eq!(one_pow_x.eval_with_vars(&[100.0]), 1.0);
    }

    #[test]
    fn test_bound_var_evaluation() {
        // Test BoundVar evaluation
        let bound_var = ASTRepr::<f64>::BoundVar(0);
        assert_eq!(bound_var.eval_with_vars(&[42.0]), 42.0);

        let bound_var_1 = ASTRepr::<f64>::BoundVar(1);
        assert_eq!(bound_var_1.eval_with_vars(&[10.0, 20.0]), 20.0);
    }

    #[test]
    #[should_panic(expected = "BoundVar index 2 is out of bounds")]
    fn test_bound_var_out_of_bounds() {
        let bound_var = ASTRepr::<f64>::BoundVar(2);
        bound_var.eval_with_vars(&[1.0, 2.0]); // Only 2 variables, index 2 is out of bounds
    }

    #[test]
    fn test_let_expression_evaluation() {
        // Test Let expression (simplified evaluation)
        let x = ASTRepr::<f64>::Variable(0);
        let const_5 = ASTRepr::<f64>::Constant(5.0);
        let let_expr = ASTRepr::Let(0, Box::new(const_5), Box::new(x.clone()));

        // Current implementation just evaluates the body with existing variables
        let result = let_expr.eval_with_vars(&[10.0]);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_lambda_evaluation_edge_cases() {
        use crate::ast::ast_repr::Lambda;

        // Test lambda with no variables (constant lambda)
        let const_body = ASTRepr::<f64>::Constant(42.0);
        let const_lambda = Lambda {
            var_indices: vec![],
            body: Box::new(const_body),
        };
        let lambda_expr = ASTRepr::Lambda(Box::new(const_lambda));

        assert_eq!(lambda_expr.eval_with_vars(&[]), 42.0);
    }

    #[test]
    #[should_panic(expected = "Cannot evaluate lambda without function application")]
    fn test_lambda_with_variables_panics() {
        use crate::ast::ast_repr::Lambda;

        // Test lambda with variables (should panic)
        let var_body = ASTRepr::<f64>::Variable(0);
        let var_lambda = Lambda {
            var_indices: vec![0],
            body: Box::new(var_body),
        };
        let lambda_expr = ASTRepr::Lambda(Box::new(var_lambda));

        lambda_expr.eval_with_vars(&[5.0]);
    }

    #[test]
    fn test_collection_evaluation_empty() {
        use crate::ast::ast_repr::Collection;

        // Test empty collection sum
        let empty_collection = Collection::<f64>::Empty;
        let sum_expr = ASTRepr::Sum(Box::new(empty_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 0.0);
    }

    #[test]
    fn test_collection_evaluation_singleton() {
        use crate::ast::ast_repr::Collection;

        // Test singleton collection sum
        let singleton_expr = ASTRepr::<f64>::Constant(5.0);
        let singleton_collection = Collection::Singleton(Box::new(singleton_expr));
        let sum_expr = ASTRepr::Sum(Box::new(singleton_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 5.0);
    }

    #[test]
    fn test_collection_evaluation_range() {
        use crate::ast::ast_repr::Collection;

        // Test range collection sum: Σ(i=1 to 3) i = 1 + 2 + 3 = 6
        let start = ASTRepr::<f64>::Constant(1.0);
        let end = ASTRepr::<f64>::Constant(3.0);
        let range_collection = Collection::Range {
            start: Box::new(start),
            end: Box::new(end),
        };
        let sum_expr = ASTRepr::Sum(Box::new(range_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 6.0);
    }

    #[test]
    fn test_collection_evaluation_variable() {
        use crate::ast::ast_repr::Collection;

        // Test variable collection (placeholder implementation returns 0)
        let var_collection = Collection::<f64>::Variable(0);
        let sum_expr = ASTRepr::Sum(Box::new(var_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 0.0);
    }

    #[test]
    fn test_collection_evaluation_union() {
        use crate::ast::ast_repr::Collection;

        // Test union collection (simplified implementation)
        let left = Collection::Singleton(Box::new(ASTRepr::<f64>::Constant(5.0)));
        let right = Collection::Singleton(Box::new(ASTRepr::<f64>::Constant(3.0)));
        let union_collection = Collection::Union {
            left: Box::new(left),
            right: Box::new(right),
        };
        let sum_expr = ASTRepr::Sum(Box::new(union_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 8.0); // 5 + 3
    }

    #[test]
    fn test_eval_with_data_basic() {
        // Test basic evaluation with data arrays
        let x = ASTRepr::<f64>::Variable(0);
        let result = x.eval_with_data(&[5.0], &[]);
        assert_eq!(result, 5.0);

        // Test with multiple parameters
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::<f64>::Variable(0)),
            Box::new(ASTRepr::<f64>::Variable(1)),
        );
        let result = expr.eval_with_data(&[3.0, 7.0], &[]);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_edge_case_evaluations() {
        // Test very large numbers
        let large_expr = ASTRepr::<f64>::Constant(1e100);
        assert_eq!(large_expr.eval_no_vars(), 1e100);

        // Test very small numbers
        let small_expr = ASTRepr::<f64>::Constant(1e-100);
        assert_eq!(small_expr.eval_no_vars(), 1e-100);

        // Test infinity handling
        let inf_expr = ASTRepr::<f64>::Constant(f64::INFINITY);
        assert!(inf_expr.eval_no_vars().is_infinite());

        // Test NaN handling
        let nan_expr = ASTRepr::<f64>::Constant(f64::NAN);
        assert!(nan_expr.eval_no_vars().is_nan());
    }

    #[test]
    fn test_eval_two_vars_comprehensive() {
        // Test all basic operations with two variables
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);

        // Addition
        let add_expr = ASTRepr::Add(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(add_expr.eval_two_vars(3.0, 4.0), 7.0);

        // Subtraction
        let sub_expr = ASTRepr::Sub(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(sub_expr.eval_two_vars(10.0, 3.0), 7.0);

        // Multiplication
        let mul_expr = ASTRepr::Mul(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(mul_expr.eval_two_vars(6.0, 7.0), 42.0);

        // Division
        let div_expr = ASTRepr::Div(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(div_expr.eval_two_vars(15.0, 3.0), 5.0);

        // Power
        let pow_expr = ASTRepr::Pow(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(pow_expr.eval_two_vars(2.0, 3.0), 8.0);
    }
}
