use crate::ast::ast_repr::ASTRepr;
use crate::ast::collection::{Collection, Lambda};
use std::collections::HashMap;

impl<T> ASTRepr<T>
where
    T: Clone + std::fmt::Debug + PartialEq,
{
    /// Evaluate the expression with the given variable values
    pub fn eval_with_vars(&self, variables: &[T]) -> Result<T, String>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + From<f64>
            + Into<f64>
            + Copy,
    {
        match self {
            ASTRepr::Constant(value) => Ok(*value),
            ASTRepr::Variable(index) => {
                if *index < variables.len() {
                    Ok(variables[*index])
                } else {
                    Err(format!("Variable index {} out of bounds", index))
                }
            }
            ASTRepr::BoundVar(index) => {
                if *index < variables.len() {
                    Ok(variables[*index])
                } else {
                    Err(format!("BoundVar index {} out of bounds", index))
                }
            }
            ASTRepr::Let(var_index, binding, body) => {
                // Evaluate the binding expression
                let binding_value = binding.eval_with_vars(variables)?;
                
                // Create extended variable context with the bound variable
                let mut extended_vars = variables.to_vec();
                if *var_index >= extended_vars.len() {
                    extended_vars.resize(*var_index + 1, T::from(0.0));
                }
                extended_vars[*var_index] = binding_value;
                
                // Evaluate the body with the extended context
                body.eval_with_vars(&extended_vars)
            }
            ASTRepr::Add(left, right) => {
                let left_val = left.eval_with_vars(variables)?;
                let right_val = right.eval_with_vars(variables)?;
                Ok(left_val + right_val)
            }
            ASTRepr::Sub(left, right) => {
                let left_val = left.eval_with_vars(variables)?;
                let right_val = right.eval_with_vars(variables)?;
                Ok(left_val - right_val)
            }
            ASTRepr::Mul(left, right) => {
                let left_val = left.eval_with_vars(variables)?;
                let right_val = right.eval_with_vars(variables)?;
                Ok(left_val * right_val)
            }
            ASTRepr::Div(left, right) => {
                let left_val = left.eval_with_vars(variables)?;
                let right_val = right.eval_with_vars(variables)?;
                Ok(left_val / right_val)
            }
            ASTRepr::Neg(inner) => {
                let inner_val = inner.eval_with_vars(variables)?;
                Ok(-inner_val)
            }
            ASTRepr::Pow(base, exponent) => {
                let base_val: f64 = base.eval_with_vars(variables)?.into();
                let exp_val: f64 = exponent.eval_with_vars(variables)?.into();
                Ok(T::from(base_val.powf(exp_val)))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_val: f64 = inner.eval_with_vars(variables)?.into();
                Ok(T::from(inner_val.sqrt()))
            }
            ASTRepr::Exp(inner) => {
                let inner_val: f64 = inner.eval_with_vars(variables)?.into();
                Ok(T::from(inner_val.exp()))
            }
            ASTRepr::Log(inner) => {
                let inner_val: f64 = inner.eval_with_vars(variables)?.into();
                Ok(T::from(inner_val.ln()))
            }
            ASTRepr::Sin(inner) => {
                let inner_val: f64 = inner.eval_with_vars(variables)?.into();
                Ok(T::from(inner_val.sin()))
            }
            ASTRepr::Cos(inner) => {
                let inner_val: f64 = inner.eval_with_vars(variables)?.into();
                Ok(T::from(inner_val.cos()))
            }
            ASTRepr::Tan(inner) => {
                let inner_val: f64 = inner.eval_with_vars(variables)?.into();
                Ok(T::from(inner_val.tan()))
            }
            ASTRepr::Sum(collection) => self.eval_collection_sum(collection, variables),
        }
    }

    fn eval_collection_sum(&self, collection: &Collection<T>, variables: &[T]) -> Result<T, String>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Neg<Output = T>
            + From<f64>
            + Into<f64>
            + Copy,
    {
        match collection {
            Collection::Range { start, end } => {
                let start_val: i32 = (*start as f64) as i32;
                let end_val: i32 = (*end as f64) as i32;
                let sum: i32 = (start_val..=end_val).sum();
                Ok(T::from(sum as f64))
            }
            Collection::DataArray(_) => {
                // For now, return zero for data arrays
                // This should be implemented properly when data evaluation is needed
                Ok(T::from(0.0))
            }
            Collection::Map { lambda, collection: inner_collection } => {
                // Evaluate the mapped collection
                match inner_collection.as_ref() {
                    Collection::Range { start, end } => {
                        let start_val: i32 = (*start as f64) as i32;
                        let end_val: i32 = (*end as f64) as i32;
                        let mut sum = T::from(0.0);
                        
                        for i in start_val..=end_val {
                            // Create extended variable context with iterator variable
                            let mut extended_vars = variables.to_vec();
                            if lambda.var_index >= extended_vars.len() {
                                extended_vars.resize(lambda.var_index + 1, T::from(0.0));
                            }
                            extended_vars[lambda.var_index] = T::from(i as f64);
                            
                            // Evaluate lambda body with iterator variable bound
                            let term_value = lambda.body.eval_with_vars(&extended_vars)?;
                            sum = sum + term_value;
                        }
                        Ok(sum)
                    }
                    _ => Err("Unsupported collection type in Map".to_string()),
                }
            }
            _ => Err("Unsupported collection type".to_string()),
        }
    }
}

impl ASTRepr<f64> {
    /// Evaluate with data arrays for summation operations
    pub fn eval_with_data(
        &self,
        params: &[f64],
        data_arrays: &[&[f64]],
    ) -> Result<f64, String> {
        self.eval_with_data_and_context(params, data_arrays, &mut HashMap::new())
    }

    fn eval_with_data_and_context(
        &self,
        params: &[f64],
        data_arrays: &[&[f64]],
        bound_vars: &mut HashMap<usize, f64>,
    ) -> Result<f64, String> {
        match self {
            ASTRepr::Constant(value) => Ok(*value),
            ASTRepr::Variable(index) => {
                if *index < params.len() {
                    Ok(params[*index])
                } else {
                    Err(format!("Variable index {} out of bounds", index))
                }
            }
            ASTRepr::BoundVar(index) => {
                if let Some(&value) = bound_vars.get(index) {
                    Ok(value)
                } else {
                    Err(format!("BoundVar {} not found in context", index))
                }
            }
            ASTRepr::Let(var_index, binding, body) => {
                // Evaluate the binding expression
                let binding_value = binding.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                
                // Add the bound variable to context
                let old_value = bound_vars.insert(*var_index, binding_value);
                
                // Evaluate the body with the extended context
                let result = body.eval_with_data_and_context(params, data_arrays, bound_vars);
                
                // Restore the old value (if any)
                match old_value {
                    Some(old_val) => { bound_vars.insert(*var_index, old_val); }
                    None => { bound_vars.remove(var_index); }
                }
                
                result
            }
            ASTRepr::Add(left, right) => {
                let left_val = left.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                let right_val = right.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(left_val + right_val)
            }
            ASTRepr::Sub(left, right) => {
                let left_val = left.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                let right_val = right.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(left_val - right_val)
            }
            ASTRepr::Mul(left, right) => {
                let left_val = left.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                let right_val = right.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(left_val * right_val)
            }
            ASTRepr::Div(left, right) => {
                let left_val = left.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                let right_val = right.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(left_val / right_val)
            }
            ASTRepr::Neg(inner) => {
                let inner_val = inner.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(-inner_val)
            }
            ASTRepr::Pow(base, exponent) => {
                let base_val = base.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                let exp_val = exponent.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(base_val.powf(exp_val))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_val = inner.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(inner_val.sqrt())
            }
            ASTRepr::Exp(inner) => {
                let inner_val = inner.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(inner_val.exp())
            }
            ASTRepr::Log(inner) => {
                let inner_val = inner.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(inner_val.ln())
            }
            ASTRepr::Sin(inner) => {
                let inner_val = inner.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(inner_val.sin())
            }
            ASTRepr::Cos(inner) => {
                let inner_val = inner.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(inner_val.cos())
            }
            ASTRepr::Tan(inner) => {
                let inner_val = inner.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                Ok(inner_val.tan())
            }
            ASTRepr::Sum(collection) => {
                self.eval_collection_sum_with_data(collection, params, data_arrays, bound_vars)
            }
        }
    }

    fn eval_collection_sum_with_data(
        &self,
        collection: &Collection<f64>,
        params: &[f64],
        data_arrays: &[&[f64]],
        bound_vars: &mut HashMap<usize, f64>,
    ) -> Result<f64, String> {
        match collection {
            Collection::Range { start, end } => {
                let start_val = *start as i32;
                let end_val = *end as i32;
                let sum: i32 = (start_val..=end_val).sum();
                Ok(sum as f64)
            }
            Collection::DataArray(data_index) => {
                if *data_index < data_arrays.len() {
                    let data = data_arrays[*data_index];
                    Ok(data.iter().sum())
                } else {
                    Err(format!("Data array index {} out of bounds", data_index))
                }
            }
            Collection::Map { lambda, collection: inner_collection } => {
                match inner_collection.as_ref() {
                    Collection::Range { start, end } => {
                        let start_val = *start as i32;
                        let end_val = *end as i32;
                        let mut sum = 0.0;
                        
                        for i in start_val..=end_val {
                            // Add iterator variable to bound context
                            let old_value = bound_vars.insert(lambda.var_index, i as f64);
                            
                            // Evaluate lambda body
                            let term_value = lambda.body.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                            sum += term_value;
                            
                            // Restore old value
                            match old_value {
                                Some(old_val) => { bound_vars.insert(lambda.var_index, old_val); }
                                None => { bound_vars.remove(&lambda.var_index); }
                            }
                        }
                        Ok(sum)
                    }
                    Collection::DataArray(data_index) => {
                        if *data_index < data_arrays.len() {
                            let data = data_arrays[*data_index];
                            let mut sum = 0.0;
                            
                            for &value in data {
                                // Add iterator variable to bound context
                                let old_value = bound_vars.insert(lambda.var_index, value);
                                
                                // Evaluate lambda body
                                let term_value = lambda.body.eval_with_data_and_context(params, data_arrays, bound_vars)?;
                                sum += term_value;
                                
                                // Restore old value
                                match old_value {
                                    Some(old_val) => { bound_vars.insert(lambda.var_index, old_val); }
                                    None => { bound_vars.remove(&lambda.var_index); }
                                }
                            }
                            Ok(sum)
                        } else {
                            Err(format!("Data array index {} out of bounds", data_index))
                        }
                    }
                    _ => Err("Unsupported inner collection type in Map".to_string()),
                }
            }
            _ => Err("Unsupported collection type".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_evaluation() {
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(5.0)),
        );
        let result = expr.eval_with_vars(&[3.0]).unwrap();
        assert_eq!(result, 8.0);
    }

    #[test]
    fn test_let_evaluation() {
        // Let x = 5 in x * x
        let expr = ASTRepr::Let(
            0,
            Box::new(ASTRepr::Constant(5.0)),
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::BoundVar(0)),
                Box::new(ASTRepr::BoundVar(0)),
            )),
        );
        let result = expr.eval_with_vars(&[]).unwrap();
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_bound_var_evaluation() {
        let expr = ASTRepr::BoundVar(0);
        let result = expr.eval_with_vars(&[42.0]).unwrap();
        assert_eq!(result, 42.0);
    }
} 