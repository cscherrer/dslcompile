//! Collection-Based Summation with Map Operations
//!
//! This module implements the Map-based summation approach that replaces
//! range-based summation with more expressive collection operations.
//! It integrates with the egglog rules in collection_summation.egg.

use crate::ast::ast_repr::ASTRepr;
use crate::error::{DSLCompileError, Result};
use std::collections::HashMap;

/// Collection types for summation operations
#[derive(Debug, Clone, PartialEq)]
pub enum Collection {
    /// Empty collection
    Empty,
    /// Single element collection
    Singleton(Box<ASTRepr<f64>>),
    /// Mathematical range [start, end]
    Range {
        start: Box<ASTRepr<f64>>,
        end: Box<ASTRepr<f64>>,
    },
    /// Set union
    Union {
        left: Box<Collection>,
        right: Box<Collection>,
    },
    /// Set intersection
    Intersection {
        left: Box<Collection>,
        right: Box<Collection>,
    },
    /// Named data array for runtime binding
    DataArray(String),
    /// Filtered collection with predicate
    Filter {
        collection: Box<Collection>,
        predicate: Box<ASTRepr<f64>>,
    },
}

/// Lambda expressions for mapping functions
#[derive(Debug, Clone, PartialEq)]
pub enum Lambda {
    /// Lambda expression: lambda var -> body
    Lambda {
        var: String,
        body: Box<ASTRepr<f64>>,
    },
    /// Identity function: lambda x -> x
    Identity,
    /// Constant function: lambda x -> c
    Constant(Box<ASTRepr<f64>>),
    /// Function composition: f ∘ g
    Compose {
        f: Box<Lambda>,
        g: Box<Lambda>,
    },
}

/// Extended AST representation with collection operations
#[derive(Debug, Clone, PartialEq)]
pub enum CollectionExpr {
    /// Sum over collection with mapping function
    Sum {
        collection: Collection,
        lambda: Lambda,
    },
    /// Map function over collection
    Map {
        lambda: Lambda,
        collection: Collection,
    },
    /// Collection size
    Size(Collection),
    /// Function application
    App {
        lambda: Lambda,
        arg: Box<ASTRepr<f64>>,
    },
    /// Regular mathematical expression
    Math(ASTRepr<f64>),
}

/// Collection-based summation optimizer using egglog
pub struct CollectionSummationOptimizer {
    /// Cache for collection analysis
    collection_cache: HashMap<String, Collection>,
    /// Lambda function cache
    lambda_cache: HashMap<String, Lambda>,
}

impl CollectionSummationOptimizer {
    /// Create a new collection summation optimizer
    pub fn new() -> Self {
        Self {
            collection_cache: HashMap::new(),
            lambda_cache: HashMap::new(),
        }
    }

    /// Convert range-based summation to collection-based summation
    pub fn convert_range_to_collection(
        &mut self,
        start: &ASTRepr<f64>,
        end: &ASTRepr<f64>,
        body: &ASTRepr<f64>,
        iter_var: usize,
    ) -> Result<CollectionExpr> {
        // Create range collection
        let range = Collection::Range {
            start: Box::new(start.clone()),
            end: Box::new(end.clone()),
        };

        // Create lambda function from body and iterator variable
        let lambda = self.create_lambda_from_body(body, iter_var)?;

        Ok(CollectionExpr::Sum {
            collection: range,
            lambda,
        })
    }

    /// Convert data-based summation to collection-based summation
    pub fn convert_data_to_collection(
        &mut self,
        data_name: &str,
        body: &ASTRepr<f64>,
        iter_var: usize,
    ) -> Result<CollectionExpr> {
        // Create data array collection
        let data_array = Collection::DataArray(data_name.to_string());

        // Create lambda function from body and iterator variable
        let lambda = self.create_lambda_from_body(body, iter_var)?;

        Ok(CollectionExpr::Sum {
            collection: data_array,
            lambda,
        })
    }

    /// Create lambda function from expression body and iterator variable
    fn create_lambda_from_body(
        &self,
        body: &ASTRepr<f64>,
        iter_var: usize,
    ) -> Result<Lambda> {
        // Analyze the body to create appropriate lambda
        match self.analyze_lambda_pattern(body, iter_var) {
            Some(lambda) => Ok(lambda),
            None => {
                // Create general lambda with variable substitution
                let var_name = format!("x_{}", iter_var);
                Ok(Lambda::Lambda {
                    var: var_name,
                    body: Box::new(body.clone()),
                })
            }
        }
    }

    /// Analyze expression patterns to create optimized lambda functions
    fn analyze_lambda_pattern(&self, body: &ASTRepr<f64>, iter_var: usize) -> Option<Lambda> {
        match body {
            // Identity: just the iterator variable
            ASTRepr::Variable(var_id) if *var_id == iter_var => Some(Lambda::Identity),
            
            // Constant: doesn't depend on iterator variable
            expr if !self.contains_variable(expr, iter_var) => {
                Some(Lambda::Constant(Box::new(expr.clone())))
            }
            
            // More complex patterns can be added here
            _ => None,
        }
    }

    /// Check if expression contains a specific variable
    fn contains_variable(&self, expr: &ASTRepr<f64>, var_id: usize) -> bool {
        match expr {
            ASTRepr::Variable(id) => *id == var_id,
            ASTRepr::Constant(_) => false,
            ASTRepr::Add(left, right) | 
            ASTRepr::Sub(left, right) | 
            ASTRepr::Mul(left, right) | 
            ASTRepr::Div(left, right) | 
            ASTRepr::Pow(left, right) => {
                self.contains_variable(left, var_id) || self.contains_variable(right, var_id)
            }
            ASTRepr::Neg(inner) | 
            ASTRepr::Ln(inner) | 
            ASTRepr::Exp(inner) | 
            ASTRepr::Sin(inner) | 
            ASTRepr::Cos(inner) | 
            ASTRepr::Sqrt(inner) => {
                self.contains_variable(inner, var_id)
            }
            ASTRepr::Sum { body, iter_var: sum_iter_var, .. } => {
                // Don't look inside nested sums with the same iterator variable
                if *sum_iter_var == var_id {
                    false
                } else {
                    self.contains_variable(body, var_id)
                }
            }
        }
    }

    /// Apply collection-based optimizations using egglog rules
    pub fn optimize_collection_expr(&mut self, expr: &CollectionExpr) -> Result<CollectionExpr> {
        match expr {
            CollectionExpr::Sum { collection, lambda } => {
                // Apply summation optimization rules
                let optimized_collection = self.optimize_collection(collection)?;
                let optimized_lambda = self.optimize_lambda(lambda)?;
                
                // Apply specific summation patterns
                self.apply_summation_patterns(&optimized_collection, &optimized_lambda)
            }
            CollectionExpr::Map { lambda, collection } => {
                // Apply map optimization rules
                let optimized_collection = self.optimize_collection(collection)?;
                let optimized_lambda = self.optimize_lambda(lambda)?;
                
                Ok(CollectionExpr::Map {
                    lambda: optimized_lambda,
                    collection: optimized_collection,
                })
            }
            other => Ok(other.clone()),
        }
    }

    /// Optimize collection operations
    fn optimize_collection(&self, collection: &Collection) -> Result<Collection> {
        match collection {
            Collection::Union { left, right } => {
                let opt_left = self.optimize_collection(left)?;
                let opt_right = self.optimize_collection(right)?;
                
                // Apply union optimization rules
                match (&opt_left, &opt_right) {
                    (Collection::Empty, right) => Ok(right.clone()),
                    (left, Collection::Empty) => Ok(left.clone()),
                    _ => Ok(Collection::Union {
                        left: Box::new(opt_left),
                        right: Box::new(opt_right),
                    }),
                }
            }
            Collection::Intersection { left, right } => {
                let opt_left = self.optimize_collection(left)?;
                let opt_right = self.optimize_collection(right)?;
                
                // Apply intersection optimization rules
                match (&opt_left, &opt_right) {
                    (Collection::Empty, _) | (_, Collection::Empty) => Ok(Collection::Empty),
                    _ => Ok(Collection::Intersection {
                        left: Box::new(opt_left),
                        right: Box::new(opt_right),
                    }),
                }
            }
            other => Ok(other.clone()),
        }
    }

    /// Optimize lambda expressions
    fn optimize_lambda(&self, lambda: &Lambda) -> Result<Lambda> {
        match lambda {
            Lambda::Compose { f, g } => {
                let opt_f = self.optimize_lambda(f)?;
                let opt_g = self.optimize_lambda(g)?;
                
                // Apply composition optimization rules
                match (&opt_f, &opt_g) {
                    (Lambda::Identity, g) => Ok(g.clone()),
                    (f, Lambda::Identity) => Ok(f.clone()),
                    _ => Ok(Lambda::Compose {
                        f: Box::new(opt_f),
                        g: Box::new(opt_g),
                    }),
                }
            }
            other => Ok(other.clone()),
        }
    }

    /// Apply specific summation optimization patterns
    fn apply_summation_patterns(
        &self,
        collection: &Collection,
        lambda: &Lambda,
    ) -> Result<CollectionExpr> {
        // Check for known summation patterns
        match (collection, lambda) {
            // Arithmetic series: Sum(Range(1, n), Identity) = n(n+1)/2
            (Collection::Range { start, end }, Lambda::Identity) => {
                if self.is_constant_one(start) {
                    return Ok(CollectionExpr::Math(
                        ASTRepr::Div(
                            Box::new(ASTRepr::Mul(
                                end.clone(),
                                Box::new(ASTRepr::Add(
                                    end.clone(),
                                    Box::new(ASTRepr::Constant(1.0)),
                                )),
                            )),
                            Box::new(ASTRepr::Constant(2.0)),
                        )
                    ));
                }
            }
            
            // Constant sum: Sum(collection, Constant(c)) = c * Size(collection)
            (coll, Lambda::Constant(c)) => {
                return Ok(CollectionExpr::Math(
                    ASTRepr::Mul(
                        c.clone(),
                        Box::new(ASTRepr::Constant(self.estimate_collection_size(coll)?)),
                    )
                ));
            }
            
            _ => {}
        }

        // Default: return optimized sum
        Ok(CollectionExpr::Sum {
            collection: collection.clone(),
            lambda: lambda.clone(),
        })
    }

    /// Check if expression is constant 1
    fn is_constant_one(&self, expr: &ASTRepr<f64>) -> bool {
        matches!(expr, ASTRepr::Constant(1.0))
    }

    /// Estimate collection size (for optimization purposes)
    fn estimate_collection_size(&self, collection: &Collection) -> Result<f64> {
        match collection {
            Collection::Empty => Ok(0.0),
            Collection::Singleton(_) => Ok(1.0),
            Collection::Range { start, end } => {
                // For constant ranges, compute size
                match (start.as_ref(), end.as_ref()) {
                    (ASTRepr::Constant(s), ASTRepr::Constant(e)) => Ok(e - s + 1.0),
                    _ => Err(DSLCompileError::Optimization(
                        "Cannot estimate size of non-constant range".to_string(),
                    )),
                }
            }
            Collection::DataArray(_) => Err(DSLCompileError::Optimization(
                "Cannot estimate size of data array at compile time".to_string(),
            )),
            _ => Err(DSLCompileError::Optimization(
                "Cannot estimate size of complex collection".to_string(),
            )),
        }
    }

    /// Convert collection expression back to standard AST
    pub fn to_ast(&self, expr: &CollectionExpr) -> Result<ASTRepr<f64>> {
        match expr {
            CollectionExpr::Math(ast) => Ok(ast.clone()),
            CollectionExpr::Sum { collection, lambda } => {
                // Convert back to Sum AST node
                self.collection_sum_to_ast(collection, lambda)
            }
            CollectionExpr::Size(collection) => {
                // Convert size operation to appropriate AST
                match collection {
                    Collection::Range { start, end } => {
                        Ok(ASTRepr::Add(
                            Box::new(ASTRepr::Sub(end.clone(), start.clone())),
                            Box::new(ASTRepr::Constant(1.0)),
                        ))
                    }
                    _ => Err(DSLCompileError::Optimization(
                        "Cannot convert complex collection size to AST".to_string(),
                    )),
                }
            }
            _ => Err(DSLCompileError::Optimization(
                "Cannot convert complex collection expression to AST".to_string(),
            )),
        }
    }

    /// Convert collection sum back to AST Sum node
    fn collection_sum_to_ast(
        &self,
        collection: &Collection,
        lambda: &Lambda,
    ) -> Result<ASTRepr<f64>> {
        match collection {
            Collection::Range { start, end } => {
                let body = self.lambda_to_ast_body(lambda, 0)?;
                Ok(ASTRepr::Sum {
                    range: crate::ast::ast_repr::SumRange::Mathematical {
                        start: start.clone(),
                        end: end.clone(),
                    },
                    body: Box::new(body),
                    iter_var: 0,
                })
            }
            Collection::DataArray(name) => {
                let body = self.lambda_to_ast_body(lambda, 0)?;
                // Create a DataParameter sum that will be evaluated with actual data
                // The data_var index 0 means the first data array in eval_with_data
                Ok(ASTRepr::Sum {
                    range: crate::ast::ast_repr::SumRange::DataParameter {
                        data_var: 0, // First data array in eval_with_data call
                    },
                    body: Box::new(body),
                    iter_var: 0, // Iterator variable for data values
                })
            }
            Collection::Union { left, right } => {
                // Convert union to sum of separate sums: Σ(f(x) for x in A ∪ B) = Σ(f(x) for x in A) + Σ(f(x) for x in B)
                let left_sum = self.collection_sum_to_ast(left, lambda)?;
                let right_sum = self.collection_sum_to_ast(right, lambda)?;
                Ok(ASTRepr::Add(
                    Box::new(left_sum),
                    Box::new(right_sum),
                ))
            }
            Collection::Intersection { left, right } => {
                // For now, convert intersection to a more complex form
                // This is a simplified implementation - real intersection would need predicate logic
                Err(DSLCompileError::Optimization(
                    "Intersection collections not yet supported in AST conversion".to_string(),
                ))
            }
            Collection::Singleton(value) => {
                // Singleton collection: just apply lambda to the single value
                let body = self.lambda_to_ast_body(lambda, 0)?;
                // Substitute the singleton value for the variable
                self.substitute_variable_in_ast(&body, 0, value)
            }
            Collection::Empty => {
                // Empty collection sums to zero
                Ok(ASTRepr::Constant(0.0))
            }
            Collection::Filter { collection, predicate } => {
                // Filtered collections are complex - for now, return error
                Err(DSLCompileError::Optimization(
                    "Filtered collections not yet supported in AST conversion".to_string(),
                ))
            }
        }
    }

    /// Substitute a variable in an AST with a given value
    fn substitute_variable_in_ast(
        &self,
        ast: &ASTRepr<f64>,
        var_id: usize,
        value: &ASTRepr<f64>,
    ) -> Result<ASTRepr<f64>> {
        match ast {
            ASTRepr::Variable(id) if *id == var_id => Ok(value.clone()),
            ASTRepr::Variable(_) | ASTRepr::Constant(_) => Ok(ast.clone()),
            ASTRepr::Add(left, right) => {
                let left_sub = self.substitute_variable_in_ast(left, var_id, value)?;
                let right_sub = self.substitute_variable_in_ast(right, var_id, value)?;
                Ok(ASTRepr::Add(Box::new(left_sub), Box::new(right_sub)))
            }
            ASTRepr::Sub(left, right) => {
                let left_sub = self.substitute_variable_in_ast(left, var_id, value)?;
                let right_sub = self.substitute_variable_in_ast(right, var_id, value)?;
                Ok(ASTRepr::Sub(Box::new(left_sub), Box::new(right_sub)))
            }
            ASTRepr::Mul(left, right) => {
                let left_sub = self.substitute_variable_in_ast(left, var_id, value)?;
                let right_sub = self.substitute_variable_in_ast(right, var_id, value)?;
                Ok(ASTRepr::Mul(Box::new(left_sub), Box::new(right_sub)))
            }
            ASTRepr::Div(left, right) => {
                let left_sub = self.substitute_variable_in_ast(left, var_id, value)?;
                let right_sub = self.substitute_variable_in_ast(right, var_id, value)?;
                Ok(ASTRepr::Div(Box::new(left_sub), Box::new(right_sub)))
            }
            ASTRepr::Pow(base, exp) => {
                let base_sub = self.substitute_variable_in_ast(base, var_id, value)?;
                let exp_sub = self.substitute_variable_in_ast(exp, var_id, value)?;
                Ok(ASTRepr::Pow(Box::new(base_sub), Box::new(exp_sub)))
            }
            ASTRepr::Sin(inner) => {
                let inner_sub = self.substitute_variable_in_ast(inner, var_id, value)?;
                Ok(ASTRepr::Sin(Box::new(inner_sub)))
            }
            ASTRepr::Cos(inner) => {
                let inner_sub = self.substitute_variable_in_ast(inner, var_id, value)?;
                Ok(ASTRepr::Cos(Box::new(inner_sub)))
            }
            ASTRepr::Ln(inner) => {
                let inner_sub = self.substitute_variable_in_ast(inner, var_id, value)?;
                Ok(ASTRepr::Ln(Box::new(inner_sub)))
            }
            ASTRepr::Exp(inner) => {
                let inner_sub = self.substitute_variable_in_ast(inner, var_id, value)?;
                Ok(ASTRepr::Exp(Box::new(inner_sub)))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_sub = self.substitute_variable_in_ast(inner, var_id, value)?;
                Ok(ASTRepr::Sqrt(Box::new(inner_sub)))
            }
            ASTRepr::Neg(inner) => {
                let inner_sub = self.substitute_variable_in_ast(inner, var_id, value)?;
                Ok(ASTRepr::Neg(Box::new(inner_sub)))
            }
            ASTRepr::Sum { range, body, iter_var } => {
                // Don't substitute inside sum bodies if they use the same variable
                if *iter_var == var_id {
                    Ok(ast.clone()) // Variable is bound by the sum
                } else {
                    let body_sub = self.substitute_variable_in_ast(body, var_id, value)?;
                    Ok(ASTRepr::Sum {
                        range: range.clone(),
                        body: Box::new(body_sub),
                        iter_var: *iter_var,
                    })
                }
            }
        }
    }

    /// Convert lambda back to AST body expression
    fn lambda_to_ast_body(&self, lambda: &Lambda, var_id: usize) -> Result<ASTRepr<f64>> {
        match lambda {
            Lambda::Identity => Ok(ASTRepr::Variable(var_id)),
            Lambda::Constant(c) => Ok(c.as_ref().clone()),
            Lambda::Lambda { body, .. } => Ok(body.as_ref().clone()),
            Lambda::Compose { f, g } => {
                // For composition, we need to substitute g into f
                // This is a simplified approach
                let g_body = self.lambda_to_ast_body(g, var_id)?;
                let f_body = self.lambda_to_ast_body(f, var_id)?;
                // TODO: Implement proper substitution
                Ok(f_body) // Simplified for now
            }
        }
    }
}

impl Default for CollectionSummationOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for converting expressions to collection-based form
pub trait IntoCollectionExpr {
    fn into_collection_expr(&self) -> Result<CollectionExpr>;
}

impl IntoCollectionExpr for ASTRepr<f64> {
    fn into_collection_expr(&self) -> Result<CollectionExpr> {
        match self {
            ASTRepr::Sum { range, body, iter_var } => {
                let mut optimizer = CollectionSummationOptimizer::new();
                match range {
                    crate::ast::ast_repr::SumRange::Mathematical { start, end } => {
                        optimizer.convert_range_to_collection(start, end, body, *iter_var)
                    }
                    crate::ast::ast_repr::SumRange::DataParameter { data_var } => {
                        let data_name = format!("data_{}", data_var);
                        optimizer.convert_data_to_collection(&data_name, body, *iter_var)
                    }
                }
            }
            other => Ok(CollectionExpr::Math(other.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_to_collection_conversion() {
        let mut optimizer = CollectionSummationOptimizer::new();
        
        let start = ASTRepr::Constant(1.0);
        let end = ASTRepr::Constant(10.0);
        let body = ASTRepr::Variable(0); // Identity function
        
        let result = optimizer.convert_range_to_collection(&start, &end, &body, 0).unwrap();
        
        match result {
            CollectionExpr::Sum { collection, lambda } => {
                assert!(matches!(collection, Collection::Range { .. }));
                assert!(matches!(lambda, Lambda::Identity));
            }
            _ => panic!("Expected Sum expression"),
        }
    }

    #[test]
    fn test_arithmetic_series_optimization() {
        let mut optimizer = CollectionSummationOptimizer::new();
        
        let collection = Collection::Range {
            start: Box::new(ASTRepr::Constant(1.0)),
            end: Box::new(ASTRepr::Constant(10.0)),
        };
        let lambda = Lambda::Identity;
        
        let result = optimizer.apply_summation_patterns(&collection, &lambda).unwrap();
        
        match result {
            CollectionExpr::Math(ASTRepr::Div(..)) => {
                // Should be n(n+1)/2 formula
            }
            _ => panic!("Expected optimized arithmetic series formula"),
        }
    }

    #[test]
    fn test_constant_sum_optimization() {
        let mut optimizer = CollectionSummationOptimizer::new();
        
        let collection = Collection::Range {
            start: Box::new(ASTRepr::Constant(1.0)),
            end: Box::new(ASTRepr::Constant(5.0)),
        };
        let lambda = Lambda::Constant(Box::new(ASTRepr::Constant(3.0)));
        
        let result = optimizer.apply_summation_patterns(&collection, &lambda).unwrap();
        
        match result {
            CollectionExpr::Math(ASTRepr::Mul(..)) => {
                // Should be c * size formula
            }
            _ => panic!("Expected optimized constant sum formula"),
        }
    }
} 