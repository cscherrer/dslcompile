//! Automatic Scope Merging Infrastructure
//!
//! This module provides the core infrastructure for automatically merging variable scopes
//! when expressions from different contexts are combined. This solves the "variable collision"
//! problem by detecting when expressions use overlapping variable indices and automatically
//! remapping them to avoid conflicts.

use crate::{
    ast::{ASTRepr, Scalar, ast_repr::{Lambda, Collection}},
    contexts::{VariableRegistry, DynamicExpr},
};
use std::{sync::Arc, cell::RefCell};

/// Information about a variable scope that needs to be merged
#[derive(Debug, Clone)]
pub struct ScopeInfo {
    /// Registry containing variable names and metadata
    pub registry: Arc<RefCell<VariableRegistry>>,
    /// Highest variable index used in this scope
    pub max_var_index: usize,
}

/// Result of merging two scopes
#[derive(Debug)]
pub struct MergedScope<T: Scalar> {
    /// The merged registry containing all variables from both scopes
    pub merged_registry: Arc<RefCell<VariableRegistry>>,
    /// The left expression with variables remapped to the merged scope
    pub left_expr: ASTRepr<T>,
    /// The right expression with variables remapped to the merged scope
    pub right_expr: ASTRepr<T>,
}

/// Core scope merging functionality
pub struct ScopeMerger;

impl ScopeMerger {
    /// Determine if two expressions need scope merging
    /// 
    /// Returns true if the expressions come from different registries (different scopes)
    /// and their variable indices might collide.
    pub fn needs_merging<T: Scalar, const SCOPE1: usize, const SCOPE2: usize>(
        left: &DynamicExpr<T, SCOPE1>, 
        right: &DynamicExpr<T, SCOPE2>
    ) -> bool {
        // Check if they have different registry addresses (different scopes)
        !Arc::ptr_eq(&left.registry, &right.registry)
    }

    /// Extract scope information from an expression
    pub fn extract_scope_info<T: Scalar, const SCOPE: usize>(expr: &DynamicExpr<T, SCOPE>) -> ScopeInfo {
        let max_var_index = Self::find_max_variable_index(&expr.ast);
        ScopeInfo {
            registry: expr.registry.clone(),
            max_var_index,
        }
    }

    /// Count the number of variables used in an expression
    fn count_variables<T: Scalar>(ast: &ASTRepr<T>) -> usize {
        let mut variables = std::collections::HashSet::new();
        Self::collect_variables(ast, &mut variables);
        variables.len()
    }

    /// Collect all variable indices used in an AST
    fn collect_variables<T: Scalar>(ast: &ASTRepr<T>, variables: &mut std::collections::HashSet<usize>) {
        match ast {
            ASTRepr::Variable(index) => {
                variables.insert(*index);
            }
            ASTRepr::Constant(_) => {}
            ASTRepr::Add(left, right) | 
            ASTRepr::Sub(left, right) | 
            ASTRepr::Mul(left, right) | 
            ASTRepr::Div(left, right) => {
                Self::collect_variables(left, variables);
                Self::collect_variables(right, variables);
            }
            ASTRepr::Neg(expr) => Self::collect_variables(expr, variables),
            ASTRepr::Sin(expr) | 
            ASTRepr::Cos(expr) | 
            ASTRepr::Exp(expr) | 
            ASTRepr::Ln(expr) |
            ASTRepr::Sqrt(expr) => Self::collect_variables(expr, variables),
            ASTRepr::Pow(base, exp) => {
                Self::collect_variables(base, variables);
                Self::collect_variables(exp, variables);
            }
            ASTRepr::Sum(_collection) => {
                // For now, assume collections don't contain variables that need remapping
            },
            ASTRepr::BoundVar(_) => {}, // Bound variables don't affect global variable indexing
            ASTRepr::Lambda(lambda) => Self::collect_variables(&lambda.body, variables),
            ASTRepr::Let(_, expr, body) => {
                Self::collect_variables(expr, variables);
                Self::collect_variables(body, variables);
            },
        }
    }

    /// Normalize variable indices in an expression to be contiguous starting from 0
    fn normalize_variable_indices<T: Scalar>(ast: &ASTRepr<T>) -> ASTRepr<T> {
        let mut variables = std::collections::HashSet::new();
        Self::collect_variables(ast, &mut variables);
        
        // Create a mapping from old indices to new contiguous indices
        let mut old_to_new: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut sorted_vars: Vec<usize> = variables.into_iter().collect();
        sorted_vars.sort();
        
        for (new_index, &old_index) in sorted_vars.iter().enumerate() {
            old_to_new.insert(old_index, new_index);
        }
        
        Self::remap_variables_with_mapping(ast, &old_to_new)
    }

    /// Remap variables using a specific mapping from old indices to new indices
    fn remap_variables_with_mapping<T: Scalar>(ast: &ASTRepr<T>, mapping: &std::collections::HashMap<usize, usize>) -> ASTRepr<T> {
        match ast {
            ASTRepr::Variable(index) => {
                let new_index = mapping.get(index).copied().unwrap_or(*index);
                ASTRepr::Variable(new_index)
            },
            ASTRepr::Constant(value) => ASTRepr::Constant(value.clone()),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(Self::remap_variables_with_mapping(left, mapping)),
                Box::new(Self::remap_variables_with_mapping(right, mapping)),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(Self::remap_variables_with_mapping(left, mapping)),
                Box::new(Self::remap_variables_with_mapping(right, mapping)),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(Self::remap_variables_with_mapping(left, mapping)),
                Box::new(Self::remap_variables_with_mapping(right, mapping)),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(Self::remap_variables_with_mapping(left, mapping)),
                Box::new(Self::remap_variables_with_mapping(right, mapping)),
            ),
            ASTRepr::Neg(expr) => ASTRepr::Neg(
                Box::new(Self::remap_variables_with_mapping(expr, mapping))
            ),
            ASTRepr::Sin(expr) => ASTRepr::Sin(
                Box::new(Self::remap_variables_with_mapping(expr, mapping))
            ),
            ASTRepr::Cos(expr) => ASTRepr::Cos(
                Box::new(Self::remap_variables_with_mapping(expr, mapping))
            ),
            ASTRepr::Exp(expr) => ASTRepr::Exp(
                Box::new(Self::remap_variables_with_mapping(expr, mapping))
            ),
            ASTRepr::Ln(expr) => ASTRepr::Ln(
                Box::new(Self::remap_variables_with_mapping(expr, mapping))
            ),
            ASTRepr::Sqrt(expr) => ASTRepr::Sqrt(
                Box::new(Self::remap_variables_with_mapping(expr, mapping))
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(Self::remap_variables_with_mapping(base, mapping)),
                Box::new(Self::remap_variables_with_mapping(exp, mapping)),
            ),
            ASTRepr::Sum(collection) => {
                // For now, assume collections don't contain variables that need remapping
                ASTRepr::Sum(collection.clone())
            },
            ASTRepr::BoundVar(index) => ASTRepr::BoundVar(*index), // Don't remap bound variables
            ASTRepr::Lambda(lambda) => ASTRepr::Lambda(Box::new(Lambda {
                var_indices: lambda.var_indices.clone(),
                body: Box::new(Self::remap_variables_with_mapping(&lambda.body, mapping)),
            })),
            ASTRepr::Let(binding_id, expr, body) => ASTRepr::Let(
                *binding_id,
                Box::new(Self::remap_variables_with_mapping(expr, mapping)),
                Box::new(Self::remap_variables_with_mapping(body, mapping)),
            ),
        }
    }

    /// Merge two scopes and remap expressions to use the merged variable space
    pub fn merge_scopes<T: Scalar, const SCOPE1: usize, const SCOPE2: usize>(
        left: &DynamicExpr<T, SCOPE1>, 
        right: &DynamicExpr<T, SCOPE2>
    ) -> MergedScope<T> {
        let left_scope = Self::extract_scope_info(left);
        let right_scope = Self::extract_scope_info(right);

        // First normalize both expressions to have contiguous indices starting from 0
        let left_normalized = Self::normalize_variable_indices(&left.ast);
        let right_normalized = Self::normalize_variable_indices(&right.ast);

        // Count variables in normalized expressions
        let left_var_count = Self::count_variables(&left_normalized);
        let right_var_count = Self::count_variables(&right_normalized);

        // DETERMINISTIC ORDERING: Use registry addresses to determine canonical order
        // This ensures that merge_scopes(A, B) == merge_scopes(B, A) in terms of variable assignment
        let left_addr = Arc::as_ptr(&left.registry) as usize;
        let right_addr = Arc::as_ptr(&right.registry) as usize;
        
        let (first_expr, first_count, second_expr, second_count, swap_needed) = if left_addr < right_addr {
            // Left registry has lower address - use left-first ordering
            (left_normalized, left_var_count, right_normalized, right_var_count, false)
        } else {
            // Right registry has lower address - use right-first ordering  
            (right_normalized, right_var_count, left_normalized, left_var_count, true)
        };

        // Create merged registry with deterministic ordering
        let merged_registry = Self::create_merged_registry_normalized(first_count, second_count);

        // Second expression gets offset by the number of variables in first expression
        let second_offset = first_count;
        let second_expr_remapped = Self::remap_variables(&second_expr, second_offset);

        // Return expressions in original left/right order, but with deterministic variable assignment
        let (final_left_expr, final_right_expr) = if swap_needed {
            // We used right-first ordering, so swap back to left/right
            (second_expr_remapped, first_expr)
        } else {
            // We used left-first ordering, keep as-is
            (first_expr, second_expr_remapped)
        };

        MergedScope {
            merged_registry,
            left_expr: final_left_expr,
            right_expr: final_right_expr,
        }
    }

    /// Create a merged registry for normalized expressions
    fn create_merged_registry_normalized(
        left_var_count: usize,
        right_var_count: usize,
    ) -> Arc<RefCell<VariableRegistry>> {
        let merged = Arc::new(RefCell::new(VariableRegistry::new()));
        let mut merged_registry = merged.borrow_mut();
        
        // Register variables for left scope (0, 1, 2, ...)
        for i in 0..left_var_count {
            let name = format!("var_{}", i);
            merged_registry.register_variable_with_index(name, i);
        }
        
        // Register variables for right scope (left_count, left_count+1, ...)
        for i in 0..right_var_count {
            let new_index = left_var_count + i;
            let name = format!("var_{}", new_index);
            merged_registry.register_variable_with_index(name, new_index);
        }
        
        drop(merged_registry);
        merged
    }

    /// Find the maximum variable index used in an AST
    fn find_max_variable_index<T: Scalar>(ast: &ASTRepr<T>) -> usize {
        Self::find_max_variable_index_recursive(ast).unwrap_or(0)
    }

    /// Helper function that returns None if no variables are found
    fn find_max_variable_index_recursive<T: Scalar>(ast: &ASTRepr<T>) -> Option<usize> {
        match ast {
            ASTRepr::Variable(index) => Some(*index),
            ASTRepr::Constant(_) => None,
            ASTRepr::Add(left, right) | 
            ASTRepr::Sub(left, right) | 
            ASTRepr::Mul(left, right) | 
            ASTRepr::Div(left, right) => {
                match (Self::find_max_variable_index_recursive(left), Self::find_max_variable_index_recursive(right)) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            ASTRepr::Neg(expr) => Self::find_max_variable_index_recursive(expr),
            ASTRepr::Sin(expr) | 
            ASTRepr::Cos(expr) | 
            ASTRepr::Exp(expr) | 
            ASTRepr::Ln(expr) |
            ASTRepr::Sqrt(expr) => Self::find_max_variable_index_recursive(expr),
            ASTRepr::Pow(base, exp) => {
                match (Self::find_max_variable_index_recursive(base), Self::find_max_variable_index_recursive(exp)) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            ASTRepr::Sum(_collection) => {
                // For now, assume collections don't contain variables that need remapping
                // This is a simplification - full implementation would need to analyze Collection
                None
            },
            ASTRepr::BoundVar(_) => None, // Bound variables don't affect global variable indexing
            ASTRepr::Lambda(lambda) => Self::find_max_variable_index_recursive(&lambda.body),
            ASTRepr::Let(_, expr, body) => {
                match (Self::find_max_variable_index_recursive(expr), Self::find_max_variable_index_recursive(body)) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            },
        }
    }

    /// Remap all variable indices in an AST by adding an offset
    fn remap_variables<T: Scalar>(ast: &ASTRepr<T>, offset: usize) -> ASTRepr<T> {
        match ast {
            ASTRepr::Variable(index) => ASTRepr::Variable(index + offset),
            ASTRepr::Constant(value) => ASTRepr::Constant(value.clone()),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(Self::remap_variables(left, offset)),
                Box::new(Self::remap_variables(right, offset)),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(Self::remap_variables(left, offset)),
                Box::new(Self::remap_variables(right, offset)),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(Self::remap_variables(left, offset)),
                Box::new(Self::remap_variables(right, offset)),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(Self::remap_variables(left, offset)),
                Box::new(Self::remap_variables(right, offset)),
            ),
            ASTRepr::Neg(expr) => ASTRepr::Neg(
                Box::new(Self::remap_variables(expr, offset))
            ),
            ASTRepr::Sin(expr) => ASTRepr::Sin(
                Box::new(Self::remap_variables(expr, offset))
            ),
            ASTRepr::Cos(expr) => ASTRepr::Cos(
                Box::new(Self::remap_variables(expr, offset))
            ),
            ASTRepr::Exp(expr) => ASTRepr::Exp(
                Box::new(Self::remap_variables(expr, offset))
            ),
            ASTRepr::Ln(expr) => ASTRepr::Ln(
                Box::new(Self::remap_variables(expr, offset))
            ),
            ASTRepr::Sqrt(expr) => ASTRepr::Sqrt(
                Box::new(Self::remap_variables(expr, offset))
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(Self::remap_variables(base, offset)),
                Box::new(Self::remap_variables(exp, offset)),
            ),
            ASTRepr::Sum(collection) => {
                // For now, assume collections don't contain variables that need remapping
                // This is a simplification - full implementation would need to handle
                // variables within Collection expressions
                ASTRepr::Sum(collection.clone())
            },
            ASTRepr::BoundVar(index) => ASTRepr::BoundVar(*index), // Don't remap bound variables
            ASTRepr::Lambda(lambda) => ASTRepr::Lambda(Box::new(Lambda {
                var_indices: lambda.var_indices.clone(),
                body: Box::new(Self::remap_variables(&lambda.body, offset)),
            })),
            ASTRepr::Let(binding_id, expr, body) => ASTRepr::Let(
                *binding_id,
                Box::new(Self::remap_variables(expr, offset)),
                Box::new(Self::remap_variables(body, offset)),
            ),
        }
    }

    /// Perform automatic scope merging for two expressions and create the combined result
    pub fn merge_and_combine<T, F, const SCOPE1: usize, const SCOPE2: usize>(
        left: &DynamicExpr<T, SCOPE1>, 
        right: &DynamicExpr<T, SCOPE2>,
        combiner: F,
    ) -> DynamicExpr<T>
    where
        T: Scalar,
        F: FnOnce(ASTRepr<T>, ASTRepr<T>) -> ASTRepr<T>,
    {
        if Self::needs_merging(left, right) {
            // Different scopes - perform merging
            let merged = Self::merge_scopes(left, right);
            let combined_ast = combiner(merged.left_expr, merged.right_expr);
            DynamicExpr::new(combined_ast, merged.merged_registry)
        } else {
            // Same scope - no merging needed
            let combined_ast = combiner(left.ast.clone(), right.ast.clone());
            DynamicExpr::new(combined_ast, left.registry.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contexts::dynamic::expression_builder::DynamicContext;
    use frunk::hlist;

    #[test]
    fn test_needs_merging_detection() {
        // Same context - should not need merging
        let mut ctx1 = DynamicContext::new();
        let x1: DynamicExpr<f64> = ctx1.var();
        let y1: DynamicExpr<f64> = ctx1.var();
        assert!(!ScopeMerger::needs_merging(&x1, &y1));

        // Different contexts - should need merging
        let mut ctx2 = DynamicContext::new();
        let x2: DynamicExpr<f64> = ctx2.var();
        assert!(ScopeMerger::needs_merging(&x1, &x2));
    }

    #[test]
    fn test_max_variable_index_detection() {
        let mut ctx = DynamicContext::new();
        let x: DynamicExpr<f64> = ctx.var(); // Variable 0
        let y: DynamicExpr<f64> = ctx.var(); // Variable 1
        let z: DynamicExpr<f64> = ctx.var(); // Variable 2

        // Simple expression: x (max index = 0)
        let scope_info = ScopeMerger::extract_scope_info(&x);
        assert_eq!(scope_info.max_var_index, 0);

        // Complex expression: x + y * z (max index = 2)
        let complex = &x + &(&y * &z);
        let scope_info = ScopeMerger::extract_scope_info(&complex);
        assert_eq!(scope_info.max_var_index, 2);
    }

    #[test]
    fn test_scope_merging_basic() {
        // Create two independent contexts
        let mut ctx1 = DynamicContext::new();
        let x1: DynamicExpr<f64> = ctx1.var(); // Variable 0 in ctx1

        let mut ctx2 = DynamicContext::new();
        let x2: DynamicExpr<f64> = ctx2.var(); // Variable 0 in ctx2 (collision!)

        // Merge scopes
        let merged = ScopeMerger::merge_scopes(&x1, &x2);

        // The key invariant is that the two expressions should use different variable indices
        // We don't care about the specific assignment, just that they're different
        let left_var_index = match &merged.left_expr {
            ASTRepr::Variable(index) => *index,
            _ => panic!("Expected Variable"),
        };

        let right_var_index = match &merged.right_expr {
            ASTRepr::Variable(index) => *index,
            _ => panic!("Expected Variable"),
        };

        // The variables should be different (no collision)
        assert_ne!(left_var_index, right_var_index);
        
        // The variables should be contiguous (0 and 1, in some order)
        let mut indices = vec![left_var_index, right_var_index];
        indices.sort();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_scope_merging_complex() {
        // Create two contexts with multiple variables
        let mut ctx1 = DynamicContext::new();
        let x1: DynamicExpr<f64> = ctx1.var(); // Variable 0
        let y1: DynamicExpr<f64> = ctx1.var(); // Variable 1
        let expr1 = &x1 + &y1; // Uses variables 0, 1

        let mut ctx2 = DynamicContext::new();
        let x2: DynamicExpr<f64> = ctx2.var(); // Variable 0 (collision!)
        let y2: DynamicExpr<f64> = ctx2.var(); // Variable 1 (collision!)
        let expr2 = &x2 * &y2; // Uses variables 0, 1

        // Merge scopes
        let merged = ScopeMerger::merge_scopes(&expr1, &expr2);

        // The merged expressions should use 4 different variables total
        let max_var_left = ScopeMerger::find_max_variable_index(&merged.left_expr);
        let max_var_right = ScopeMerger::find_max_variable_index(&merged.right_expr);
        let overall_max = max_var_left.max(max_var_right);
        
        // Should use variables 0, 1, 2, 3 (so max index is 3)
        assert_eq!(overall_max, 3);
        
        // Verify that left and right expressions use disjoint variable sets
        let mut left_vars = std::collections::HashSet::new();
        let mut right_vars = std::collections::HashSet::new();
        
        ScopeMerger::collect_variables(&merged.left_expr, &mut left_vars);
        ScopeMerger::collect_variables(&merged.right_expr, &mut right_vars);
        
        // Should have no overlap
        assert!(left_vars.is_disjoint(&right_vars));
        
        // Should have 2 variables each
        assert_eq!(left_vars.len(), 2);
        assert_eq!(right_vars.len(), 2);
    }

    #[test]
    fn test_merge_and_combine() {
        // Create expressions from different contexts
        let mut ctx1 = DynamicContext::new();
        let x1: DynamicExpr<f64> = ctx1.var();
        let expr1 = &x1 * 2.0; // 2*x1

        let mut ctx2 = DynamicContext::new();
        let x2: DynamicExpr<f64> = ctx2.var();
        let expr2 = &x2 * 3.0; // 3*x2

        // Combine with automatic scope merging
        let combined = ScopeMerger::merge_and_combine(&expr1, &expr2, |left, right| {
            ASTRepr::Add(Box::new(left), Box::new(right))
        });

        // The combined expression should use variables 0 and 1
        assert_eq!(ScopeMerger::find_max_variable_index(&combined.ast), 1);

        // Create a new context that can handle the merged scope
        let temp_ctx = DynamicContext::new();
        // Verify evaluation works correctly with merged scope
        let result = temp_ctx.eval(&combined, hlist![4.0, 5.0]);
        
        // The result should be either 2*4 + 3*5 = 23 or 2*5 + 3*4 = 22
        // depending on the deterministic ordering based on memory addresses
        assert!(result == 23.0 || result == 22.0, "Expected 22.0 or 23.0, got {}", result);
    }
}