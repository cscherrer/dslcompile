//! Automatic Scope Merging Infrastructure
//!
//! This module provides the core infrastructure for automatically merging variable scopes
//! when expressions from different contexts are combined. This solves the "variable collision"
//! problem by detecting when expressions use overlapping variable indices and automatically
//! remapping them to avoid conflicts.

use crate::{
    ast::{ASTRepr, Scalar, ast_repr::{Lambda, Collection}},
    contexts::{VariableRegistry, TypedBuilderExpr},
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
    pub fn needs_merging<T: Scalar>(
        left: &TypedBuilderExpr<T>, 
        right: &TypedBuilderExpr<T>
    ) -> bool {
        // Check if they have different registry addresses (different scopes)
        !Arc::ptr_eq(&left.registry, &right.registry)
    }

    /// Extract scope information from an expression
    pub fn extract_scope_info<T: Scalar>(expr: &TypedBuilderExpr<T>) -> ScopeInfo {
        let max_var_index = Self::find_max_variable_index(&expr.ast);
        ScopeInfo {
            registry: expr.registry.clone(),
            max_var_index,
        }
    }

    /// Merge two scopes and remap expressions to use the merged variable space
    pub fn merge_scopes<T: Scalar>(
        left: &TypedBuilderExpr<T>, 
        right: &TypedBuilderExpr<T>
    ) -> MergedScope<T> {
        let left_scope = Self::extract_scope_info(left);
        let right_scope = Self::extract_scope_info(right);

        // Create merged registry
        let merged_registry = Self::create_merged_registry(&left_scope, &right_scope);

        // Calculate variable remapping for right expression
        // Left expression keeps its indices (0, 1, 2, ...)
        // Right expression gets remapped to (left_max + 1, left_max + 2, ...)
        let right_offset = left_scope.max_var_index + 1;

        // Remap expressions
        let left_expr = left.ast.clone(); // No remapping needed for left
        let right_expr = Self::remap_variables(&right.ast, right_offset);

        MergedScope {
            merged_registry,
            left_expr,
            right_expr,
        }
    }

    /// Find the maximum variable index used in an AST
    fn find_max_variable_index<T: Scalar>(ast: &ASTRepr<T>) -> usize {
        match ast {
            ASTRepr::Variable(index) => *index,
            ASTRepr::Constant(_) => 0,
            ASTRepr::Add(left, right) | 
            ASTRepr::Sub(left, right) | 
            ASTRepr::Mul(left, right) | 
            ASTRepr::Div(left, right) => {
                Self::find_max_variable_index(left).max(Self::find_max_variable_index(right))
            }
            ASTRepr::Neg(expr) => Self::find_max_variable_index(expr),
            ASTRepr::Sin(expr) | 
            ASTRepr::Cos(expr) | 
            ASTRepr::Exp(expr) | 
            ASTRepr::Ln(expr) |
            ASTRepr::Sqrt(expr) => Self::find_max_variable_index(expr),
            ASTRepr::Pow(base, exp) => {
                Self::find_max_variable_index(base).max(Self::find_max_variable_index(exp))
            }
            ASTRepr::Sum(_collection) => {
                // For now, assume collections don't contain variables that need remapping
                // This is a simplification - full implementation would need to analyze Collection
                0
            },
            ASTRepr::BoundVar(_) => 0, // Bound variables don't affect global variable indexing
            ASTRepr::Lambda(lambda) => Self::find_max_variable_index(&lambda.body),
            ASTRepr::Let(_, expr, body) => {
                Self::find_max_variable_index(expr).max(Self::find_max_variable_index(body))
            },
        }
    }

    /// Create a merged registry combining variables from both scopes
    fn create_merged_registry(
        left_scope: &ScopeInfo, 
        right_scope: &ScopeInfo
    ) -> Arc<RefCell<VariableRegistry>> {
        let merged = Arc::new(RefCell::new(VariableRegistry::new()));
        
        // Copy variables from left scope (keep same indices)
        {
            let left_registry = left_scope.registry.borrow();
            let mut merged_registry = merged.borrow_mut();
            
            for (name, index) in left_registry.name_to_index_mapping() {
                merged_registry.register_variable_with_index(name.clone(), index);
            }
        }

        // Copy variables from right scope (with offset indices)
        {
            let right_registry = right_scope.registry.borrow();
            let mut merged_registry = merged.borrow_mut();
            let offset = left_scope.max_var_index + 1;
            
            for (name, index) in right_registry.name_to_index_mapping() {
                let new_index = index + offset;
                // Use a modified name to avoid conflicts (could be improved with better naming strategy)
                let new_name = format!("{}_scope2", name);
                merged_registry.register_variable_with_index(new_name, new_index);
            }
        }

        merged
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
    pub fn merge_and_combine<T, F>(
        left: &TypedBuilderExpr<T>, 
        right: &TypedBuilderExpr<T>,
        combiner: F,
    ) -> TypedBuilderExpr<T>
    where
        T: Scalar,
        F: FnOnce(ASTRepr<T>, ASTRepr<T>) -> ASTRepr<T>,
    {
        if Self::needs_merging(left, right) {
            // Different scopes - perform merging
            let merged = Self::merge_scopes(left, right);
            let combined_ast = combiner(merged.left_expr, merged.right_expr);
            TypedBuilderExpr::new(combined_ast, merged.merged_registry)
        } else {
            // Same scope - no merging needed
            let combined_ast = combiner(left.ast.clone(), right.ast.clone());
            TypedBuilderExpr::new(combined_ast, left.registry.clone())
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
        let mut ctx1 = DynamicContext::<f64>::new();
        let x1: TypedBuilderExpr<f64> = ctx1.var();
        let y1: TypedBuilderExpr<f64> = ctx1.var();
        assert!(!ScopeMerger::needs_merging(&x1, &y1));

        // Different contexts - should need merging
        let mut ctx2 = DynamicContext::<f64>::new();
        let x2: TypedBuilderExpr<f64> = ctx2.var();
        assert!(ScopeMerger::needs_merging(&x1, &x2));
    }

    #[test]
    fn test_max_variable_index_detection() {
        let mut ctx = DynamicContext::<f64>::new();
        let x: TypedBuilderExpr<f64> = ctx.var(); // Variable 0
        let y: TypedBuilderExpr<f64> = ctx.var(); // Variable 1
        let z: TypedBuilderExpr<f64> = ctx.var(); // Variable 2

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
        let mut ctx1 = DynamicContext::<f64>::new();
        let x1: TypedBuilderExpr<f64> = ctx1.var(); // Variable 0 in ctx1

        let mut ctx2 = DynamicContext::<f64>::new();
        let x2: TypedBuilderExpr<f64> = ctx2.var(); // Variable 0 in ctx2 (collision!)

        // Merge scopes
        let merged = ScopeMerger::merge_scopes(&x1, &x2);

        // Check that left expression is unchanged
        match &merged.left_expr {
            ASTRepr::Variable(index) => assert_eq!(*index, 0),
            _ => panic!("Expected Variable"),
        }

        // Check that right expression is remapped
        match &merged.right_expr {
            ASTRepr::Variable(index) => assert_eq!(*index, 1), // Should be offset to 1
            _ => panic!("Expected Variable"),
        }
    }

    #[test]
    fn test_scope_merging_complex() {
        // Create two contexts with multiple variables
        let mut ctx1 = DynamicContext::<f64>::new();
        let x1: TypedBuilderExpr<f64> = ctx1.var(); // Variable 0
        let y1: TypedBuilderExpr<f64> = ctx1.var(); // Variable 1
        let expr1 = &x1 + &y1; // Uses variables 0, 1

        let mut ctx2 = DynamicContext::<f64>::new();
        let x2: TypedBuilderExpr<f64> = ctx2.var(); // Variable 0 (collision!)
        let y2: TypedBuilderExpr<f64> = ctx2.var(); // Variable 1 (collision!)
        let expr2 = &x2 * &y2; // Uses variables 0, 1

        // Merge scopes
        let merged = ScopeMerger::merge_scopes(&expr1, &expr2);

        // Left expression should use variables 0, 1
        assert_eq!(ScopeMerger::find_max_variable_index(&merged.left_expr), 1);
        
        // Right expression should use variables 2, 3 (offset by 2)
        assert_eq!(ScopeMerger::find_max_variable_index(&merged.right_expr), 3);
    }

    #[test]
    fn test_merge_and_combine() {
        // Create expressions from different contexts
        let mut ctx1 = DynamicContext::<f64>::new();
        let x1: TypedBuilderExpr<f64> = ctx1.var();
        let expr1 = &x1 * 2.0; // 2*x1

        let mut ctx2 = DynamicContext::<f64>::new();
        let x2: TypedBuilderExpr<f64> = ctx2.var();
        let expr2 = &x2 * 3.0; // 3*x2

        // Combine with automatic scope merging
        let combined = ScopeMerger::merge_and_combine(&expr1, &expr2, |left, right| {
            ASTRepr::Add(Box::new(left), Box::new(right))
        });

        // The combined expression should use variables 0 and 1
        assert_eq!(ScopeMerger::find_max_variable_index(&combined.ast), 1);

        // Create a new context that can handle the merged scope
        let temp_ctx = DynamicContext::<f64>::new();
        // Verify evaluation works correctly with merged scope
        let result = temp_ctx.eval(&combined, hlist![4.0, 5.0]);
        // Should be 2*4 + 3*5 = 8 + 15 = 23
        assert_eq!(result, 23.0);
    }
}