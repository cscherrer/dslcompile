//! Summation Support for DSLCompile
//!
//! This module provides unified HList-based summation capabilities that eliminate
//! the `Constant` architecture in favor of treating all inputs as typed variables
//! in the same `HList` structure.
//!
//! ## Key Components
//!
//! - `IntoHListSummationRange`: Unified trait for converting inputs to summation
//! - Range summation implementations for mathematical ranges
//! - Data array summation implementations for Vec<T> and slice types
//! - Type-safe summation that preserves heterogeneous structure

use crate::{
    ast::{
        ExpressionType, Scalar,
        ast_repr::{ASTRepr, Collection, Lambda},
    },
    contexts::dynamic::expression_builder::{DynamicContext, DynamicExpr},
};

/// Trait for HList-based summation that eliminates `Constant` architecture
///
/// This trait provides a unified approach where all inputs (mathematical ranges,
/// data vectors, etc.) are treated as typed variables in the same `HList` rather
/// than artificial `Constant` separation.
pub trait IntoHListSummationRange<T: Scalar + ExpressionType> {
    /// Convert input to `HList` summation, creating appropriate Variable references
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<T, SCOPE>
    where
        F: FnOnce(DynamicExpr<T, SCOPE>) -> DynamicExpr<T, SCOPE>,
        T: num_traits::FromPrimitive + Copy;
}

/// Implementation for mathematical ranges - creates Range collection (no `Constant`)
impl<T: Scalar + ExpressionType + num_traits::FromPrimitive> IntoHListSummationRange<T>
    for std::ops::RangeInclusive<T>
{
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<T, SCOPE>
    where
        F: FnOnce(DynamicExpr<T, SCOPE>) -> DynamicExpr<T, SCOPE>,
        T: num_traits::FromPrimitive + Copy,
    {
        let start = *self.start();
        let end = *self.end();

        // Create iterator variable using De Bruijn index for the lambda
        // De Bruijn indices provide canonical representation and prevent variable capture
        let iter_var_id = 0; // De Bruijn index: 0 = innermost bound variable in single-argument lambdas

        // Create iterator variable expression using BoundVar for lambda body
        let iter_var = DynamicExpr::new(ASTRepr::BoundVar(iter_var_id), ctx.registry.clone());

        // Apply the function to the iterator variable
        let body = f(iter_var);

        // Create the lambda that maps over the range
        let lambda = Lambda {
            var_indices: vec![iter_var_id],
            body: Box::new(body.ast),
        };

        // Create the underlying range collection
        let range_collection = Collection::Range {
            start: Box::new(ASTRepr::Constant(start)),
            end: Box::new(ASTRepr::Constant(end)),
        };

        // Create Map collection that applies lambda to range
        let map_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(range_collection),
        };

        DynamicExpr::new(ASTRepr::Sum(Box::new(map_collection)), ctx.registry.clone())
    }
}

/// Implementation for integer ranges with f64 context - converts integers to f64
impl IntoHListSummationRange<f64> for std::ops::RangeInclusive<i32> {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Convert integer range to f64 range
        let start = f64::from(*self.start());
        let end = f64::from(*self.end());

        // Create iterator variable using De Bruijn index for the lambda
        // De Bruijn indices provide canonical representation and prevent variable capture
        let iter_var_id = 0; // De Bruijn index: 0 = innermost bound variable in single-argument lambdas

        // Create iterator variable expression using BoundVar for lambda body
        let iter_var = DynamicExpr::new(ASTRepr::BoundVar(iter_var_id), ctx.registry.clone());

        // Apply the function to the iterator variable
        let body = f(iter_var);

        // Create the lambda that maps over the range
        let lambda = Lambda {
            var_indices: vec![iter_var_id],
            body: Box::new(body.ast),
        };

        // Create the underlying range collection
        let range_collection = Collection::Range {
            start: Box::new(ASTRepr::Constant(start)),
            end: Box::new(ASTRepr::Constant(end)),
        };

        // Create Map collection that applies lambda to range
        let map_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(range_collection),
        };

        DynamicExpr::new(ASTRepr::Sum(Box::new(map_collection)), ctx.registry.clone())
    }
}

/// Implementation for data vectors - creates explicit singleton collections
impl IntoHListSummationRange<f64> for Vec<f64> {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        if self.is_empty() {
            // Empty data array - return sum of empty collection
            return DynamicExpr::new(
                ASTRepr::Sum(Box::new(Collection::Empty)),
                ctx.registry.clone(),
            );
        }

        // Create iterator variable using De Bruijn index for the lambda
        // De Bruijn indices provide canonical representation and prevent variable capture
        let iter_var_id = 0; // De Bruijn index: 0 = innermost bound variable in single-argument lambdas

        // Create iterator variable expression using BoundVar for lambda body
        let iter_var = DynamicExpr::new(ASTRepr::BoundVar(iter_var_id), ctx.registry.clone());

        // Apply the function to the iterator variable
        let body = f(iter_var);

        // Create the lambda that maps over the data
        let lambda = Lambda {
            var_indices: vec![iter_var_id],
            body: Box::new(body.ast),
        };

        // For data arrays, embed the data directly in the AST
        // This avoids variable indexing issues and makes evaluation simpler
        let data_collection = Collection::Constant(self);

        // Create Map collection that applies lambda to the data array
        let map_collection = Collection::Map {
            lambda: Box::new(lambda),
            collection: Box::new(data_collection),
        };

        DynamicExpr::new(ASTRepr::Sum(Box::new(map_collection)), ctx.registry.clone())
    }
}

/// Implementation for data slices - creates `Constant` collection (transitional approach)
impl IntoHListSummationRange<f64> for &Vec<f64> {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Clone and delegate to Vec<f64> implementation
        self.clone().into_hlist_summation(ctx, f)
    }
}

/// Implementation for f64 slices - converts to Vec and delegates
impl IntoHListSummationRange<f64> for &[f64] {
    fn into_hlist_summation<F, const SCOPE: usize>(
        self,
        ctx: &mut DynamicContext<SCOPE>,
        f: F,
    ) -> DynamicExpr<f64, SCOPE>
    where
        F: FnOnce(DynamicExpr<f64, SCOPE>) -> DynamicExpr<f64, SCOPE>,
        f64: num_traits::FromPrimitive + Copy,
    {
        // Convert slice to Vec and delegate
        self.to_vec().into_hlist_summation(ctx, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contexts::dynamic::expression_builder::DynamicContext;

    #[test]
    fn test_unified_sum_api() {
        let mut ctx = DynamicContext::new();

        // Test 1: Range summation using the unified API
        let range_sum = ctx.sum(1..=5, |x| x * 2);
        println!("Range sum AST: {:?}", range_sum.as_ast());

        // Test 2: Parametric summation using the unified API
        let param: DynamicExpr<f64, 0> = ctx.var();
        let param_sum = ctx.sum(1..=3, |x| x * param.clone());
        println!("Parametric sum AST: {:?}", param_sum.as_ast());

        // Verify the AST structure is correct (Sum(Map{lambda, collection}))
        match range_sum.as_ast() {
            ASTRepr::Sum(collection) => match collection.as_ref() {
                Collection::Map {
                    lambda: _,
                    collection: _,
                } => {
                    println!("✅ Correct structure: Sum(Map{{lambda, collection}})");
                }
                _ => panic!("❌ Expected Map collection"),
            },
            _ => panic!("❌ Expected Sum AST"),
        }
    }

    #[test]
    fn test_vector_summation() {
        let mut ctx = DynamicContext::new();

        // Test vector summation
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector_sum = ctx.sum(data, |x| x * 2.0);

        // Verify the AST structure
        match vector_sum.as_ast() {
            ASTRepr::Sum(collection) => match collection.as_ref() {
                Collection::Map {
                    lambda: _,
                    collection: inner,
                } => match inner.as_ref() {
                    Collection::Constant(data) => {
                        assert_eq!(data, &vec![1.0, 2.0, 3.0, 4.0, 5.0]);
                        println!("✅ Constant collection contains correct data");
                    }
                    _ => panic!("❌ Expected Constant collection"),
                },
                _ => panic!("❌ Expected Map collection"),
            },
            _ => panic!("❌ Expected Sum AST"),
        }
    }

    #[test]
    fn test_slice_summation() {
        let mut ctx = DynamicContext::new();

        // Test slice summation
        let data = [2.0, 4.0, 6.0];
        let slice_sum = ctx.sum(&data[..], |x| x / 2.0);

        // Should work the same as vector summation
        match slice_sum.as_ast() {
            ASTRepr::Sum(_) => {
                println!("✅ Slice summation creates valid Sum AST");
            }
            _ => panic!("❌ Expected Sum AST"),
        }
    }
}
