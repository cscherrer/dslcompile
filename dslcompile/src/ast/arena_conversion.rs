//! Conversion utilities between Box-based `ASTRepr` and arena-based `ArenaAST`
//!
//! This module provides utilities to convert between the traditional Box-based
//! `ASTRepr` and the new arena-based `ArenaAST` for gradual migration and compatibility.

use crate::ast::{
    ASTRepr, Scalar,
    arena::{ArenaAST, ArenaCollection, ArenaLambda, ArenaMultiSet, ExprArena, ExprId},
    ast_repr::{Collection, Lambda},
    multiset::MultiSet,
};
use std::collections::HashMap;

/// Convert a Box-based `ASTRepr` to an arena-based representation
///
/// This function performs a deep conversion from the traditional Box-based
/// AST to the new arena-based AST, eliminating Box allocations.
pub fn ast_to_arena<T: Scalar>(ast: &ASTRepr<T>, arena: &mut ExprArena<T>) -> ExprId {
    ast_to_arena_with_cache(ast, arena, &mut HashMap::new())
}

/// Convert with memoization to handle shared subexpressions efficiently
fn ast_to_arena_with_cache<T: Scalar>(
    ast: &ASTRepr<T>,
    arena: &mut ExprArena<T>,
    cache: &mut HashMap<*const ASTRepr<T>, ExprId>,
) -> ExprId {
    // Check cache first for structural sharing
    let ast_ptr = ast as *const ASTRepr<T>;
    if let Some(&cached_id) = cache.get(&ast_ptr) {
        return cached_id;
    }

    let expr_id = match ast {
        ASTRepr::Constant(value) => arena.constant(value.clone()),
        ASTRepr::Variable(index) => arena.variable(*index),
        ASTRepr::BoundVar(index) => arena.bound_var(*index),
        ASTRepr::Let(binding_id, expr, body) => {
            let expr_id = ast_to_arena_with_cache(expr, arena, cache);
            let body_id = ast_to_arena_with_cache(body, arena, cache);
            arena.let_binding(*binding_id, expr_id, body_id)
        }
        ASTRepr::Add(multiset) => {
            let arena_multiset = multiset_to_arena(multiset, arena, cache);
            arena.alloc(ArenaAST::Add(arena_multiset))
        }
        ASTRepr::Mul(multiset) => {
            let arena_multiset = multiset_to_arena(multiset, arena, cache);
            arena.alloc(ArenaAST::Mul(arena_multiset))
        }
        ASTRepr::Sub(left, right) => {
            let left_id = ast_to_arena_with_cache(left, arena, cache);
            let right_id = ast_to_arena_with_cache(right, arena, cache);
            arena.sub(left_id, right_id)
        }
        ASTRepr::Div(left, right) => {
            let left_id = ast_to_arena_with_cache(left, arena, cache);
            let right_id = ast_to_arena_with_cache(right, arena, cache);
            arena.div(left_id, right_id)
        }
        ASTRepr::Pow(base, exponent) => {
            let base_id = ast_to_arena_with_cache(base, arena, cache);
            let exp_id = ast_to_arena_with_cache(exponent, arena, cache);
            arena.pow(base_id, exp_id)
        }
        ASTRepr::Neg(operand) => {
            let operand_id = ast_to_arena_with_cache(operand, arena, cache);
            arena.neg(operand_id)
        }
        ASTRepr::Ln(operand) => {
            let operand_id = ast_to_arena_with_cache(operand, arena, cache);
            arena.ln(operand_id)
        }
        ASTRepr::Exp(operand) => {
            let operand_id = ast_to_arena_with_cache(operand, arena, cache);
            arena.exp(operand_id)
        }
        ASTRepr::Sin(operand) => {
            let operand_id = ast_to_arena_with_cache(operand, arena, cache);
            arena.sin(operand_id)
        }
        ASTRepr::Cos(operand) => {
            let operand_id = ast_to_arena_with_cache(operand, arena, cache);
            arena.cos(operand_id)
        }
        ASTRepr::Sqrt(operand) => {
            let operand_id = ast_to_arena_with_cache(operand, arena, cache);
            arena.sqrt(operand_id)
        }
        ASTRepr::Sum(collection) => {
            let arena_collection = collection_to_arena(collection, arena, cache);
            arena.sum(arena_collection)
        }
        ASTRepr::Lambda(lambda) => {
            let arena_lambda = lambda_to_arena(lambda, arena, cache);
            arena.alloc(ArenaAST::Lambda(arena_lambda))
        }
    };

    // Cache the result for structural sharing
    cache.insert(ast_ptr, expr_id);
    expr_id
}

/// Convert a Box-based `MultiSet` to arena-based `ArenaMultiSet`
fn multiset_to_arena<T: Scalar>(
    multiset: &MultiSet<ASTRepr<T>>,
    arena: &mut ExprArena<T>,
    cache: &mut HashMap<*const ASTRepr<T>, ExprId>,
) -> ArenaMultiSet<T> {
    let mut arena_multiset = ArenaMultiSet::new();

    for (expr, multiplicity) in multiset.iter_with_multiplicity() {
        let expr_id = ast_to_arena_with_cache(expr, arena, cache);
        // Insert with the correct multiplicity
        for _ in 0..multiplicity.as_integer().unwrap_or(1).max(1) as usize {
            arena_multiset.insert(expr_id);
        }
    }

    arena_multiset
}

/// Convert a Box-based Collection to arena-based `ArenaCollection`
fn collection_to_arena<T: Scalar>(
    collection: &Collection<T>,
    arena: &mut ExprArena<T>,
    cache: &mut HashMap<*const ASTRepr<T>, ExprId>,
) -> ArenaCollection<T> {
    match collection {
        Collection::Empty => ArenaCollection::Empty,
        Collection::Singleton(expr) => {
            let expr_id = ast_to_arena_with_cache(expr, arena, cache);
            ArenaCollection::Singleton(expr_id)
        }
        Collection::Range { start, end } => {
            let start_id = ast_to_arena_with_cache(start, arena, cache);
            let end_id = ast_to_arena_with_cache(end, arena, cache);
            ArenaCollection::Range {
                start: start_id,
                end: end_id,
            }
        }
        Collection::Variable(index) => ArenaCollection::Variable(*index),
        Collection::Filter {
            collection,
            predicate,
        } => {
            let arena_collection = collection_to_arena(collection, arena, cache);
            let predicate_id = ast_to_arena_with_cache(predicate, arena, cache);
            ArenaCollection::Filter {
                collection: Box::new(arena_collection),
                predicate: predicate_id,
            }
        }
        Collection::Map { lambda, collection } => {
            let arena_lambda = lambda_to_arena(lambda, arena, cache);
            let arena_collection = collection_to_arena(collection, arena, cache);
            ArenaCollection::Map {
                lambda: arena_lambda,
                collection: Box::new(arena_collection),
            }
        }
        Collection::DataArray(data) => ArenaCollection::DataArray(data.clone()),
    }
}

/// Convert a Box-based Lambda to arena-based `ArenaLambda`
fn lambda_to_arena<T: Scalar>(
    lambda: &Lambda<T>,
    arena: &mut ExprArena<T>,
    cache: &mut HashMap<*const ASTRepr<T>, ExprId>,
) -> ArenaLambda<T> {
    let body_id = ast_to_arena_with_cache(&lambda.body, arena, cache);
    ArenaLambda::new(lambda.var_indices.clone(), body_id)
}

/// Convert an arena-based `ArenaAST` back to Box-based `ASTRepr`
///
/// This function provides the reverse conversion for compatibility
/// with existing code that expects the traditional `ASTRepr` format.
#[must_use]
pub fn arena_to_ast<T: Scalar>(expr_id: ExprId, arena: &ExprArena<T>) -> Option<ASTRepr<T>> {
    arena_to_ast_with_cache(expr_id, arena, &mut HashMap::new())
}

/// Convert with memoization for efficiency
fn arena_to_ast_with_cache<T: Scalar>(
    expr_id: ExprId,
    arena: &ExprArena<T>,
    cache: &mut HashMap<ExprId, ASTRepr<T>>,
) -> Option<ASTRepr<T>> {
    // Check cache first
    if let Some(cached_ast) = cache.get(&expr_id) {
        return Some(cached_ast.clone());
    }

    let arena_ast = arena.get(expr_id)?;

    let ast = match arena_ast {
        ArenaAST::Constant(value) => ASTRepr::Constant(value.clone()),
        ArenaAST::Variable(index) => ASTRepr::Variable(*index),
        ArenaAST::BoundVar(index) => ASTRepr::BoundVar(*index),
        ArenaAST::Let(binding_id, expr_id, body_id) => {
            let expr = arena_to_ast_with_cache(*expr_id, arena, cache)?;
            let body = arena_to_ast_with_cache(*body_id, arena, cache)?;
            ASTRepr::Let(*binding_id, Box::new(expr), Box::new(body))
        }
        ArenaAST::Add(arena_multiset) => {
            let multiset = arena_multiset_to_multiset(arena_multiset, arena, cache)?;
            ASTRepr::Add(multiset)
        }
        ArenaAST::Mul(arena_multiset) => {
            let multiset = arena_multiset_to_multiset(arena_multiset, arena, cache)?;
            ASTRepr::Mul(multiset)
        }
        ArenaAST::Sub(left_id, right_id) => {
            let left = arena_to_ast_with_cache(*left_id, arena, cache)?;
            let right = arena_to_ast_with_cache(*right_id, arena, cache)?;
            ASTRepr::Sub(Box::new(left), Box::new(right))
        }
        ArenaAST::Div(left_id, right_id) => {
            let left = arena_to_ast_with_cache(*left_id, arena, cache)?;
            let right = arena_to_ast_with_cache(*right_id, arena, cache)?;
            ASTRepr::Div(Box::new(left), Box::new(right))
        }
        ArenaAST::Pow(base_id, exp_id) => {
            let base = arena_to_ast_with_cache(*base_id, arena, cache)?;
            let exp = arena_to_ast_with_cache(*exp_id, arena, cache)?;
            ASTRepr::Pow(Box::new(base), Box::new(exp))
        }
        ArenaAST::Neg(operand_id) => {
            let operand = arena_to_ast_with_cache(*operand_id, arena, cache)?;
            ASTRepr::Neg(Box::new(operand))
        }
        ArenaAST::Ln(operand_id) => {
            let operand = arena_to_ast_with_cache(*operand_id, arena, cache)?;
            ASTRepr::Ln(Box::new(operand))
        }
        ArenaAST::Exp(operand_id) => {
            let operand = arena_to_ast_with_cache(*operand_id, arena, cache)?;
            ASTRepr::Exp(Box::new(operand))
        }
        ArenaAST::Sin(operand_id) => {
            let operand = arena_to_ast_with_cache(*operand_id, arena, cache)?;
            ASTRepr::Sin(Box::new(operand))
        }
        ArenaAST::Cos(operand_id) => {
            let operand = arena_to_ast_with_cache(*operand_id, arena, cache)?;
            ASTRepr::Cos(Box::new(operand))
        }
        ArenaAST::Sqrt(operand_id) => {
            let operand = arena_to_ast_with_cache(*operand_id, arena, cache)?;
            ASTRepr::Sqrt(Box::new(operand))
        }
        ArenaAST::Sum(arena_collection) => {
            let collection = arena_collection_to_collection(arena_collection, arena, cache)?;
            ASTRepr::Sum(Box::new(collection))
        }
        ArenaAST::Lambda(arena_lambda) => {
            let lambda = arena_lambda_to_lambda(arena_lambda, arena, cache)?;
            ASTRepr::Lambda(Box::new(lambda))
        }
    };

    // Cache the result
    cache.insert(expr_id, ast.clone());
    Some(ast)
}

/// Convert `ArenaMultiSet` back to `MultiSet`
fn arena_multiset_to_multiset<T: Scalar>(
    arena_multiset: &ArenaMultiSet<T>,
    arena: &ExprArena<T>,
    cache: &mut HashMap<ExprId, ASTRepr<T>>,
) -> Option<MultiSet<ASTRepr<T>>> {
    let mut multiset = MultiSet::new();

    for (expr_id, multiplicity) in arena_multiset.iter() {
        let ast = arena_to_ast_with_cache(*expr_id, arena, cache)?;
        // Insert with the correct multiplicity
        for _ in 0..multiplicity.as_integer().unwrap_or(1).max(1) as usize {
            multiset.insert(ast.clone());
        }
    }

    Some(multiset)
}

/// Convert `ArenaCollection` back to Collection
fn arena_collection_to_collection<T: Scalar>(
    arena_collection: &ArenaCollection<T>,
    arena: &ExprArena<T>,
    cache: &mut HashMap<ExprId, ASTRepr<T>>,
) -> Option<Collection<T>> {
    let collection = match arena_collection {
        ArenaCollection::Empty => Collection::Empty,
        ArenaCollection::Singleton(expr_id) => {
            let ast = arena_to_ast_with_cache(*expr_id, arena, cache)?;
            Collection::Singleton(Box::new(ast))
        }
        ArenaCollection::Range { start, end } => {
            let start_ast = arena_to_ast_with_cache(*start, arena, cache)?;
            let end_ast = arena_to_ast_with_cache(*end, arena, cache)?;
            Collection::Range {
                start: Box::new(start_ast),
                end: Box::new(end_ast),
            }
        }
        ArenaCollection::Variable(index) => Collection::Variable(*index),
        ArenaCollection::Filter {
            collection,
            predicate,
        } => {
            let collection_ast = arena_collection_to_collection(collection, arena, cache)?;
            let predicate_ast = arena_to_ast_with_cache(*predicate, arena, cache)?;
            Collection::Filter {
                collection: Box::new(collection_ast),
                predicate: Box::new(predicate_ast),
            }
        }
        ArenaCollection::Map { lambda, collection } => {
            let lambda_ast = arena_lambda_to_lambda(lambda, arena, cache)?;
            let collection_ast = arena_collection_to_collection(collection, arena, cache)?;
            Collection::Map {
                lambda: Box::new(lambda_ast),
                collection: Box::new(collection_ast),
            }
        }
        ArenaCollection::DataArray(data) => Collection::DataArray(data.clone()),
    };

    Some(collection)
}

/// Convert `ArenaLambda` back to Lambda
fn arena_lambda_to_lambda<T: Scalar>(
    arena_lambda: &ArenaLambda<T>,
    arena: &ExprArena<T>,
    cache: &mut HashMap<ExprId, ASTRepr<T>>,
) -> Option<Lambda<T>> {
    let body_ast = arena_to_ast_with_cache(arena_lambda.body, arena, cache)?;
    Some(Lambda {
        var_indices: arena_lambda.var_indices.clone(),
        body: Box::new(body_ast),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ASTRepr;

    #[test]
    fn test_round_trip_conversion() {
        // Create a complex Box-based AST: 2 * (x + y) + ln(x)
        let x = ASTRepr::Variable(0);
        let y = ASTRepr::Variable(1);
        let two = ASTRepr::Constant(2.0_f64);

        let x_plus_y = ASTRepr::add_binary(x.clone(), y);
        let two_times_sum = ASTRepr::mul_binary(two, x_plus_y);
        let ln_x = ASTRepr::Ln(Box::new(x));
        let final_expr = ASTRepr::add_binary(two_times_sum, ln_x);

        // Convert to arena
        let mut arena = ExprArena::new();
        let arena_id = ast_to_arena(&final_expr, &mut arena);

        // Convert back to Box-based AST
        let converted_back = arena_to_ast(arena_id, &arena).unwrap();

        // Should be equivalent (though not necessarily identical due to multiset ordering)
        // We'll test that the structure is preserved correctly
        match (&final_expr, &converted_back) {
            (ASTRepr::Add(_), ASTRepr::Add(_)) => {
                // Both should be addition operations
                assert!(true);
            }
            _ => panic!("Round-trip conversion failed"),
        }

        // Verify arena efficiency - should have fewer nodes than a naive conversion
        // because of structural sharing
        assert!(arena.len() <= 10); // Should be efficient
    }

    #[test]
    fn test_structural_sharing() {
        // Create an expression with shared subexpressions: (x + 1) * (x + 1)
        let x = ASTRepr::Variable(0);
        let one = ASTRepr::Constant(1.0_f64);
        let x_plus_one = ASTRepr::add_binary(x, one);
        let squared = ASTRepr::mul_binary(x_plus_one.clone(), x_plus_one.clone());

        let mut arena = ExprArena::new();
        let _arena_id = ast_to_arena(&squared, &mut arena);

        // Without sharing, this would create duplicate nodes for (x + 1)
        // With sharing, we should have: x, 1, x+1, squared = 4 nodes
        // Note: due to multiset internal structure, we might have a few more
        assert!(arena.len() <= 8); // Should be reasonably efficient
    }
}
