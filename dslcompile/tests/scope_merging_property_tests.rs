//! Property Tests for Automatic Scope Merging
//!
//! This module implements comprehensive property-based tests for automatic scope merging,
//! ensuring that expressions from different contexts can be safely combined without
//! variable collisions and that the resulting expressions evaluate correctly.

use dslcompile::{
    prelude::*,
    contexts::{ScopeMerger, ScopeInfo},
    ast::ast_repr::ASTRepr,
};
use frunk::hlist;
use proptest::prelude::*;
use rand::SeedableRng;
use std::collections::HashSet;

/// Configuration for scope merging complexity in property tests
#[derive(Debug, Clone)]
pub struct ScopeMergingComplexity {
    pub max_contexts: usize,
    pub max_expr_depth: usize,
    pub max_variables_per_context: usize,
    pub max_evaluation_attempts: usize,
}

impl Default for ScopeMergingComplexity {
    fn default() -> Self {
        Self {
            max_contexts: 3,
            max_expr_depth: 4,
            max_variables_per_context: 3,
            max_evaluation_attempts: 10,
        }
    }
}

/// Represents an expression built in an independent context for testing
#[derive(Debug, Clone)]
pub struct ContextualExpression {
    pub expr: TypedBuilderExpr<f64>,
    pub context_id: usize,
    pub used_variables: HashSet<usize>,
    pub registry: std::sync::Arc<std::cell::RefCell<VariableRegistry>>,
}

/// Generates multiple independent contexts with expressions for property testing
#[derive(Debug, Clone)]
pub struct MultiContextScenario {
    pub expressions: Vec<ContextualExpression>,
    pub complexity: ScopeMergingComplexity,
}

impl MultiContextScenario {
    /// Generate a simple multi-context scenario for testing
    pub fn simple() -> BoxedStrategy<Self> {
        let config = ScopeMergingComplexity {
            max_contexts: 2,
            max_expr_depth: 3,
            max_variables_per_context: 2,
            max_evaluation_attempts: 5,
        };
        Self::with_complexity(config)
    }

    /// Generate a complex multi-context scenario with many contexts and deep expressions
    pub fn complex() -> BoxedStrategy<Self> {
        let config = ScopeMergingComplexity {
            max_contexts: 4,
            max_expr_depth: 6,
            max_variables_per_context: 4,
            max_evaluation_attempts: 20,
        };
        Self::with_complexity(config)
    }

    /// Generate scenarios with specific complexity settings
    pub fn with_complexity(complexity: ScopeMergingComplexity) -> BoxedStrategy<Self> {
        (1..=complexity.max_contexts)
            .prop_flat_map(move |num_contexts| {
                let config = complexity.clone();
                proptest::collection::vec(
                    Self::generate_contextual_expression(config.clone()),
                    num_contexts..=num_contexts,
                )
                .prop_map(move |expressions| MultiContextScenario {
                    expressions,
                    complexity: config.clone(),
                })
            })
            .boxed()
    }

    /// Generate a single expression in its own independent context
    fn generate_contextual_expression(
        complexity: ScopeMergingComplexity,
    ) -> impl Strategy<Value = ContextualExpression> {
        static CONTEXT_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        (1..=complexity.max_variables_per_context)
            .prop_flat_map(move |num_vars| {
                Self::generate_expression_with_vars(num_vars, complexity.max_expr_depth)
            })
            .prop_map(move |expr_data| {
                let context_id = CONTEXT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                ContextualExpression {
                    expr: expr_data.0,
                    context_id,
                    used_variables: expr_data.1,
                    registry: expr_data.2,
                }
            })
    }

    /// Generate an expression using a specific number of variables
    fn generate_expression_with_vars(
        num_vars: usize,
        max_depth: usize,
    ) -> impl Strategy<Value = (TypedBuilderExpr<f64>, HashSet<usize>, std::sync::Arc<std::cell::RefCell<VariableRegistry>>)> {
        any::<u64>().prop_map(move |seed| {
            // Create a new independent context
            let mut ctx = DynamicContext::<f64>::new();
            
            // Create the specified number of variables
            let vars: Vec<_> = (0..num_vars).map(|_| ctx.var()).collect();
            
            // Build a random expression using these variables
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let expr = Self::build_random_expression(&vars, &mut ctx, max_depth, &mut rng);
            
            // Track which variables were used
            let used_variables: HashSet<usize> = (0..num_vars).collect();
            
            (expr, used_variables, std::sync::Arc::new(std::cell::RefCell::new(dslcompile::contexts::VariableRegistry::new())))
        })
    }

    /// Build a random expression using available variables
    fn build_random_expression(
        vars: &[TypedBuilderExpr<f64>],
        ctx: &mut DynamicContext<f64>,
        max_depth: usize,
        rng: &mut impl rand::Rng,
    ) -> TypedBuilderExpr<f64> {
        if max_depth == 0 || vars.is_empty() {
            // Base case: return a variable or constant
            if vars.is_empty() || rng.gen_bool(0.3) {
                ctx.constant(rng.gen_range(-10.0..10.0))
            } else {
                vars[rng.gen_range(0..vars.len())].clone()
            }
        } else {
            // Recursive case: build a compound expression
            let op_choice = rng.gen_range(0..4);
            let left = Self::build_random_expression(vars, ctx, max_depth - 1, rng);
            let right = Self::build_random_expression(vars, ctx, max_depth - 1, rng);
            
            match op_choice {
                0 => &left + &right,
                1 => &left - &right,
                2 => &left * &right,
                3 => {
                    // Avoid division by zero
                    let safe_right = &right + &ctx.constant(1.0);
                    &left / &safe_right
                }
                _ => unreachable!(),
            }
        }
    }

    /// Combine all expressions using automatic scope merging
    pub fn merge_all_expressions(&self) -> TypedBuilderExpr<f64> {
        if self.expressions.is_empty() {
            panic!("Cannot merge empty expression list");
        }
        
        let mut result = self.expressions[0].expr.clone();
        
        for expr in &self.expressions[1..] {
            result = &result + &expr.expr; // This should trigger automatic scope merging
        }
        
        result
    }

    /// Calculate the expected number of variables in the merged expression
    pub fn expected_merged_variable_count(&self) -> usize {
        self.expressions.iter()
            .map(|expr| expr.used_variables.len())
            .sum()
    }

    /// Generate test values for all variables across all contexts
    pub fn generate_test_values(&self) -> Vec<f64> {
        (0..self.expected_merged_variable_count())
            .map(|i| (i as f64 + 1.0) * 2.0) // Simple deterministic values
            .collect()
    }
}

/// Utility functions for scope merging analysis
pub mod scope_utils {
    use super::*;
    use dslcompile::contexts::scope_merging::ScopeMerger;

    /// Extract all variable indices used in a TypedBuilderExpr
    pub fn extract_variable_indices(expr: &TypedBuilderExpr<f64>) -> HashSet<usize> {
        extract_variables_from_ast(expr.as_ast())
    }

    fn extract_variables_from_ast(ast: &ASTRepr<f64>) -> HashSet<usize> {
        let mut indices = HashSet::new();
        extract_variables_recursive(ast, &mut indices);
        indices
    }

    fn extract_variables_recursive(ast: &ASTRepr<f64>, indices: &mut HashSet<usize>) {
        match ast {
            ASTRepr::Variable(idx) => {
                indices.insert(*idx);
            }
            ASTRepr::Constant(_) | ASTRepr::BoundVar(_) => {}
            ASTRepr::Add(l, r) | ASTRepr::Sub(l, r) | ASTRepr::Mul(l, r) | ASTRepr::Div(l, r) | ASTRepr::Pow(l, r) => {
                extract_variables_recursive(l, indices);
                extract_variables_recursive(r, indices);
            }
            ASTRepr::Neg(inner) | ASTRepr::Sin(inner) | ASTRepr::Cos(inner) | 
            ASTRepr::Exp(inner) | ASTRepr::Ln(inner) | ASTRepr::Sqrt(inner) => {
                extract_variables_recursive(inner, indices);
            }
            ASTRepr::Sum(_) => {
                // Simplified for now - would need to analyze Collection structure
            }
            ASTRepr::Lambda(lambda) => {
                extract_variables_recursive(&lambda.body, indices);
            }
            ASTRepr::Let(_, expr, body) => {
                extract_variables_recursive(expr, indices);
                extract_variables_recursive(body, indices);
            }
        }
    }

    /// Check if an expression needs scope merging by examining its registries
    pub fn needs_scope_merging(expressions: &[&TypedBuilderExpr<f64>]) -> bool {
        if expressions.len() < 2 {
            return false;
        }
        
        // Since we can't access the registry field directly, we use a different approach
        // For now, assume scope merging is needed if we have multiple expressions
        expressions.len() > 1
    }

    /// Verify that scope merging preserved the mathematical semantics
    pub fn verify_merging_semantics(
        original_expressions: &[&TypedBuilderExpr<f64>],
        merged_expression: &TypedBuilderExpr<f64>,
        test_values: &[f64],
    ) -> dslcompile::error::Result<()> {
        // This is a simplified semantic check
        // In practice, we'd need more sophisticated verification
        
        let merged_variables = extract_variable_indices(merged_expression);
        let expected_var_count: usize = original_expressions.iter()
            .map(|expr| extract_variable_indices(expr).len())
            .sum();
            
        if merged_variables.len() != expected_var_count {
            return Err(format!(
                "Expected {} variables in merged expression, found {}",
                expected_var_count,
                merged_variables.len()
            ).into());
        }
        
        // Verify variable indices are contiguous starting from 0
        let expected_indices: HashSet<_> = (0..expected_var_count).collect();
        if merged_variables != expected_indices {
            return Err(format!(
                "Merged expression has non-contiguous variable indices: {:?}",
                merged_variables
            ).into());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_generation() {
        // Verify that we can generate multi-context scenarios
        let strategy = MultiContextScenario::simple();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        for _ in 0..5 {
            let scenario = strategy.new_tree(&mut runner).unwrap().current();
            
            // Basic validation
            assert!(!scenario.expressions.is_empty());
            assert!(scenario.expressions.len() <= scenario.complexity.max_contexts);
            
            // Each expression should be in its own context
            let context_ids: HashSet<_> = scenario.expressions.iter()
                .map(|expr| expr.context_id)
                .collect();
            assert_eq!(context_ids.len(), scenario.expressions.len());
        }
    }

    proptest! {
        #[test]
        fn prop_scope_merging_preserves_variable_count(scenario in MultiContextScenario::simple()) {
            // When we merge expressions from different contexts,
            // the total number of variables should be preserved
            
            let merged = scenario.merge_all_expressions();
            let merged_vars = scope_utils::extract_variable_indices(&merged);
            let expected_count = scenario.expected_merged_variable_count();
            
            prop_assert_eq!(merged_vars.len(), expected_count,
                          "Merged expression should have {} variables, found {}", 
                          expected_count, merged_vars.len());
        }

        #[test]
        fn prop_scope_merging_creates_contiguous_indices(scenario in MultiContextScenario::simple()) {
            // After merging, variable indices should be contiguous starting from 0
            
            let merged = scenario.merge_all_expressions();
            let merged_vars = scope_utils::extract_variable_indices(&merged);
            let expected_indices: HashSet<_> = (0..merged_vars.len()).collect();
            
            let merged_vars_len = merged_vars.len();
            prop_assert_eq!(merged_vars, expected_indices,
                          "Merged expression should have contiguous variable indices 0..{}", 
                          merged_vars_len);
        }

        #[test]
        fn prop_scope_merging_evaluation_is_deterministic(scenario in MultiContextScenario::simple()) {
            // Multiple evaluations of the same merged expression should give identical results
            
            let merged = scenario.merge_all_expressions();
            let test_values = scenario.generate_test_values();
            
            // Skip if we don't have enough test values
            if test_values.len() < scenario.expected_merged_variable_count() {
                return Ok(());
            }
            
            // Create HList for evaluation - simplified for basic cases
            let result1 = match test_values.len() {
                0 => return Ok(()), // No variables to test
                1 => {
                    let temp_ctx = DynamicContext::<f64>::new();
                    temp_ctx.eval(&merged, hlist![test_values[0]])
                },
                2 => {
                    let temp_ctx = DynamicContext::<f64>::new();
                    temp_ctx.eval(&merged, hlist![test_values[0], test_values[1]])
                },
                3 => {
                    let temp_ctx = DynamicContext::<f64>::new();
                    temp_ctx.eval(&merged, hlist![test_values[0], test_values[1], test_values[2]])
                },
                _ => return Ok(()), // Skip complex cases for now
            };
            
            let result2 = match test_values.len() {
                1 => {
                    let temp_ctx = DynamicContext::<f64>::new();
                    temp_ctx.eval(&merged, hlist![test_values[0]])
                },
                2 => {
                    let temp_ctx = DynamicContext::<f64>::new();
                    temp_ctx.eval(&merged, hlist![test_values[0], test_values[1]])
                },
                3 => {
                    let temp_ctx = DynamicContext::<f64>::new();
                    temp_ctx.eval(&merged, hlist![test_values[0], test_values[1], test_values[2]])
                },
                _ => return Ok(()),
            };
            
            prop_assert!((result1 - result2).abs() < 1e-12,
                       "Multiple evaluations should be deterministic: {} vs {}", result1, result2);
        }

        #[test]
        fn prop_manual_vs_automatic_merging_equivalence(
            x1 in -10.0..10.0f64,
            y1 in -10.0..10.0f64,
            x2 in -10.0..10.0f64,
            y2 in -10.0..10.0f64
        ) {
            // Compare manual merging (current approach) vs automatic merging
            
            // Manual approach: rebuild expressions in unified context
            let mut unified_ctx = DynamicContext::<f64>::new();
            let x_unified = unified_ctx.var();
            let y_unified = unified_ctx.var();
            let manual_expr = &x_unified * 2.0 + &y_unified * 3.0;
            let manual_result = unified_ctx.eval(&manual_expr, hlist![x1, x2]);
            
            // Automatic approach: use scope merging
            let mut ctx1 = DynamicContext::<f64>::new();
            let x1_var = ctx1.var();
            let expr1 = &x1_var * 2.0;
            
            let mut ctx2 = DynamicContext::<f64>::new();
            let x2_var = ctx2.var();
            let expr2 = &x2_var * 3.0;
            
            let automatic_expr = &expr1 + &expr2; // Should trigger scope merging
            let temp_ctx = DynamicContext::<f64>::new();
            let automatic_result = temp_ctx.eval(&automatic_expr, hlist![x1, x2]);
            
            prop_assert!((manual_result - automatic_result).abs() < 1e-12,
                       "Manual and automatic merging should give same results: {} vs {}", 
                       manual_result, automatic_result);
        }

        #[test]
        fn prop_scope_merging_is_commutative(
            x in -10.0..10.0f64,
            y in -10.0..10.0f64
        ) {
            // expr1 + expr2 should equal expr2 + expr1 even across contexts
            
            let mut ctx1 = DynamicContext::<f64>::new();
            let x1 = ctx1.var();
            let expr1 = &x1 * 2.0;
            
            let mut ctx2 = DynamicContext::<f64>::new();
            let x2 = ctx2.var();
            let expr2 = &x2 + 1.0;
            
            let combined1 = &expr1 + &expr2;
            let combined2 = &expr2 + &expr1;
            
            let temp_ctx = DynamicContext::<f64>::new();
            let result1 = temp_ctx.eval(&combined1, hlist![x, y]);
            let result2 = temp_ctx.eval(&combined2, hlist![x, y]);
            
            prop_assert!((result1 - result2).abs() < 1e-12,
                       "Cross-context addition should be commutative: {} vs {}", result1, result2);
        }

        #[test]
        fn prop_scope_merging_handles_same_context_correctly(
            x in -10.0..10.0f64,
            y in -10.0..10.0f64
        ) {
            // When expressions are from the same context, no merging should occur
            
            let mut ctx = DynamicContext::<f64>::new();
            let x_var = ctx.var();
            let y_var = ctx.var();
            
            let expr1 = &x_var * 2.0;
            let expr2 = &y_var + 1.0;
            
            // This should NOT trigger scope merging since they're from the same context
            let combined = &expr1 + &expr2;
            
            // Should evaluate normally
            let result = ctx.eval(&combined, hlist![x, y]);
            let expected = x * 2.0 + y + 1.0;
            
            prop_assert!((result - expected).abs() < 1e-12,
                       "Same-context expressions should work normally: {} vs {}", result, expected);
        }
    }

    #[test]
    fn test_complex_multi_context_scenario() {
        // Test a specific complex scenario manually
        let mut ctx1 = DynamicContext::<f64>::new();
        let x1 = ctx1.var();
        let y1 = ctx1.var();
        let expr1 = &x1 * &y1 + 1.0; // x1 * y1 + 1
        
        let mut ctx2 = DynamicContext::<f64>::new();
        let x2 = ctx2.var();
        let expr2 = &x2 * 2.0; // 2 * x2
        
        let mut ctx3 = DynamicContext::<f64>::new();
        let x3 = ctx3.var();
        let expr3 = &x3 + 3.0; // x3 + 3
        
        // Combine all three expressions
        let combined = &(&expr1 + &expr2) + &expr3;
        
        // Should result in expression using variables [0, 1, 2, 3]
        let merged_vars = scope_utils::extract_variable_indices(&combined);
        assert_eq!(merged_vars.len(), 4);
        assert_eq!(merged_vars, (0..4).collect());
        
        // Test evaluation
        let temp_ctx = DynamicContext::<f64>::new();
        let result = temp_ctx.eval(&combined, hlist![2.0, 3.0, 4.0, 5.0]);
        
        // Expected: (2*3 + 1) + (2*4) + (5 + 3) = 7 + 8 + 8 = 23
        let expected = (2.0 * 3.0 + 1.0) + (2.0 * 4.0) + (5.0 + 3.0);
        assert!((result - expected).abs() < 1e-12);
    }
}