use dslcompile::{
    ast::{
        advanced::{pretty_ast},
        runtime::VariableRegistry,
        ASTRepr,
    },
    frunk::hlist,
    interval_domain::{IntervalDomain, IntervalDomainAnalyzer},
    symbolic::symbolic::SymbolicOptimizer,
    contexts::dynamic::DynamicContext,
    DSLCompileError,
};

#[cfg(test)]
mod lambda_variable_binding_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    /// Test that lambda parameters are correctly bound as BoundVar, not free Variable
    #[test]
    fn test_lambda_variable_binding_correctness() {
        let mut ctx = DynamicContext::new();
        
        // Create parameters
        let mu = ctx.var();    // Variable(0) 
        let sigma = ctx.var(); // Variable(1)
        
        // Create summation expression with lambda
        let data_placeholder: &[f64] = &[];
        let sum_expr = ctx.sum(data_placeholder, |x| {
            // x should be BoundVar(0), not Variable(2)
            let centered = x - &mu;           // (x - μ)
            let standardized = &centered / &sigma; // (x - μ) / σ
            &standardized * &standardized    // ((x - μ) / σ)²
        });
        
        // Convert to AST and analyze variables
        let ast = ctx.to_ast(&sum_expr);
        let mut free_vars = HashSet::new();
        collect_free_variables(&ast, &mut free_vars);
        
        // Should only have 3 free variables: μ=0, σ=1, data_collection=2
        assert_eq!(free_vars.len(), 3, "Expected exactly 3 free variables (μ, σ, data), found: {:?}", free_vars);
        assert!(free_vars.contains(&0), "Missing μ variable");
        assert!(free_vars.contains(&1), "Missing σ variable"); 
        assert!(free_vars.contains(&2), "Missing data collection variable");
        
        // Verify the AST structure contains a proper lambda with BoundVar
        assert!(contains_bound_var(&ast), "AST should contain BoundVar for lambda parameter");
        
        println!("✅ Lambda variable binding test passed: {} free variables found", free_vars.len());
    }

    /// Property test: Lambda expressions should always have correct variable count
    proptest! {
        #[test]
        fn prop_lambda_variable_count_is_correct(
            mu_val in -10.0..10.0f64,
            sigma_val in 0.1..10.0f64,
            data_size in 1..10usize
        ) {
            let mut ctx = DynamicContext::new();
            
            // Create parameters
            let mu = ctx.var();    // Variable(0)
            let sigma = ctx.var(); // Variable(1) 
            
            // Create test data
            let test_data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();
            
            // Create summation with lambda
            let sum_expr = ctx.sum(test_data.as_slice(), |x| {
                let diff = x - &mu;
                &diff / &sigma
            });
            
            // Analyze AST
            let ast = ctx.to_ast(&sum_expr);
            let mut free_vars = HashSet::new();
            collect_free_variables(&ast, &mut free_vars);
            
            // Property: Should always have exactly 3 free variables regardless of data size
            prop_assert_eq!(free_vars.len(), 3, "Lambda expression should have exactly 3 free variables");
            prop_assert!(free_vars.contains(&0), "Should contain μ variable");
            prop_assert!(free_vars.contains(&1), "Should contain σ variable");
            prop_assert!(free_vars.contains(&2), "Should contain data variable");
        }
    }

    /// Test that evaluation works correctly with bound variables
    #[test]
    fn test_lambda_evaluation_correctness() {
        let mut ctx = DynamicContext::new();
        
        let mu = ctx.var();
        let sigma = ctx.var();
        
        // Create simple summation: sum over [1,2,3] of (x - mu) / sigma
        let test_data = vec![1.0, 2.0, 3.0];
        let sum_expr = ctx.sum(test_data.as_slice(), |x| {
            let diff = x - &mu;
            &diff / &sigma
        });
        
        // Evaluate with mu=0, sigma=1
        let result = ctx.eval(&sum_expr, hlist![0.0, 1.0]);
        
        // Expected: (1-0)/1 + (2-0)/1 + (3-0)/1 = 1 + 2 + 3 = 6
        let expected = 6.0;
        assert!((result - expected).abs() < 1e-10, 
               "Expected {}, got {}", expected, result);
               
        println!("✅ Lambda evaluation test passed: result = {}", result);
    }

    /// Helper function to collect free variables (not bound variables)
    fn collect_free_variables(ast: &ASTRepr<f64>, vars: &mut HashSet<usize>) {
        match ast {
            ASTRepr::Variable(index) => {
                vars.insert(*index);
            }
            ASTRepr::BoundVar(_) => {
                // BoundVar should NOT be counted as a free variable
            }
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                collect_free_variables(left, vars);
                collect_free_variables(right, vars);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                collect_free_variables(inner, vars);
            }
            ASTRepr::Sum(collection) => {
                collect_free_variables_from_collection(collection, vars);
            }
            ASTRepr::Lambda(lambda) => {
                collect_free_variables(&lambda.body, vars);
            }
            ASTRepr::Constant(_) => {}
            ASTRepr::Let(_, expr, body) => {
                collect_free_variables(expr, vars);
                collect_free_variables(body, vars);
            }
        }
    }

    /// Helper function for collections
    fn collect_free_variables_from_collection(collection: &dslcompile::ast::ast_repr::Collection<f64>, vars: &mut HashSet<usize>) {
        use dslcompile::ast::ast_repr::Collection;
        match collection {
            Collection::Empty => {}
            Collection::Singleton(expr) => collect_free_variables(expr, vars),
            Collection::Range { start, end } => {
                collect_free_variables(start, vars);
                collect_free_variables(end, vars);
            }
            Collection::Variable(index) => {
                vars.insert(*index);
            }
            Collection::Filter { collection, predicate } => {
                collect_free_variables_from_collection(collection, vars);
                collect_free_variables(predicate, vars);
            }
            Collection::Map { lambda, collection } => {
                collect_free_variables(&lambda.body, vars);
                collect_free_variables_from_collection(collection, vars);
            }
        }
    }

    /// Helper function to check if AST contains BoundVar
    fn contains_bound_var(ast: &ASTRepr<f64>) -> bool {
        match ast {
            ASTRepr::BoundVar(_) => true,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                contains_bound_var(left) || contains_bound_var(right)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                contains_bound_var(inner)
            }
            ASTRepr::Sum(collection) => {
                contains_bound_var_in_collection(collection)
            }
            ASTRepr::Lambda(lambda) => {
                contains_bound_var(&lambda.body)
            }
            ASTRepr::Let(_, expr, body) => {
                contains_bound_var(expr) || contains_bound_var(body)
            }
            _ => false,
        }
    }

    /// Helper for collections
    fn contains_bound_var_in_collection(collection: &dslcompile::ast::ast_repr::Collection<f64>) -> bool {
        use dslcompile::ast::ast_repr::Collection;
        match collection {
            Collection::Map { lambda, collection } => {
                contains_bound_var(&lambda.body) || contains_bound_var_in_collection(collection)
            }
            Collection::Filter { collection, predicate } => {
                contains_bound_var_in_collection(collection) || contains_bound_var(predicate)
            }
            Collection::Singleton(expr) => contains_bound_var(expr),
            Collection::Range { start, end } => {
                contains_bound_var(start) || contains_bound_var(end)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod egg_round_trip_tests {
    use super::*;
    use proptest::prelude::*;
    
    #[cfg(feature = "optimization")]
    use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

    /// Property test: MathLang ↔ ASTRepr round-trip semantic equivalence
    /// This ensures that converting AST → MathLang → AST preserves semantic meaning
    #[cfg(feature = "optimization")]
    proptest! {
        #[test]
        fn prop_egg_round_trip_semantic_equivalence(
            // Use smaller ranges to avoid numerical overflow in complex expressions
            var_count in 1..=3usize,
            expr_complexity in 1..=5usize
        ) {
            let mut ctx = DynamicContext::new();
            
            // Create a simple expression to test round-trip conversion
            // Start with basic arithmetic expressions that are supported by egg optimizer
            let vars: Vec<_> = (0..var_count).map(|_| ctx.var()).collect();
            
            // Build expression based on complexity
            let expr = match expr_complexity {
                1 => {
                    // Simple variable
                    vars[0].clone()
                }
                2 => {
                    // Simple addition: x + y or x + constant
                    if vars.len() > 1 {
                        &vars[0] + &vars[1]
                    } else {
                        &vars[0] + 2.0
                    }
                }
                3 => {
                    // Multiplication: x * y or x * constant
                    if vars.len() > 1 {
                        &vars[0] * &vars[1]
                    } else {
                        &vars[0] * 3.0
                    }
                }
                4 => {
                    // Mixed: x * 2 + y
                    if vars.len() > 1 {
                        &vars[0] * 2.0 + &vars[1]
                    } else {
                        &vars[0] * 2.0 + 1.0
                    }
                }
                _ => {
                    // Complex: (x + y) * (x - z) + constant
                    if vars.len() >= 3 {
                        (&vars[0] + &vars[1]) * (&vars[0] - &vars[2]) + 5.0
                    } else if vars.len() == 2 {
                        (&vars[0] + &vars[1]) * (&vars[0] - 1.0) + 5.0
                    } else {
                        (&vars[0] + 1.0) * (&vars[0] - 1.0) + 5.0
                    }
                }
            };
            
            // Convert to AST
            let original_ast = ctx.to_ast(&expr);
            
            // Perform round-trip: AST → MathLang → AST
            let round_trip_result = optimize_simple_sum_splitting(&original_ast);
            prop_assert!(round_trip_result.is_ok(), "Round-trip conversion should succeed");
            
            let optimized_ast = round_trip_result.unwrap();
            
            // Test semantic equivalence by evaluating both expressions with test values
            let test_values: Vec<f64> = (0..var_count).map(|i| (i as f64 + 1.0) * 0.5).collect();
            
            // Original expression evaluation
            let original_result = match var_count {
                1 => ctx.eval(&expr, hlist![test_values[0]]),
                2 => ctx.eval(&expr, hlist![test_values[0], test_values[1]]),
                3 => ctx.eval(&expr, hlist![test_values[0], test_values[1], test_values[2]]),
                _ => unreachable!()
            };
            
            // Optimized expression evaluation
            let optimized_result = optimized_ast.eval_with_vars(&test_values);
            
            // Check semantic equivalence (allow small numerical differences)
            let tolerance = 1e-10;
            prop_assert!(
                (original_result - optimized_result).abs() < tolerance,
                "Round-trip should preserve semantic equivalence: original={}, optimized={}, diff={}",
                original_result,
                optimized_result,
                (original_result - optimized_result).abs()
            );
            
            // Additional check: both should have same free variables
            let mut original_vars = HashSet::new();
            collect_free_variables(&original_ast, &mut original_vars);
            
            let mut optimized_vars = HashSet::new();
            collect_free_variables(&optimized_ast, &mut optimized_vars);
            
            prop_assert_eq!(
                original_vars, optimized_vars,
                "Round-trip should preserve free variables: original={:?}, optimized={:?}",
                original_vars, optimized_vars
            );
        }
    }

    /// Test round-trip with sum expressions specifically
    #[cfg(feature = "optimization")]
    #[test]
    fn test_sum_round_trip_semantic_equivalence() {
        let mut ctx = DynamicContext::new();
        let mu = ctx.var();
        let sigma = ctx.var();
        
        // Create a sum expression similar to what the probabilistic demo uses
        let data = vec![1.0, 2.0, 3.0];
        let sum_expr = ctx.sum(data.clone(), |x| {
            let centered = x - &mu;
            &centered / &sigma
        });
        
        let original_ast = ctx.to_ast(&sum_expr);
        
        // Perform round-trip
        let result = optimize_simple_sum_splitting(&original_ast);
        assert!(result.is_ok(), "Sum round-trip should succeed");
        
        let optimized_ast = result.unwrap();
        
        // Test semantic equivalence
        let test_mu = 1.0;
        let test_sigma = 2.0;
        
        let original_result = ctx.eval(&sum_expr, hlist![test_mu, test_sigma]);
        let optimized_result = optimized_ast.eval_with_vars(&[test_mu, test_sigma]);
        
        // Expected: (1-1)/2 + (2-1)/2 + (3-1)/2 = 0 + 0.5 + 1.0 = 1.5
        let expected = (1.0 - test_mu) / test_sigma 
                      + (2.0 - test_mu) / test_sigma 
                      + (3.0 - test_mu) / test_sigma;
        
        assert!((original_result - expected).abs() < 1e-10, 
               "Original should match expected: {} vs {}", original_result, expected);
        assert!((optimized_result - expected).abs() < 1e-10,
               "Optimized should match expected: {} vs {}", optimized_result, expected);
        assert!((original_result - optimized_result).abs() < 1e-10,
               "Round-trip should preserve sum semantics: {} vs {}", original_result, optimized_result);
        
        println!("✅ Sum round-trip test passed: {} ≈ {} ≈ {}", 
                 original_result, optimized_result, expected);
    }

    /// Test round-trip with collection variables specifically
    #[cfg(feature = "optimization")]
    #[test] 
    fn test_collection_identity_preservation() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        
        // Create multiple sum expressions with different data to test collection identity preservation
        let data1 = vec![1.0, 2.0];
        let data2 = vec![3.0, 4.0];
        
        let sum1 = ctx.sum(data1.clone(), |item| &x * item);
        let sum2 = ctx.sum(data2.clone(), |item| &x * item);
        
        // Create compound expression: sum1 + sum2
        let compound = &sum1 + &sum2;
        let compound_ast = ctx.to_ast(&compound);
        
        // Round-trip through egg optimizer
        let result = optimize_simple_sum_splitting(&compound_ast);
        assert!(result.is_ok(), "Collection identity round-trip should succeed");
        
        let optimized_ast = result.unwrap();
        
        // Test semantic equivalence
        let test_x = 2.0;
        let original_result = ctx.eval(&compound, hlist![test_x]);
        let optimized_result = optimized_ast.eval_with_vars(&[test_x]);
        
        // Expected: x*(1+2) + x*(3+4) = x*3 + x*7 = x*10 = 2*10 = 20
        let expected = test_x * (1.0 + 2.0) + test_x * (3.0 + 4.0);
        
        assert!((original_result - expected).abs() < 1e-10,
               "Original should match expected: {} vs {}", original_result, expected);
        assert!((optimized_result - expected).abs() < 1e-10,
               "Optimized should match expected: {} vs {}", optimized_result, expected);
        assert!((original_result - optimized_result).abs() < 1e-10,
               "Round-trip should preserve collection identity: {} vs {}", original_result, optimized_result);
        
        println!("✅ Collection identity test passed: {} ≈ {} ≈ {}", 
                 original_result, optimized_result, expected);
    }
} 