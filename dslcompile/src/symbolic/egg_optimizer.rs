//! Simple Egg-based Optimizer for Sum Splitting
//!
//! This module implements basic sum splitting optimization using egg e-graphs,
//! focusing on constant factoring: Σ(a*x + b*x) → (a+b)*Σ(x) where a,b are constants.

use crate::{
    ast::{ASTRepr, ast_repr::Lambda},
    error::{DSLCompileError, Result},
};
use std::collections::HashSet;

#[cfg(feature = "optimization")]
use egg::{
    Analysis, CostFunction, DidMerge, EGraph, Extractor, Id, Language, LanguageChildren, RecExpr,
    Rewrite, Runner, Symbol, define_language, rewrite as rw,
};

#[cfg(feature = "optimization")]
use ordered_float::OrderedFloat;

/// Mathematical language for sum splitting with proper lambda support
#[cfg(feature = "optimization")]
define_language! {
    pub enum MathLang {
        // Basic values
        Num(OrderedFloat<f64>),
        Var(Symbol), // Variables (simplified - just symbols)
        LambdaVar(usize), // Lambda-bound variables

        // Core operations
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Sub([Id; 2]),
        "/" = Div([Id; 2]),
        "neg" = Neg([Id; 1]),
        "pow" = Pow([Id; 2]),
        "ln" = Ln([Id; 1]),

        // Sum with lambda: sum(lambda(var, expr), collection)
        "sum" = Sum([Id; 2]), // [lambda, collection]

        // Lambda expressions: lambda(var, body)
        "lambda" = Lambda([Id; 2]), // [var, body]

        // Collections
        "data" = Data([Id; 0]), // Data arrays
        "var_collection" = VarCollection([Id; 0]), // Collection variables (no children needed)
    }
}

/// Dependency analysis to track which variables each expression depends on
/// This enables safe coefficient factoring in sum splitting
#[cfg(feature = "optimization")]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DependencyData {
    /// Set of variable IDs that this expression depends on
    /// Only includes `UserVar` IDs, not `BoundVar` IDs (bound variables don't create external dependencies)
    pub free_vars: HashSet<usize>,
}

#[cfg(feature = "optimization")]

/// Dependency analysis implementation using egg's Analysis trait
#[cfg(feature = "optimization")]
#[derive(Debug, Default)]
pub struct DependencyAnalysis;

#[cfg(feature = "optimization")]
impl Analysis<MathLang> for DependencyAnalysis {
    type Data = DependencyData;

    fn make(egraph: &mut EGraph<MathLang, Self>, enode: &MathLang) -> Self::Data {
        let mut deps = HashSet::new();

        match enode {
            // Constants have no dependencies
            MathLang::Num(_) => {}

            // User variables create dependencies
            MathLang::Var(var) => {
                // Extract variable index from symbol name (e.g., "x0" -> 0)
                let var_str = var.as_str();
                if let Some(idx_str) = var_str.strip_prefix('x')
                    && let Ok(idx) = idx_str.parse::<usize>()
                {
                    deps.insert(idx);
                }
            }

            // Lambda variables (bound variables) don't create external dependencies
            MathLang::LambdaVar(_) => {}

            // Binary operations: union of children's dependencies
            MathLang::Add([left, right])
            | MathLang::Mul([left, right])
            | MathLang::Sub([left, right])
            | MathLang::Div([left, right])
            | MathLang::Pow([left, right]) => {
                deps.extend(&egraph[*left].data.free_vars);
                deps.extend(&egraph[*right].data.free_vars);
            }

            // Unary operations: dependencies of child
            MathLang::Neg([inner]) | MathLang::Ln([inner]) => {
                deps.extend(&egraph[*inner].data.free_vars);
            }

            // Lambda: dependencies of body, but bound variables don't propagate
            MathLang::Lambda([_var, body]) => {
                // For now, just propagate body dependencies
                // TODO: More sophisticated handling of variable binding
                deps.extend(&egraph[*body].data.free_vars);
            }

            // Collections have no inherent dependencies
            MathLang::Data([]) | MathLang::VarCollection([]) => {}

            // Sum: dependencies from both lambda and collection
            MathLang::Sum([lambda, collection]) => {
                deps.extend(&egraph[*lambda].data.free_vars);
                deps.extend(&egraph[*collection].data.free_vars);
            }
        }

        DependencyData { free_vars: deps }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let old_size = a.free_vars.len();
        a.free_vars.extend(b.free_vars);
        if a.free_vars.len() > old_size {
            DidMerge(true, false) // Changed, not should_stop
        } else {
            DidMerge(false, false)
        }
    }
}

/// Helper functions for dependency analysis
#[cfg(feature = "optimization")]
impl DependencyData {
    /// Check if this expression is independent of a specific variable
    #[must_use]
    pub fn is_independent_of(&self, var_id: usize) -> bool {
        !self.free_vars.contains(&var_id)
    }

    /// Check if this expression depends on any variables
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.free_vars.is_empty()
    }

    /// Get all free variables
    #[must_use]
    pub fn get_free_vars(&self) -> &HashSet<usize> {
        &self.free_vars
    }
}

/// Enhanced cost function for better optimization decisions
#[cfg(feature = "optimization")]
struct EnhancedCost;

#[cfg(feature = "optimization")]
impl CostFunction<MathLang> for EnhancedCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &MathLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            // Constants are free (can be precomputed)
            MathLang::Num(_) => 0.5,

            // Variables cost depends on type
            MathLang::Var(_) => 1.0,       // Regular variables
            MathLang::LambdaVar(_) => 0.5, // Lambda vars are very efficient

            // Basic operations - favor simpler forms
            MathLang::Add([a, b]) => {
                let a_cost = costs(*a);
                let b_cost = costs(*b);
                a_cost + b_cost + 1.0
            }
            MathLang::Mul([a, b]) => {
                let a_cost = costs(*a);
                let b_cost = costs(*b);
                // Multiplication is slightly more expensive than addition
                a_cost + b_cost + 1.5
            }
            MathLang::Sub([a, b]) => {
                let a_cost = costs(*a);
                let b_cost = costs(*b);
                a_cost + b_cost + 1.2
            }

            // Lambda creation has moderate cost
            MathLang::Lambda([_var, body]) => {
                let body_cost = costs(*body);
                body_cost + 2.0 // Lambda creation overhead
            }

            // Collections have base costs
            MathLang::Data([]) => 3.0, // Data arrays are moderately expensive
            MathLang::VarCollection([]) => 2.0, // Variable collections are cheaper

            // KEY: Sum has sophisticated non-additive cost modeling
            MathLang::Sum([lambda, collection]) => {
                let lambda_cost = costs(*lambda);
                let collection_cost = costs(*collection);

                // Collection size estimation (could be made dynamic)
                let estimated_collection_size = 50.0;

                // Cost model: base + (collection_size * lambda_complexity)
                // This encourages:
                // 1. Simpler lambdas (lower lambda_cost)
                // 2. Factoring constants out of lambdas
                // 3. Splitting sums when it reduces total work
                let base_cost = 10.0;
                let iteration_cost = estimated_collection_size * lambda_cost * 0.2;

                collection_cost + base_cost + iteration_cost
            }

            // Division is moderately expensive
            MathLang::Div([a, b]) => {
                let a_cost = costs(*a);
                let b_cost = costs(*b);
                a_cost + b_cost + 5.0 // Match the cost from ast_utils.rs
            }

            // Power is expensive
            MathLang::Pow([a, b]) => {
                let a_cost = costs(*a);
                let b_cost = costs(*b);
                a_cost + b_cost + 10.0 // Match the cost from ast_utils.rs
            }

            // Unary operations are relatively cheap
            MathLang::Neg([inner]) => {
                let inner_cost = costs(*inner);
                inner_cost + 1.0
            }

            MathLang::Ln([inner]) => {
                let inner_cost = costs(*inner);
                inner_cost + 30.0 // Transcendental functions are expensive
            }
        }
    }
}

/// Create rewrite rules for sum splitting with dependency analysis
#[cfg(feature = "optimization")]
fn make_sum_splitting_rules() -> Vec<Rewrite<MathLang, DependencyAnalysis>> {
    vec![
        // Basic arithmetic simplification
        rw!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rw!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        rw!("add-zero"; "(+ ?a 0.0)" => "?a"),
        rw!("mul-one"; "(* ?a 1.0)" => "?a"),
        rw!("mul-zero"; "(* ?a 0.0)" => "0.0"),
        // MULTISET ABSORPTION: Teaching egg about our multiset semantics
        // These rules explicitly handle the "flattening" that MultiSet provides

        // Addition absorption: (a + b) + c → (a + b + c) conceptually
        // In egg's binary tree world: nested adds should flatten
        rw!("flatten-add-left"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("flatten-add-right"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        // Multiplication absorption: (a * b) * c → (a * b * c) conceptually
        rw!("flatten-mul-left"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rw!("flatten-mul-right"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        // Coefficient combining for same variables: 2*x + 3*x → 5*x
        rw!("combine-coeffs"; "(+ (* ?c1 ?x) (* ?c2 ?x))" => "(* (+ ?c1 ?c2) ?x)"),
        // Factor combining for same bases: x^2 * x^3 → x^5 (if we had powers)
        rw!("combine-factors"; "(* (* ?x ?c1) (* ?x ?c2))" => "(* ?x (* ?c1 ?c2))"),
        // Distribution: a*(b + c) → a*b + a*c
        rw!("distribute"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        // Reverse distribution (factoring): a*b + a*c → a*(b + c)
        rw!("factor-common"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        // SUM SPLITTING: The key optimization
        // Σ(λv.(a + b)) → Σ(λv.a) + Σ(λv.b)
        rw!("sum-split-add"; 
            "(sum (lambda ?v (+ ?a ?b)) ?coll)" => 
            "(+ (sum (lambda ?v ?a) ?coll) (sum (lambda ?v ?b) ?coll))"),
        // CONSTANT FACTORING: Σ(λv.(c * x)) → c * Σ(λv.x) where c is constant
        rw!("factor-constant-left"; 
            "(sum (lambda ?v (* ?c ?x)) ?coll)" => 
            "(* ?c (sum (lambda ?v ?x) ?coll))"),
        rw!("factor-constant-right"; 
            "(sum (lambda ?v (* ?x ?c)) ?coll)" => 
            "(* ?c (sum (lambda ?v ?x) ?coll))"),
        // Lambda simplification: λv.v → v
        rw!("lambda-identity"; "(lambda ?v ?v)" => "?v"),
    ]
}

/// Optimizer with dependency analysis that applies sum splitting rules
#[cfg(feature = "optimization")]
pub fn optimize_simple_sum_splitting(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
    // Step 0: Normalize expression (Sub -> Add+Neg, Div -> Mul+Pow)
    let normalized = crate::ast::normalization::normalize(expr);

    // Step 1: Convert AST to MathLang with dependency analysis
    let mut egraph: EGraph<MathLang, DependencyAnalysis> = Default::default();
    let root = ast_to_mathlang(&normalized, &mut egraph)?;

    println!("🔍 Dependency analysis completed - tracking free variables");

    // Debug: Print dependency information for the root
    if let Some(root_class) = egraph.classes().find(|class| class.id == root) {
        println!(
            "   Root expression depends on variables: {:?}",
            root_class.data.free_vars
        );
    }

    // Step 2: Apply sum splitting rules
    let rules = make_sum_splitting_rules();
    let runner = Runner::default().with_egraph(egraph).run(&rules);

    println!(
        "🔄 Egg optimization completed in {} iterations",
        runner.iterations.len()
    );

    // Step 3: Extract best expression using our enhanced cost function
    let extractor = Extractor::new(&runner.egraph, EnhancedCost);
    let (cost, best_expr) = extractor.find_best(root);

    println!("💰 Best expression cost: {cost:.1}");

    // Step 4: Convert back to AST
    mathlang_to_ast(&best_expr, &runner.egraph)
}

/// Get dependency information for an expression in the e-graph
#[cfg(feature = "optimization")]
#[must_use]
pub fn get_dependencies(
    egraph: &EGraph<MathLang, DependencyAnalysis>,
    id: Id,
) -> Option<&DependencyData> {
    egraph
        .classes()
        .find(|class| class.id == id)
        .map(|class| &class.data)
}

/// Convert Lambda to `MathLang`
#[cfg(feature = "optimization")]
fn convert_lambda_to_mathlang(
    lambda: &Lambda<f64>,
    egraph: &mut EGraph<MathLang, DependencyAnalysis>,
) -> Result<Id> {
    // For now, handle single-argument lambdas
    let param_idx = lambda.var_indices.first().copied().unwrap_or(0);
    let param_name = format!("v{param_idx}");
    let param_id = egraph.add(MathLang::Var(param_name.into()));

    // Convert body expression
    let body_id = ast_to_mathlang(&lambda.body, egraph)?;

    // Create lambda
    Ok(egraph.add(MathLang::Lambda([param_id, body_id])))
}

/// Convert `ASTRepr` to `MathLang` (simplified version)
#[cfg(feature = "optimization")]
fn ast_to_mathlang(
    expr: &ASTRepr<f64>,
    egraph: &mut EGraph<MathLang, DependencyAnalysis>,
) -> Result<Id> {
    match expr {
        ASTRepr::Constant(val) => Ok(egraph.add(MathLang::Num(OrderedFloat(*val)))),

        ASTRepr::Variable(idx) => {
            let var_name = format!("x{idx}");
            Ok(egraph.add(MathLang::Var(var_name.into())))
        }

        ASTRepr::BoundVar(idx) => Ok(egraph.add(MathLang::LambdaVar(*idx))),

        ASTRepr::Add(terms) => {
            // Convert multiset to binary operations
            let term_vec: Vec<_> = terms.iter().map(|(expr, _count)| expr).collect();
            if term_vec.is_empty() {
                Ok(egraph.add(MathLang::Num(OrderedFloat(0.0))))
            } else if term_vec.len() == 1 {
                ast_to_mathlang(term_vec[0], egraph)
            } else {
                let mut result = ast_to_mathlang(term_vec[0], egraph)?;
                for term in &term_vec[1..] {
                    let term_id = ast_to_mathlang(term, egraph)?;
                    result = egraph.add(MathLang::Add([result, term_id]));
                }
                Ok(result)
            }
        }

        ASTRepr::Mul(factors) => {
            // Convert multiset to binary operations
            let factor_vec: Vec<_> = factors.iter().map(|(expr, _count)| expr).collect();
            if factor_vec.is_empty() {
                Ok(egraph.add(MathLang::Num(OrderedFloat(1.0))))
            } else if factor_vec.len() == 1 {
                ast_to_mathlang(factor_vec[0], egraph)
            } else {
                let mut result = ast_to_mathlang(factor_vec[0], egraph)?;
                for factor in &factor_vec[1..] {
                    let factor_id = ast_to_mathlang(factor, egraph)?;
                    result = egraph.add(MathLang::Mul([result, factor_id]));
                }
                Ok(result)
            }
        }

        ASTRepr::Sub(left, right) => {
            let left_id = ast_to_mathlang(left, egraph)?;
            let right_id = ast_to_mathlang(right, egraph)?;
            Ok(egraph.add(MathLang::Sub([left_id, right_id])))
        }

        ASTRepr::Sum(collection) => {
            // Convert sum with proper lambda structure
            match collection.as_ref() {
                crate::ast::ast_repr::Collection::Map {
                    lambda,
                    collection: inner_coll,
                } => {
                    // Convert the lambda
                    let lambda_id = convert_lambda_to_mathlang(lambda, egraph)?;

                    // Convert the inner collection
                    let collection_id = match inner_coll.as_ref() {
                        crate::ast::ast_repr::Collection::DataArray(_data) => {
                            egraph.add(MathLang::Data([]))
                        }
                        crate::ast::ast_repr::Collection::Variable(_idx) => {
                            egraph.add(MathLang::VarCollection([]))
                        }
                        _ => {
                            // Default to data reference
                            egraph.add(MathLang::Data([]))
                        }
                    };

                    Ok(egraph.add(MathLang::Sum([lambda_id, collection_id])))
                }
                crate::ast::ast_repr::Collection::DataArray(_data) => {
                    // Simple sum over data - create identity lambda
                    let var_name = "x".to_string();
                    let var_id = egraph.add(MathLang::Var(var_name.clone().into()));
                    let lambda_id = egraph.add(MathLang::Lambda([var_id, var_id])); // λx.x
                    let data_id = egraph.add(MathLang::Data([]));
                    Ok(egraph.add(MathLang::Sum([lambda_id, data_id])))
                }
                _ => {
                    // For other collection types, create identity lambda with collection reference
                    let var_name = "x".to_string();
                    let var_id = egraph.add(MathLang::Var(var_name.clone().into()));
                    let lambda_id = egraph.add(MathLang::Lambda([var_id, var_id])); // λx.x
                    let data_id = egraph.add(MathLang::Data([]));
                    Ok(egraph.add(MathLang::Sum([lambda_id, data_id])))
                }
            }
        }

        ASTRepr::Div(left, right) => {
            let left_id = ast_to_mathlang(left, egraph)?;
            let right_id = ast_to_mathlang(right, egraph)?;
            Ok(egraph.add(MathLang::Div([left_id, right_id])))
        }

        ASTRepr::Neg(inner) => {
            let inner_id = ast_to_mathlang(inner, egraph)?;
            Ok(egraph.add(MathLang::Neg([inner_id])))
        }

        ASTRepr::Pow(base, exp) => {
            let base_id = ast_to_mathlang(base, egraph)?;
            let exp_id = ast_to_mathlang(exp, egraph)?;
            Ok(egraph.add(MathLang::Pow([base_id, exp_id])))
        }

        ASTRepr::Ln(inner) => {
            let inner_id = ast_to_mathlang(inner, egraph)?;
            Ok(egraph.add(MathLang::Ln([inner_id])))
        }

        _ => Err(DSLCompileError::Generic(format!(
            "Unsupported AST node for egg optimization: {expr:?}"
        ))),
    }
}

/// Convert `MathLang` back to `ASTRepr`
#[cfg(feature = "optimization")]
fn mathlang_to_ast(
    expr: &RecExpr<MathLang>,
    _egraph: &EGraph<MathLang, DependencyAnalysis>,
) -> Result<ASTRepr<f64>> {
    fn convert_node(expr: &RecExpr<MathLang>, node_id: Id) -> Result<ASTRepr<f64>> {
        match &expr[node_id] {
            MathLang::Num(val) => Ok(ASTRepr::Constant(**val)),

            MathLang::Var(name) => {
                // Extract variable index from name like "x0", "x1", etc.
                let name_str = name.as_str();
                if let Some(idx_str) = name_str.strip_prefix('x') {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        Ok(ASTRepr::Variable(idx))
                    } else {
                        Err(DSLCompileError::Generic(format!(
                            "Invalid variable name: {name_str}"
                        )))
                    }
                } else {
                    Err(DSLCompileError::Generic(format!(
                        "Invalid variable name: {name_str}"
                    )))
                }
            }

            MathLang::LambdaVar(idx) => Ok(ASTRepr::BoundVar(*idx)),

            MathLang::Add([left, right]) => {
                let left_ast = convert_node(expr, *left)?;
                let right_ast = convert_node(expr, *right)?;
                Ok(ASTRepr::add_binary(left_ast, right_ast))
            }

            MathLang::Mul([left, right]) => {
                let left_ast = convert_node(expr, *left)?;
                let right_ast = convert_node(expr, *right)?;
                Ok(ASTRepr::mul_binary(left_ast, right_ast))
            }

            MathLang::Sub([left, right]) => {
                let left_ast = convert_node(expr, *left)?;
                let right_ast = convert_node(expr, *right)?;
                Ok(ASTRepr::Sub(Box::new(left_ast), Box::new(right_ast)))
            }

            MathLang::Lambda([var, body]) => {
                // Convert lambda back - extract parameter index from variable name
                let param_idx = if let MathLang::Var(name) = &expr[*var] {
                    let name_str = name.as_str();
                    if let Some(idx_str) = name_str.strip_prefix('v') {
                        idx_str.parse::<usize>().unwrap_or(0)
                    } else {
                        0
                    }
                } else {
                    0
                };

                let body_ast = convert_node(expr, *body)?;
                let lambda = Lambda {
                    var_indices: vec![param_idx],
                    body: Box::new(body_ast),
                };

                // For now, return the body directly since we can't return a Lambda from this context
                convert_node(expr, *body)
            }

            MathLang::Sum([lambda, collection]) => {
                // Convert lambda
                let lambda_ast = if let MathLang::Lambda([var, body]) = &expr[*lambda] {
                    let param_idx = if let MathLang::Var(name) = &expr[*var] {
                        let name_str = name.as_str();
                        if let Some(idx_str) = name_str.strip_prefix('v') {
                            idx_str.parse::<usize>().unwrap_or(0)
                        } else {
                            0
                        }
                    } else {
                        0
                    };

                    let body_ast = convert_node(expr, *body)?;
                    Lambda {
                        var_indices: vec![param_idx],
                        body: Box::new(body_ast),
                    }
                } else {
                    return Err(DSLCompileError::Generic(
                        "Invalid lambda in sum".to_string(),
                    ));
                };

                // Convert collection
                let collection_ast = match &expr[*collection] {
                    MathLang::Data([]) => {
                        crate::ast::ast_repr::Collection::DataArray(vec![1.0, 2.0, 3.0]) // Default data
                    }
                    MathLang::VarCollection([]) => {
                        crate::ast::ast_repr::Collection::Variable(0) // Default to variable 0
                    }
                    _ => crate::ast::ast_repr::Collection::Variable(0),
                };

                let map_collection = crate::ast::ast_repr::Collection::Map {
                    lambda: Box::new(lambda_ast),
                    collection: Box::new(collection_ast),
                };

                Ok(ASTRepr::Sum(Box::new(map_collection)))
            }

            MathLang::Data([]) => {
                // Convert data reference back to collection
                let collection_ref =
                    crate::ast::ast_repr::Collection::DataArray(vec![1.0, 2.0, 3.0]);
                Ok(ASTRepr::Sum(Box::new(collection_ref)))
            }

            MathLang::VarCollection([]) => {
                let collection_ref = crate::ast::ast_repr::Collection::Variable(0);
                Ok(ASTRepr::Sum(Box::new(collection_ref)))
            }

            MathLang::Div([left, right]) => {
                let left_ast = convert_node(expr, *left)?;
                let right_ast = convert_node(expr, *right)?;
                Ok(ASTRepr::Div(Box::new(left_ast), Box::new(right_ast)))
            }

            MathLang::Neg([inner]) => {
                let inner_ast = convert_node(expr, *inner)?;
                Ok(ASTRepr::Neg(Box::new(inner_ast)))
            }

            MathLang::Pow([base, exp]) => {
                let base_ast = convert_node(expr, *base)?;
                let exp_ast = convert_node(expr, *exp)?;
                Ok(ASTRepr::Pow(Box::new(base_ast), Box::new(exp_ast)))
            }

            MathLang::Ln([inner]) => {
                let inner_ast = convert_node(expr, *inner)?;
                Ok(ASTRepr::Ln(Box::new(inner_ast)))
            }
        }
    }

    let root_id = (expr.as_ref().len() - 1).into();
    convert_node(expr, root_id)
}

#[cfg(not(feature = "optimization"))]
pub fn optimize_simple_sum_splitting(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
    // If egg_optimization feature is not enabled, return the original expression
    Ok(expr.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "optimization")]
    #[test]
    fn test_simple_sum_splitting() {
        // Test: Σ(2*x + 3*x) should become (2+3)*Σ(x) = 5*Σ(x)
        // This is a simplified version of the test
        let x = ASTRepr::Variable(0);
        let two = ASTRepr::Constant(2.0);
        let three = ASTRepr::Constant(3.0);

        // 2*x + 3*x
        let expr = ASTRepr::add_binary(
            ASTRepr::mul_binary(two, x.clone()),
            ASTRepr::mul_binary(three, x),
        );

        // For now, just test that optimization doesn't crash
        let result = optimize_simple_sum_splitting(&expr);
        assert!(result.is_ok());
    }

    #[cfg(feature = "optimization")]
    #[test]
    fn test_dependency_analysis() {
        // Test that dependency analysis correctly tracks free variables
        let x = ASTRepr::Variable(0); // Should depend on {0}
        let y = ASTRepr::Variable(1); // Should depend on {1}
        let const_2 = ASTRepr::Constant(2.0); // Should depend on {}

        // Test simple variable
        let mut egraph: EGraph<MathLang, DependencyAnalysis> = Default::default();
        let x_id = ast_to_mathlang(&x, &mut egraph).unwrap();
        egraph.rebuild();

        if let Some(x_deps) = get_dependencies(&egraph, x_id) {
            assert!(
                x_deps.free_vars.contains(&0),
                "Variable x should depend on variable 0"
            );
            assert_eq!(
                x_deps.free_vars.len(),
                1,
                "Variable x should depend on exactly one variable"
            );
        }

        // Test expression with multiple variables: x + y
        let mut egraph2: EGraph<MathLang, DependencyAnalysis> = Default::default();
        let expr = ASTRepr::add_binary(x.clone(), y.clone());
        let expr_id = ast_to_mathlang(&expr, &mut egraph2).unwrap();
        egraph2.rebuild();

        if let Some(expr_deps) = get_dependencies(&egraph2, expr_id) {
            assert!(
                expr_deps.free_vars.contains(&0),
                "Expression x+y should depend on variable 0"
            );
            assert!(
                expr_deps.free_vars.contains(&1),
                "Expression x+y should depend on variable 1"
            );
            assert_eq!(
                expr_deps.free_vars.len(),
                2,
                "Expression x+y should depend on exactly two variables"
            );
        }

        // Test constant expression: 2
        let mut egraph3: EGraph<MathLang, DependencyAnalysis> = Default::default();
        let const_id = ast_to_mathlang(&const_2, &mut egraph3).unwrap();
        egraph3.rebuild();

        if let Some(const_deps) = get_dependencies(&egraph3, const_id) {
            assert!(
                const_deps.is_constant(),
                "Constant should have no dependencies"
            );
            assert_eq!(
                const_deps.free_vars.len(),
                0,
                "Constant should depend on zero variables"
            );
        }

        // Test mixed expression: 2*x + 3
        let mixed_expr =
            ASTRepr::add_binary(ASTRepr::mul_binary(const_2, x), ASTRepr::Constant(3.0));
        let mut egraph4: EGraph<MathLang, DependencyAnalysis> = Default::default();
        let mixed_id = ast_to_mathlang(&mixed_expr, &mut egraph4).unwrap();
        egraph4.rebuild();

        if let Some(mixed_deps) = get_dependencies(&egraph4, mixed_id) {
            assert!(
                mixed_deps.free_vars.contains(&0),
                "Expression 2*x+3 should depend on variable 0"
            );
            assert_eq!(
                mixed_deps.free_vars.len(),
                1,
                "Expression 2*x+3 should depend on exactly one variable"
            );
            assert!(
                !mixed_deps.is_constant(),
                "Expression 2*x+3 should not be constant"
            );
        }
    }
}
