//! Egg-based E-Graph Optimizer with Non-Additive Cost Functions
//!
//! This module implements direct e-graph manipulation using the egg crate,
//! providing sophisticated cost modeling for mathematical expressions with
//! particular focus on summation operations that require non-additive costs.

use crate::{
    ast::ASTRepr,
    error::{DSLCompileError, Result},
};
use std::collections::{BTreeSet, HashMap};

#[cfg(feature = "egg_optimization")]
use egg::{*, rewrite as rw};

#[cfg(feature = "egg_optimization")]
use ordered_float::OrderedFloat;

/// Mathematical language definition for egg e-graph
#[cfg(feature = "egg_optimization")]
define_language! {
    /// Mathematical expression language supporting all DSLCompile operations
    pub enum MathLang {
        // Basic values - using Symbol for identifiers instead of raw integers
        Num(OrderedFloat<f64>),
        UserVar(Symbol),
        BoundVar(Symbol),
        
        // Binary operations
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Sub([Id; 2]),
        "/" = Div([Id; 2]),
        "^" = Pow([Id; 2]),
        
        // Unary operations
        "neg" = Neg([Id; 1]),
        "ln" = Ln([Id; 1]),
        "exp" = Exp([Id; 1]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        
        // Collection operations (simplified for egg)
        "sum" = Sum([Id; 1]),
        "range" = Range([Id; 2]),
        "singleton" = Singleton([Id; 1]),
        
        // Let bindings for CSE
        "let" = Let([Id; 3]), // (let var_id expr body)
        
        // Placeholder for complex collections that need runtime evaluation
        "collection_ref" = CollectionRef(Symbol),
    }
}

/// Variable dependency analysis for mathematical expressions
#[cfg(feature = "egg_optimization")]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct DependencyAnalysis;

#[cfg(feature = "egg_optimization")]
impl Analysis<MathLang> for DependencyAnalysis {
    type Data = BTreeSet<usize>;
    
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let old_len = to.len();
        to.extend(from);
        DidMerge(to.len() > old_len, false)
    }
    
    fn make(egraph: &EGraph<MathLang, Self>, enode: &MathLang) -> Self::Data {
        let mut deps = BTreeSet::new();
        
        match enode {
            MathLang::UserVar(var_symbol) => {
                // Extract variable index from symbol
                if let Ok(var_id) = var_symbol.as_str().parse::<usize>() {
                    deps.insert(var_id);
                }
            }
            MathLang::Add([a, b]) | MathLang::Mul([a, b]) | 
            MathLang::Sub([a, b]) | MathLang::Div([a, b]) | 
            MathLang::Pow([a, b]) | MathLang::Range([a, b]) => {
                deps.extend(&egraph[*a].data);
                deps.extend(&egraph[*b].data);
            }
            MathLang::Neg([a]) | MathLang::Ln([a]) | MathLang::Exp([a]) | 
            MathLang::Sin([a]) | MathLang::Cos([a]) | MathLang::Sqrt([a]) | 
            MathLang::Sum([a]) | MathLang::Singleton([a]) => {
                deps.extend(&egraph[*a].data);
            }
            MathLang::Let([_var, expr, body]) => {
                // For Let expressions, include dependencies from both expr and body
                // Note: In a full implementation, we'd need to handle variable scoping
                deps.extend(&egraph[*expr].data);
                deps.extend(&egraph[*body].data);
            }
            MathLang::CollectionRef(_) => {
                // Collection references have no variable dependencies
                // (they're resolved at runtime)
            }
            _ => {} // Constants and bound variables have no free dependencies
        }
        
        deps
    }
}

/// Domain analysis information for expressions
#[cfg(feature = "egg_optimization")]
#[derive(Debug, Clone, PartialEq)]
pub enum DomainInfo {
    Unknown,
    Positive,
    NonNegative,
    NonZero,
    Constant(f64),
}

/// Coupling pattern analysis for summation optimization
#[cfg(feature = "egg_optimization")]
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingPattern {
    Independent,   // No shared variables
    Simple,       // Simple variable sharing
    Complex,      // Complex interdependencies
}

/// Non-additive cost function for mathematical expressions with summation awareness
#[cfg(feature = "egg_optimization")]
pub struct SummationCostFunction<'a> {
    egraph: &'a EGraph<MathLang, DependencyAnalysis>,
    collection_size_estimates: HashMap<Id, usize>,
    operation_costs: HashMap<&'static str, f64>,
    default_collection_size: usize,
}

#[cfg(feature = "egg_optimization")]
impl<'a> SummationCostFunction<'a> {
    pub fn new(egraph: &'a EGraph<MathLang, DependencyAnalysis>) -> Self {
        let mut operation_costs = HashMap::new();
        
        // Base operation costs
        operation_costs.insert("add", 1.0);
        operation_costs.insert("mul", 2.0);
        operation_costs.insert("sub", 1.0);
        operation_costs.insert("div", 5.0);
        operation_costs.insert("pow", 10.0);
        operation_costs.insert("neg", 0.5);
        operation_costs.insert("ln", 15.0);
        operation_costs.insert("exp", 15.0);
        operation_costs.insert("sin", 12.0);
        operation_costs.insert("cos", 12.0);
        operation_costs.insert("sqrt", 8.0);
        operation_costs.insert("sum", 1000.0); // Base summation setup cost
        
        Self {
            egraph,
            collection_size_estimates: HashMap::new(),
            operation_costs,
            default_collection_size: 100, // Conservative default
        }
    }
    
    /// Estimate collection size for cost calculation
    fn estimate_collection_size(&mut self, collection_id: Id) -> usize {
        if let Some(&cached_size) = self.collection_size_estimates.get(&collection_id) {
            return cached_size;
        }
        
        // Look at the e-class to find concrete representations
        let eclass = &self.egraph[collection_id];
        for node in &eclass.nodes {
            match node {
                MathLang::Range([start, end]) => {
                    // Try to evaluate range size if both endpoints are constants
                    if let (Some(start_val), Some(end_val)) = (
                        self.try_evaluate_constant(*start),
                        self.try_evaluate_constant(*end)
                    ) {
                        let size = (end_val - start_val + 1.0).max(0.0) as usize;
                        let capped_size = size.min(10000); // Cap at reasonable maximum
                        self.collection_size_estimates.insert(collection_id, capped_size);
                        return capped_size;
                    }
                }
                MathLang::Singleton(_) => {
                    self.collection_size_estimates.insert(collection_id, 1);
                    return 1;
                }
                MathLang::CollectionRef(_) => {
                    // Use default for runtime collections
                    self.collection_size_estimates.insert(collection_id, self.default_collection_size);
                    return self.default_collection_size;
                }
                _ => {}
            }
        }
        
        // Default estimate for unknown collections
        self.collection_size_estimates.insert(collection_id, self.default_collection_size);
        self.default_collection_size
    }
    
    /// Try to evaluate a constant value from an e-class
    fn try_evaluate_constant(&self, id: Id) -> Option<f64> {
        let eclass = &self.egraph[id];
        for node in &eclass.nodes {
            if let MathLang::Num(val) = node {
                return Some(val.into_inner());
            }
        }
        None
    }
    
    /// Analyze coupling patterns for summation cost modeling
    fn analyze_coupling_pattern(&self, sum_id: Id) -> CouplingPattern {
        let dependencies = &self.egraph[sum_id].data;
        
        // Analyze coupling based on dependency complexity
        match dependencies.len() {
            0 => CouplingPattern::Independent,
            1 => CouplingPattern::Simple,
            2 => CouplingPattern::Simple,
            _ => CouplingPattern::Complex,
        }
    }
    
    /// Check if an expression is provably positive
    fn is_provably_positive(&self, id: Id) -> bool {
        let eclass = &self.egraph[id];
        for node in &eclass.nodes {
            match node {
                MathLang::Num(val) => return val.into_inner() > 0.0,
                MathLang::Exp(_) => return true, // exp(x) > 0 always
                _ => {}
            }
        }
        false
    }
    
    /// Get domain information for an expression
    fn get_domain_info(&self, id: Id) -> DomainInfo {
        let eclass = &self.egraph[id];
        for node in &eclass.nodes {
            match node {
                MathLang::Num(val) => {
                    let v = val.into_inner();
                    return DomainInfo::Constant(v);
                }
                MathLang::Exp(_) => return DomainInfo::Positive,
                MathLang::Sqrt(_) => return DomainInfo::NonNegative,
                _ => {}
            }
        }
        DomainInfo::Unknown
    }
}

#[cfg(feature = "egg_optimization")]
impl<'a> CostFunction<MathLang> for SummationCostFunction<'a> {
    type Cost = f64;
    
    fn cost<C>(&mut self, enode: &MathLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            // Constants and variables have minimal cost
            MathLang::Num(_) => 0.1,
            MathLang::UserVar(_) => 0.5,
            MathLang::BoundVar(_) => 0.3,
            MathLang::CollectionRef(_) => 1.0,
            
            // Binary operations: sum child costs + operation cost
            MathLang::Add([a, b]) => {
                costs(*a) + costs(*b) + self.operation_costs["add"]
            }
            MathLang::Mul([a, b]) => {
                costs(*a) + costs(*b) + self.operation_costs["mul"]
            }
            MathLang::Sub([a, b]) => {
                costs(*a) + costs(*b) + self.operation_costs["sub"]
            }
            MathLang::Div([a, b]) => {
                costs(*a) + costs(*b) + self.operation_costs["div"]
            }
            MathLang::Pow([a, b]) => {
                let base_cost = costs(*a);
                let exp_cost = costs(*b);
                
                // Power operations can be expensive, especially with variable exponents
                let power_multiplier = if self.try_evaluate_constant(*b).is_some() {
                    1.0 // Constant exponent is cheaper
                } else {
                    3.0 // Variable exponent is more expensive
                };
                
                base_cost + exp_cost + self.operation_costs["pow"] * power_multiplier
            }
            
            // Unary operations
            MathLang::Neg([a]) => costs(*a) + self.operation_costs["neg"],
            MathLang::Ln([a]) => {
                let inner_cost = costs(*a);
                let domain_penalty = if self.is_provably_positive(*a) {
                    0.0 // Safe ln operation
                } else {
                    100.0 // Potentially unsafe ln operation
                };
                inner_cost + self.operation_costs["ln"] + domain_penalty
            }
            MathLang::Exp([a]) => costs(*a) + self.operation_costs["exp"],
            MathLang::Sin([a]) => costs(*a) + self.operation_costs["sin"],
            MathLang::Cos([a]) => costs(*a) + self.operation_costs["cos"],
            MathLang::Sqrt([a]) => {
                let inner_cost = costs(*a);
                let domain_penalty = match self.get_domain_info(*a) {
                    DomainInfo::NonNegative | DomainInfo::Positive => 0.0,
                    DomainInfo::Constant(v) if v >= 0.0 => 0.0,
                    _ => 50.0, // Potentially unsafe sqrt operation
                };
                inner_cost + self.operation_costs["sqrt"] + domain_penalty
            }
            
            // Collection operations
            MathLang::Range([start, end]) => {
                costs(*start) + costs(*end) + 5.0
            }
            MathLang::Singleton([a]) => {
                costs(*a) + 2.0
            }
            
            // Let bindings (CSE)
            MathLang::Let([_var, expr, body]) => {
                let expr_cost = costs(*expr);
                let body_cost = costs(*body);
                
                // Let bindings can reduce cost by enabling reuse
                // Give a slight bonus for CSE opportunities
                expr_cost + body_cost - 5.0
            }
            
            // SUMMATION: The key non-additive cost case
            MathLang::Sum([collection]) => {
                let inner_cost = costs(*collection);
                let collection_size = self.estimate_collection_size(*collection);
                let coupling_pattern = self.analyze_coupling_pattern(*collection);
                
                // Coupling multiplier based on variable interdependencies
                let coupling_multiplier = match coupling_pattern {
                    CouplingPattern::Independent => 1.0,
                    CouplingPattern::Simple => 1.5,
                    CouplingPattern::Complex => 3.0,
                };
                
                // NON-ADDITIVE COST CALCULATION:
                // Total cost = base_summation_cost + (collection_size * inner_complexity * coupling)
                let base_cost = self.operation_costs["sum"];
                let iteration_cost = (collection_size as f64) * inner_cost * coupling_multiplier;
                
                println!(
                    "💰 Summation cost: base={:.1}, size={}, inner={:.1}, coupling={:.1}x → total={:.1}",
                    base_cost, collection_size, inner_cost, coupling_multiplier, 
                    base_cost + iteration_cost
                );
                
                base_cost + iteration_cost
            }
        }
    }
}

/// Egg-based optimizer with non-additive cost functions
#[cfg(feature = "egg_optimization")]
pub struct EggOptimizer {
    rules: Vec<Rewrite<MathLang, DependencyAnalysis>>,
}

#[cfg(feature = "egg_optimization")]
impl EggOptimizer {
    pub fn new() -> Self {
        Self {
            rules: make_mathematical_rules(),
        }
    }
    
    /// Optimize an expression using egg with custom cost functions
    pub fn optimize(&self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // Convert AST to egg expression
        let mut egraph: EGraph<MathLang, DependencyAnalysis> = Default::default();
        let root_id = self.ast_to_egg(&mut egraph, expr)?;
        
        // Run optimization with rewrite rules
        let runner = Runner::default()
            .with_egraph(egraph)
            .run(&self.rules);
        
        println!("🔬 Egg optimization completed in {} iterations", runner.iterations.len());
        
        // Extract best expression using custom summation cost function
        let cost_fn = SummationCostFunction::new(&runner.egraph);
        let extractor = Extractor::new(&runner.egraph, cost_fn);
        let (best_cost, best_expr) = extractor.find_best(root_id);
        
        println!("💰 Best expression cost: {:.2}", best_cost);
        
        // Convert back to AST
        self.egg_to_ast(&best_expr)
    }
    
    /// Convert DSLCompile AST to egg e-graph
    fn ast_to_egg(&self, egraph: &mut EGraph<MathLang, DependencyAnalysis>, expr: &ASTRepr<f64>) -> Result<Id> {
        match expr {
            ASTRepr::Constant(val) => {
                Ok(egraph.add(MathLang::Num(OrderedFloat(*val))))
            }
            ASTRepr::Variable(idx) => {
                let var_symbol = format!("v{}", idx).into();
                Ok(egraph.add(MathLang::UserVar(var_symbol)))
            }
            ASTRepr::BoundVar(idx) => {
                let bound_symbol = format!("b{}", idx).into();
                Ok(egraph.add(MathLang::BoundVar(bound_symbol)))
            }
            ASTRepr::Add(terms) => {
                // Convert multiset to binary operations
                let term_ids: Result<Vec<_>> = terms.elements()
                    .map(|term| self.ast_to_egg(egraph, term))
                    .collect();
                let term_ids = term_ids?;
                
                if term_ids.is_empty() {
                    Ok(egraph.add(MathLang::Num(OrderedFloat(0.0))))
                } else if term_ids.len() == 1 {
                    Ok(term_ids[0])
                } else {
                    // Chain binary additions: ((a + b) + c) + d
                    let mut result = term_ids[0];
                    for &term_id in &term_ids[1..] {
                        result = egraph.add(MathLang::Add([result, term_id]));
                    }
                    Ok(result)
                }
            }
            ASTRepr::Mul(factors) => {
                // Convert multiset to binary operations
                let factor_ids: Result<Vec<_>> = factors.elements()
                    .map(|factor| self.ast_to_egg(egraph, factor))
                    .collect();
                let factor_ids = factor_ids?;
                
                if factor_ids.is_empty() {
                    Ok(egraph.add(MathLang::Num(OrderedFloat(1.0))))
                } else if factor_ids.len() == 1 {
                    Ok(factor_ids[0])
                } else {
                    // Chain binary multiplications: ((a * b) * c) * d
                    let mut result = factor_ids[0];
                    for &factor_id in &factor_ids[1..] {
                        result = egraph.add(MathLang::Mul([result, factor_id]));
                    }
                    Ok(result)
                }
            }
            ASTRepr::Sub(left, right) => {
                let left_id = self.ast_to_egg(egraph, left)?;
                let right_id = self.ast_to_egg(egraph, right)?;
                Ok(egraph.add(MathLang::Sub([left_id, right_id])))
            }
            ASTRepr::Div(left, right) => {
                let left_id = self.ast_to_egg(egraph, left)?;
                let right_id = self.ast_to_egg(egraph, right)?;
                Ok(egraph.add(MathLang::Div([left_id, right_id])))
            }
            ASTRepr::Pow(base, exp) => {
                let base_id = self.ast_to_egg(egraph, base)?;
                let exp_id = self.ast_to_egg(egraph, exp)?;
                Ok(egraph.add(MathLang::Pow([base_id, exp_id])))
            }
            ASTRepr::Neg(inner) => {
                let inner_id = self.ast_to_egg(egraph, inner)?;
                Ok(egraph.add(MathLang::Neg([inner_id])))
            }
            ASTRepr::Ln(inner) => {
                let inner_id = self.ast_to_egg(egraph, inner)?;
                Ok(egraph.add(MathLang::Ln([inner_id])))
            }
            ASTRepr::Exp(inner) => {
                let inner_id = self.ast_to_egg(egraph, inner)?;
                Ok(egraph.add(MathLang::Exp([inner_id])))
            }
            ASTRepr::Sin(inner) => {
                let inner_id = self.ast_to_egg(egraph, inner)?;
                Ok(egraph.add(MathLang::Sin([inner_id])))
            }
            ASTRepr::Cos(inner) => {
                let inner_id = self.ast_to_egg(egraph, inner)?;
                Ok(egraph.add(MathLang::Cos([inner_id])))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_id = self.ast_to_egg(egraph, inner)?;
                Ok(egraph.add(MathLang::Sqrt([inner_id])))
            }
            ASTRepr::Let(var_id, expr, body) => {
                let bound_symbol = format!("b{}", var_id).into();
                let var_node = egraph.add(MathLang::BoundVar(bound_symbol));
                let expr_id = self.ast_to_egg(egraph, expr)?;
                let body_id = self.ast_to_egg(egraph, body)?;
                Ok(egraph.add(MathLang::Let([var_node, expr_id, body_id])))
            }
            
            // Simplified collection handling for now
            ASTRepr::Sum(collection) => {
                let collection_id = self.collection_to_egg(egraph, collection)?;
                Ok(egraph.add(MathLang::Sum([collection_id])))
            }
            
            // Lambda expressions need special handling - simplified for now
            ASTRepr::Lambda(_lambda) => {
                // For now, treat as a collection reference
                // In a full implementation, we'd need proper lambda support
                let lambda_symbol = "lambda0".into();
                Ok(egraph.add(MathLang::CollectionRef(lambda_symbol)))
            }
        }
    }
    
    /// Convert collection to egg representation (simplified)
    fn collection_to_egg(&self, egraph: &mut EGraph<MathLang, DependencyAnalysis>, collection: &crate::ast::ast_repr::Collection<f64>) -> Result<Id> {
        use crate::ast::ast_repr::Collection;
        
        match collection {
            Collection::Empty => {
                // Empty collection as range [0, 0)
                let zero = egraph.add(MathLang::Num(OrderedFloat(0.0)));
                Ok(egraph.add(MathLang::Range([zero, zero])))
            }
            Collection::Singleton(expr) => {
                let expr_id = self.ast_to_egg(egraph, expr)?;
                Ok(egraph.add(MathLang::Singleton([expr_id])))
            }
            Collection::Range { start, end } => {
                let start_id = self.ast_to_egg(egraph, start)?;
                let end_id = self.ast_to_egg(egraph, end)?;
                Ok(egraph.add(MathLang::Range([start_id, end_id])))
            }
            Collection::Variable(index) => {
                let coll_symbol = format!("c{}", index).into();
                Ok(egraph.add(MathLang::CollectionRef(coll_symbol)))
            }
            Collection::DataArray(_) => {
                // Data arrays become collection references
                let coll_symbol = "data0".into();
                Ok(egraph.add(MathLang::CollectionRef(coll_symbol)))
            }
            Collection::Map { lambda: _, collection } => {
                // Simplified: just use the collection for now
                // In a full implementation, we'd need proper lambda/map support
                self.collection_to_egg(egraph, collection)
            }
            Collection::Filter { collection, predicate: _ } => {
                // Simplified: just use the collection for now
                self.collection_to_egg(egraph, collection)
            }
        }
    }
    
    /// Convert egg expression back to DSLCompile AST
    fn egg_to_ast(&self, expr: &RecExpr<MathLang>) -> Result<ASTRepr<f64>> {
        let node = expr.as_ref().last().unwrap();
        
        match node {
            MathLang::Num(val) => Ok(ASTRepr::Constant(val.into_inner())),
            MathLang::UserVar(var_symbol) => {
                // Extract variable index from symbol (v123 -> 123)
                if let Some(idx_str) = var_symbol.as_str().strip_prefix('v') {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        Ok(ASTRepr::Variable(idx))
                    } else {
                        Err(DSLCompileError::Generic(format!("Invalid variable symbol: {}", var_symbol)))
                    }
                } else {
                    Err(DSLCompileError::Generic(format!("Invalid variable symbol format: {}", var_symbol)))
                }
            }
            MathLang::BoundVar(bound_symbol) => {
                // Extract bound variable index from symbol (b123 -> 123)
                if let Some(idx_str) = bound_symbol.as_str().strip_prefix('b') {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        Ok(ASTRepr::BoundVar(idx))
                    } else {
                        Err(DSLCompileError::Generic(format!("Invalid bound variable symbol: {}", bound_symbol)))
                    }
                } else {
                    Err(DSLCompileError::Generic(format!("Invalid bound variable symbol format: {}", bound_symbol)))
                }
            }
            
            MathLang::Add([a, b]) => {
                let left = self.extract_subexpr(expr, *a)?;
                let right = self.extract_subexpr(expr, *b)?;
                Ok(ASTRepr::add_from_array([left, right]))
            }
            MathLang::Mul([a, b]) => {
                let left = self.extract_subexpr(expr, *a)?;
                let right = self.extract_subexpr(expr, *b)?;
                Ok(ASTRepr::mul_from_array([left, right]))
            }
            MathLang::Sub([a, b]) => {
                let left = self.extract_subexpr(expr, *a)?;
                let right = self.extract_subexpr(expr, *b)?;
                Ok(ASTRepr::Sub(Box::new(left), Box::new(right)))
            }
            MathLang::Div([a, b]) => {
                let left = self.extract_subexpr(expr, *a)?;
                let right = self.extract_subexpr(expr, *b)?;
                Ok(ASTRepr::Div(Box::new(left), Box::new(right)))
            }
            MathLang::Pow([a, b]) => {
                let base = self.extract_subexpr(expr, *a)?;
                let exp = self.extract_subexpr(expr, *b)?;
                Ok(ASTRepr::Pow(Box::new(base), Box::new(exp)))
            }
            MathLang::Neg([a]) => {
                let inner = self.extract_subexpr(expr, *a)?;
                Ok(ASTRepr::Neg(Box::new(inner)))
            }
            MathLang::Ln([a]) => {
                let inner = self.extract_subexpr(expr, *a)?;
                Ok(ASTRepr::Ln(Box::new(inner)))
            }
            MathLang::Exp([a]) => {
                let inner = self.extract_subexpr(expr, *a)?;
                Ok(ASTRepr::Exp(Box::new(inner)))
            }
            MathLang::Sin([a]) => {
                let inner = self.extract_subexpr(expr, *a)?;
                Ok(ASTRepr::Sin(Box::new(inner)))
            }
            MathLang::Cos([a]) => {
                let inner = self.extract_subexpr(expr, *a)?;
                Ok(ASTRepr::Cos(Box::new(inner)))
            }
            MathLang::Sqrt([a]) => {
                let inner = self.extract_subexpr(expr, *a)?;
                Ok(ASTRepr::Sqrt(Box::new(inner)))
            }
            
            // Simplified collection reconstruction
            MathLang::Sum([a]) => {
                // Create a simplified sum for now
                let collection = crate::ast::ast_repr::Collection::Variable(0);
                Ok(ASTRepr::Sum(Box::new(collection)))
            }
            
            _ => Err(DSLCompileError::Generic(
                "Unsupported egg expression in conversion".to_string()
            ))
        }
    }
    
    /// Extract subexpression from RecExpr
    fn extract_subexpr(&self, expr: &RecExpr<MathLang>, id: Id) -> Result<ASTRepr<f64>> {
        // Create a sub-expression from the given id
        let mut sub_expr = RecExpr::default();
        let mut id_map = HashMap::new();
        self.extract_recursive(expr, id, &mut sub_expr, &mut id_map);
        
        self.egg_to_ast(&sub_expr)
    }
    
    /// Recursively extract expression nodes
    fn extract_recursive(&self, expr: &RecExpr<MathLang>, id: Id, target: &mut RecExpr<MathLang>, id_map: &mut HashMap<Id, Id>) {
        if id_map.contains_key(&id) {
            return;
        }
        
        let node = &expr[id];
        
        // Recursively extract children first
        for child_id in node.children() {
            self.extract_recursive(expr, *child_id, target, id_map);
        }
        
        // Create new node with updated child IDs
        let new_node = match node {
            MathLang::Add([a, b]) => MathLang::Add([id_map[a], id_map[b]]),
            MathLang::Mul([a, b]) => MathLang::Mul([id_map[a], id_map[b]]),
            MathLang::Sub([a, b]) => MathLang::Sub([id_map[a], id_map[b]]),
            MathLang::Div([a, b]) => MathLang::Div([id_map[a], id_map[b]]),
            MathLang::Pow([a, b]) => MathLang::Pow([id_map[a], id_map[b]]),
            MathLang::Range([a, b]) => MathLang::Range([id_map[a], id_map[b]]),
            MathLang::Neg([a]) => MathLang::Neg([id_map[a]]),
            MathLang::Ln([a]) => MathLang::Ln([id_map[a]]),
            MathLang::Exp([a]) => MathLang::Exp([id_map[a]]),
            MathLang::Sin([a]) => MathLang::Sin([id_map[a]]),
            MathLang::Cos([a]) => MathLang::Cos([id_map[a]]),
            MathLang::Sqrt([a]) => MathLang::Sqrt([id_map[a]]),
            MathLang::Sum([a]) => MathLang::Sum([id_map[a]]),
            MathLang::Singleton([a]) => MathLang::Singleton([id_map[a]]),
            MathLang::Let([a, b, c]) => MathLang::Let([id_map[a], id_map[b], id_map[c]]),
            // Leaf nodes stay the same
            _ => node.clone(),
        };
        
        let new_id = target.add(new_node);
        id_map.insert(id, new_id);
    }
}

/// Create mathematical rewrite rules for egg
#[cfg(feature = "egg_optimization")]
fn make_mathematical_rules() -> Vec<Rewrite<MathLang, DependencyAnalysis>> {
    vec![
        // Arithmetic identities
        rw!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rw!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        rw!("add-assoc"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("mul-assoc"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        
        // Zero and one identities
        rw!("add-zero"; "(+ ?a 0.0)" => "?a"),
        rw!("add-zero-rev"; "?a" => "(+ ?a 0.0)"),
        rw!("mul-zero"; "(* ?a 0.0)" => "0.0"),
        rw!("mul-one"; "(* ?a 1.0)" => "?a"),
        rw!("mul-one-rev"; "?a" => "(* ?a 1.0)"),
        
        // Negation rules
        rw!("neg-neg"; "(neg (neg ?a))" => "?a"),
        rw!("neg-zero"; "(neg 0.0)" => "0.0"),
        rw!("sub-to-add"; "(- ?a ?b)" => "(+ ?a (neg ?b))"),
        
        // Power rules
        rw!("pow-one"; "(^ ?x 1.0)" => "?x"),
        rw!("pow-zero"; "(^ ?x 0.0)" => "1.0"),
        rw!("pow-mul"; "(^ ?x (+ ?a ?b))" => "(* (^ ?x ?a) (^ ?x ?b))"),
        
        // Logarithm and exponential rules (with safety conditions)
        rw!("ln-exp"; "(ln (exp ?x))" => "?x"),
        // rw!("exp-ln"; "(exp (ln ?x))" => "?x" if is_positive("?x")),
        // rw!("ln-mul"; "(ln (* ?a ?b))" => "(+ (ln ?a) (ln ?b))" 
        //     if is_positive("?a") if is_positive("?b")),
        // rw!("ln-pow"; "(ln (^ ?a ?b))" => "(* ?b (ln ?a))" if is_positive("?a")),
        
        // Distributivity (controlled expansion)
        rw!("distribute"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("factor"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        
        // Simplify nested operations
        rw!("mul-div"; "(* ?a (/ 1.0 ?b))" => "(/ ?a ?b)"),
        rw!("div-mul"; "(/ ?a (/ 1.0 ?b))" => "(* ?a ?b)"),
        
        // Trigonometric identities
        rw!("sin-neg"; "(sin (neg ?x))" => "(neg (sin ?x))"),
        rw!("cos-neg"; "(cos (neg ?x))" => "(cos ?x)"),
        
        // Square root simplifications
        // rw!("sqrt-square"; "(sqrt (* ?x ?x))" => "?x" if is_non_negative("?x")),
        // rw!("sqrt-mul"; "(sqrt (* ?a ?b))" => "(* (sqrt ?a) (sqrt ?b))" 
        //     if is_non_negative("?a") if is_non_negative("?b")),
    ]
}

/// Conditional function to check if expression is positive
#[cfg(feature = "egg_optimization")]
fn is_positive(var: &str) -> impl Fn(&mut EGraph<MathLang, DependencyAnalysis>, Id, &Subst) -> bool {
    let _var_name = var.to_string();
    move |egraph, id, _subst| {
        let eclass = &egraph[id];
        for node in &eclass.nodes {
            match node {
                MathLang::Num(val) => return val.into_inner() > 0.0,
                MathLang::Exp(_) => return true, // exp(x) > 0 always
                _ => {}
            }
        }
        false // Conservative: assume not provably positive
    }
}

/// Conditional function to check if expression is non-negative
#[cfg(feature = "egg_optimization")]
fn is_non_negative(var: &str) -> impl Fn(&mut EGraph<MathLang, DependencyAnalysis>, Id, &Subst) -> bool {
    let _var_name = var.to_string();
    move |egraph, id, _subst| {
        let eclass = &egraph[id];
        for node in &eclass.nodes {
            match node {
                MathLang::Num(val) => return val.into_inner() >= 0.0,
                MathLang::Exp(_) => return true,  // exp(x) >= 0 always
                MathLang::Sqrt(_) => return true, // sqrt(x) >= 0 by definition
                _ => {}
            }
        }
        false // Conservative: assume not provably non-negative
    }
}

/// Fallback implementation when egg_optimization feature is not enabled
#[cfg(not(feature = "egg_optimization"))]
pub struct EggOptimizer;

#[cfg(not(feature = "egg_optimization"))]
impl EggOptimizer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn optimize(&self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        Ok(expr.clone())
    }
}

/// Helper function to create and use the egg optimizer
pub fn optimize_with_egg(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
    let optimizer = EggOptimizer::new();
    optimizer.optimize(expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "egg_optimization")]
    #[test]
    fn test_egg_optimizer_creation() {
        let optimizer = EggOptimizer::new();
        assert!(!optimizer.rules.is_empty());
    }
    
    #[test]
    fn test_basic_optimization() {
        let expr = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(0.0),
        ]);
        
        let result = optimize_with_egg(&expr);
        assert!(result.is_ok());
    }
    
    #[cfg(feature = "egg_optimization")]
    #[test]
    fn test_summation_cost_function() {
        let mut egraph: EGraph<MathLang, DependencyAnalysis> = Default::default();
        
        // Create a simple range [1, 10]
        let one = egraph.add(MathLang::Num(OrderedFloat(1.0)));
        let ten = egraph.add(MathLang::Num(OrderedFloat(10.0)));
        let range = egraph.add(MathLang::Range([one, ten]));
        let sum_expr = egraph.add(MathLang::Sum([range]));
        
        let mut cost_fn = SummationCostFunction::new(&egraph);
        let cost = cost_fn.cost(&MathLang::Sum([range]), |_| 5.0);
        
        // Should have non-additive cost: base_cost + (size * inner_cost * coupling)
        assert!(cost > 1000.0); // Greater than just base cost
        println!("Summation cost: {}", cost);
    }
    
    #[cfg(feature = "egg_optimization")]
    #[test]
    fn test_domain_aware_costs() {
        let mut egraph: EGraph<MathLang, DependencyAnalysis> = Default::default();
        
        // Test ln of positive constant (safe)
        let positive = egraph.add(MathLang::Num(OrderedFloat(5.0)));
        let ln_safe = egraph.add(MathLang::Ln([positive]));
        
        // Test ln of variable (potentially unsafe)
        let variable = egraph.add(MathLang::UserVar(0));
        let ln_unsafe = egraph.add(MathLang::Ln([variable]));
        
        let mut cost_fn = SummationCostFunction::new(&egraph);
        
        let safe_cost = cost_fn.cost(&MathLang::Ln([positive]), |_| 1.0);
        let unsafe_cost = cost_fn.cost(&MathLang::Ln([variable]), |_| 1.0);
        
        // Unsafe ln should have higher cost due to domain penalty
        assert!(unsafe_cost > safe_cost);
        println!("Safe ln cost: {}, Unsafe ln cost: {}", safe_cost, unsafe_cost);
    }
}