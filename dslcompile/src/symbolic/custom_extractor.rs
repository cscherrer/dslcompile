//! Enhanced Cost Analysis with Multiplicative Costs
//!
//! This module provides cost analysis for mathematical expressions with special focus on
//! multiplicative cost modeling for Sum and Map operations, inspired by egglog-experimental.
//!
//! ## Key Problems Addressed
//!
//! 1. **Summation Traversal Coupling**: When summations couple data traversal with parameter access
//! 2. **Multiplicative Cost Scaling**: Sum/Map operations have costs that multiply with iteration count
//! 3. **Dynamic Cost Assignment**: Runtime cost adjustment based on expression complexity
//!
//! ## Cost Model Types
//!
//! - **ADDITIVE**: Simple operations like `x + y` have linear cost growth
//! - **MULTIPLICATIVE**: `Σ(i=1 to n) f(i)` has cost = `n * cost(f(i)) * coupling_factor`
//! - **EXPONENTIAL**: Nested structures like `Σ(Σ(...))` compound multiplicatively
//!
//! ## Coupling Examples
//!
//! - **HIGH COUPLING**: `Σ(i=1 to n) k*i²` - cost = `n * base_cost * 1000` (external param access)
//! - **LOW COUPLING**: `Σ(i=1 to n) i²` - cost = `n * base_cost * 1` (range-only access)
//! - **EXPANDABLE**: `(x+y)²` → `x² + 2xy + y²` - expansion reduces coupling

use crate::error::Result;
use crate::final_tagless::ASTRepr;
use std::collections::HashMap;

/// Patterns of summation traversal coupling in mathematical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingPattern {
    /// Decoupled: Summations that only use range variables (efficient)
    Decoupled {
        /// Range variables used (e.g., [0] for single summation over variable 0)
        range_vars: Vec<usize>,
        /// Estimated operations count
        operation_count: usize,
    },
    /// Coupled: Summations that access external parameters during traversal (expensive)
    Coupled {
        /// External parameters accessed during summation (variable indices)
        external_params: Vec<usize>,
        /// Range variables (variable indices)
        range_vars: Vec<usize>,
        /// Base cost multiplier for coupled operations
        cost_multiplier: f64,
        /// Estimated operations count
        operation_count: usize,
        /// Multiplicative factor for iterative operations (new)
        iteration_multiplier: f64,
        /// Estimated range size for multiplicative cost calculation
        estimated_range_size: usize,
    },
    /// Expandable: Expressions that can be expanded to reduce coupling
    Expandable {
        /// Original coupled expression
        original: String,
        /// Expanded decoupled form
        expanded: String,
        /// Cost reduction from expansion
        cost_reduction: f64,
    },
}

/// Enhanced cost analyzer with multiplicative cost modeling
pub struct EnhancedCostAnalyzer {
    /// Cache for coupling pattern analysis
    pattern_cache: HashMap<String, CouplingPattern>,
    /// Cache for cost calculations
    cost_cache: HashMap<String, f64>,
    /// Dynamic cost table for runtime cost assignment (egglog-style)
    dynamic_cost_table: HashMap<String, f64>,
    /// Multiplicative cost factors for different operation types
    operation_multipliers: HashMap<String, f64>,
}

impl EnhancedCostAnalyzer {
    /// Create a new enhanced cost analyzer with multiplicative cost support
    #[must_use]
    pub fn new() -> Self {
        let mut operation_multipliers = HashMap::new();
        
        // Set multiplicative factors for different operations (inspired by egglog-experimental)
        operation_multipliers.insert("Sum".to_string(), 1000.0);  // High multiplier for summations
        operation_multipliers.insert("Map".to_string(), 500.0);   // Medium multiplier for maps
        operation_multipliers.insert("Pow".to_string(), 100.0);   // Power operations are expensive
        operation_multipliers.insert("Mul".to_string(), 10.0);    // Multiplication compounds
        operation_multipliers.insert("Add".to_string(), 1.0);     // Addition is cheap
        
        Self {
            pattern_cache: HashMap::new(),
            cost_cache: HashMap::new(),
            dynamic_cost_table: HashMap::new(),
            operation_multipliers,
        }
    }
    
    /// Set dynamic cost for specific expression pattern (egglog-experimental style)
    pub fn set_dynamic_cost(&mut self, pattern: &str, cost: f64) {
        self.dynamic_cost_table.insert(pattern.to_string(), cost);
    }
    
    /// Get multiplicative factor for operation type
    pub fn get_operation_multiplier(&self, op_type: &str) -> f64 {
        self.operation_multipliers.get(op_type).copied().unwrap_or(1.0)
    }

    /// Analyze coupling patterns in an expression
    pub fn analyze_coupling(&mut self, expr: &ASTRepr<f64>) -> Result<CouplingPattern> {
        let expr_key = format!("{expr:?}");

        // Check cache first
        if let Some(pattern) = self.pattern_cache.get(&expr_key) {
            return Ok(pattern.clone());
        }

        // Analyze based on expression structure
        let pattern = match expr {
            // High coupling: Power operations often couple data and parameters
            ASTRepr::Pow(base, exp) => {
                // Check if this looks like a summation coupling pattern
                if self.involves_multiple_variables(base) || self.involves_multiple_variables(exp) {
                    CouplingPattern::Coupled {
                        external_params: vec![1], // Assume parameter coupling
                        range_vars: vec![0],
                        cost_multiplier: 1000.0,
                        operation_count: self.count_operations(expr),
                        iteration_multiplier: self.get_operation_multiplier("Pow"),
                        estimated_range_size: 1, // Single operation
                    }
                } else {
                    CouplingPattern::Decoupled {
                        range_vars: vec![0],
                        operation_count: self.count_operations(expr),
                    }
                }
            }

            // Low coupling: Addition and multiplication are typically decoupled
            ASTRepr::Add(_) | ASTRepr::Mul(_) => CouplingPattern::Decoupled {
                range_vars: vec![0],
                operation_count: self.count_operations(expr),
            },

            // Sum operations: High multiplicative cost potential
            ASTRepr::Sum(collection) => {
                let range_size = self.estimate_collection_size(collection);
                let inner_complexity = self.collection_involves_multiple_variables(collection);
                
                if inner_complexity {
                    CouplingPattern::Coupled {
                        external_params: vec![1], // Assume external parameter access
                        range_vars: vec![0],
                        cost_multiplier: 1000.0,
                        operation_count: range_size, // Operations proportional to collection size
                        iteration_multiplier: self.get_operation_multiplier("Sum"),
                        estimated_range_size: range_size,
                    }
                } else {
                    CouplingPattern::Decoupled {
                        range_vars: vec![0],
                        operation_count: range_size,
                    }
                }
            }
            
            // Default: Assume decoupled for simple operations
            _ => CouplingPattern::Decoupled {
                range_vars: vec![0],
                operation_count: self.count_operations(expr),
            },
        };

        // Cache the result
        self.pattern_cache.insert(expr_key, pattern.clone());
        Ok(pattern)
    }

    /// Calculate enhanced cost with multiplicative factors
    pub fn calculate_enhanced_cost(&mut self, expr: &ASTRepr<f64>) -> Result<f64> {
        let expr_key = format!("{expr:?}");

        // Check dynamic cost table first (egglog-experimental style)
        if let Some(dynamic_cost) = self.dynamic_cost_table.get(&expr_key) {
            return Ok(*dynamic_cost);
        }

        // Check cache
        if let Some(cost) = self.cost_cache.get(&expr_key) {
            return Ok(*cost);
        }

        // Analyze coupling pattern
        let pattern = self.analyze_coupling(expr)?;

        // Calculate cost with multiplicative factors
        let cost = match pattern {
            CouplingPattern::Decoupled {
                operation_count, ..
            } => 1.0 + (operation_count as f64) * 0.1,
            
            CouplingPattern::Coupled {
                cost_multiplier,
                operation_count,
                iteration_multiplier,
                estimated_range_size,
                ..
            } => {
                // Multiplicative cost: base * operations * range * iteration_factor
                let base_cost = cost_multiplier + (operation_count as f64) * 10.0;
                let multiplicative_factor = iteration_multiplier * (estimated_range_size as f64);
                base_cost * multiplicative_factor.max(1.0)
            }
            
            CouplingPattern::Expandable { cost_reduction, .. } => 10.0 * (1.0 - cost_reduction),
        };

        // Cache the result
        self.cost_cache.insert(expr_key, cost);
        Ok(cost)
    }
    
    /// Estimate collection size for multiplicative cost calculation
    fn estimate_collection_size(&self, collection: &crate::ast::ast_repr::Collection<f64>) -> usize {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Empty => 0,
            Collection::Singleton(_) => 1,
            Collection::Range { .. } => 100, // Default estimate for ranges
            Collection::Variable(_) => 50,   // Default estimate for data arrays
            Collection::Filter { collection, .. } => self.estimate_collection_size(collection) / 2, // Assume filtering reduces by half
            Collection::Map { collection, .. } => self.estimate_collection_size(collection), // Same size as input
            Collection::Union(collections) => {
                collections.iter().map(|c| self.estimate_collection_size(c)).sum()
            }
        }
    }
    
    /// Check if a collection involves multiple variables (indicating potential coupling)
    fn collection_involves_multiple_variables(&self, collection: &crate::ast::ast_repr::Collection<f64>) -> bool {
        use crate::ast::ast_repr::Collection;
        match collection {
            Collection::Empty => false,
            Collection::Singleton(expr) => self.involves_multiple_variables(expr),
            Collection::Range { start, end } => {
                self.involves_multiple_variables(start) || self.involves_multiple_variables(end)
            }
            Collection::Variable(_) => false, // Single variable reference
            Collection::Filter { collection, predicate } => {
                self.collection_involves_multiple_variables(collection) 
                || self.involves_multiple_variables(predicate)
            }
            Collection::Map { lambda, collection } => {
                // Check if lambda body involves multiple variables
                self.involves_multiple_variables(&lambda.body) 
                || self.collection_involves_multiple_variables(collection)
            }
            Collection::Union(collections) => {
                collections.iter().any(|c| self.collection_involves_multiple_variables(c))
            }
        }
    }

    /// Get a detailed coupling analysis report
    #[must_use]
    pub fn get_coupling_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Summation Traversal Coupling Analysis Report ===\n");

        report.push_str(&format!("Analyzed {} patterns\n", self.pattern_cache.len()));
        report.push_str(&format!(
            "Cached {} cost calculations\n",
            self.cost_cache.len()
        ));

        report.push_str("\nCoupling Patterns Found:\n");
        for (expr_key, pattern) in &self.pattern_cache {
            report.push_str(&format!("  {expr_key}: {pattern:?}\n"));
        }

        report.push_str("\nCost Analysis:\n");
        for (expr_key, cost) in &self.cost_cache {
            report.push_str(&format!("  {expr_key}: {cost:.2}\n"));
        }

        report
    }

    /// Check if an expression involves multiple variables (indicating potential coupling)
    fn involves_multiple_variables(&self, expr: &ASTRepr<f64>) -> bool {
        let mut variables = std::collections::HashSet::new();
        self.collect_variables(expr, &mut variables);
        variables.len() > 1
    }

    /// Collect all variables used in an expression
    fn collect_variables(
        &self,
        expr: &ASTRepr<f64>,
        variables: &mut std::collections::HashSet<usize>,
    ) {
        match expr {
            ASTRepr::Variable(index) => {
                variables.insert(*index);
            }
            ASTRepr::Add(terms) => {
                for term in terms {
                    self.collect_variables(term, variables);
                }
            }
            ASTRepr::Mul(factors) => {
                for factor in factors {
                    self.collect_variables(factor, variables);
                }
            }
            ASTRepr::Sub(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.collect_variables(left, variables);
                self.collect_variables(right, variables);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                self.collect_variables(inner, variables);
            }
            ASTRepr::Constant(_) => {
                // Constants don't contribute variables
            }
        }
    }

    /// Count the number of operations in an expression
    fn count_operations(&self, expr: &ASTRepr<f64>) -> usize {
        match expr {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
            ASTRepr::Add(terms) => {
                terms.iter().map(|term| self.count_operations(term)).sum::<usize>() + terms.len().saturating_sub(1)
            }
            ASTRepr::Mul(factors) => {
                factors.iter().map(|factor| self.count_operations(factor)).sum::<usize>() + factors.len().saturating_sub(1)
            }
            ASTRepr::Sub(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                1 + self.count_operations(left) + self.count_operations(right)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => 1 + self.count_operations(inner),
        }
    }
}

impl Default for EnhancedCostAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze summation traversal coupling for an expression
pub fn analyze_summation_coupling(expr: &ASTRepr<f64>) -> Result<(CouplingPattern, f64)> {
    let mut analyzer = SummationCouplingAnalyzer::new();
    let pattern = analyzer.analyze_coupling(expr)?;
    let cost = analyzer.calculate_coupling_cost(expr)?;
    Ok((pattern, cost))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_pattern_analysis() {
        let mut analyzer = EnhancedCostAnalyzer::new();

        // Test high coupling: power with multiple variables
        let coupled_expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );

        let pattern = analyzer.analyze_coupling(&coupled_expr).unwrap();

        match pattern {
            CouplingPattern::Coupled {
                cost_multiplier, ..
            } => {
                assert!(
                    cost_multiplier > 100.0,
                    "Power operations with multiple variables should have high coupling cost"
                );
            }
            _ => panic!("Power operations with multiple variables should be classified as coupled"),
        }
    }

    #[test]
    fn test_cost_calculation() {
        let mut analyzer = EnhancedCostAnalyzer::new();

        // Test low coupling: simple addition
        let decoupled_expr = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(1.0),
        ]);

        let cost = analyzer.calculate_enhanced_cost(&decoupled_expr).unwrap();

        assert!(
            cost < 10.0,
            "Simple addition should have low coupling cost, got {cost}"
        );
    }

    #[test]
    fn test_variable_analysis() {
        let analyzer = EnhancedCostAnalyzer::new();

        // Test single variable
        let single_var = ASTRepr::Variable(0);
        assert!(!analyzer.involves_multiple_variables(&single_var));

        // Test multiple variables
        let multi_var = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Variable(1),
        ]);
        assert!(analyzer.involves_multiple_variables(&multi_var));
    }

    #[test]
    fn test_operation_counting() {
        let analyzer = EnhancedCostAnalyzer::new();

        // Test simple expression
        let simple = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(1.0),
        ]);
        assert_eq!(analyzer.count_operations(&simple), 1);

        // Test complex expression
        let complex = ASTRepr::mul_from_array([
            ASTRepr::add_from_array([
                ASTRepr::Variable(0),
                ASTRepr::Variable(1),
            ]),
            ASTRepr::Constant(2.0),
        ]);
        assert_eq!(analyzer.count_operations(&complex), 2);
    }

    #[test]
    fn test_coupling_analysis_function() {
        // Test the convenience function
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::add_from_array([
                ASTRepr::Variable(0),
                ASTRepr::Variable(1),
            ])),
            Box::new(ASTRepr::Constant(2.0)),
        );

        let (pattern, cost) = analyze_enhanced_cost(&expr).unwrap();

        // Should detect coupling due to multiple variables in power expression
        match pattern {
            CouplingPattern::Coupled { .. } => {
                assert!(cost > 100.0, "Coupled expression should have high cost");
            }
            _ => panic!("Expression with multiple variables should be detected as coupled"),
        }
    }
    
    #[test]
    fn test_multiplicative_cost_for_sum() {
        use crate::ast::ast_repr::Collection;
        
        // Test multiplicative cost calculation for Sum operations
        // Create a Sum with a Map that involves multiple variables (creates coupling)
        let map_collection = Collection::Map {
            lambda: Box::new(crate::ast::ast_repr::Lambda {
                param: 0, // Range variable
                body: ASTRepr::Mul(crate::ast::multiset::MultiSet::from_iter([
                    ASTRepr::Variable(0), // Range variable
                    ASTRepr::Variable(1), // External parameter - creates coupling
                ])),
            }),
            collection: Box::new(Collection::Range {
                start: Box::new(ASTRepr::Constant(1.0)),
                end: Box::new(ASTRepr::Constant(10.0)),
            }),
        };
        
        let sum_expr = ASTRepr::Sum(Box::new(map_collection));

        let (pattern, cost) = analyze_enhanced_cost(&sum_expr).unwrap();
        
        match pattern {
            CouplingPattern::Coupled { iteration_multiplier, estimated_range_size, .. } => {
                assert!(iteration_multiplier > 100.0, "Sum should have high iteration multiplier");
                assert!(estimated_range_size > 1, "Range size should be estimated");
                assert!(cost > 10000.0, "Multiplicative cost should be very high for coupled sum, got {cost}");
            }
            _ => panic!("Sum with external parameter access should be detected as coupled"),
        }
    }
    
    #[test]
    fn test_dynamic_cost_assignment() {
        let mut analyzer = EnhancedCostAnalyzer::new();
        
        // Test dynamic cost assignment (egglog-experimental style)
        analyzer.set_dynamic_cost("high_priority_expr", 50000.0);
        
        let expr_key = "high_priority_expr";
        let cost = analyzer.dynamic_cost_table.get(expr_key).unwrap();
        assert_eq!(*cost, 50000.0, "Dynamic cost should be settable");
    }
    
    #[test]
    fn test_operation_multipliers() {
        let analyzer = EnhancedCostAnalyzer::new();
        
        // Test that different operations have appropriate multipliers
        assert_eq!(analyzer.get_operation_multiplier("Sum"), 1000.0);
        assert_eq!(analyzer.get_operation_multiplier("Map"), 500.0);
        assert_eq!(analyzer.get_operation_multiplier("Pow"), 100.0);
        assert_eq!(analyzer.get_operation_multiplier("Add"), 1.0);
        assert_eq!(analyzer.get_operation_multiplier("Unknown"), 1.0); // Default
    }
}
