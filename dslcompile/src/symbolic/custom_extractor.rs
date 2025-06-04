//! Summation Traversal Coupling Cost Analysis
//!
//! This module provides cost analysis for summation expressions to identify
//! when summations couple data traversal with parameter access, making them expensive.
//!
//! ## Key Problem: Summation Traversal Coupling
//!
//! The core issue we're solving is when summation expressions couple two types of traversal:
//! - **Data Traversal**: Iterating through summation range (e.g., `i` in `Σ(i=1 to n)`)
//! - **Parameter Access**: Accessing variables outside the range (e.g., `k` in `Σ(i=1 to n) k*i²`)
//!
//! ## Coupling Examples
//!
//! - **HIGH COUPLING**: `Σ(i=1 to n) k*i²` - requires accessing external parameter `k` during each iteration
//! - **LOW COUPLING**: `Σ(i=1 to n) i²` - only uses range variable, enables closed-form optimization
//! - **EXPANDABLE**: `(x+y)²` → `x² + 2xy + y²` - expansion enables sufficient statistics discovery

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
        /// Cost multiplier for coupled operations
        cost_multiplier: f64,
        /// Estimated operations count
        operation_count: usize,
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

/// Analyze summation traversal coupling in expressions
pub struct SummationCouplingAnalyzer {
    /// Cache for coupling pattern analysis
    pattern_cache: HashMap<String, CouplingPattern>,
    /// Cache for cost calculations
    cost_cache: HashMap<String, f64>,
}

impl SummationCouplingAnalyzer {
    /// Create a new coupling analyzer
    #[must_use]
    pub fn new() -> Self {
        Self {
            pattern_cache: HashMap::new(),
            cost_cache: HashMap::new(),
        }
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
                    }
                } else {
                    CouplingPattern::Decoupled {
                        range_vars: vec![0],
                        operation_count: self.count_operations(expr),
                    }
                }
            }

            // Low coupling: Addition and multiplication are typically decoupled
            ASTRepr::Add(_, _) | ASTRepr::Mul(_, _) => CouplingPattern::Decoupled {
                range_vars: vec![0],
                operation_count: self.count_operations(expr),
            },

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

    /// Calculate cost based on coupling analysis
    pub fn calculate_coupling_cost(&mut self, expr: &ASTRepr<f64>) -> Result<f64> {
        let expr_key = format!("{expr:?}");

        // Check cache first
        if let Some(cost) = self.cost_cache.get(&expr_key) {
            return Ok(*cost);
        }

        // Analyze coupling pattern
        let pattern = self.analyze_coupling(expr)?;

        // Calculate cost based on pattern
        let cost = match pattern {
            CouplingPattern::Decoupled {
                operation_count, ..
            } => 1.0 + (operation_count as f64) * 0.1,
            CouplingPattern::Coupled {
                cost_multiplier,
                operation_count,
                ..
            } => cost_multiplier + (operation_count as f64) * 10.0,
            CouplingPattern::Expandable { cost_reduction, .. } => 10.0 * (1.0 - cost_reduction),
        };

        // Cache the result
        self.cost_cache.insert(expr_key, cost);
        Ok(cost)
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
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
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
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
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

impl Default for SummationCouplingAnalyzer {
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
        let mut analyzer = SummationCouplingAnalyzer::new();

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
        let mut analyzer = SummationCouplingAnalyzer::new();

        // Test low coupling: simple addition
        let decoupled_expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(1.0)),
        );

        let cost = analyzer.calculate_coupling_cost(&decoupled_expr).unwrap();

        assert!(
            cost < 10.0,
            "Simple addition should have low coupling cost, got {cost}"
        );
    }

    #[test]
    fn test_variable_analysis() {
        let analyzer = SummationCouplingAnalyzer::new();

        // Test single variable
        let single_var = ASTRepr::Variable(0);
        assert!(!analyzer.involves_multiple_variables(&single_var));

        // Test multiple variables
        let multi_var = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );
        assert!(analyzer.involves_multiple_variables(&multi_var));
    }

    #[test]
    fn test_operation_counting() {
        let analyzer = SummationCouplingAnalyzer::new();

        // Test simple expression
        let simple = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(1.0)),
        );
        assert_eq!(analyzer.count_operations(&simple), 1);

        // Test complex expression
        let complex = ASTRepr::Mul(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Variable(1)),
            )),
            Box::new(ASTRepr::Constant(2.0)),
        );
        assert_eq!(analyzer.count_operations(&complex), 2);
    }

    #[test]
    fn test_coupling_analysis_function() {
        // Test the convenience function
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Variable(1)),
            )),
            Box::new(ASTRepr::Constant(2.0)),
        );

        let (pattern, cost) = analyze_summation_coupling(&expr).unwrap();

        // Should detect coupling due to multiple variables in power expression
        match pattern {
            CouplingPattern::Coupled { .. } => {
                assert!(cost > 100.0, "Coupled expression should have high cost");
            }
            _ => panic!("Expression with multiple variables should be detected as coupled"),
        }
    }
}
