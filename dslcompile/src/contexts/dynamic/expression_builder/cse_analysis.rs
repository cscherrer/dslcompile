//! Advanced CSE Analysis for DynamicContext
//!
//! This module provides enhanced Common Subexpression Elimination (CSE) analysis
//! with cost visibility and automatic optimization suggestions. It integrates with
//! egglog cost analysis to provide intelligent CSE decisions.

use crate::{
    ast::{Scalar, ast_repr::ASTRepr},
    contexts::dynamic::expression_builder::{DynamicContext, DynamicExpr},
};
use std::collections::{HashMap, HashSet};

/// CSE analysis result containing cost information and optimization suggestions
#[derive(Debug, Clone)]
pub struct CSEAnalysis {
    /// Original expression cost
    pub original_cost: f64,
    /// Estimated cost after CSE optimization
    pub optimized_cost: f64,
    /// Potential savings from CSE
    pub savings: f64,
    /// Detected subexpressions that could benefit from CSE
    pub candidates: Vec<CSECandidate>,
    /// Detailed cost breakdown
    pub cost_breakdown: CostBreakdown,
}

/// A candidate subexpression for CSE optimization
#[derive(Debug, Clone)]
pub struct CSECandidate {
    /// The subexpression AST
    pub expression: String, // AST debug representation
    /// Number of times this subexpression appears
    pub frequency: usize,
    /// Cost of computing this subexpression once
    pub computation_cost: f64,
    /// Total savings if this subexpression is eliminated
    pub potential_savings: f64,
    /// Complexity score (higher = more complex, better CSE candidate)
    pub complexity_score: f64,
}

/// Detailed cost breakdown for transparency
#[derive(Debug, Clone)]
pub struct CostBreakdown {
    /// Cost from basic operations (add, mul, etc.)
    pub operation_cost: f64,
    /// Cost from transcendental functions (sin, ln, etc.)
    pub transcendental_cost: f64,
    /// Cost from summations
    pub summation_cost: f64,
    /// Cost from Let expressions (existing CSE)
    pub cse_cost: f64,
    /// Cost from variables and constants
    pub variable_cost: f64,
}

impl CostBreakdown {
    /// Calculate total cost
    #[must_use]
    pub fn total(&self) -> f64 {
        self.operation_cost
            + self.transcendental_cost
            + self.summation_cost
            + self.cse_cost
            + self.variable_cost
    }
}

/// Advanced CSE analyzer with cost visibility
pub struct CSEAnalyzer {
    /// Threshold for considering CSE beneficial (minimum savings)
    cost_threshold: f64,
    /// Minimum frequency for CSE candidacy
    frequency_threshold: usize,
    /// Weight for complexity in CSE decisions
    complexity_weight: f64,
}

impl Default for CSEAnalyzer {
    fn default() -> Self {
        Self {
            cost_threshold: 5.0,    // Minimum 5-unit cost savings
            frequency_threshold: 2, // Must appear at least twice
            complexity_weight: 1.5, // Complexity multiplier
        }
    }
}

impl CSEAnalyzer {
    /// Create analyzer with custom thresholds
    #[must_use]
    pub fn new(cost_threshold: f64, frequency_threshold: usize, complexity_weight: f64) -> Self {
        Self {
            cost_threshold,
            frequency_threshold,
            complexity_weight,
        }
    }

    /// Analyze expression for CSE opportunities
    pub fn analyze<T: Scalar + Clone>(&self, expr: &ASTRepr<T>) -> CSEAnalysis {
        let original_cost = self.calculate_total_cost(expr);
        let cost_breakdown = self.analyze_cost_breakdown(expr);
        let candidates = self.find_cse_candidates(expr);

        // Calculate optimized cost based on CSE candidates
        let optimized_cost =
            original_cost - candidates.iter().map(|c| c.potential_savings).sum::<f64>();

        let savings = original_cost - optimized_cost;

        CSEAnalysis {
            original_cost,
            optimized_cost,
            savings,
            candidates,
            cost_breakdown,
        }
    }

    /// Suggest automatic CSE optimization with cost justification
    pub fn suggest_optimizations<T: Scalar + Clone>(
        &self,
        expr: &ASTRepr<T>,
    ) -> Vec<CSEOptimization> {
        let analysis = self.analyze(expr);

        analysis
            .candidates
            .into_iter()
            .filter(|candidate| {
                candidate.potential_savings >= self.cost_threshold
                    && candidate.frequency >= self.frequency_threshold
            })
            .map(|candidate| CSEOptimization {
                target_expression: candidate.expression.clone(),
                frequency: candidate.frequency,
                cost_savings: candidate.potential_savings,
                complexity_score: candidate.complexity_score,
                recommended_action: if candidate.potential_savings > 20.0 {
                    CSEAction::HighPriority
                } else if candidate.potential_savings > 10.0 {
                    CSEAction::MediumPriority
                } else {
                    CSEAction::LowPriority
                },
            })
            .collect()
    }

    /// Calculate total expression cost using summation-aware analysis
    fn calculate_total_cost<T: Scalar + Clone>(&self, expr: &ASTRepr<T>) -> f64 {
        use crate::ast::ast_utils::visitors::SummationAwareCostVisitor;
        SummationAwareCostVisitor::compute_cost_with_domain_size(expr, 50) as f64
    }

    /// Analyze cost breakdown by operation type
    fn analyze_cost_breakdown<T: Scalar + Clone>(&self, expr: &ASTRepr<T>) -> CostBreakdown {
        let mut breakdown = CostBreakdown {
            operation_cost: 0.0,
            transcendental_cost: 0.0,
            summation_cost: 0.0,
            cse_cost: 0.0,
            variable_cost: 0.0,
        };

        self.analyze_breakdown_recursive(expr, &mut breakdown);
        breakdown
    }

    /// Recursive helper for cost breakdown analysis
    fn analyze_breakdown_recursive<T: Scalar + Clone>(
        &self,
        expr: &ASTRepr<T>,
        breakdown: &mut CostBreakdown,
    ) {
        match expr {
            ASTRepr::Constant(_) => breakdown.variable_cost += 0.5,
            ASTRepr::Variable(_) => breakdown.variable_cost += 1.0,
            ASTRepr::BoundVar(_) => breakdown.variable_cost += 0.5,

            ASTRepr::Add(terms) => {
                breakdown.operation_cost += terms.len() as f64;
                for (term, _count) in terms.iter() {
                    self.analyze_breakdown_recursive(term, breakdown);
                }
            }
            ASTRepr::Mul(factors) => {
                breakdown.operation_cost += factors.len() as f64 * 1.5;
                for (factor, _count) in factors.iter() {
                    self.analyze_breakdown_recursive(factor, breakdown);
                }
            }
            ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) => {
                breakdown.operation_cost += if matches!(expr, ASTRepr::Div(_, _)) {
                    5.0
                } else {
                    1.2
                };
                self.analyze_breakdown_recursive(left, breakdown);
                self.analyze_breakdown_recursive(right, breakdown);
            }
            ASTRepr::Pow(base, exp) => {
                breakdown.operation_cost += 10.0;
                self.analyze_breakdown_recursive(base, breakdown);
                self.analyze_breakdown_recursive(exp, breakdown);
            }

            // Transcendental functions
            ASTRepr::Sin(inner) | ASTRepr::Cos(inner) => {
                breakdown.transcendental_cost += 25.0;
                self.analyze_breakdown_recursive(inner, breakdown);
            }
            ASTRepr::Ln(inner) | ASTRepr::Exp(inner) => {
                breakdown.transcendental_cost += 30.0;
                self.analyze_breakdown_recursive(inner, breakdown);
            }
            ASTRepr::Sqrt(inner) => {
                breakdown.transcendental_cost += 15.0;
                self.analyze_breakdown_recursive(inner, breakdown);
            }
            ASTRepr::Neg(inner) => {
                breakdown.operation_cost += 1.0;
                self.analyze_breakdown_recursive(inner, breakdown);
            }

            // CSE-related
            ASTRepr::Let(_, expr, body) => {
                breakdown.cse_cost += 2.0; // Let binding overhead
                self.analyze_breakdown_recursive(expr, breakdown);
                self.analyze_breakdown_recursive(body, breakdown);
            }

            // Summations
            ASTRepr::Sum(collection) => {
                breakdown.summation_cost += 50.0; // Base summation cost
                // TODO: Analyze collection cost recursively
            }

            ASTRepr::Lambda(lambda) => {
                breakdown.operation_cost += 2.0; // Lambda creation cost
                self.analyze_breakdown_recursive(&lambda.body, breakdown);
            }
        }
    }

    /// Find potential CSE candidates by detecting repeated subexpressions
    fn find_cse_candidates<T: Scalar + Clone>(&self, expr: &ASTRepr<T>) -> Vec<CSECandidate> {
        let mut subexpr_counts: HashMap<String, (usize, f64)> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();

        self.collect_subexpressions(expr, &mut subexpr_counts, &mut visited);

        subexpr_counts
            .into_iter()
            .filter_map(|(expr_str, (count, cost))| {
                if count >= self.frequency_threshold && cost > 1.0 {
                    let complexity_score = self.calculate_complexity_score(&expr_str, cost);
                    let potential_savings = (count - 1) as f64 * cost * self.complexity_weight;

                    Some(CSECandidate {
                        expression: expr_str,
                        frequency: count,
                        computation_cost: cost,
                        potential_savings,
                        complexity_score,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Collect subexpressions and their frequencies
    fn collect_subexpressions<T: Scalar + Clone>(
        &self,
        expr: &ASTRepr<T>,
        counts: &mut HashMap<String, (usize, f64)>,
        visited: &mut HashSet<String>,
    ) {
        let expr_str = format!("{expr:?}");
        let expr_cost = self.estimate_subexpression_cost(expr);

        // Only consider non-trivial subexpressions
        if expr_cost > 1.0 && !visited.contains(&expr_str) {
            let entry = counts.entry(expr_str.clone()).or_insert((0, expr_cost));
            entry.0 += 1;
            visited.insert(expr_str);
        }

        // Recursively analyze subexpressions
        match expr {
            ASTRepr::Add(terms) => {
                for (term, _count) in terms.iter() {
                    self.collect_subexpressions(term, counts, visited);
                }
            }
            ASTRepr::Mul(factors) => {
                for (factor, _count) in factors.iter() {
                    self.collect_subexpressions(factor, counts, visited);
                }
            }
            ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) | ASTRepr::Pow(left, right) => {
                self.collect_subexpressions(left, counts, visited);
                self.collect_subexpressions(right, counts, visited);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sqrt(inner) => {
                self.collect_subexpressions(inner, counts, visited);
            }
            ASTRepr::Let(_, expr, body) => {
                self.collect_subexpressions(expr, counts, visited);
                self.collect_subexpressions(body, counts, visited);
            }
            ASTRepr::Lambda(lambda) => {
                self.collect_subexpressions(&lambda.body, counts, visited);
            }
            // Leaves don't have subexpressions
            ASTRepr::Constant(_)
            | ASTRepr::Variable(_)
            | ASTRepr::BoundVar(_)
            | ASTRepr::Sum(_) => {}
        }
    }

    /// Estimate the computational cost of a subexpression
    fn estimate_subexpression_cost<T: Scalar + Clone>(&self, expr: &ASTRepr<T>) -> f64 {
        match expr {
            ASTRepr::Constant(_) => 0.5,
            ASTRepr::Variable(_) => 1.0,
            ASTRepr::BoundVar(_) => 0.5,
            ASTRepr::Add(_) => 1.0,
            ASTRepr::Mul(_) => 1.5,
            ASTRepr::Sub(_, _) => 1.2,
            ASTRepr::Div(_, _) => 5.0,
            ASTRepr::Pow(_, _) => 10.0,
            ASTRepr::Neg(_) => 1.0,
            ASTRepr::Sin(_) | ASTRepr::Cos(_) => 25.0,
            ASTRepr::Ln(_) | ASTRepr::Exp(_) => 30.0,
            ASTRepr::Sqrt(_) => 15.0,
            ASTRepr::Let(_, _, _) => 2.0,
            ASTRepr::Sum(_) => 50.0,
            ASTRepr::Lambda(_) => 2.0,
        }
    }

    /// Calculate complexity score for CSE prioritization
    fn calculate_complexity_score(&self, _expr_str: &str, cost: f64) -> f64 {
        // Simple heuristic: higher cost = higher complexity
        // Could be enhanced with AST depth analysis, operation counting, etc.
        cost * 1.2
    }
}

/// CSE optimization recommendation
#[derive(Debug, Clone)]
pub struct CSEOptimization {
    /// The target expression to optimize
    pub target_expression: String,
    /// How many times this expression appears
    pub frequency: usize,
    /// Estimated cost savings
    pub cost_savings: f64,
    /// Complexity score for prioritization
    pub complexity_score: f64,
    /// Recommended action level
    pub recommended_action: CSEAction,
}

/// CSE action priority levels
#[derive(Debug, Clone)]
pub enum CSEAction {
    /// High priority optimization (>20 cost units saved)
    HighPriority,
    /// Medium priority optimization (>10 cost units saved)
    MediumPriority,
    /// Low priority optimization (>5 cost units saved)
    LowPriority,
}

/// Integration with `DynamicContext` for automatic CSE analysis
impl<const SCOPE: usize> DynamicContext<SCOPE> {
    /// Analyze expression for CSE opportunities with cost visibility
    pub fn analyze_cse<T: Scalar + Clone>(&self, expr: &DynamicExpr<T, SCOPE>) -> CSEAnalysis {
        CSEAnalyzer::default().analyze(&expr.ast)
    }

    /// Get CSE optimization suggestions with cost justification
    pub fn suggest_cse_optimizations<T: Scalar + Clone>(
        &self,
        expr: &DynamicExpr<T, SCOPE>,
    ) -> Vec<CSEOptimization> {
        CSEAnalyzer::default().suggest_optimizations(&expr.ast)
    }

    /// Analyze expression with custom CSE parameters
    pub fn analyze_cse_with_thresholds<T: Scalar + Clone>(
        &self,
        expr: &DynamicExpr<T, SCOPE>,
        cost_threshold: f64,
        frequency_threshold: usize,
        complexity_weight: f64,
    ) -> CSEAnalysis {
        CSEAnalyzer::new(cost_threshold, frequency_threshold, complexity_weight).analyze(&expr.ast)
    }

    /// Apply automatic CSE optimization based on cost analysis
    /// Returns the optimized expression with CSE applied to high-value candidates
    pub fn auto_cse<T: Scalar + Clone>(
        &mut self,
        expr: DynamicExpr<T, SCOPE>,
    ) -> DynamicExpr<T, SCOPE> {
        let optimizations = self.suggest_cse_optimizations(&expr);

        // For now, return the original expression
        // TODO: Implement automatic CSE application based on analysis
        // This would require sophisticated AST rewriting capabilities

        if optimizations.is_empty() {
            println!("No beneficial CSE optimizations found");
        } else {
            println!(
                "Found {} CSE optimization opportunities:",
                optimizations.len()
            );
            for opt in &optimizations {
                println!(
                    "  - {:?}: {:.1} cost savings (frequency: {})",
                    opt.recommended_action, opt.cost_savings, opt.frequency
                );
            }
        }

        expr
    }
}
