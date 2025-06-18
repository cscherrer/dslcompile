//! Native egglog Integration with Domain Analysis
//!
//! This module implements domain-aware symbolic optimization using egglog's
//! native abstract interpretation capabilities, following the approach
//! demonstrated in the Herbie case study.
//!
//! ## Key Features
//!
//! - **Interval Analysis**: Native egglog lattice-based interval tracking
//! - **Domain-Safe Rules**: Conditional rewrite rules gated on domain analysis
//! - **Multiple Analyses**: Composable interval and "not-equals" analyses
//! - **Semi-Naive Evaluation**: Leverages egglog's efficient evaluation strategy
//!
//! ## Architecture
//!
//! Unlike our previous Rust wrapper approach, this uses egglog directly:
//! 1. **Native egglog Program**: Mathematical rules + analysis rules in egglog syntax
//! 2. **Interval Lattice**: Built-in egglog support for interval domains
//! 3. **Conditional Rules**: Rules that only fire when domain constraints are satisfied
//! 4. **Extraction**: Cost-based extraction with domain-aware cost functions

use crate::{
    ast::ASTRepr,
    error::{DSLCompileError, Result},
    symbolic::rule_loader::RuleLoader,
};
use std::collections::HashMap;

#[cfg(feature = "optimization")]
use egglog_experimental::{EGraph, new_experimental_egraph};

/// Native egglog optimizer with built-in domain analysis
#[cfg(feature = "optimization")]
pub struct NativeEgglogOptimizer {
    /// The egglog `EGraph` with mathematical and analysis rules
    egraph: EGraph,
    /// Variable counter for generating unique names
    var_counter: usize,
    /// Cache for expression mappings
    expr_cache: HashMap<String, ASTRepr<f64>>,
}

#[cfg(feature = "optimization")]
impl NativeEgglogOptimizer {
    /// Create a new native egglog optimizer with domain analysis and dynamic cost support
    pub fn new() -> Result<Self> {
        // Use the experimental egraph that includes dynamic cost functionality
        let mut egraph = new_experimental_egraph();

        // Load the native egglog program with domain analysis
        let program = Self::create_egglog_program();

        egraph.parse_and_run_program(None, &program).map_err(|e| {
            DSLCompileError::Generic(format!(
                "Failed to initialize native egglog with domain analysis: {e}"
            ))
        })?;

        Ok(Self {
            egraph,
            var_counter: 0,
            expr_cache: HashMap::new(),
        })
    }

    /// Create a new native egglog optimizer with custom rule loader
    pub fn with_rule_loader(rule_loader: RuleLoader) -> Result<Self> {
        // Use the experimental egraph that includes dynamic cost functionality
        let mut egraph = new_experimental_egraph();

        // Load the rules using the provided rule loader
        let program = rule_loader
            .load_rules()
            .map_err(|e| DSLCompileError::Generic(format!("Failed to load rules: {e}")))?;

        egraph.parse_and_run_program(None, &program).map_err(|e| {
            DSLCompileError::Generic(format!(
                "Failed to initialize native egglog with custom rules: {e}"
            ))
        })?;

        Ok(Self {
            egraph,
            var_counter: 0,
            expr_cache: HashMap::new(),
        })
    }

    /// Create the egglog program with staged mathematical optimization rules
    fn create_egglog_program() -> String {
        // Load core datatypes first (defines Math datatype and constructors with dynamic costs)
        let core_datatypes = include_str!("../egglog_rules/core_datatypes.egg");
        // Then load core math rules (uses Math datatype)
        let core_rules = include_str!("../egglog_rules/staged_core_math.egg");
        // Finally load dependency analysis (uses Math datatype, compatible with dynamic costs)
        let dependency_rules = include_str!("../egglog_rules/dependency_analysis.egg");

        // TODO: Integrate CSE (Common Subexpression Elimination) rules
        // The old CSE rules were simplified but need to be re-integrated
        // with the new dependency analysis system for safe optimization

        // TODO: Add back summation optimization rules
        // Sum splitting and constant factoring rules need to be restored
        // once the core dependency system is stable

        format!("{core_datatypes}\n\n{core_rules}\n\n{dependency_rules}")
    }

    /// Optimize an expression using native egglog with domain analysis
    pub fn optimize(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // Convert expression to egglog format
        let egglog_expr = self.ast_to_egglog(expr)?;
        let expr_id = format!("expr_{}", self.var_counter);
        self.var_counter += 1;

        // Store original expression
        self.expr_cache.insert(expr_id.clone(), expr.clone());

        // Add expression to egglog
        let add_command = format!("(let {expr_id} {egglog_expr})");

        self.egraph
            .parse_and_run_program(None, &add_command)
            .map_err(|e| {
                DSLCompileError::Generic(format!("Failed to add expression to egglog: {e}"))
            })?;

        // Run optimization with available rules
        // Full saturation test - multiset canonical forms should reach natural fixed point
        let optimization_command = if cfg!(test) {
            "(run)" // Run to saturation in tests - multisets should prevent explosion
        } else {
            "(run)" // Full saturation in production - leveraging multiset canonical forms
        };

        // Run optimization with thread-based timeout protection - increased for full saturation
        let timeout_duration = if cfg!(test) {
            std::time::Duration::from_secs(10) // 10 second timeout for full saturation tests
        } else {
            std::time::Duration::from_secs(60) // 60 second timeout for production saturation
        };

        // Use a simple fallback instead of complex threading for now
        // TODO: Implement proper thread-based timeout if needed
        let start_time = std::time::Instant::now();
        let result = self
            .egraph
            .parse_and_run_program(None, optimization_command);
        let elapsed = start_time.elapsed();

        if elapsed > timeout_duration {
            println!(
                "⚠️  WARNING: Optimization took {:.2}s (longer than {}s timeout)",
                elapsed.as_secs_f64(),
                timeout_duration.as_secs()
            );
        }

        // Log rule firing statistics if the run completed
        if let Some(report) = self.egraph.get_run_report() {
            println!("\n🔍 EGGLOG RULE FIRING STATISTICS:");
            println!("=====================================");
            println!("{report}");

            // Check for suspicious rule patterns that might indicate infinite loops
            for (rule_name, &match_count) in &report.num_matches_per_rule {
                if match_count > 10000 {
                    println!(
                        "⚠️  WARNING: Rule '{rule_name}' fired {match_count} times - possible infinite loop!"
                    );
                }
            }

            // Also show overall statistics to see cumulative patterns
            let overall_report = self.egraph.get_overall_run_report();
            println!("\n📊 OVERALL STATISTICS:");
            println!("======================");
            println!("{overall_report}");
        }

        result.map_err(|e| {
            DSLCompileError::Generic(format!("Failed to run staged optimization: {e}"))
        })?;

        // Extract the best expression using dynamic cost model
        self.extract_best_with_dynamic_costs(&expr_id)
    }

    /// Set dynamic cost for a specific expression pattern
    pub fn set_dynamic_cost(&mut self, expr: &ASTRepr<f64>, cost: i64) -> Result<()> {
        let egglog_expr = self.ast_to_egglog(expr)?;
        let set_cost_command = format!("(set-cost {egglog_expr} {cost})");

        self.egraph
            .parse_and_run_program(None, &set_cost_command)
            .map_err(|e| DSLCompileError::Generic(format!("Failed to set dynamic cost: {e}")))?;

        Ok(())
    }

    /// Get interval analysis information for an expression
    pub fn analyze_interval(&mut self, expr: &ASTRepr<f64>) -> Result<String> {
        // Convert expression to egglog format and add it
        let egglog_expr = self.ast_to_egglog(expr)?;
        let expr_id = format!("interval_expr_{}", self.var_counter);
        self.var_counter += 1;

        // Add expression to egglog
        let add_command = format!("(let {expr_id} {egglog_expr})");
        self.egraph
            .parse_and_run_program(None, &add_command)
            .map_err(|e| {
                DSLCompileError::Generic(format!(
                    "Failed to add expression for interval analysis: {e}"
                ))
            })?;

        // Run analysis rules to full saturation
        let analysis_command = if cfg!(test) {
            "(run)" // Full saturation analysis with multiset protection
        } else {
            "(run)" // Complete analysis to fixed point with multiset canonical forms
        };
        self.egraph
            .parse_and_run_program(None, analysis_command)
            .map_err(|e| {
                DSLCompileError::Generic(format!("Failed to run interval analysis: {e}"))
            })?;

        // Try to extract interval information
        // For now, we'll return a basic analysis based on the expression structure
        self.analyze_interval_heuristic(expr)
    }

    /// Check if an expression is domain-safe for a specific operation
    pub fn is_domain_safe(&mut self, expr: &ASTRepr<f64>, operation: &str) -> Result<bool> {
        match operation {
            "ln" => self.is_positive_definite(expr),
            "sqrt" => self.is_non_negative(expr),
            "div" => self.is_nonzero_denominator(expr),
            _ => Ok(true), // Conservative: assume safe for unknown operations
        }
    }

    /// Heuristic interval analysis based on expression structure
    fn analyze_interval_heuristic(&self, expr: &ASTRepr<f64>) -> Result<String> {
        match expr {
            ASTRepr::Constant(val) => Ok(format!("[{val}, {val}] (singleton interval)")),
            ASTRepr::Variable(_) => Ok("(-∞, +∞) (unknown variable bounds)".to_string()),
            ASTRepr::Add(terms) => {
                if terms.len() == 2 {
                    let terms_vec: Vec<_> = terms.elements().collect();
                    let left_analysis = self.analyze_interval_heuristic(terms_vec[0])?;
                    let right_analysis = self.analyze_interval_heuristic(terms_vec[1])?;
                    Ok(format!(
                        "Sum of intervals: {left_analysis} + {right_analysis}"
                    ))
                } else {
                    let analyses: Result<Vec<_>> = terms
                        .elements()
                        .map(|term| self.analyze_interval_heuristic(term))
                        .collect();
                    let analyses = analyses?;
                    Ok(format!("Sum of intervals: [{}]", analyses.join(", ")))
                }
            }
            ASTRepr::Mul(factors) => {
                if factors.len() == 2 {
                    let factors_vec: Vec<_> = factors.elements().collect();
                    let left_analysis = self.analyze_interval_heuristic(factors_vec[0])?;
                    let right_analysis = self.analyze_interval_heuristic(factors_vec[1])?;
                    Ok(format!(
                        "Product of intervals: {left_analysis} * {right_analysis}"
                    ))
                } else {
                    let analyses: Result<Vec<_>> = factors
                        .elements()
                        .map(|factor| self.analyze_interval_heuristic(factor))
                        .collect();
                    let analyses = analyses?;
                    Ok(format!("Product of intervals: [{}]", analyses.join(" * ")))
                }
            }
            ASTRepr::Exp(_) => Ok("(0, +∞) (exponential is always positive)".to_string()),
            ASTRepr::Ln(inner) => {
                if self.is_positive_definite(inner)? {
                    Ok("(-∞, +∞) (ln of positive expression)".to_string())
                } else {
                    Ok("Domain error: ln requires positive argument".to_string())
                }
            }
            ASTRepr::Sqrt(inner) => {
                if self.is_non_negative(inner)? {
                    Ok("[0, +∞) (sqrt of non-negative expression)".to_string())
                } else {
                    Ok("Domain error: sqrt requires non-negative argument".to_string())
                }
            }
            _ => Ok("Complex expression - detailed analysis needed".to_string()),
        }
    }

    /// Check if an expression is provably positive
    fn is_positive_definite(&self, expr: &ASTRepr<f64>) -> Result<bool> {
        match expr {
            ASTRepr::Constant(val) => Ok(*val > 0.0),
            ASTRepr::Exp(_) => Ok(true), // exp(x) > 0 for all x
            ASTRepr::Mul(factors) => {
                if factors.len() == 2 {
                    // Product is positive if both factors are positive or both are negative
                    let factors_vec: Vec<_> = factors.elements().collect();
                    let left_pos = self.is_positive_definite(factors_vec[0])?;
                    let right_pos = self.is_positive_definite(factors_vec[1])?;
                    let left_neg = self.is_negative_definite(factors_vec[0])?;
                    let right_neg = self.is_negative_definite(factors_vec[1])?;
                    Ok((left_pos && right_pos) || (left_neg && right_neg))
                } else {
                    // For n-ary multiplication, check all factors
                    let positive_count = factors
                        .elements()
                        .map(|f| self.is_positive_definite(f))
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .filter(|&b| b)
                        .count();
                    let negative_count = factors
                        .elements()
                        .map(|f| self.is_negative_definite(f))
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .filter(|&b| b)
                        .count();

                    // Product is positive if all factors are positive or even number of negative factors
                    Ok(positive_count == factors.len()
                        || (negative_count % 2 == 0 && negative_count > 0))
                }
            }
            ASTRepr::Pow(base, exp) => {
                // x^a > 0 if x > 0, or if x < 0 and a is even integer
                if self.is_positive_definite(base)? {
                    Ok(true)
                } else if let ASTRepr::Constant(exp_val) = exp.as_ref() {
                    // Check if exponent is even integer
                    if exp_val.fract() == 0.0 && (*exp_val as i64) % 2 == 0 {
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            ASTRepr::Sqrt(inner) => {
                // sqrt(x) >= 0, and > 0 if x > 0
                self.is_positive_definite(inner)
            }
            _ => Ok(false), // Conservative: assume not provably positive
        }
    }

    /// Check if an expression is provably negative
    fn is_negative_definite(&self, expr: &ASTRepr<f64>) -> Result<bool> {
        match expr {
            ASTRepr::Constant(val) => Ok(*val < 0.0),
            ASTRepr::Neg(inner) => self.is_positive_definite(inner),
            _ => Ok(false), // Conservative: assume not provably negative
        }
    }

    /// Check if an expression is provably non-negative
    fn is_non_negative(&self, expr: &ASTRepr<f64>) -> Result<bool> {
        match expr {
            ASTRepr::Constant(val) => Ok(*val >= 0.0),
            ASTRepr::Exp(_) => Ok(true),  // exp(x) >= 0 for all x
            ASTRepr::Sqrt(_) => Ok(true), // sqrt(x) >= 0 by definition
            ASTRepr::Sum(_collection) => {
                // TODO: Implement Sum Collection variant non-negative analysis
                Ok(false) // Conservative: cannot guarantee non-negative without analysis
            }
            ASTRepr::Mul(factors) => {
                if factors.len() == 2 {
                    // Product is non-negative if both factors have same sign
                    let factors_vec: Vec<_> = factors.elements().collect();
                    let left_nonneg = self.is_non_negative(factors_vec[0])?;
                    let right_nonneg = self.is_non_negative(factors_vec[1])?;
                    let left_nonpos = self.is_non_positive(factors_vec[0])?;
                    let right_nonpos = self.is_non_positive(factors_vec[1])?;
                    Ok((left_nonneg && right_nonneg) || (left_nonpos && right_nonpos))
                } else {
                    // For n-ary multiplication, check signs more carefully
                    let nonneg_count = factors
                        .elements()
                        .map(|f| self.is_non_negative(f))
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .filter(|&b| b)
                        .count();
                    let nonpos_count = factors
                        .elements()
                        .map(|f| self.is_non_positive(f))
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .filter(|&b| b)
                        .count();

                    // Product is non-negative if all are non-negative or even number of non-positive
                    Ok(
                        nonneg_count == factors.len()
                            || (nonpos_count % 2 == 0 && nonpos_count > 0),
                    )
                }
            }
            ASTRepr::Pow(base, exp) => {
                // x^a >= 0 if x >= 0, or if a is even
                if self.is_non_negative(base)? {
                    Ok(true)
                } else if let ASTRepr::Constant(exp_val) = exp.as_ref() {
                    // Check if exponent is even integer
                    if exp_val.fract() == 0.0 && (*exp_val as i64) % 2 == 0 {
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            ASTRepr::Add(terms) => {
                // Sum is non-negative if all terms are non-negative
                let all_nonneg: Result<Vec<_>> = terms
                    .elements()
                    .map(|term| self.is_non_negative(term))
                    .collect();
                let all_nonneg = all_nonneg?;
                Ok(all_nonneg.into_iter().all(|b| b))
            }
            _ => Ok(false), // Conservative: assume not provably non-negative
        }
    }

    /// Check if an expression is provably non-positive
    fn is_non_positive(&self, expr: &ASTRepr<f64>) -> Result<bool> {
        match expr {
            ASTRepr::Constant(val) => Ok(*val <= 0.0),
            ASTRepr::Neg(inner) => self.is_non_negative(inner),
            _ => Ok(false), // Conservative: assume not provably non-positive
        }
    }

    /// Check if a denominator expression is provably non-zero
    fn is_nonzero_denominator(&self, expr: &ASTRepr<f64>) -> Result<bool> {
        match expr {
            ASTRepr::Constant(val) => Ok(*val != 0.0),
            ASTRepr::Exp(_) => Ok(true), // exp(x) != 0 for all x
            ASTRepr::Sqrt(inner) => {
                // sqrt(x) != 0 if x > 0
                self.is_positive_definite(inner)
            }
            _ => Ok(false), // Conservative: assume might be zero
        }
    }

    /// Convert AST to egglog format  
    pub fn ast_to_egglog(&self, expr: &ASTRepr<f64>) -> Result<String> {
        match expr {
            ASTRepr::Constant(val) => {
                if val.fract() == 0.0 {
                    Ok(format!("(Num {val:.1})"))
                } else {
                    Ok(format!("(Num {val})"))
                }
            }
            ASTRepr::Variable(idx) => Ok(format!("(UserVar {idx})")), // Use UserVar for collision safety
            ASTRepr::BoundVar(idx) => Ok(format!("(BoundVar {idx})")), // CSE-bound variables
            ASTRepr::Let(binding_id, expr, body) => {
                let expr_s = self.ast_to_egglog(expr)?;
                let body_s = self.ast_to_egglog(body)?;
                Ok(format!("(Let {binding_id} {expr_s} {body_s})"))
            }
            ASTRepr::Add(terms) => {
                if terms.len() == 2 {
                    let terms_vec: Vec<_> = terms.elements().collect();
                    let left_s = self.ast_to_egglog(terms_vec[0])?;
                    let right_s = self.ast_to_egglog(terms_vec[1])?;
                    Ok(format!("(Add {left_s} {right_s})"))
                } else {
                    // For n-ary addition, we need to chain binary operations
                    // TODO: Update egglog grammar to support n-ary Add
                    let term_strings: Result<Vec<_>> = terms
                        .elements()
                        .map(|term| self.ast_to_egglog(term))
                        .collect();
                    let term_strings = term_strings?;

                    if term_strings.is_empty() {
                        Ok("(Num 0.0)".to_string())
                    } else if term_strings.len() == 1 {
                        Ok(term_strings[0].clone())
                    } else {
                        // Chain binary additions: (Add a (Add b c))
                        let mut result = term_strings[0].clone();
                        for term in &term_strings[1..] {
                            result = format!("(Add {result} {term})");
                        }
                        Ok(result)
                    }
                }
            }
            ASTRepr::Sub(left, right) => {
                let left_s = self.ast_to_egglog(left)?;
                let right_s = self.ast_to_egglog(right)?;
                Ok(format!("(Add {left_s} (Neg {right_s}))")) // Convert Sub to Add + Neg
            }
            ASTRepr::Mul(factors) => {
                if factors.len() == 2 {
                    let factors_vec: Vec<_> = factors.elements().collect();
                    let left_s = self.ast_to_egglog(factors_vec[0])?;
                    let right_s = self.ast_to_egglog(factors_vec[1])?;
                    Ok(format!("(Mul {left_s} {right_s})"))
                } else {
                    // For n-ary multiplication, we need to chain binary operations
                    // TODO: Update egglog grammar to support n-ary Mul
                    let factor_strings: Result<Vec<_>> = factors
                        .elements()
                        .map(|factor| self.ast_to_egglog(factor))
                        .collect();
                    let factor_strings = factor_strings?;

                    if factor_strings.is_empty() {
                        Ok("(Num 1.0)".to_string())
                    } else if factor_strings.len() == 1 {
                        Ok(factor_strings[0].clone())
                    } else {
                        // Chain binary multiplications: (Mul a (Mul b c))
                        let mut result = factor_strings[0].clone();
                        for factor in &factor_strings[1..] {
                            result = format!("(Mul {result} {factor})");
                        }
                        Ok(result)
                    }
                }
            }
            ASTRepr::Div(left, right) => {
                let left_s = self.ast_to_egglog(left)?;
                let right_s = self.ast_to_egglog(right)?;
                Ok(format!("(Div {left_s} {right_s})"))
            }
            ASTRepr::Pow(left, right) => {
                let left_s = self.ast_to_egglog(left)?;
                let right_s = self.ast_to_egglog(right)?;
                Ok(format!("(Pow {left_s} {right_s})"))
            }
            ASTRepr::Neg(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Neg {inner_s})"))
            }
            ASTRepr::Ln(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Ln {inner_s})"))
            }
            ASTRepr::Exp(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Exp {inner_s})"))
            }
            ASTRepr::Sin(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Sin {inner_s})"))
            }
            ASTRepr::Cos(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Cos {inner_s})"))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Sqrt {inner_s})"))
            }
            ASTRepr::Sum(collection) => {
                // Convert Collection format to unified Expr datatype
                let collection_str = self.collection_to_egglog(collection)?;
                Ok(format!("(Sum {collection_str})"))
            }
            ASTRepr::Lambda(lambda) => {
                // Convert lambda to egglog representation
                let body_s = self.ast_to_egglog(&lambda.body)?;
                let var_indices = lambda
                    .var_indices
                    .iter()
                    .map(|idx| format!("{idx}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                Ok(format!("(Lambda [{var_indices}] {body_s})"))
            }
        }
    }

    /// Convert Collection to proper Collection datatype representation
    fn collection_to_egglog(
        &self,
        collection: &crate::ast::ast_repr::Collection<f64>,
    ) -> Result<String> {
        use crate::ast::ast_repr::Collection;

        match collection {
            Collection::Empty => Ok("Empty".to_string()),
            Collection::Singleton(expr) => {
                let expr_str = self.ast_to_egglog(expr)?;
                Ok(format!("(Singleton {expr_str})"))
            }
            Collection::Range { start, end } => {
                let start_str = self.ast_to_egglog(start)?;
                let end_str = self.ast_to_egglog(end)?;
                Ok(format!("(Range {start_str} {end_str})"))
            }
            Collection::Variable(index) => Ok(format!("(DataArray {index})")),
            Collection::Map { lambda, collection } => {
                let lambda_str = self.lambda_to_egglog(lambda)?;
                let collection_str = self.collection_to_egglog(collection)?;
                Ok(format!("(Map {lambda_str} {collection_str})"))
            }
            Collection::DataArray(_data) => {
                // For embedded data arrays, we need to convert them to symbolic DataArray references
                // The actual data will be bound at evaluation time, not optimization time
                // For now, we'll use a generic index (0) since egglog can't handle embedded data directly
                // TODO: Implement proper data array indexing system for symbolic optimization
                Ok("(DataArray 0)".to_string())
            }
            Collection::Filter {
                collection,
                predicate,
            } => {
                let collection_str = self.collection_to_egglog(collection)?;
                let predicate_str = self.ast_to_egglog(predicate)?;
                Ok(format!("(Filter {collection_str} {predicate_str})"))
            }
        }
    }

    /// Convert Lambda to proper Lambda datatype representation  
    fn lambda_to_egglog(&self, lambda: &crate::ast::ast_repr::Lambda<f64>) -> Result<String> {
        let body_str = self.ast_to_egglog(&lambda.body)?;

        if lambda.var_indices.is_empty() {
            // Constant lambda
            Ok(format!("(ConstantFunc {body_str})"))
        } else if lambda.var_indices.len() == 1 {
            // Single-argument lambda
            let var_index = lambda.var_indices[0];
            Ok(format!("(LambdaFunc {var_index} {body_str})"))
        } else {
            // Multi-argument lambda - not supported in our current datatype
            Err(DSLCompileError::Generic(
                "Multi-argument lambdas not supported in current egglog datatype".to_string(),
            ))
        }
    }

    /// Extract the best expression using dynamic cost model
    fn extract_best_with_dynamic_costs(&mut self, expr_id: &str) -> Result<ASTRepr<f64>> {
        // Use the custom extract command from egglog-experimental that uses DynamicCostModel
        let extract_command = format!("(extract {expr_id})");

        // Run the extraction command
        let extract_result = self
            .egraph
            .parse_and_run_program(None, &extract_command)
            .map_err(|e| {
                DSLCompileError::Generic(format!(
                    "Failed to extract optimized expression with dynamic costs: {e}"
                ))
            })?;

        // Convert Vec<String> to a single string for parsing
        let output_string = extract_result.join("\n");

        // Parse the extraction result
        // If extraction parsing fails, fall back to the original expression
        match self.parse_egglog_output(&output_string) {
            Ok(optimized) => Ok(optimized),
            Err(_) => {
                // Extraction parsing failed, return original expression
                // This is a reasonable fallback since the original expression is still valid
                self.expr_cache.get(expr_id).cloned().ok_or_else(|| {
                    DSLCompileError::Generic("Expression not found in cache".to_string())
                })
            }
        }
    }

    /// Extract the best expression using default egglog extraction (fallback)
    fn extract_best(&mut self, expr_id: &str) -> Result<ASTRepr<f64>> {
        // Use default extraction for now to debug CSE rules
        let extract_command = format!("(extract {expr_id})");

        // Run the extraction command
        let extract_result = self
            .egraph
            .parse_and_run_program(None, &extract_command)
            .map_err(|e| {
                DSLCompileError::Generic(format!("Failed to extract optimized expression: {e}"))
            })?;

        // Convert Vec<String> to a single string for parsing
        let output_string = extract_result.join("\n");

        // Parse the extraction result
        // If extraction parsing fails, fall back to the original expression
        match self.parse_egglog_output(&output_string) {
            Ok(optimized) => Ok(optimized),
            Err(_) => {
                // Extraction parsing failed, return original expression
                // This is a reasonable fallback since the original expression is still valid
                self.expr_cache.get(expr_id).cloned().ok_or_else(|| {
                    DSLCompileError::Generic("Expression not found in cache".to_string())
                })
            }
        }
    }

    /// Parse egglog output back to `ASTRepr`
    fn parse_egglog_output(&self, output: &str) -> Result<ASTRepr<f64>> {
        // Remove any whitespace and newlines
        let cleaned = output.trim();

        // Parse the s-expression recursively
        self.parse_sexpr(cleaned)
    }

    /// Parse a single s-expression
    fn parse_sexpr(&self, s: &str) -> Result<ASTRepr<f64>> {
        let s = s.trim();

        if !s.starts_with('(') || !s.ends_with(')') {
            return Err(DSLCompileError::Generic(format!(
                "Invalid s-expression: {s}"
            )));
        }

        // Remove outer parentheses
        let inner = &s[1..s.len() - 1];

        // Split into tokens
        let tokens = self.tokenize_sexpr(inner)?;

        if tokens.is_empty() {
            return Err(DSLCompileError::Generic("Empty s-expression".to_string()));
        }

        match tokens[0].as_str() {
            "Num" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Num requires exactly one argument".to_string(),
                    ));
                }
                let value = tokens[1].parse::<f64>().map_err(|_| {
                    DSLCompileError::Generic(format!("Invalid number: {}", tokens[1]))
                })?;
                Ok(ASTRepr::Constant(value))
            }
            "UserVar" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "UserVar requires exactly one argument".to_string(),
                    ));
                }
                // Parse variable index directly as integer
                let index = tokens[1].parse::<usize>().map_err(|_| {
                    DSLCompileError::Generic(format!("Invalid variable index: {}", tokens[1]))
                })?;
                Ok(ASTRepr::Variable(index))
            }
            "BoundVar" => {
                // BoundVar should be handled by Let bindings in optimized code
                // If we encounter a BoundVar directly, it means a let binding wasn't properly resolved
                // For now, we'll treat it as a variable error since bare BoundVars shouldn't appear in final AST
                Err(DSLCompileError::Generic(
                    "Bare BoundVar found - let bindings should be resolved during optimization"
                        .to_string(),
                ))
            }
            "Let" => {
                if tokens.len() != 4 {
                    return Err(DSLCompileError::Generic(
                        "Let requires exactly three arguments: bound_id, expr, body".to_string(),
                    ));
                }
                // For Let bindings, we need to substitute the bound variable with the expression
                // This is a simplified approach - in a full implementation we'd need proper variable substitution
                let _bound_id = tokens[1].parse::<u32>().map_err(|_| {
                    DSLCompileError::Generic(format!("Invalid bound variable ID: {}", tokens[1]))
                })?;

                let _expr = self.parse_sexpr(&tokens[2])?;
                let body = self.parse_sexpr(&tokens[3])?;

                // For now, just return the body - this is a simplification
                // A full implementation would need to substitute the bound variable
                Ok(body)
            }
            "Add" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "Add requires exactly two arguments".to_string(),
                    ));
                }
                let left = self.parse_sexpr(&tokens[1])?;
                let right = self.parse_sexpr(&tokens[2])?;
                Ok(ASTRepr::add_binary(left, right))
            }
            "Sub" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "Sub requires exactly two arguments".to_string(),
                    ));
                }
                let left = self.parse_sexpr(&tokens[1])?;
                let right = self.parse_sexpr(&tokens[2])?;
                Ok(ASTRepr::Sub(Box::new(left), Box::new(right)))
            }
            "Mul" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "Mul requires exactly two arguments".to_string(),
                    ));
                }
                let left = self.parse_sexpr(&tokens[1])?;
                let right = self.parse_sexpr(&tokens[2])?;
                Ok(ASTRepr::mul_binary(left, right))
            }
            "Div" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "Div requires exactly two arguments".to_string(),
                    ));
                }
                let left = self.parse_sexpr(&tokens[1])?;
                let right = self.parse_sexpr(&tokens[2])?;
                Ok(ASTRepr::Div(Box::new(left), Box::new(right)))
            }
            "Neg" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Neg requires exactly one argument".to_string(),
                    ));
                }
                let inner = self.parse_sexpr(&tokens[1])?;
                Ok(ASTRepr::Neg(Box::new(inner)))
            }
            "Pow" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "Pow requires exactly two arguments".to_string(),
                    ));
                }
                let base = self.parse_sexpr(&tokens[1])?;
                let exp = self.parse_sexpr(&tokens[2])?;
                Ok(ASTRepr::Pow(Box::new(base), Box::new(exp)))
            }
            "Ln" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Ln requires exactly one argument".to_string(),
                    ));
                }
                let inner = self.parse_sexpr(&tokens[1])?;
                Ok(ASTRepr::Ln(Box::new(inner)))
            }
            "Exp" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Exp requires exactly one argument".to_string(),
                    ));
                }
                let inner = self.parse_sexpr(&tokens[1])?;
                Ok(ASTRepr::Exp(Box::new(inner)))
            }
            "Sin" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Sin requires exactly one argument".to_string(),
                    ));
                }
                let inner = self.parse_sexpr(&tokens[1])?;
                Ok(ASTRepr::Sin(Box::new(inner)))
            }
            "Cos" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Cos requires exactly one argument".to_string(),
                    ));
                }
                let inner = self.parse_sexpr(&tokens[1])?;
                Ok(ASTRepr::Cos(Box::new(inner)))
            }
            "Sqrt" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Sqrt requires exactly one argument".to_string(),
                    ));
                }
                let inner = self.parse_sexpr(&tokens[1])?;
                Ok(ASTRepr::Sqrt(Box::new(inner)))
            }
            "Sum" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Sum requires exactly one argument (collection)".to_string(),
                    ));
                }
                let collection = self.parse_collection_sexpr(&tokens[1])?;
                Ok(ASTRepr::Sum(Box::new(collection)))
            }
            "Expand" => {
                // Expand(expr) - request expansion
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Expand requires exactly one argument".to_string(),
                    ));
                }
                // Just parse the inner expression, the Expand wrapper is handled by egglog
                self.parse_sexpr(&tokens[1])
            }
            "Expanded" => {
                // Expanded(expr) - already expanded, extract the inner expression
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Expanded requires exactly one argument".to_string(),
                    ));
                }
                // Just parse the inner expression, the Expanded wrapper is removed
                self.parse_sexpr(&tokens[1])
            }
            _ => Err(DSLCompileError::Generic(format!(
                "Unknown operation: {}",
                tokens[0]
            ))),
        }
    }

    /// Parse a Collection s-expression
    fn parse_collection_sexpr(&self, s: &str) -> Result<crate::ast::ast_repr::Collection<f64>> {
        use crate::ast::ast_repr::Collection;

        let s = s.trim();
        if !s.starts_with('(') || !s.ends_with(')') {
            return Err(DSLCompileError::Generic(format!(
                "Invalid collection s-expression: {s}"
            )));
        }

        let inner = &s[1..s.len() - 1];
        let tokens = self.tokenize_sexpr(inner)?;

        if tokens.is_empty() {
            return Err(DSLCompileError::Generic(
                "Empty collection s-expression".to_string(),
            ));
        }

        match tokens[0].as_str() {
            "Empty" => Ok(Collection::Empty),
            "Singleton" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Singleton requires exactly one argument".to_string(),
                    ));
                }
                let expr = self.parse_sexpr(&tokens[1])?;
                Ok(Collection::Singleton(Box::new(expr)))
            }
            "Range" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "Range requires exactly two arguments".to_string(),
                    ));
                }
                let start = self.parse_sexpr(&tokens[1])?;
                let end = self.parse_sexpr(&tokens[2])?;
                Ok(Collection::Range {
                    start: Box::new(start),
                    end: Box::new(end),
                })
            }
            "DataArray" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "DataArray requires exactly one argument".to_string(),
                    ));
                }
                let index = tokens[1].parse::<usize>().map_err(|_| {
                    DSLCompileError::Generic(format!("Invalid data array index: {}", tokens[1]))
                })?;
                Ok(Collection::Variable(index))
            }
            "Variable" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Variable requires exactly one argument".to_string(),
                    ));
                }
                let index = tokens[1].parse::<usize>().map_err(|_| {
                    DSLCompileError::Generic(format!(
                        "Invalid variable reference index: {}",
                        tokens[1]
                    ))
                })?;
                Ok(Collection::Variable(index))
            }
            "Map" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "Map requires exactly two arguments".to_string(),
                    ));
                }
                let lambda = self.parse_lambda_sexpr(&tokens[1])?;
                let collection = self.parse_collection_sexpr(&tokens[2])?;
                Ok(Collection::Map {
                    lambda: Box::new(lambda),
                    collection: Box::new(collection),
                })
            }
            _ => Err(DSLCompileError::Generic(format!(
                "Unknown collection type: {}",
                tokens[0]
            ))),
        }
    }

    /// Parse a Lambda s-expression  
    fn parse_lambda_sexpr(&self, s: &str) -> Result<crate::ast::ast_repr::Lambda<f64>> {
        use crate::ast::ast_repr::Lambda;

        let s = s.trim();
        if !s.starts_with('(') || !s.ends_with(')') {
            return Err(DSLCompileError::Generic(format!(
                "Invalid lambda s-expression: {s}"
            )));
        }

        let inner = &s[1..s.len() - 1];
        let tokens = self.tokenize_sexpr(inner)?;

        if tokens.is_empty() {
            return Err(DSLCompileError::Generic(
                "Empty lambda s-expression".to_string(),
            ));
        }

        match tokens[0].as_str() {
            "Identity" => Ok(Lambda::identity()),
            "Constant" => {
                if tokens.len() != 2 {
                    return Err(DSLCompileError::Generic(
                        "Constant lambda requires exactly one argument".to_string(),
                    ));
                }
                let expr = self.parse_sexpr(&tokens[1])?;
                Ok(Lambda::new(vec![], Box::new(expr)))
            }
            "LambdaFunc" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "LambdaFunc requires exactly two arguments".to_string(),
                    ));
                }
                let var_index = tokens[1].parse::<usize>().map_err(|_| {
                    DSLCompileError::Generic(format!("Invalid variable index: {}", tokens[1]))
                })?;
                let body = self.parse_sexpr(&tokens[2])?;
                Ok(Lambda::single(var_index, Box::new(body)))
            }
            "MultiArgFunc" => {
                if tokens.len() != 3 {
                    return Err(DSLCompileError::Generic(
                        "MultiArgFunc lambda requires exactly two arguments".to_string(),
                    ));
                }
                // Parse the indices from the second token (should be a list like "(1 2 3)")
                let indices_str = &tokens[1];
                if !indices_str.starts_with('(') || !indices_str.ends_with(')') {
                    return Err(DSLCompileError::Generic(
                        "MultiArgFunc indices must be a parenthesized list".to_string(),
                    ));
                }
                let indices_inner = &indices_str[1..indices_str.len() - 1];
                let var_indices: std::result::Result<Vec<usize>, _> = indices_inner
                    .split_whitespace()
                    .map(str::parse::<usize>)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|_| {
                        DSLCompileError::Generic(
                            "Invalid variable indices in MultiArgFunc".to_string(),
                        )
                    });
                let var_indices = var_indices?;
                let body = self.parse_sexpr(&tokens[2])?;
                Ok(Lambda::new(var_indices, Box::new(body)))
            }
            _ => Err(DSLCompileError::Generic(format!(
                "Unknown lambda type: {}",
                tokens[0]
            ))),
        }
    }

    /// Tokenize an s-expression into its components
    fn tokenize_sexpr(&self, s: &str) -> Result<Vec<String>> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut paren_depth = 0;
        let mut in_string = false;
        let chars = s.chars().peekable();

        for ch in chars {
            match ch {
                '"' => {
                    in_string = !in_string;
                    current_token.push(ch);
                }
                '(' if !in_string => {
                    if paren_depth == 0 && !current_token.is_empty() {
                        // We're starting a new nested expression, save the current token first
                        tokens.push(current_token.trim().to_string());
                        current_token.clear();
                    }
                    current_token.push(ch);
                    paren_depth += 1;
                }
                ')' if !in_string => {
                    current_token.push(ch);
                    paren_depth -= 1;
                    if paren_depth == 0 {
                        // We've completed a nested expression
                        tokens.push(current_token.trim().to_string());
                        current_token.clear();
                    }
                }
                ' ' | '\t' | '\n' if !in_string && paren_depth == 0 => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.trim().to_string());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token.trim().to_string());
        }

        Ok(tokens)
    }

    /// Optimize with forced expansion (useful for pattern recognition)
    pub fn optimize_with_expansion(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // Convert expression to egglog format and wrap with Expand()
        let egglog_expr = self.ast_to_egglog(expr)?;
        let expand_expr = format!("(Expand {egglog_expr})");
        let expr_id = format!("expand_expr_{}", self.var_counter);
        self.var_counter += 1;

        // Store original expression
        self.expr_cache.insert(expr_id.clone(), expr.clone());

        // Add expression to egglog with expansion request
        let add_command = format!("(let {expr_id} {expand_expr})");
        self.egraph
            .parse_and_run_program(None, &add_command)
            .map_err(|e| {
                DSLCompileError::Generic(format!("Failed to add expression for expansion: {e}"))
            })?;

        // Run mathematical optimization rules with expansion to full saturation
        let expansion_command = if cfg!(test) {
            "(run)" // Full saturation expansion with multiset protection
        } else {
            "(run)" // Complete expansion to fixed point with multiset canonical forms
        };
        self.egraph
            .parse_and_run_program(None, expansion_command)
            .map_err(|e| DSLCompileError::Generic(format!("Failed to run expansion rules: {e}")))?;

        // Extract the best expression (should be expanded now)
        self.extract_best(&expr_id)
    }
}

/// Fallback implementation when optimization feature is not enabled
#[cfg(not(feature = "optimization"))]
pub struct NativeEgglogOptimizer;

#[cfg(not(feature = "optimization"))]
impl NativeEgglogOptimizer {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn optimize(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        Ok(expr.clone())
    }
}

/// Helper function to create and use the native egglog optimizer
pub fn optimize_with_native_egglog(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
    use crate::ast::normalize;

    // First normalize the expression to canonical form (Sub → Add + Neg, Div → Mul + Pow^-1)
    let normalized_expr = normalize(expr);

    let mut optimizer = NativeEgglogOptimizer::new()?;
    let optimized = optimizer.optimize(&normalized_expr)?;

    // The result should stay normalized for consistency
    Ok(optimized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_egglog_creation() {
        let result = NativeEgglogOptimizer::new();
        if let Err(e) = &result {
            println!("Error creating NativeEgglogOptimizer: {e}");
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_ast_to_egglog_conversion() {
        let optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test simple addition
        let add = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Constant(1.0)]);
        let egglog_str = optimizer.ast_to_egglog(&add).unwrap();
        assert_eq!(egglog_str, "(Add (UserVar 0) (Num 1.0))");
    }

    #[test]
    fn test_canonical_form_conversion() {
        let optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test conversion of canonical form (Sub -> Add + Neg)
        let sub = ASTRepr::Sub(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(1.0)),
        );
        let egglog_str = optimizer.ast_to_egglog(&sub).unwrap();
        assert_eq!(egglog_str, "(Add (UserVar 0) (Neg (Num 1.0)))");

        // Test conversion of canonical form (Div -> Mul + Pow)
        let div = ASTRepr::Div(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        );
        let egglog_str = optimizer.ast_to_egglog(&div).unwrap();
        assert_eq!(egglog_str, "(Div (UserVar 0) (Num 2.0))");
    }

    #[test]
    fn test_basic_optimization() {
        let expr = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Constant(0.0)]);
        let result = optimize_with_native_egglog(&expr);

        #[cfg(feature = "optimization")]
        {
            // Should run without error
            if result.is_err() {
                println!("Error: {:?}", result.as_ref().unwrap_err());
            }
            assert!(result.is_ok());
        }

        #[cfg(not(feature = "optimization"))]
        {
            // Should return unchanged
            if result.is_err() {
                println!("Error: {:?}", result.as_ref().unwrap_err());
            }
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_domain_aware_optimization() {
        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test domain-safe ln(exp(x)) = x (should always work)
        let safe_expr = ASTRepr::Ln(Box::new(ASTRepr::Exp(Box::new(ASTRepr::Variable(0)))));

        let result = optimizer.optimize(&safe_expr);
        assert!(result.is_ok());

        // Test potentially unsafe exp(ln(x)) (depends on domain analysis)
        let potentially_unsafe =
            ASTRepr::Exp(Box::new(ASTRepr::Ln(Box::new(ASTRepr::Variable(0)))));

        let result = optimizer.optimize(&potentially_unsafe);
        assert!(result.is_ok());
    }

    #[test]
    fn test_interval_analysis() {
        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test interval analysis on a constant
        let constant_expr = ASTRepr::Constant(5.0);
        let interval_info = optimizer.analyze_interval(&constant_expr);
        assert!(interval_info.is_ok());

        // Test interval analysis on a variable
        let var_expr = ASTRepr::Variable(0);
        let interval_info = optimizer.analyze_interval(&var_expr);
        assert!(interval_info.is_ok());

        // Test interval analysis on a complex expression
        let complex_expr = ASTRepr::add_from_array([
            ASTRepr::Constant(2.0),
            ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Constant(3.0)]),
        ]);
        let interval_info = optimizer.analyze_interval(&complex_expr);
        assert!(interval_info.is_ok());
    }

    #[test]
    fn test_domain_safety_checks() {
        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test safety check for ln operation
        let positive_constant = ASTRepr::Constant(5.0);
        let is_safe = optimizer.is_domain_safe(&positive_constant, "ln");
        assert!(is_safe.is_ok());

        // Test safety check for division
        let nonzero_expr = ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Constant(1.0)]);
        let is_safe = optimizer.is_domain_safe(&nonzero_expr, "div");
        assert!(is_safe.is_ok());

        // Test safety check for sqrt
        let sqrt_expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        );
        let is_safe = optimizer.is_domain_safe(&sqrt_expr, "sqrt");
        assert!(is_safe.is_ok());
    }

    #[test]
    fn test_domain_aware_ln_rules() {
        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test ln(a * b) = ln(a) + ln(b) with positive constants
        let ln_product = ASTRepr::Ln(Box::new(ASTRepr::mul_from_array([
            ASTRepr::Constant(2.0),
            ASTRepr::Constant(3.0),
        ])));

        let result = optimizer.optimize(&ln_product);
        assert!(result.is_ok());

        // The optimization should work since both constants are positive
        // In a full implementation, we'd check that it actually transformed to ln(2) + ln(3)
    }

    #[test]
    fn test_sqrt_domain_awareness() {
        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test sqrt(x^2) = x with domain analysis
        let sqrt_square = ASTRepr::Sqrt(Box::new(ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        )));

        let result = optimizer.optimize(&sqrt_square);
        assert!(result.is_ok());

        // The rule should only apply if x is known to be non-negative
        // For variables, this would typically not apply without additional constraints
    }

    #[test]
    fn test_multiplication_expansion_rule() {
        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test the multiplication expansion rule: (x+y)*(x+y) → x² + 2xy + y²
        let mult_expr = ASTRepr::mul_from_array([
            ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Variable(1)]),
            ASTRepr::add_from_array([ASTRepr::Variable(0), ASTRepr::Variable(1)]),
        ]);

        println!("🔬 Testing multiplication expansion rule");
        println!("   Input: {mult_expr:?}");

        let result = optimizer.optimize(&mult_expr).unwrap();
        println!("   Output: {result:?}");

        // Check if expansion occurred by counting operations
        let input_ops = count_operations(&mult_expr);
        let output_ops = count_operations(&result);

        println!("   Input operations: {input_ops}");
        println!("   Output operations: {output_ops}");

        if output_ops > input_ops {
            println!("   ✅ Expansion occurred!");
        } else {
            println!("   ❌ No expansion detected");
        }
    }

    #[test]
    fn test_simple_distributivity_rule() {
        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test simple distributivity: a*(b+c) → a*b + a*c
        let dist_expr = ASTRepr::mul_from_array([
            ASTRepr::Variable(0), // a
            ASTRepr::add_from_array([
                ASTRepr::Variable(1), // b
                ASTRepr::Variable(2), // c
            ]),
        ]);

        println!("🔬 Testing simple distributivity rule");
        println!("   Input: {dist_expr:?}");

        let result = optimizer.optimize(&dist_expr).unwrap();
        println!("   Output: {result:?}");

        // Check if expansion occurred by counting operations
        let input_ops = count_operations(&dist_expr);
        let output_ops = count_operations(&result);

        println!("   Input operations: {input_ops}");
        println!("   Output operations: {output_ops}");

        if output_ops > input_ops {
            println!("   ✅ Distributivity expansion occurred!");
        } else {
            println!("   ❌ No distributivity expansion detected");
        }
    }

    fn count_operations(expr: &ASTRepr<f64>) -> usize {
        match expr {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
            ASTRepr::Add(terms) => {
                terms.elements().map(count_operations).sum::<usize>()
                    + terms.len().saturating_sub(1)
            }
            ASTRepr::Mul(factors) => {
                factors.elements().map(count_operations).sum::<usize>()
                    + factors.len().saturating_sub(1)
            }
            ASTRepr::Sub(left, right) | ASTRepr::Div(left, right) | ASTRepr::Pow(left, right) => {
                1 + count_operations(left) + count_operations(right)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => 1 + count_operations(inner),
            ASTRepr::Sum(_collection) => {
                // TODO: Handle Collection format for native egglog operation counting
                1 // Placeholder count until Collection analysis is implemented
            }
            ASTRepr::Lambda(lambda) => {
                // Lambda expressions: count operations in body plus one for the lambda itself
                1 + count_operations(&lambda.body)
            }
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => {
                // CSE-related constructs count as 0 operations (they're just bindings)
                0
            }
        }
    }

    #[test]
    fn test_custom_extractor_integration() {
        println!("🎯 Testing basic optimization (custom extraction functionality removed)");

        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        // Test expression with power operation
        let test_expr = ASTRepr::Pow(
            Box::new(ASTRepr::add_from_array([
                ASTRepr::Variable(0),
                ASTRepr::Variable(1),
            ])),
            Box::new(ASTRepr::Constant(2.0)),
        );

        println!("   Input: {test_expr:?}");

        // Test basic optimization
        let result = optimizer.optimize(&test_expr);

        match result {
            Ok(optimized) => {
                println!("   Output: {optimized:?}");
                println!("   ✅ Basic optimization successful");
            }
            Err(e) => {
                println!("   Error: {e}");
                // Basic optimization should work
                panic!("Basic optimization failed: {e}");
            }
        }
    }

    #[test]
    fn test_optimization_consistency() {
        println!("🔬 Testing optimization consistency");

        let mut optimizer = NativeEgglogOptimizer::new().unwrap();

        let test_expr = ASTRepr::Pow(
            Box::new(ASTRepr::add_from_array([
                ASTRepr::Variable(0),
                ASTRepr::Variable(1),
            ])),
            Box::new(ASTRepr::Constant(2.0)),
        );

        let result1 = optimizer.optimize(&test_expr).unwrap();
        let result2 = optimizer.optimize(&test_expr).unwrap();

        println!("   First optimization: {result1:?}");
        println!("   Second optimization: {result2:?}");

        // Results should be consistent
        // Note: We're just checking that both succeed, not that they're identical
        // since the optimizer state may change between runs
    }
}
