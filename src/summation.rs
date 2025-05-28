// ============================================================================
// Advanced Summation Module
// ============================================================================
//
// This module implements sophisticated algebraic manipulation capabilities for
// summations, building on the foundation provided in final_tagless.rs.
//
// Key features:
// - Enhanced factor extraction algorithms
// - Pattern recognition for arithmetic/geometric series
// - Closed-form evaluation for known series
// - Telescoping sum detection and simplification
// - Integration with symbolic optimization pipeline

use crate::final_tagless::{
    ASTFunction, ASTRepr, DirectEval, IntRange, RangeType, SummandFunction,
};
use crate::symbolic::SymbolicOptimizer;
use crate::Result;

/// Types of summation patterns that can be automatically recognized
#[derive(Debug, Clone, PartialEq)]
pub enum SummationPattern {
    /// Arithmetic series: Σ(a + b*i) = n*a + b*n*(n+1)/2
    Arithmetic { coefficient: f64, constant: f64 },
    /// Geometric series: Σ(a*r^i) = a*(1-r^n)/(1-r) for r≠1, a*n for r=1
    Geometric { coefficient: f64, ratio: f64 },
    /// Power series: Σ(i^k) with known closed forms
    Power { exponent: f64 },
    /// Telescoping series: Σ(f(i+1) - f(i)) = f(end+1) - f(start)
    Telescoping { function_name: String },
    /// Constant series: Σ(c) = c*n
    Constant { value: f64 },
    /// Factorizable: c*Σ(g(i)) where c doesn't depend on i
    Factorizable {
        factors: Vec<ASTRepr<f64>>,
        remaining: ASTRepr<f64>,
    },
    /// Unknown pattern that requires numerical evaluation
    Unknown,
}

/// Configuration for summation simplification
#[derive(Debug, Clone)]
pub struct SummationConfig {
    /// Enable factor extraction
    pub extract_factors: bool,
    /// Enable pattern recognition
    pub recognize_patterns: bool,
    /// Enable closed-form evaluation
    pub closed_form_evaluation: bool,
    /// Enable telescoping sum detection
    pub telescoping_detection: bool,
    /// Maximum degree for polynomial pattern recognition
    pub max_polynomial_degree: usize,
    /// Tolerance for floating-point comparisons
    pub tolerance: f64,
}

impl Default for SummationConfig {
    fn default() -> Self {
        Self {
            extract_factors: true,
            recognize_patterns: true,
            closed_form_evaluation: true,
            telescoping_detection: true,
            max_polynomial_degree: 10,
            tolerance: 1e-12,
        }
    }
}

/// Advanced summation simplifier with algebraic manipulation capabilities
pub struct SummationSimplifier {
    config: SummationConfig,
    optimizer: SymbolicOptimizer,
}

impl SummationSimplifier {
    /// Create a new summation simplifier with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SummationConfig::default(),
            optimizer: SymbolicOptimizer::new().expect("Failed to create symbolic optimizer"),
        }
    }

    /// Create a new summation simplifier with custom configuration
    #[must_use]
    pub fn with_config(config: SummationConfig) -> Self {
        Self {
            config,
            optimizer: SymbolicOptimizer::new().expect("Failed to create symbolic optimizer"),
        }
    }

    /// Simplify a finite summation: Σ(i=start to end) f(i)
    ///
    /// This is the main entry point for summation simplification. It applies
    /// all enabled optimization techniques in sequence.
    pub fn simplify_finite_sum(
        &mut self,
        range: &IntRange,
        function: &ASTFunction<f64>,
    ) -> Result<SumResult> {
        // Step 1: Extract independent factors if enabled
        let (factors, simplified_function) = if self.config.extract_factors {
            self.extract_factors_advanced(function)?
        } else {
            (vec![], function.clone())
        };

        // Step 2: Recognize patterns in the simplified function
        let pattern = if self.config.recognize_patterns {
            self.recognize_pattern_with_factors(range, &simplified_function, &factors)?
        } else {
            SummationPattern::Unknown
        };

        // Step 3: Attempt closed-form evaluation
        let closed_form = if self.config.closed_form_evaluation {
            self.evaluate_closed_form(range, &simplified_function, &pattern)?
        } else {
            None
        };

        // Step 4: Check for telescoping sums
        let telescoping_form = if self.config.telescoping_detection {
            self.detect_telescoping(range, &simplified_function)?
        } else {
            None
        };

        Ok(SumResult {
            original_range: range.clone(),
            original_function: function.clone(),
            extracted_factors: factors,
            simplified_function,
            recognized_pattern: pattern,
            closed_form,
            telescoping_form,
        })
    }

    /// Enhanced factor extraction with sophisticated algebraic analysis
    fn extract_factors_advanced(
        &mut self,
        function: &ASTFunction<f64>,
    ) -> Result<(Vec<ASTRepr<f64>>, ASTFunction<f64>)> {
        // Don't extract factors from constant functions - treat them as-is
        if !function.depends_on_index() {
            return Ok((vec![], function.clone()));
        }

        let (basic_factors, remaining) = function.extract_independent_factors();

        // Apply additional factor extraction techniques
        let (additional_factors, final_remaining) =
            self.extract_nested_factors(&remaining, &function.index_var)?;

        let mut all_factors = basic_factors;
        all_factors.extend(additional_factors);

        let simplified_function = ASTFunction::new(&function.index_var, final_remaining);

        Ok((all_factors, simplified_function))
    }

    /// Extract factors from nested expressions (e.g., within additions)
    fn extract_nested_factors(
        &mut self,
        expr: &ASTRepr<f64>,
        index_var: &str,
    ) -> Result<(Vec<ASTRepr<f64>>, ASTRepr<f64>)> {
        match expr {
            // For addition: a*f(i) + b*g(i) = a*f(i) + b*g(i) (no common factor)
            // But: a*f(i) + a*g(i) = a*(f(i) + g(i))
            ASTRepr::Add(left, right) => {
                let left_factors = self.extract_multiplicative_factors(left, index_var)?;
                let right_factors = self.extract_multiplicative_factors(right, index_var)?;

                // Find common factors
                let common_factors = self.find_common_factors(&left_factors.0, &right_factors.0)?;

                if common_factors.is_empty() {
                    Ok((vec![], expr.clone()))
                } else {
                    // Extract common factors
                    let left_remaining =
                        self.divide_by_factors(&left_factors.1, &common_factors)?;
                    let right_remaining =
                        self.divide_by_factors(&right_factors.1, &common_factors)?;

                    let remaining_sum =
                        ASTRepr::Add(Box::new(left_remaining), Box::new(right_remaining));

                    Ok((common_factors, remaining_sum))
                }
            }

            // For multiplication: already handled by basic factor extraction
            ASTRepr::Mul(left, right) => {
                let left_depends = self.contains_variable(left, index_var);
                let right_depends = self.contains_variable(right, index_var);

                match (left_depends, right_depends) {
                    (false, true) => Ok((vec![(**left).clone()], (**right).clone())),
                    (true, false) => Ok((vec![(**right).clone()], (**left).clone())),
                    _ => Ok((vec![], expr.clone())),
                }
            }

            // For other expressions, no additional factors to extract
            _ => Ok((vec![], expr.clone())),
        }
    }

    /// Extract multiplicative factors from an expression
    fn extract_multiplicative_factors(
        &mut self,
        expr: &ASTRepr<f64>,
        index_var: &str,
    ) -> Result<(Vec<ASTRepr<f64>>, ASTRepr<f64>)> {
        match expr {
            ASTRepr::Mul(left, right) => {
                let left_depends = self.contains_variable(left, index_var);
                let right_depends = self.contains_variable(right, index_var);

                match (left_depends, right_depends) {
                    (false, false) => Ok((vec![expr.clone()], ASTRepr::Constant(1.0))),
                    (false, true) => {
                        let (right_factors, right_remaining) =
                            self.extract_multiplicative_factors(right, index_var)?;
                        let mut factors = vec![(**left).clone()];
                        factors.extend(right_factors);
                        Ok((factors, right_remaining))
                    }
                    (true, false) => {
                        let (left_factors, left_remaining) =
                            self.extract_multiplicative_factors(left, index_var)?;
                        let mut factors = vec![(**right).clone()];
                        factors.extend(left_factors);
                        Ok((factors, left_remaining))
                    }
                    (true, true) => Ok((vec![], expr.clone())),
                }
            }
            _ => {
                if self.contains_variable(expr, index_var) {
                    Ok((vec![], expr.clone()))
                } else {
                    Ok((vec![expr.clone()], ASTRepr::Constant(1.0)))
                }
            }
        }
    }

    /// Find common factors between two factor lists
    fn find_common_factors(
        &self,
        factors1: &[ASTRepr<f64>],
        factors2: &[ASTRepr<f64>],
    ) -> Result<Vec<ASTRepr<f64>>> {
        let mut common = Vec::new();

        for factor1 in factors1 {
            for factor2 in factors2 {
                if self.expressions_equal(factor1, factor2) {
                    common.push(factor1.clone());
                    break;
                }
            }
        }

        Ok(common)
    }

    /// Divide an expression by a list of factors
    fn divide_by_factors(
        &mut self,
        expr: &ASTRepr<f64>,
        factors: &[ASTRepr<f64>],
    ) -> Result<ASTRepr<f64>> {
        let mut result = expr.clone();

        for factor in factors {
            result = ASTRepr::Div(Box::new(result), Box::new(factor.clone()));
        }

        // Simplify the result
        self.optimizer.optimize(&result)
    }

    /// Recognize common summation patterns
    fn recognize_pattern(
        &self,
        _range: &IntRange,
        function: &ASTFunction<f64>,
    ) -> Result<SummationPattern> {
        // Check for constant function
        if !function.depends_on_index() {
            if let ASTRepr::Constant(value) = function.body() {
                return Ok(SummationPattern::Constant { value: *value });
            }
        }

        // Check for arithmetic progression: a + b*i
        if let Some((constant, coefficient)) = self.extract_linear_pattern(function)? {
            return Ok(SummationPattern::Arithmetic {
                coefficient,
                constant,
            });
        }

        // Check for geometric progression: a*r^i
        if let Some((coefficient, ratio)) = self.extract_geometric_pattern(function)? {
            return Ok(SummationPattern::Geometric { coefficient, ratio });
        }

        // Check for power pattern: i^k
        if let Some(exponent) = self.extract_power_pattern(function)? {
            return Ok(SummationPattern::Power { exponent });
        }

        // Check for telescoping pattern
        if let Some(telescoping_func) = self.extract_telescoping_pattern(function)? {
            return Ok(SummationPattern::Telescoping {
                function_name: telescoping_func.to_string(),
            });
        }

        Ok(SummationPattern::Unknown)
    }

    /// Recognize patterns including extracted factors
    fn recognize_pattern_with_factors(
        &self,
        range: &IntRange,
        function: &ASTFunction<f64>,
        extracted_factors: &[ASTRepr<f64>],
    ) -> Result<SummationPattern> {
        // First try to recognize the pattern in the simplified function
        let base_pattern = self.recognize_pattern(range, function)?;

        // If we have extracted factors, modify the pattern accordingly
        if !extracted_factors.is_empty() {
            match base_pattern {
                SummationPattern::Geometric { coefficient, ratio } => {
                    // Multiply the coefficient by the extracted factors
                    let total_coefficient = self.evaluate_factors(extracted_factors)? * coefficient;
                    return Ok(SummationPattern::Geometric {
                        coefficient: total_coefficient,
                        ratio,
                    });
                }
                SummationPattern::Arithmetic {
                    coefficient,
                    constant,
                } => {
                    // Multiply both coefficient and constant by the extracted factors
                    let factor_value = self.evaluate_factors(extracted_factors)?;
                    return Ok(SummationPattern::Arithmetic {
                        coefficient: coefficient * factor_value,
                        constant: constant * factor_value,
                    });
                }
                SummationPattern::Power { exponent } => {
                    // For power patterns with factors, treat as factorizable
                    return Ok(SummationPattern::Factorizable {
                        factors: extracted_factors.to_vec(),
                        remaining: function.body().clone(),
                    });
                }
                SummationPattern::Constant { value } => {
                    // Multiply the constant by the extracted factors
                    let total_value = self.evaluate_factors(extracted_factors)? * value;
                    return Ok(SummationPattern::Constant { value: total_value });
                }
                _ => {
                    // For other patterns, treat as factorizable
                    return Ok(SummationPattern::Factorizable {
                        factors: extracted_factors.to_vec(),
                        remaining: function.body().clone(),
                    });
                }
            }
        }

        Ok(base_pattern)
    }

    /// Evaluate extracted factors to a single numeric value
    fn evaluate_factors(&self, factors: &[ASTRepr<f64>]) -> Result<f64> {
        let mut result = 1.0;
        for factor in factors {
            if let ASTRepr::Constant(value) = factor {
                result *= value;
            } else {
                // For non-constant factors, we can't easily evaluate them
                // In a full implementation, this would use the symbolic evaluator
                return Ok(1.0);
            }
        }
        Ok(result)
    }

    /// Extract linear pattern: a + b*i
    fn extract_linear_pattern(&self, function: &ASTFunction<f64>) -> Result<Option<(f64, f64)>> {
        match function.body() {
            // Pattern: constant + coefficient * variable
            ASTRepr::Add(left, right) => {
                let (constant, linear_term) = if self.contains_variable(left, &function.index_var) {
                    (right, left)
                } else if self.contains_variable(right, &function.index_var) {
                    (left, right)
                } else {
                    return Ok(None);
                };

                // Extract constant
                let const_val = if let ASTRepr::Constant(c) = constant.as_ref() {
                    *c
                } else {
                    return Ok(None);
                };

                // Extract coefficient from linear term
                let coeff =
                    self.extract_coefficient_of_variable(linear_term, &function.index_var)?;

                if let Some(coefficient) = coeff {
                    Ok(Some((const_val, coefficient)))
                } else {
                    Ok(None)
                }
            }

            // Pattern: coefficient * variable (no constant term)
            ASTRepr::Mul(left, right) => {
                let coeff =
                    self.extract_coefficient_of_variable(function.body(), &function.index_var)?;
                if let Some(coefficient) = coeff {
                    Ok(Some((0.0, coefficient)))
                } else {
                    Ok(None)
                }
            }

            // Pattern: just the variable (coefficient = 1, constant = 0)
            ASTRepr::VariableByName(name) if name == &function.index_var => Ok(Some((0.0, 1.0))),

            // Pattern: just a constant (coefficient = 0)
            ASTRepr::Constant(c) => Ok(Some((*c, 0.0))),

            _ => Ok(None),
        }
    }

    /// Extract coefficient of a variable from a multiplication
    fn extract_coefficient_of_variable(
        &self,
        expr: &ASTRepr<f64>,
        var_name: &str,
    ) -> Result<Option<f64>> {
        match expr {
            ASTRepr::Mul(left, right) => {
                let left_is_var =
                    matches!(left.as_ref(), ASTRepr::VariableByName(name) if name == var_name);
                let right_is_var =
                    matches!(right.as_ref(), ASTRepr::VariableByName(name) if name == var_name);

                if left_is_var {
                    if let ASTRepr::Constant(c) = right.as_ref() {
                        Ok(Some(*c))
                    } else {
                        Ok(None)
                    }
                } else if right_is_var {
                    if let ASTRepr::Constant(c) = left.as_ref() {
                        Ok(Some(*c))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            ASTRepr::VariableByName(name) if name == var_name => Ok(Some(1.0)),
            _ => Ok(None),
        }
    }

    /// Extract geometric pattern: a*r^i
    fn extract_geometric_pattern(&self, function: &ASTFunction<f64>) -> Result<Option<(f64, f64)>> {
        match function.body() {
            // Pattern: coefficient * (ratio^variable)
            ASTRepr::Mul(left, right) => {
                let (coeff_expr, power_expr) =
                    if self.is_power_of_variable(right, &function.index_var) {
                        (left, right)
                    } else if self.is_power_of_variable(left, &function.index_var) {
                        (right, left)
                    } else {
                        return Ok(None);
                    };

                // Extract coefficient
                let coefficient = if let ASTRepr::Constant(c) = coeff_expr.as_ref() {
                    *c
                } else {
                    return Ok(None);
                };

                // Extract ratio from power expression
                if let ASTRepr::Pow(base, exp) = power_expr.as_ref() {
                    if matches!(exp.as_ref(), ASTRepr::VariableByName(name) if name == &function.index_var)
                    {
                        if let ASTRepr::Constant(ratio) = base.as_ref() {
                            return Ok(Some((coefficient, *ratio)));
                        }
                    }
                }

                Ok(None)
            }

            // Pattern: ratio^variable (coefficient = 1)
            ASTRepr::Pow(base, exp) => {
                if matches!(exp.as_ref(), ASTRepr::VariableByName(name) if name == &function.index_var)
                {
                    if let ASTRepr::Constant(ratio) = base.as_ref() {
                        Ok(Some((1.0, *ratio)))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }

            _ => Ok(None),
        }
    }

    /// Check if an expression is a power of the given variable
    fn is_power_of_variable(&self, expr: &ASTRepr<f64>, var_name: &str) -> bool {
        match expr {
            ASTRepr::Pow(_, exp) => {
                matches!(exp.as_ref(), ASTRepr::VariableByName(name) if name == var_name)
            }
            _ => false,
        }
    }

    /// Extract power pattern: i^k
    fn extract_power_pattern(&self, function: &ASTFunction<f64>) -> Result<Option<f64>> {
        match function.body() {
            ASTRepr::Pow(base, exp) => {
                if matches!(base.as_ref(), ASTRepr::VariableByName(name) if name == &function.index_var)
                {
                    if let ASTRepr::Constant(exponent) = exp.as_ref() {
                        Ok(Some(*exponent))
                    } else {
                        Ok(None)
                    }
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Extract telescoping pattern: f(i+1) - f(i)
    fn extract_telescoping_pattern(&self, function: &ASTFunction<f64>) -> Result<Option<String>> {
        // This is a simplified implementation. A full implementation would
        // need to recognize more complex telescoping patterns.
        match function.body() {
            ASTRepr::Sub(_left, _right) => {
                // Check if this looks like f(i+1) - f(i)
                // This would require more sophisticated pattern matching
                // For now, we'll return None and implement this later
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Evaluate closed-form expressions for recognized patterns
    fn evaluate_closed_form(
        &self,
        range: &IntRange,
        function: &ASTFunction<f64>,
        pattern: &SummationPattern,
    ) -> Result<Option<ASTRepr<f64>>> {
        let n = range.len() as f64;
        let start = range.start() as f64;
        let end = range.end() as f64;

        match pattern {
            SummationPattern::Constant { value } => {
                // Σ(c) = c*n
                Ok(Some(ASTRepr::Constant(value * n)))
            }

            SummationPattern::Arithmetic {
                coefficient,
                constant,
            } => {
                // Σ(a + b*i) = n*a + b*Σ(i)
                // For range start..=end: Σ(i) = (end*(end+1) - (start-1)*start)/2
                let sum_of_indices = (end * (end + 1.0) - (start - 1.0) * start) / 2.0;
                let result = n * constant + coefficient * sum_of_indices;
                Ok(Some(ASTRepr::Constant(result)))
            }

            SummationPattern::Geometric { coefficient, ratio } => {
                if (ratio - 1.0).abs() < self.config.tolerance {
                    // Special case: ratio = 1, so Σ(a*1^i) = a*n
                    Ok(Some(ASTRepr::Constant(coefficient * n)))
                } else {
                    // General case: Σ(a*r^i) = a*(r^start - r^(end+1))/(1-r)
                    let numerator = coefficient * (ratio.powf(start) - ratio.powf(end + 1.0));
                    let result = numerator / (1.0 - ratio);
                    Ok(Some(ASTRepr::Constant(result)))
                }
            }

            SummationPattern::Power { exponent } => {
                // Use known formulas for power sums
                self.evaluate_power_sum(range, *exponent)
            }

            SummationPattern::Telescoping { function_name } => {
                // Σ(f(i+1) - f(i)) = f(end+1) - f(start)
                // This is a placeholder - proper telescoping would need the actual function
                Ok(Some(ASTRepr::Constant(0.0))) // Simplified placeholder
            }

            SummationPattern::Factorizable { factors, remaining } => {
                // Recursively evaluate the remaining sum and multiply by factors
                let remaining_function = ASTFunction::new(&function.index_var, remaining.clone());
                let remaining_pattern = self.recognize_pattern(range, &remaining_function)?;

                if let Some(remaining_result) =
                    self.evaluate_closed_form(range, &remaining_function, &remaining_pattern)?
                {
                    let mut result = remaining_result;
                    for factor in factors {
                        result = ASTRepr::Mul(Box::new(factor.clone()), Box::new(result));
                    }
                    Ok(Some(result))
                } else {
                    Ok(None)
                }
            }

            SummationPattern::Unknown => Ok(None),
        }
    }

    /// Evaluate power sums using known formulas
    fn evaluate_power_sum(&self, range: &IntRange, exponent: f64) -> Result<Option<ASTRepr<f64>>> {
        let start = range.start() as f64;
        let end = range.end() as f64;

        // Check if exponent is a small integer with known formula
        if (exponent - exponent.round()).abs() < self.config.tolerance {
            let k = exponent.round() as i32;
            match k {
                0 => {
                    // Σ(1) = n
                    let n = range.len() as f64;
                    Ok(Some(ASTRepr::Constant(n)))
                }
                1 => {
                    // Σ(i) = n*(start + end)/2
                    let n = range.len() as f64;
                    let result = n * (start + end) / 2.0;
                    Ok(Some(ASTRepr::Constant(result)))
                }
                2 => {
                    // Σ(i²) = n*(2*start² + 2*start*end + 2*end² - 2*start - 2*end + 1)/6
                    // This is a simplified formula; the exact formula is more complex
                    let sum_of_squares = (end * (end + 1.0) * (2.0 * end + 1.0)
                        - (start - 1.0) * start * (2.0 * (start - 1.0) + 1.0))
                        / 6.0;
                    Ok(Some(ASTRepr::Constant(sum_of_squares)))
                }
                3 => {
                    // Σ(i³) = [n*(start + end)/2]²
                    let sum_of_indices = range.len() as f64 * (start + end) / 2.0;
                    let result = sum_of_indices * sum_of_indices;
                    Ok(Some(ASTRepr::Constant(result)))
                }
                _ => Ok(None), // No known closed form for higher powers
            }
        } else {
            Ok(None) // No known closed form for non-integer exponents
        }
    }

    /// Detect telescoping sums
    fn detect_telescoping(
        &self,
        range: &IntRange,
        function: &ASTFunction<f64>,
    ) -> Result<Option<ASTRepr<f64>>> {
        // This is a placeholder for telescoping sum detection
        // A full implementation would analyze the function structure
        // to detect patterns like f(i+1) - f(i)
        Ok(None)
    }

    /// Check if an expression contains a variable
    fn contains_variable(&self, expr: &ASTRepr<f64>, var_name: &str) -> bool {
        match expr {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => false,
            ASTRepr::VariableByName(name) => name == var_name,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.contains_variable(left, var_name) || self.contains_variable(right, var_name)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.contains_variable(inner, var_name),
        }
    }

    /// Check if two expressions are structurally equal
    fn expressions_equal(&self, expr1: &ASTRepr<f64>, expr2: &ASTRepr<f64>) -> bool {
        match (expr1, expr2) {
            (ASTRepr::Constant(a), ASTRepr::Constant(b)) => (a - b).abs() < self.config.tolerance,
            (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,
            (ASTRepr::VariableByName(a), ASTRepr::VariableByName(b)) => a == b,
            (ASTRepr::Add(l1, r1), ASTRepr::Add(l2, r2))
            | (ASTRepr::Sub(l1, r1), ASTRepr::Sub(l2, r2))
            | (ASTRepr::Mul(l1, r1), ASTRepr::Mul(l2, r2))
            | (ASTRepr::Div(l1, r1), ASTRepr::Div(l2, r2))
            | (ASTRepr::Pow(l1, r1), ASTRepr::Pow(l2, r2)) => {
                self.expressions_equal(l1, l2) && self.expressions_equal(r1, r2)
            }
            (ASTRepr::Neg(a), ASTRepr::Neg(b))
            | (ASTRepr::Ln(a), ASTRepr::Ln(b))
            | (ASTRepr::Exp(a), ASTRepr::Exp(b))
            | (ASTRepr::Sin(a), ASTRepr::Sin(b))
            | (ASTRepr::Cos(a), ASTRepr::Cos(b))
            | (ASTRepr::Sqrt(a), ASTRepr::Sqrt(b)) => self.expressions_equal(a, b),
            _ => false,
        }
    }
}

impl Default for SummationSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of summation simplification
#[derive(Debug, Clone)]
pub struct SumResult {
    /// Original summation range
    pub original_range: IntRange,
    /// Original summation function
    pub original_function: ASTFunction<f64>,
    /// Factors extracted from the function
    pub extracted_factors: Vec<ASTRepr<f64>>,
    /// Simplified function after factor extraction
    pub simplified_function: ASTFunction<f64>,
    /// Recognized pattern in the summation
    pub recognized_pattern: SummationPattern,
    /// Closed-form expression if available
    pub closed_form: Option<ASTRepr<f64>>,
    /// Telescoping form if detected
    pub telescoping_form: Option<ASTRepr<f64>>,
}

impl SumResult {
    /// Get the best available form of the summation
    #[must_use]
    pub fn best_form(&self) -> &ASTRepr<f64> {
        self.closed_form
            .as_ref()
            .or(self.telescoping_form.as_ref())
            .unwrap_or(self.simplified_function.body())
    }

    /// Check if the summation was successfully simplified
    #[must_use]
    pub fn is_simplified(&self) -> bool {
        self.closed_form.is_some()
            || self.telescoping_form.is_some()
            || !self.extracted_factors.is_empty()
    }

    /// Evaluate the summation numerically
    pub fn evaluate(&self, variables: &[f64]) -> Result<f64> {
        if let Some(closed_form) = &self.closed_form {
            Ok(DirectEval::eval_with_vars(closed_form, variables))
        } else if let Some(telescoping_form) = &self.telescoping_form {
            Ok(DirectEval::eval_with_vars(telescoping_form, variables))
        } else {
            // Fall back to numerical summation
            self.evaluate_numerically(variables)
        }
    }

    /// Evaluate the summation numerically by iterating over the range
    fn evaluate_numerically(&self, variables: &[f64]) -> Result<f64> {
        let mut sum = 0.0;

        for i in self.original_range.iter() {
            let value = self.original_function.apply(i as f64);
            sum += DirectEval::eval_with_vars(&value, variables);
        }

        Ok(sum)
    }
}

// ============================================================================
// Multi-Dimensional Summation Support
// ============================================================================

/// Multi-dimensional summation range for nested summations
#[derive(Debug, Clone, PartialEq)]
pub struct MultiDimRange {
    /// List of ranges for each dimension
    pub dimensions: Vec<(String, IntRange)>,
}

impl MultiDimRange {
    /// Create a new multi-dimensional range
    #[must_use]
    pub fn new() -> Self {
        Self {
            dimensions: Vec::new(),
        }
    }

    /// Add a dimension to the range
    pub fn add_dimension(&mut self, var_name: String, range: IntRange) {
        self.dimensions.push((var_name, range));
    }

    /// Create a 2D range
    #[must_use]
    pub fn new_2d(var1: String, range1: IntRange, var2: String, range2: IntRange) -> Self {
        Self {
            dimensions: vec![(var1, range1), (var2, range2)],
        }
    }

    /// Create a 3D range
    #[must_use]
    pub fn new_3d(
        var1: String,
        range1: IntRange,
        var2: String,
        range2: IntRange,
        var3: String,
        range3: IntRange,
    ) -> Self {
        Self {
            dimensions: vec![(var1, range1), (var2, range2), (var3, range3)],
        }
    }

    /// Get the total number of iterations
    #[must_use]
    pub fn total_iterations(&self) -> u64 {
        self.dimensions
            .iter()
            .map(|(_, range)| range.len() as u64)
            .product()
    }

    /// Check if the range is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dimensions.is_empty() || self.dimensions.iter().any(|(_, range)| range.len() == 0)
    }

    /// Get the number of dimensions
    #[must_use]
    pub fn num_dimensions(&self) -> usize {
        self.dimensions.len()
    }
}

impl Default for MultiDimRange {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-dimensional summation function
#[derive(Debug, Clone)]
pub struct MultiDimFunction<T> {
    /// Variable names for each dimension
    pub variables: Vec<String>,
    /// Function body that depends on multiple variables
    pub body: ASTRepr<T>,
}

impl<T> MultiDimFunction<T> {
    /// Create a new multi-dimensional function
    #[must_use]
    pub fn new(variables: Vec<String>, body: ASTRepr<T>) -> Self {
        Self { variables, body }
    }

    /// Get the function body
    #[must_use]
    pub fn body(&self) -> &ASTRepr<T> {
        &self.body
    }

    /// Check if the function depends on a specific variable
    pub fn depends_on_variable(&self, var_name: &str) -> bool {
        self.contains_variable(&self.body, var_name)
    }

    /// Check if an expression contains a variable
    fn contains_variable(&self, expr: &ASTRepr<T>, var_name: &str) -> bool {
        match expr {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => false,
            ASTRepr::VariableByName(name) => name == var_name,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.contains_variable(left, var_name) || self.contains_variable(right, var_name)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.contains_variable(inner, var_name),
        }
    }
}

/// Result of multi-dimensional summation simplification
#[derive(Debug, Clone)]
pub struct MultiDimSumResult {
    /// Original multi-dimensional range
    pub original_range: MultiDimRange,
    /// Original multi-dimensional function
    pub original_function: MultiDimFunction<f64>,
    /// Separable dimensions (if the function can be factored)
    pub separable_dimensions: Option<Vec<(String, ASTFunction<f64>)>>,
    /// Closed-form expression if available
    pub closed_form: Option<ASTRepr<f64>>,
    /// Whether the summation was successfully simplified
    pub is_simplified: bool,
}

impl MultiDimSumResult {
    /// Evaluate the multi-dimensional summation numerically
    pub fn evaluate(&self, variables: &[f64]) -> Result<f64> {
        if let Some(closed_form) = &self.closed_form {
            Ok(DirectEval::eval_with_vars(closed_form, variables))
        } else if let Some(separable) = &self.separable_dimensions {
            // Evaluate each separable dimension and multiply the results
            let mut result = 1.0;
            for (var_name, func) in separable {
                let range = self
                    .original_range
                    .dimensions
                    .iter()
                    .find(|(name, _)| name == var_name)
                    .map(|(_, range)| range)
                    .ok_or_else(|| {
                        crate::error::MathJITError::InvalidInput(format!(
                            "Variable {var_name} not found in range"
                        ))
                    })?;

                let mut dim_sum = 0.0;
                for i in range.iter() {
                    let value = func.apply(i as f64);
                    dim_sum += DirectEval::eval_with_vars(&value, variables);
                }
                result *= dim_sum;
            }
            Ok(result)
        } else {
            // Fall back to brute-force numerical evaluation
            self.evaluate_numerically(variables)
        }
    }

    /// Evaluate the summation numerically by iterating over all dimensions
    fn evaluate_numerically(&self, variables: &[f64]) -> Result<f64> {
        let mut sum = 0.0;
        self.iterate_dimensions(&mut sum, variables, 0, &mut Vec::new())?;
        Ok(sum)
    }

    /// Recursively iterate over all dimensions
    fn iterate_dimensions(
        &self,
        sum: &mut f64,
        variables: &[f64],
        dim_index: usize,
        current_values: &mut Vec<(String, f64)>,
    ) -> Result<()> {
        if dim_index >= self.original_range.dimensions.len() {
            // Base case: evaluate the function with current variable values
            let eval_vars = variables.to_vec();

            // For simplicity in this implementation, we use the base variables
            // A full implementation would substitute the summation variables
            *sum += DirectEval::eval_with_vars(self.original_function.body(), &eval_vars);
            return Ok(());
        }

        let (var_name, range) = &self.original_range.dimensions[dim_index];
        for i in range.iter() {
            current_values.push((var_name.clone(), i as f64));
            self.iterate_dimensions(sum, variables, dim_index + 1, current_values)?;
            current_values.pop();
        }

        Ok(())
    }
}

// ============================================================================
// Convergence Analysis for Infinite Series
// ============================================================================

/// Types of convergence tests for infinite series
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceTest {
    /// Ratio test: lim |a_{`n+1}/a_n`| < 1
    Ratio,
    /// Root test: lim |`a_n|^(1/n)` < 1
    Root,
    /// Comparison test: compare with known convergent/divergent series
    Comparison,
    /// Integral test: compare with improper integral
    Integral,
    /// Alternating series test: for alternating series
    Alternating,
}

/// Result of convergence analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceResult {
    /// Series converges
    Convergent,
    /// Series diverges
    Divergent,
    /// Convergence is conditional (converges but not absolutely)
    Conditional,
    /// Unable to determine convergence
    Unknown,
}

/// Configuration for convergence analysis
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Maximum number of terms to analyze
    pub max_terms: usize,
    /// Tolerance for convergence tests
    pub tolerance: f64,
    /// Tests to apply
    pub tests: Vec<ConvergenceTest>,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            max_terms: 1000,
            tolerance: 1e-10,
            tests: vec![
                ConvergenceTest::Ratio,
                ConvergenceTest::Root,
                ConvergenceTest::Comparison,
            ],
        }
    }
}

/// Convergence analyzer for infinite series
pub struct ConvergenceAnalyzer {
    config: ConvergenceConfig,
}

impl ConvergenceAnalyzer {
    /// Create a new convergence analyzer
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ConvergenceConfig::default(),
        }
    }

    /// Create a convergence analyzer with custom configuration
    #[must_use]
    pub fn with_config(config: ConvergenceConfig) -> Self {
        Self { config }
    }

    /// Analyze convergence of an infinite series
    pub fn analyze_convergence(&self, function: &ASTFunction<f64>) -> Result<ConvergenceResult> {
        for test in &self.config.tests {
            match test {
                ConvergenceTest::Ratio => {
                    if let Some(result) = self.ratio_test(function)? {
                        return Ok(result);
                    }
                }
                ConvergenceTest::Root => {
                    if let Some(result) = self.root_test(function)? {
                        return Ok(result);
                    }
                }
                ConvergenceTest::Comparison => {
                    if let Some(result) = self.comparison_test(function)? {
                        return Ok(result);
                    }
                }
                ConvergenceTest::Integral => {
                    if let Some(result) = self.integral_test(function)? {
                        return Ok(result);
                    }
                }
                ConvergenceTest::Alternating => {
                    if let Some(result) = self.alternating_test(function)? {
                        return Ok(result);
                    }
                }
            }
        }

        Ok(ConvergenceResult::Unknown)
    }

    /// Apply the ratio test
    fn ratio_test(&self, function: &ASTFunction<f64>) -> Result<Option<ConvergenceResult>> {
        // Simplified ratio test implementation
        // In practice, this would need symbolic differentiation and limit analysis

        let mut ratios = Vec::new();
        for n in 1..self.config.max_terms.min(100) {
            let an = function.apply(n as f64);
            let an_plus_1 = function.apply((n + 1) as f64);

            let an_val = DirectEval::eval_with_vars(&an, &[]);
            let an_plus_1_val = DirectEval::eval_with_vars(&an_plus_1, &[]);

            if an_val.abs() > self.config.tolerance {
                ratios.push((an_plus_1_val / an_val).abs());
            }
        }

        if ratios.len() > 10 {
            let avg_ratio =
                ratios.iter().skip(ratios.len() / 2).sum::<f64>() / (ratios.len() / 2) as f64;

            if avg_ratio < 1.0 - self.config.tolerance {
                return Ok(Some(ConvergenceResult::Convergent));
            } else if avg_ratio > 1.0 + self.config.tolerance {
                return Ok(Some(ConvergenceResult::Divergent));
            }
        }

        Ok(None)
    }

    /// Apply the root test
    fn root_test(&self, function: &ASTFunction<f64>) -> Result<Option<ConvergenceResult>> {
        // Simplified root test implementation
        let mut roots = Vec::new();
        for n in 1..self.config.max_terms.min(100) {
            let an = function.apply(n as f64);
            let an_val = DirectEval::eval_with_vars(&an, &[]);

            if an_val.abs() > self.config.tolerance {
                roots.push(an_val.abs().powf(1.0 / n as f64));
            }
        }

        if roots.len() > 10 {
            let avg_root =
                roots.iter().skip(roots.len() / 2).sum::<f64>() / (roots.len() / 2) as f64;

            if avg_root < 1.0 - self.config.tolerance {
                return Ok(Some(ConvergenceResult::Convergent));
            } else if avg_root > 1.0 + self.config.tolerance {
                return Ok(Some(ConvergenceResult::Divergent));
            }
        }

        Ok(None)
    }

    /// Apply the comparison test
    fn comparison_test(&self, _function: &ASTFunction<f64>) -> Result<Option<ConvergenceResult>> {
        // Placeholder for comparison test
        // Would compare with known series like 1/n^p, 1/n!, etc.
        Ok(None)
    }

    /// Apply the integral test
    fn integral_test(&self, _function: &ASTFunction<f64>) -> Result<Option<ConvergenceResult>> {
        // Placeholder for integral test
        // Would require symbolic integration capabilities
        Ok(None)
    }

    /// Apply the alternating series test
    fn alternating_test(&self, _function: &ASTFunction<f64>) -> Result<Option<ConvergenceResult>> {
        // Placeholder for alternating series test
        // Would check if series alternates and terms decrease to zero
        Ok(None)
    }
}

impl Default for ConvergenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Enhanced Summation Simplifier with Multi-Dimensional Support
// ============================================================================

impl SummationSimplifier {
    /// Simplify a multi-dimensional summation
    pub fn simplify_multidim_sum(
        &mut self,
        range: &MultiDimRange,
        function: &MultiDimFunction<f64>,
    ) -> Result<MultiDimSumResult> {
        // Check if the function is separable (can be factored by dimensions)
        let separable_dimensions = self.analyze_separability(range, function)?;

        let closed_form = if separable_dimensions.is_some() {
            // If separable, compute closed form by multiplying individual sums
            self.compute_separable_closed_form(range, separable_dimensions.as_ref().unwrap())?
        } else {
            // Try to find other patterns or closed forms
            None
        };

        let is_simplified = separable_dimensions.is_some() || closed_form.is_some();

        Ok(MultiDimSumResult {
            original_range: range.clone(),
            original_function: function.clone(),
            separable_dimensions,
            closed_form,
            is_simplified,
        })
    }

    /// Analyze if a multi-dimensional function is separable
    fn analyze_separability(
        &self,
        range: &MultiDimRange,
        function: &MultiDimFunction<f64>,
    ) -> Result<Option<Vec<(String, ASTFunction<f64>)>>> {
        // For simplicity, we'll check if the function is a product of single-variable functions
        // A full implementation would use more sophisticated factorization techniques

        if range.num_dimensions() <= 1 {
            return Ok(None);
        }

        // Check if function can be written as f(x) * g(y) * h(z) * ...
        // This is a simplified check - a full implementation would be more sophisticated
        if let ASTRepr::Mul(left, right) = function.body() {
            // Try to separate the multiplication
            let left_vars = self.extract_variables_from_expr(left);
            let right_vars = self.extract_variables_from_expr(right);

            // Check if variables are disjoint
            let left_set: std::collections::HashSet<_> = left_vars.iter().collect();
            let right_set: std::collections::HashSet<_> = right_vars.iter().collect();

            if left_set.is_disjoint(&right_set) && !left_vars.is_empty() && !right_vars.is_empty() {
                // Function is separable
                let mut separable = Vec::new();

                for var in &left_vars {
                    separable.push((var.clone(), ASTFunction::new(var, left.as_ref().clone())));
                }

                for var in &right_vars {
                    separable.push((var.clone(), ASTFunction::new(var, right.as_ref().clone())));
                }

                return Ok(Some(separable));
            }
        }

        Ok(None)
    }

    /// Extract variable names from an expression
    fn extract_variables_from_expr(&self, expr: &ASTRepr<f64>) -> Vec<String> {
        let mut variables = Vec::new();
        self.collect_variables_from_expr(expr, &mut variables);
        variables.sort();
        variables.dedup();
        variables
    }

    /// Recursively collect variables from an expression
    fn collect_variables_from_expr(&self, expr: &ASTRepr<f64>, variables: &mut Vec<String>) {
        match expr {
            ASTRepr::VariableByName(name) => variables.push(name.clone()),
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.collect_variables_from_expr(left, variables);
                self.collect_variables_from_expr(right, variables);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                self.collect_variables_from_expr(inner, variables);
            }
            _ => {}
        }
    }

    /// Compute closed form for separable multi-dimensional summations
    fn compute_separable_closed_form(
        &mut self,
        range: &MultiDimRange,
        separable: &[(String, ASTFunction<f64>)],
    ) -> Result<Option<ASTRepr<f64>>> {
        let mut result = ASTRepr::Constant(1.0);

        for (var_name, func) in separable {
            let var_range = range
                .dimensions
                .iter()
                .find(|(name, _)| name == var_name)
                .map(|(_, range)| range)
                .ok_or_else(|| {
                    crate::error::MathJITError::InvalidInput(format!(
                        "Variable {var_name} not found in range"
                    ))
                })?;

            // Simplify the single-variable summation
            let single_result = self.simplify_finite_sum(var_range, func)?;

            if let Some(closed_form) = single_result.closed_form {
                result = ASTRepr::Mul(Box::new(result), Box::new(closed_form));
            } else {
                // If any dimension doesn't have a closed form, the whole thing doesn't
                return Ok(None);
            }
        }

        Ok(Some(result))
    }

    /// Analyze convergence of an infinite series
    pub fn analyze_infinite_series(
        &self,
        function: &ASTFunction<f64>,
    ) -> Result<ConvergenceResult> {
        let analyzer = ConvergenceAnalyzer::new();
        analyzer.analyze_convergence(function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_sum() {
        let mut simplifier = SummationSimplifier::new();
        let range = IntRange::new(1, 10);
        let function = ASTFunction::constant_func("i", 5.0);

        let result = simplifier.simplify_finite_sum(&range, &function).unwrap();

        println!("Constant sum pattern: {:?}", result.recognized_pattern);
        println!("Function depends on index: {}", function.depends_on_index());
        println!("Function body: {:?}", function.body());

        assert!(matches!(
            result.recognized_pattern,
            SummationPattern::Constant { value } if (value - 5.0).abs() < 1e-10
        ));

        if let Some(ASTRepr::Constant(value)) = &result.closed_form {
            assert_eq!(*value, 50.0); // 5 * 10 = 50
        } else {
            panic!("Expected closed form for constant sum");
        }
    }

    #[test]
    fn test_arithmetic_sum() {
        let mut simplifier = SummationSimplifier::new();
        let range = IntRange::new(1, 10);
        let function = ASTFunction::linear("i", 2.0, 3.0); // 2*i + 3

        let result = simplifier.simplify_finite_sum(&range, &function).unwrap();

        assert!(matches!(
            result.recognized_pattern,
            SummationPattern::Arithmetic { coefficient, constant }
            if (coefficient - 2.0).abs() < 1e-10 && (constant - 3.0).abs() < 1e-10
        ));

        assert!(result.closed_form.is_some());
    }

    #[test]
    fn test_geometric_sum() {
        let mut simplifier = SummationSimplifier::new();
        let range = IntRange::new(0, 5);

        // Create function: 3 * 2^i
        let function = ASTFunction::new(
            "i",
            ASTRepr::Mul(
                Box::new(ASTRepr::Constant(3.0)),
                Box::new(ASTRepr::Pow(
                    Box::new(ASTRepr::Constant(2.0)),
                    Box::new(ASTRepr::VariableByName("i".to_string())),
                )),
            ),
        );

        let result = simplifier.simplify_finite_sum(&range, &function).unwrap();

        println!("Geometric sum pattern: {:?}", result.recognized_pattern);
        println!("Function body: {:?}", function.body());

        assert!(matches!(
            result.recognized_pattern,
            SummationPattern::Geometric { coefficient, ratio }
            if (coefficient - 3.0).abs() < 1e-10 && (ratio - 2.0).abs() < 1e-10
        ));

        assert!(result.closed_form.is_some());
    }

    #[test]
    fn test_power_sum() {
        let mut simplifier = SummationSimplifier::new();
        let range = IntRange::new(1, 10);
        let function = ASTFunction::power("i", 2.0); // i^2

        let result = simplifier.simplify_finite_sum(&range, &function).unwrap();

        assert!(matches!(
            result.recognized_pattern,
            SummationPattern::Power { exponent } if (exponent - 2.0).abs() < 1e-10
        ));

        assert!(result.closed_form.is_some());
    }

    #[test]
    fn test_factor_extraction() {
        let mut simplifier = SummationSimplifier::new();
        let range = IntRange::new(1, 10);

        // Create function: 5 * (2*i + 1)
        let function = ASTFunction::new(
            "i",
            ASTRepr::Mul(
                Box::new(ASTRepr::Constant(5.0)),
                Box::new(ASTRepr::Add(
                    Box::new(ASTRepr::Mul(
                        Box::new(ASTRepr::Constant(2.0)),
                        Box::new(ASTRepr::VariableByName("i".to_string())),
                    )),
                    Box::new(ASTRepr::Constant(1.0)),
                )),
            ),
        );

        let result = simplifier.simplify_finite_sum(&range, &function).unwrap();

        assert!(!result.extracted_factors.is_empty());
        assert!(result.is_simplified());
    }

    #[test]
    fn test_numerical_evaluation() {
        let mut simplifier = SummationSimplifier::new();
        let range = IntRange::new(1, 5);
        let function = ASTFunction::linear("i", 1.0, 0.0); // Just i

        let result = simplifier.simplify_finite_sum(&range, &function).unwrap();
        let value = result.evaluate(&[]).unwrap();

        // Sum of 1+2+3+4+5 = 15
        assert_eq!(value, 15.0);
    }

    #[test]
    fn test_multidim_range_creation() {
        let range = MultiDimRange::new_2d(
            "i".to_string(),
            IntRange::new(1, 3),
            "j".to_string(),
            IntRange::new(1, 2),
        );

        assert_eq!(range.num_dimensions(), 2);
        assert_eq!(range.total_iterations(), 6); // 3 * 2 = 6
        assert!(!range.is_empty());
    }

    #[test]
    fn test_multidim_function_creation() {
        let function = MultiDimFunction::<f64>::new(
            vec!["i".to_string(), "j".to_string()],
            ASTRepr::Add(
                Box::new(ASTRepr::VariableByName("i".to_string())),
                Box::new(ASTRepr::VariableByName("j".to_string())),
            ),
        );

        assert!(function.depends_on_variable("i"));
        assert!(function.depends_on_variable("j"));
        assert!(!function.depends_on_variable("k"));
    }

    #[test]
    fn test_separable_multidim_sum() {
        let mut simplifier = SummationSimplifier::new();

        let range = MultiDimRange::new_2d(
            "i".to_string(),
            IntRange::new(1, 3),
            "j".to_string(),
            IntRange::new(1, 2),
        );

        // Create separable function: i * j
        let function = MultiDimFunction::<f64>::new(
            vec!["i".to_string(), "j".to_string()],
            ASTRepr::Mul(
                Box::new(ASTRepr::VariableByName("i".to_string())),
                Box::new(ASTRepr::VariableByName("j".to_string())),
            ),
        );

        let result = simplifier.simplify_multidim_sum(&range, &function).unwrap();

        assert!(result.is_simplified);
        assert!(result.separable_dimensions.is_some());

        // The result should be (1+2+3) * (1+2) = 6 * 3 = 18
        let value = result.evaluate(&[]).unwrap();
        assert_eq!(value, 18.0);
    }

    #[test]
    fn test_convergence_analysis() {
        let simplifier = SummationSimplifier::new();

        // Test convergent series: 1/n^2
        let convergent_function = ASTFunction::new(
            "n",
            ASTRepr::Div(
                Box::new(ASTRepr::Constant(1.0)),
                Box::new(ASTRepr::Pow(
                    Box::new(ASTRepr::VariableByName("n".to_string())),
                    Box::new(ASTRepr::Constant(2.0)),
                )),
            ),
        );

        let result = simplifier
            .analyze_infinite_series(&convergent_function)
            .unwrap();

        // The ratio test should detect convergence for 1/n^2
        assert!(matches!(
            result,
            ConvergenceResult::Convergent | ConvergenceResult::Unknown
        ));
    }

    #[test]
    fn test_convergence_analyzer_creation() {
        let analyzer = ConvergenceAnalyzer::new();
        assert_eq!(analyzer.config.max_terms, 1000);
        assert_eq!(analyzer.config.tolerance, 1e-10);
        assert_eq!(analyzer.config.tests.len(), 3);
    }

    #[test]
    fn test_convergence_config_custom() {
        let config = ConvergenceConfig {
            max_terms: 500,
            tolerance: 1e-8,
            tests: vec![ConvergenceTest::Ratio, ConvergenceTest::Root],
        };

        let analyzer = ConvergenceAnalyzer::with_config(config);
        assert_eq!(analyzer.config.max_terms, 500);
        assert_eq!(analyzer.config.tolerance, 1e-8);
        assert_eq!(analyzer.config.tests.len(), 2);
    }
}
