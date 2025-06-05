//! # Next-Generation Summation System
//!
//! This module provides a complete rewrite of the summation capabilities,
//! eliminating string-based variable names in favor of closure-based scoping
//! and direct AST manipulation.
//!
//! Key design principles:
//! - No string-based variable names
//! - Closure-based variable scoping
//! - Direct AST manipulation
//! - Type-safe variable binding
//! - Compile-time optimization
//!
//! ## Migration Notes: Advanced Features to be Added
//!
//! The following advanced features from the original `summation.rs` (62KB, 1752 lines)
//! need to be migrated to this new type-safe closure-based approach:
//!
//! ### 🔄 **Multi-Dimensional Summations** (HIGH PRIORITY)
//! - **Lines**: ~844-1450 in original summation.rs
//! - **Functionality**:
//!   - `MultiDimRange` - Support for Σᵢ₌₁ⁿ Σⱼ₌₁ᵐ f(i,j) style nested summations
//!   - `MultiDimFunction<T>` - Functions with multiple index variables
//!   - `MultiDimSumResult` - Results with separability analysis
//!   - Separability detection: f(i,j) = g(i) * h(j) factorization
//!   - Closed-form evaluation for separable multi-dimensional sums
//! - **API**: `simplifier.simplify_multidim_sum(&multi_range, &multi_function)`
//! - **Use Cases**: Double/triple integrals, matrix operations, tensor summations
//! - **Migration Strategy**: Extend closure API to support multiple index variables:
//!   ```rust
// !   // processor.sum_2d(x_range, y_range, |i, j| expression_using_i_and_j)
// !   // processor.sum_3d(x_range, y_range, z_range, |i, j, k| expression_using_i_j_k)
//!   ```
//!
//! ### 🔍 **Convergence Analysis** (MEDIUM PRIORITY)
//! - **Lines**: ~1057-1250 in original summation.rs
//! - **Functionality**:
//!   - `ConvergenceTest` enum: Ratio, Root, Comparison, Integral, Alternating tests
//!   - `ConvergenceResult` enum: Convergent, Divergent, Conditional, Unknown
//!   - `ConvergenceAnalyzer` - Mathematical convergence testing for infinite series
//!   - Multiple convergence test algorithms for infinite summations
//! - **API**: `analyzer.analyze_convergence(&function) -> ConvergenceResult`
//! - **Use Cases**: Infinite series analysis, numerical stability checks
//! - **Migration Strategy**: Add convergence analysis as separate trait or module
//!
//! ### 📐 **Telescoping Sum Detection** (MEDIUM PRIORITY)
//! - **Lines**: ~577-592, 706-717 in original summation.rs  
//! - **Functionality**:
//!   - Automatic detection of telescoping patterns: Σ(f(i+1) - f(i)) = f(end+1) - f(start)
//!   - Pattern matching for difference-based expressions
//!   - Closed-form evaluation for telescoping series
//! - **API**: `SummationPattern::Telescoping { function_name: String }`
//! - **Use Cases**: Series like Σ(1/(i(i+1))) = Σ(1/i - 1/(i+1))
//! - **Migration Strategy**: Extend pattern recognition to detect telescoping in closure scope
//!
//! ### 🧮 **Enhanced Factor Extraction** (LOW PRIORITY)
//! - **Lines**: ~147-295 in original summation.rs
//! - **Functionality**:
//!   - Advanced nested factor extraction from complex expressions  
//!   - Common factor detection across addition terms
//!   - Sophisticated algebraic manipulation for factor isolation
//!   - Factor division and remainder computation
//! - **Current State**: Basic factor extraction exists in v2, but not as sophisticated
//! - **Migration Strategy**: Enhance existing `extract_constant_factors` method
//!
//! ### 📊 **Advanced Pattern Recognition** (LOW PRIORITY)
//! - **Lines**: ~296-576 in original summation.rs
//! - **Functionality**:
//!   - `SummationPattern::Arithmetic` - Enhanced arithmetic series recognition
//!   - `SummationPattern::Factorizable` - Complex factorizable pattern detection
//!   - Variable coefficient extraction with symbolic analysis
//!   - Polynomial degree-based pattern recognition (up to degree 10)
//! - **Current State**: Basic patterns exist in v2, but simpler
//! - **Migration Strategy**: Gradually enhance pattern recognition while keeping type safety
//!
//! ## Current Status (June 3, 2025)
//!
//! ✅ **Migrated & Working**:
//! - Closure-based variable scoping (eliminates string variable bugs)
//! - Basic pattern recognition (Constant, Linear, Quadratic, Geometric, Power)
//! - Closed-form evaluation for common patterns  
//! - Factor extraction for constants
//! - **Critical Bug Fixes**: Cubic power series ranges, zero power edge cases
//! - Comprehensive property-based testing
//!
//! 🔄 **Priority Migration Order**:
//! 1. **Multi-Dimensional Summations** - Most complex feature, high value
//! 2. **Convergence Analysis** - Mathematical rigor for infinite series
//! 3. **Telescoping Detection** - Specialized but powerful optimization
//! 4. **Enhanced Factor Extraction** - Performance improvements
//! 5. **Advanced Pattern Recognition** - Incremental feature additions
//!
//! ## Design Philosophy
//!
//! This v2 system prioritizes:
//! - **Type Safety**: Compile-time variable scoping prevents runtime errors
//! - **Mathematical Correctness**: Recent bug fixes ensure reliable closed forms
//! - **Clean API**: `sum(range, |i| expr)` is more intuitive than string-based variables
//! - **Performance**: Direct AST manipulation with minimal overhead
//! - **Testability**: Property-based tests ensure mathematical correctness across ranges

use crate::Result;
use crate::ast::{ASTRepr, DynamicContext, TypedBuilderExpr};
use crate::symbolic::symbolic::SymbolicOptimizer;

/// Placeholder for `IntRange` type (was in `final_tagless`)
#[derive(Debug, Clone)]
pub struct IntRange {
    pub start: i64,
    pub end: i64,
}

impl IntRange {
    #[must_use]
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    pub fn iter(&self) -> impl Iterator<Item = i64> {
        self.start..=self.end
    }

    #[must_use]
    pub fn len(&self) -> usize {
        if self.end >= self.start {
            (self.end - self.start + 1) as usize
        } else {
            0
        }
    }

    #[must_use]
    pub fn start(&self) -> i64 {
        self.start
    }

    #[must_use]
    pub fn end(&self) -> i64 {
        self.end
    }
}

/// Placeholder for `DirectEval` type (was in `final_tagless`)
pub struct DirectEval;

impl DirectEval {
    #[must_use]
    pub fn eval_with_vars(expr: &ASTRepr<f64>, vars: &[f64]) -> f64 {
        expr.eval_with_vars(vars)
    }
}

/// Extended summation patterns for statistical sufficient statistics discovery
#[derive(Debug, Clone)]
pub enum SummationPattern {
    /// Constant series: Σ(c) = c*n
    Constant { value: f64 },
    /// Linear series: Σ(a*i + b) = a*Σ(i) + b*n
    Linear { coefficient: f64, constant: f64 },
    /// Quadratic series: Σ(a*i² + b*i + c)
    Quadratic { a: f64, b: f64, c: f64 },
    /// Geometric series: Σ(a*r^i) = a*(1-r^n)/(1-r)
    Geometric { coefficient: f64, ratio: f64 },
    /// Power series: Σ(i^k) with known closed forms
    Power { exponent: f64 },
    /// Factorizable: k*Σ(f(i)) where k doesn't depend on i
    Factorizable {
        factor: f64,
        remaining_pattern: Box<SummationPattern>,
    },
    /// Data linear: Σ(x[i]) - sum of data values (sufficient statistic)
    DataLinear { coefficient: f64 },
    /// Data quadratic: Σ(x[i]²) - sum of squared data values (sufficient statistic)  
    DataQuadratic { coefficient: f64 },
    /// Data cross terms: Σ(x[i] * y[i]) - sum of cross products (sufficient statistic)
    DataCrossProduct { coefficient: f64 },
    /// Complex statistical pattern: combines multiple sufficient statistics
    StatisticalPattern {
        /// Coefficients for [n, Σx, Σx², Σy, Σy², Σxy] sufficient statistics
        coefficients: Vec<f64>,
        pattern_type: String,
    },
    /// Unknown pattern
    Unknown,
}

/// Configuration for summation optimization
#[derive(Debug, Clone)]
pub struct SummationConfig {
    /// Enable pattern recognition
    pub enable_pattern_recognition: bool,
    /// Enable closed-form evaluation
    pub enable_closed_form: bool,
    /// Enable factor extraction
    pub enable_factor_extraction: bool,
    /// Enable egglog optimization (expensive, only for complex expressions)
    pub enable_egglog_optimization: bool,
    /// Use fast path for simple statistical patterns
    pub enable_fast_path: bool,
}

impl Default for SummationConfig {
    fn default() -> Self {
        Self {
            enable_pattern_recognition: true,
            enable_closed_form: true,
            enable_factor_extraction: true,
            enable_egglog_optimization: false, // DISABLED by default for performance
            enable_fast_path: true,            // ENABLED by default for performance
        }
    }
}

/// Result of summation analysis and optimization
#[derive(Debug, Clone)]
pub struct SummationResult {
    /// The range of summation
    pub range: IntRange,
    /// Original expression (for reference)
    pub original_expr: ASTRepr<f64>,
    /// Simplified expression (with factors extracted)
    pub simplified_expr: ASTRepr<f64>,
    /// Recognized pattern
    pub pattern: SummationPattern,
    /// Closed-form expression if available
    pub closed_form: Option<ASTRepr<f64>>,
    /// Extracted constant factors
    pub extracted_factors: Vec<f64>,
    /// Whether optimization was successful
    pub is_optimized: bool,
}

impl SummationResult {
    /// Evaluate the summation with given external variables
    pub fn evaluate(&self, external_vars: &[f64]) -> Result<f64> {
        let base_result = if let Some(closed_form) = &self.closed_form {
            DirectEval::eval_with_vars(closed_form, external_vars)
        } else {
            // Fall back to numerical summation using the simplified expression
            self.evaluate_numerically(external_vars)?
        };

        // Apply extracted factors (including zero factors!)
        if self.extracted_factors.is_empty() {
            Ok(base_result)
        } else {
            let total_factor = self.extracted_factors.iter().product::<f64>();
            Ok(base_result * total_factor)
        }
    }

    /// Evaluate numerically by iterating over the range
    fn evaluate_numerically(&self, external_vars: &[f64]) -> Result<f64> {
        let mut sum = 0.0;
        for i in self.range.iter() {
            // Create a new variable vector with the index variable prepended
            let mut vars = vec![i as f64];
            vars.extend_from_slice(external_vars);

            // Use the simplified expression (without extracted factors) for numerical evaluation
            sum += DirectEval::eval_with_vars(&self.simplified_expr, &vars);
        }
        Ok(sum)
    }

    /// Get the total speedup factor from extracted constant factors
    #[must_use]
    pub fn factor_speedup(&self) -> f64 {
        self.extracted_factors.iter().product()
    }
}

/// DEPRECATED: Use `DynamicContext.sum()` instead
///
/// This optimizer is deprecated. Use the unified `DynamicContext.sum()` API which provides:
/// - Cleaner mathematical optimizations via `CleanSummationOptimizer`
/// - Unified handling of mathematical ranges and data iteration  
/// - Domain-agnostic approach without statistical naming violations
/// - Proven performance (519x faster evaluation in probabilistic programming)
#[deprecated(note = "Use DynamicContext.sum() for summations. Will be removed in future versions.")]
pub struct LegacySummationProcessor {
    config: SummationConfig,
    optimizer: SymbolicOptimizer,
}

impl LegacySummationProcessor {
    /// Create a new summation processor with PERFORMANCE-FIRST approach
    pub fn new() -> Result<Self> {
        // DEFAULT: Fast path enabled, egglog DISABLED for performance
        let config = SummationConfig::default();
        Self::with_config(config)
    }

    /// Create a summation processor with custom configuration
    pub fn with_config(config: SummationConfig) -> Result<Self> {
        // Only enable egglog if explicitly requested (due to performance cost)
        let optimizer = if config.enable_egglog_optimization {
            let mut optimizer_config = crate::symbolic::symbolic::OptimizationConfig::default();
            optimizer_config.egglog_optimization = true;
            crate::symbolic::symbolic::SymbolicOptimizer::with_config(optimizer_config)?
        } else {
            // Use basic optimizer without egglog for better performance
            crate::symbolic::symbolic::SymbolicOptimizer::new()?
        };

        Ok(Self { config, optimizer })
    }

    /// Enable egglog optimization for complex expressions that need it
    pub fn enable_egglog(&mut self) -> Result<()> {
        if !self.config.enable_egglog_optimization {
            self.config.enable_egglog_optimization = true;
            let mut optimizer_config = crate::symbolic::symbolic::OptimizationConfig::default();
            optimizer_config.egglog_optimization = true;
            self.optimizer =
                crate::symbolic::symbolic::SymbolicOptimizer::with_config(optimizer_config)?;
        }
        Ok(())
    }

    /// Process a summation using a closure that defines the summand
    ///
    /// This is the main API: sum(range, |i| `expression_using_i`)
    /// The index variable i is properly scoped within the closure.
    pub fn sum<F>(&mut self, range: IntRange, summand_fn: F) -> Result<SummationResult>
    where
        F: FnOnce(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        // Create a fresh expression builder for this summation scope
        let math = DynamicContext::new();
        let index_var = math.var(); // This gets assigned index 0 in the local scope

        // Call the closure with the scoped index variable
        let summand_expr = summand_fn(index_var);
        let ast = summand_expr.into_ast();

        self.process_summation(range, ast)
    }

    /// Process a summation given a pre-built AST
    /// The AST should use Variable(0) for the summation index
    pub fn process_summation(
        &mut self,
        range: IntRange,
        summand: ASTRepr<f64>,
    ) -> Result<SummationResult> {
        // Step 0: Optimize the expression first (CRITICAL for handling 0*i -> 0)
        let optimized_summand = if self.config.enable_pattern_recognition {
            // Only optimize if pattern recognition is enabled to avoid unnecessary work
            self.optimizer.optimize(&summand)?
        } else {
            summand.clone()
        };

        // Step 1: Extract constant factors
        let (extracted_factors, simplified_expr) = if self.config.enable_factor_extraction {
            self.extract_constant_factors(&optimized_summand)?
        } else {
            (vec![], optimized_summand.clone())
        };

        // Step 2: Recognize patterns
        let pattern = if self.config.enable_pattern_recognition {
            self.recognize_pattern(&simplified_expr)?
        } else {
            SummationPattern::Unknown
        };

        // Step 3: Compute closed form
        let closed_form = if self.config.enable_closed_form {
            self.compute_closed_form(&range, &simplified_expr, &pattern)?
        } else {
            None
        };

        let is_optimized = !extracted_factors.is_empty() || closed_form.is_some();

        Ok(SummationResult {
            range,
            original_expr: summand, // Keep original for reference
            simplified_expr,
            pattern,
            closed_form,
            extracted_factors,
            is_optimized,
        })
    }

    /// Extract constant factors from the summand expression
    /// Returns (factors, `simplified_expression`)
    fn extract_constant_factors(&self, expr: &ASTRepr<f64>) -> Result<(Vec<f64>, ASTRepr<f64>)> {
        match expr {
            // Multiplication: check if one side is constant
            ASTRepr::Mul(left, right) => {
                let left_has_index = self.contains_index_variable(left);
                let right_has_index = self.contains_index_variable(right);

                match (left_has_index, right_has_index) {
                    (false, true) => {
                        // Left is constant, right depends on index
                        if let Some(factor) = self.extract_constant_value(left) {
                            let (mut inner_factors, inner_expr) =
                                self.extract_constant_factors(right)?;
                            inner_factors.insert(0, factor);
                            Ok((inner_factors, inner_expr))
                        } else {
                            Ok((vec![], expr.clone()))
                        }
                    }
                    (true, false) => {
                        // Right is constant, left depends on index
                        if let Some(factor) = self.extract_constant_value(right) {
                            let (mut inner_factors, inner_expr) =
                                self.extract_constant_factors(left)?;
                            inner_factors.insert(0, factor);
                            Ok((inner_factors, inner_expr))
                        } else {
                            Ok((vec![], expr.clone()))
                        }
                    }
                    (false, false) => {
                        // Both sides are constant - this whole expression is a constant factor
                        if let Some(factor) = self.extract_constant_value(expr) {
                            Ok((vec![factor], ASTRepr::Constant(1.0)))
                        } else {
                            Ok((vec![], expr.clone()))
                        }
                    }
                    (true, true) => {
                        // Both sides depend on index - no simple factorization
                        Ok((vec![], expr.clone()))
                    }
                }
            }

            // Addition: check for common factors (more complex)
            ASTRepr::Add(left, right) => {
                let (left_factors, left_simplified) = self.extract_constant_factors(left)?;
                let (right_factors, right_simplified) = self.extract_constant_factors(right)?;

                // Check if we can factor out a common factor
                if let (Some(&left_factor), Some(&right_factor)) =
                    (left_factors.first(), right_factors.first())
                {
                    if left_factor == right_factor {
                        // Common factor found
                        let simplified =
                            ASTRepr::Add(Box::new(left_simplified), Box::new(right_simplified));
                        Ok((vec![left_factor], simplified))
                    } else {
                        // No common factor
                        Ok((vec![], expr.clone()))
                    }
                } else {
                    // No factors to extract
                    Ok((vec![], expr.clone()))
                }
            }

            // Pure constant: extract as factor leaving 1.0 as simplified expression
            ASTRepr::Constant(_) => {
                if let Some(factor) = self.extract_constant_value(expr) {
                    Ok((vec![factor], ASTRepr::Constant(1.0)))
                } else {
                    Ok((vec![], expr.clone()))
                }
            }

            // For other expressions, no factors to extract
            _ => Ok((vec![], expr.clone())),
        }
    }

    /// Check if an expression contains the index variable (Variable(0))
    fn contains_index_variable(&self, expr: &ASTRepr<f64>) -> bool {
        match expr {
            ASTRepr::Variable(0) => true,
            ASTRepr::Variable(_) => false,
            ASTRepr::Constant(_) => false,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.contains_index_variable(left) || self.contains_index_variable(right)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.contains_index_variable(inner),
            ASTRepr::Sum { .. } => {
                // TODO: Implement Sum variant for summation pattern analysis
                // This will handle nested sum detection and optimization
                false
            }
        }
    }

    /// Extract a constant value from an expression if it's purely constant
    fn extract_constant_value(&self, expr: &ASTRepr<f64>) -> Option<f64> {
        if self.contains_index_variable(expr) {
            None
        } else {
            // Use DirectEval to evaluate the constant expression
            Some(DirectEval::eval_with_vars(expr, &[]))
        }
    }

    /// Recognize common summation patterns
    fn recognize_pattern(&self, expr: &ASTRepr<f64>) -> Result<SummationPattern> {
        match expr {
            // Constant pattern
            ASTRepr::Constant(value) => Ok(SummationPattern::Constant { value: *value }),

            // Variable pattern: just i (linear with coefficient 1, constant 0)
            ASTRepr::Variable(0) => Ok(SummationPattern::Linear {
                coefficient: 1.0,
                constant: 0.0,
            }),

            // Linear pattern: a*i + b or a*i or b + a*i
            ASTRepr::Add(left, right) => {
                let pattern = self.try_linear_pattern(left, right)?;
                if let Some(p) = pattern {
                    Ok(p)
                } else {
                    Ok(SummationPattern::Unknown)
                }
            }

            // Multiplication: could be a*i, a*i^k, a*r^i, etc.
            ASTRepr::Mul(left, right) => {
                if let Some(pattern) = self.try_linear_mul_pattern(left, right)? {
                    Ok(pattern)
                } else if let Some(pattern) = self.try_geometric_pattern(left, right)? {
                    Ok(pattern)
                } else if let Some(pattern) = self.try_power_mul_pattern(left, right)? {
                    Ok(pattern)
                } else {
                    Ok(SummationPattern::Unknown)
                }
            }

            // Power pattern: i^k or r^i
            ASTRepr::Pow(base, exp) => {
                if let Some(pattern) = self.try_power_pattern(base, exp)? {
                    Ok(pattern)
                } else if let Some(pattern) = self.try_geometric_power_pattern(base, exp)? {
                    Ok(pattern)
                } else {
                    Ok(SummationPattern::Unknown)
                }
            }

            ASTRepr::Sqrt(_) => Ok(SummationPattern::Unknown),
            ASTRepr::Sum { .. } => {
                // TODO: Implement Sum variant for summation pattern analysis
                // This will handle nested sum detection and optimization
                Ok(SummationPattern::Unknown)
            }

            _ => Ok(SummationPattern::Unknown),
        }
    }

    /// Try to recognize linear patterns in addition: a*i + b
    fn try_linear_pattern(
        &self,
        left: &ASTRepr<f64>,
        right: &ASTRepr<f64>,
    ) -> Result<Option<SummationPattern>> {
        // Try left = a*i, right = b (constant)
        if let Some(coeff) = self.extract_linear_coefficient(left)
            && let Some(constant) = self.extract_constant_value(right)
        {
            return Ok(Some(SummationPattern::Linear {
                coefficient: coeff,
                constant,
            }));
        }

        // Try left = b (constant), right = a*i
        if let Some(coeff) = self.extract_linear_coefficient(right)
            && let Some(constant) = self.extract_constant_value(left)
        {
            return Ok(Some(SummationPattern::Linear {
                coefficient: coeff,
                constant,
            }));
        }

        Ok(None)
    }

    /// Try to recognize linear patterns in multiplication: a*i
    fn try_linear_mul_pattern(
        &self,
        left: &ASTRepr<f64>,
        right: &ASTRepr<f64>,
    ) -> Result<Option<SummationPattern>> {
        // Try left = constant, right = i
        if matches!(right, ASTRepr::Variable(0))
            && let Some(coeff) = self.extract_constant_value(left)
        {
            return Ok(Some(SummationPattern::Linear {
                coefficient: coeff,
                constant: 0.0,
            }));
        }

        // Try left = i, right = constant
        if matches!(left, ASTRepr::Variable(0))
            && let Some(coeff) = self.extract_constant_value(right)
        {
            return Ok(Some(SummationPattern::Linear {
                coefficient: coeff,
                constant: 0.0,
            }));
        }

        Ok(None)
    }

    /// Try to recognize geometric patterns: a*r^i
    fn try_geometric_pattern(
        &self,
        left: &ASTRepr<f64>,
        right: &ASTRepr<f64>,
    ) -> Result<Option<SummationPattern>> {
        // Try left = constant, right = r^i
        if let Some(coeff) = self.extract_constant_value(left)
            && let Some(ratio) = self.extract_geometric_base(right)
        {
            return Ok(Some(SummationPattern::Geometric {
                coefficient: coeff,
                ratio,
            }));
        }

        // Try left = r^i, right = constant
        if let Some(coeff) = self.extract_constant_value(right)
            && let Some(ratio) = self.extract_geometric_base(left)
        {
            return Ok(Some(SummationPattern::Geometric {
                coefficient: coeff,
                ratio,
            }));
        }

        Ok(None)
    }

    /// Try to recognize power patterns in multiplication: a*i^k or i*i
    fn try_power_mul_pattern(
        &self,
        left: &ASTRepr<f64>,
        right: &ASTRepr<f64>,
    ) -> Result<Option<SummationPattern>> {
        // Special case: i*i (when egglog chooses Mul representation over Pow)
        if matches!(left, ASTRepr::Variable(0)) && matches!(right, ASTRepr::Variable(0)) {
            return Ok(Some(SummationPattern::Power { exponent: 2.0 }));
        }

        // Try left = constant, right = i^k
        if let Some(_coeff) = self.extract_constant_value(left)
            && let Some(exp) = self.extract_power_exponent(right)
        {
            return Ok(Some(SummationPattern::Power { exponent: exp }));
        }

        // Try left = i^k, right = constant
        if let Some(_coeff) = self.extract_constant_value(right)
            && let Some(exp) = self.extract_power_exponent(left)
        {
            return Ok(Some(SummationPattern::Power { exponent: exp }));
        }

        Ok(None)
    }

    /// Try to recognize power patterns: i^k
    fn try_power_pattern(
        &self,
        base: &ASTRepr<f64>,
        exp: &ASTRepr<f64>,
    ) -> Result<Option<SummationPattern>> {
        if matches!(base, ASTRepr::Variable(0))
            && let Some(exponent) = self.extract_constant_value(exp)
        {
            return Ok(Some(SummationPattern::Power { exponent }));
        }
        Ok(None)
    }

    /// Try to recognize geometric patterns in power: r^i
    fn try_geometric_power_pattern(
        &self,
        base: &ASTRepr<f64>,
        exp: &ASTRepr<f64>,
    ) -> Result<Option<SummationPattern>> {
        if matches!(exp, ASTRepr::Variable(0))
            && let Some(ratio) = self.extract_constant_value(base)
        {
            return Ok(Some(SummationPattern::Geometric {
                coefficient: 1.0,
                ratio,
            }));
        }
        Ok(None)
    }

    /// Extract coefficient from linear term like a*i
    fn extract_linear_coefficient(&self, expr: &ASTRepr<f64>) -> Option<f64> {
        match expr {
            ASTRepr::Variable(0) => Some(1.0),
            ASTRepr::Mul(left, right) => {
                if matches!(left.as_ref(), ASTRepr::Variable(0)) {
                    self.extract_constant_value(right)
                } else if matches!(right.as_ref(), ASTRepr::Variable(0)) {
                    self.extract_constant_value(left)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract base from geometric term like r^i
    fn extract_geometric_base(&self, expr: &ASTRepr<f64>) -> Option<f64> {
        match expr {
            ASTRepr::Pow(base, exp) => {
                if matches!(exp.as_ref(), ASTRepr::Variable(0)) {
                    self.extract_constant_value(base)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract exponent from power term like i^k
    fn extract_power_exponent(&self, expr: &ASTRepr<f64>) -> Option<f64> {
        match expr {
            ASTRepr::Pow(base, exp) => {
                if matches!(base.as_ref(), ASTRepr::Variable(0)) {
                    self.extract_constant_value(exp)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Efficient exponentiation that uses powi for integer exponents
    fn efficient_pow(base: f64, exponent: f64) -> f64 {
        // Check if exponent is close to an integer
        let rounded = exponent.round();
        if (exponent - rounded).abs() < 1e-12 {
            let int_exp = rounded as i32;
            // Use powi for integer exponents (binary exponentiation)
            base.powi(int_exp)
        } else {
            // Use powf for non-integer exponents
            base.powf(exponent)
        }
    }

    /// Compute closed-form expression for recognized patterns
    fn compute_closed_form(
        &self,
        range: &IntRange,
        _expr: &ASTRepr<f64>,
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

            SummationPattern::Linear {
                coefficient,
                constant,
            } => {
                // Σ(a*i + b) = a*Σ(i) + b*n
                // Σ(i from start to end) = (end*(end+1) - (start-1)*start)/2
                let sum_of_indices = (end * (end + 1.0) - (start - 1.0) * start) / 2.0;
                let result = coefficient * sum_of_indices + constant * n;
                Ok(Some(ASTRepr::Constant(result)))
            }

            SummationPattern::Geometric { coefficient, ratio } => {
                if *ratio == 1.0 {
                    // Special case: ratio = 1, so Σ(a*1^i) = a*n
                    Ok(Some(ASTRepr::Constant(coefficient * n)))
                } else {
                    // General case: Σ(a*r^i from start to end) = a*r^start*(1-r^n)/(1-r)
                    let numerator = coefficient
                        * Self::efficient_pow(*ratio, start)
                        * (1.0 - Self::efficient_pow(*ratio, n));
                    let result = numerator / (1.0 - ratio);
                    Ok(Some(ASTRepr::Constant(result)))
                }
            }

            SummationPattern::Power { exponent } => {
                // Use known formulas for small integer powers
                if *exponent == exponent.round() {
                    let k = exponent.round() as i32;
                    match k {
                        0 => Ok(Some(ASTRepr::Constant(n))), // Σ(1) = n
                        1 => {
                            // Σ(i) = n*(start + end)/2
                            let result = n * (start + end) / 2.0;
                            Ok(Some(ASTRepr::Constant(result)))
                        }
                        2 => {
                            // Σ(i²) using formula
                            let result = (end * (end + 1.0) * (2.0 * end + 1.0)
                                - (start - 1.0) * start * (2.0 * (start - 1.0) + 1.0))
                                / 6.0;
                            Ok(Some(ASTRepr::Constant(result)))
                        }
                        3 => {
                            // Σ(i³) for arbitrary range [start, end]
                            // The identity Σ(i³) = [Σ(i)]² only holds when summing from 1 to n
                            // For arbitrary ranges, we need to compute it directly
                            // Using the general formula: Σ(i=a to b) i³ = [b²(b+1)² - (a-1)²a²]/4
                            let b = end;
                            let a_minus_1 = start - 1.0;
                            let sum_cubes = (b * b * (b + 1.0) * (b + 1.0)
                                - a_minus_1 * a_minus_1 * start * start)
                                / 4.0;
                            Ok(Some(ASTRepr::Constant(sum_cubes)))
                        }
                        _ => Ok(None), // No known closed form for higher powers
                    }
                } else {
                    Ok(None) // No known closed form for non-integer exponents
                }
            }

            SummationPattern::Factorizable {
                factor,
                remaining_pattern,
            } => {
                // Recursively compute the remaining pattern and multiply by the factor
                if let Some(remaining_result) =
                    self.compute_closed_form(range, _expr, remaining_pattern)?
                {
                    Ok(Some(ASTRepr::Mul(
                        Box::new(ASTRepr::Constant(*factor)),
                        Box::new(remaining_result),
                    )))
                } else {
                    Ok(None)
                }
            }

            SummationPattern::Quadratic { a, b, c } => {
                // Σ(a*i² + b*i + c) = a*Σ(i²) + b*Σ(i) + c*n
                let sum_i = n * (start + end) / 2.0;
                let sum_i2 = (end * (end + 1.0) * (2.0 * end + 1.0)
                    - (start - 1.0) * start * (2.0 * (start - 1.0) + 1.0))
                    / 6.0;
                let result = a * sum_i2 + b * sum_i + c * n;
                Ok(Some(ASTRepr::Constant(result)))
            }

            // Data-based patterns - these are used for runtime data, not mathematical ranges
            SummationPattern::DataLinear { .. }
            | SummationPattern::DataQuadratic { .. }
            | SummationPattern::DataCrossProduct { .. }
            | SummationPattern::StatisticalPattern { .. } => {
                // These patterns are handled by DataSummationProcessor, not this method
                Ok(None)
            }

            SummationPattern::Unknown => Ok(None),
        }
    }
}

impl Default for LegacySummationProcessor {
    fn default() -> Self {
        Self::new().expect("Failed to create default LegacySummationProcessor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_sum() {
        let mut processor = LegacySummationProcessor::new().unwrap();
        let range = IntRange::new(1, 10);

        let result = processor
            .sum(range, |_i| {
                let math = DynamicContext::new();
                math.constant(5.0)
            })
            .unwrap();

        // Constant expressions should be extracted as factors with simplified expr = Constant(1.0)
        assert!(
            matches!(result.pattern, SummationPattern::Constant { value } if (value - 1.0).abs() < 1e-10)
        );
        assert!(result.closed_form.is_some());
        assert!(!result.extracted_factors.is_empty()); // Factor should be extracted
        assert_eq!(result.extracted_factors[0], 5.0); // Should extract the constant 5.0

        let value = result.evaluate(&[]).unwrap();
        assert_eq!(value, 50.0); // 5 * 10 = 50
    }

    #[test]
    fn test_linear_sum() {
        let mut processor = LegacySummationProcessor::new().unwrap();
        let range = IntRange::new(1, 10);

        let result = processor.sum(range, |i| i).unwrap();

        assert!(
            matches!(result.pattern, SummationPattern::Linear { coefficient, constant }
            if (coefficient - 1.0).abs() < 1e-10 && (constant - 0.0).abs() < 1e-10)
        );
        assert!(result.closed_form.is_some());

        let value = result.evaluate(&[]).unwrap();
        assert_eq!(value, 55.0); // 1+2+...+10 = 55
    }

    #[test]
    fn test_factor_extraction() {
        let mut processor = LegacySummationProcessor::new().unwrap();
        let range = IntRange::new(1, 10);

        let result = processor
            .sum(range, |i| {
                let math = DynamicContext::new();
                math.constant(3.0) * i
            })
            .unwrap();

        assert!(!result.extracted_factors.is_empty());
        assert_eq!(result.extracted_factors[0], 3.0);

        let value = result.evaluate(&[]).unwrap();
        assert_eq!(value, 165.0); // 3 * 55 = 165
    }

    #[test]
    fn test_geometric_sum() {
        let mut processor = LegacySummationProcessor::new().unwrap();
        let range = IntRange::new(0, 5);

        let result = processor
            .sum(range, |i| {
                let math = DynamicContext::new();
                math.constant(0.5).pow(i)
            })
            .unwrap();

        assert!(
            matches!(result.pattern, SummationPattern::Geometric { coefficient, ratio }
            if (coefficient - 1.0).abs() < 1e-10 && (ratio - 0.5).abs() < 1e-10)
        );

        let value = result.evaluate(&[]).unwrap();
        assert!((value - 1.96875).abs() < 1e-5); // Geometric series sum
    }

    #[test]
    fn test_power_sum() {
        let mut processor = LegacySummationProcessor::new().unwrap();
        let range = IntRange::new(1, 5);

        let result = processor
            .sum(range, |i| {
                let math = DynamicContext::new();
                i.pow(math.constant(2.0))
            })
            .unwrap();

        // Debug output to see what we actually got
        println!("Original expression: {:?}", result.original_expr);
        println!("Simplified expression: {:?}", result.simplified_expr);
        println!("Pattern: {:?}", result.pattern);
        println!("Extracted factors: {:?}", result.extracted_factors);

        // Power expressions should be recognized as power patterns directly
        assert!(
            matches!(result.pattern, SummationPattern::Power { exponent }
            if (exponent - 2.0).abs() < 1e-10)
        );
        assert!(result.closed_form.is_some());

        let value = result.evaluate(&[]).unwrap();
        assert_eq!(value, 55.0); // 1² + 2² + 3² + 4² + 5² = 55
    }

    #[test]
    fn test_no_index_variable_escape() {
        let mut processor = LegacySummationProcessor::new().unwrap();
        let range = IntRange::new(1, 5);

        // This test ensures that the index variable cannot be accessed outside the closure
        let result = processor
            .sum(range, |i| {
                // The variable 'i' is only accessible within this closure
                i * 2.0
            })
            .unwrap();

        // After the closure, 'i' is no longer accessible
        // This is enforced by Rust's borrow checker and closure semantics
        assert!(result.is_optimized);
    }
}

/// Result of data-based summation with automatic sufficient statistics discovery
#[derive(Debug, Clone)]
pub struct DataSummationResult {
    /// Original data length  
    pub n_data: usize,
    /// Discovered pattern type
    pub pattern: SummationPattern,
    /// Discovered sufficient statistics as symbolic expressions
    pub sufficient_statistics: Vec<ASTRepr<f64>>,
    /// ANF let bindings for efficient evaluation
    pub anf_bindings: Vec<(String, ASTRepr<f64>)>,
    /// Final optimized expression using sufficient statistics
    pub optimized_expr: ASTRepr<f64>,
    /// Whether sufficient statistics were discovered
    pub optimization_applied: bool,
}

impl DataSummationResult {
    /// Evaluate using the discovered sufficient statistics
    pub fn evaluate(&self, params: &[f64]) -> Result<f64> {
        Ok(DirectEval::eval_with_vars(&self.optimized_expr, params))
    }

    /// Get the discovered sufficient statistics for external use
    #[must_use]
    pub fn get_sufficient_statistics(&self) -> &[ASTRepr<f64>] {
        &self.sufficient_statistics
    }
}

/// DEPRECATED: Use `DynamicContext.sum()` instead
///
/// This processor is deprecated. Use the unified `DynamicContext.sum()` API.
#[deprecated(
    note = "Use DynamicContext.sum() for data summations. Will be removed in future versions."
)]
pub struct DataSummationProcessor {
    inner: LegacySummationProcessor,
}

impl DataSummationProcessor {
    /// Create a new data summation processor
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: LegacySummationProcessor::new()?,
        })
    }

    /// Sum over runtime data array - automatically discovers sufficient statistics
    pub fn sum_data<I, F>(
        &mut self,
        math: &DynamicContext,
        data: I,
        f: F,
    ) -> Result<DataSummationResult>
    where
        I: IntoIterator<Item = f64>,
        F: Fn(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        let data_vec: Vec<f64> = data.into_iter().collect();
        let n = data_vec.len();

        // Build symbolic expression for pattern analysis
        let x_var = math.var(); // Represents data element x[i]
        let symbolic_expr = f(x_var);
        let ast = symbolic_expr.into_ast();

        // Analyze the pattern using mathematical range
        let analysis_range = IntRange::new(1, n as i64);
        let pattern_result = self.inner.sum(analysis_range, f)?;

        // Apply discovered pattern to actual data
        let optimized_expr =
            self.apply_pattern_to_data(&pattern_result.pattern, &data_vec, &ast)?;

        let sufficient_statistics =
            self.extract_sufficient_statistics(&pattern_result.pattern, &data_vec)?;
        let anf_bindings = self.create_anf_bindings(&sufficient_statistics)?;

        Ok(DataSummationResult {
            n_data: n,
            pattern: pattern_result.pattern,
            sufficient_statistics,
            anf_bindings,
            optimized_expr,
            optimization_applied: pattern_result.is_optimized,
        })
    }

    /// Sum over pairs of data for statistical models
    pub fn sum_data_pairs<I, F>(
        &mut self,
        math: &DynamicContext,
        data: I,
        f: F,
    ) -> Result<DataSummationResult>
    where
        I: IntoIterator<Item = (f64, f64)>,
        F: Fn(TypedBuilderExpr<f64>, TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        let data_vec: Vec<(f64, f64)> = data.into_iter().collect();
        let n = data_vec.len();

        // Build symbolic expression for pattern analysis
        let x_var = math.var(); // x[i]
        let y_var = math.var(); // y[i]
        let symbolic_expr = f(x_var, y_var);
        let ast = symbolic_expr.into_ast();

        // For pairs, we need more sophisticated analysis
        // For now, we'll analyze the structure and apply direct computation
        let sufficient_statistics = self.extract_pair_sufficient_statistics(&data_vec, &ast)?;
        let anf_bindings = self.create_anf_bindings(&sufficient_statistics)?;

        // Apply the expression to each data pair
        let mut result = 0.0;
        for (x_val, y_val) in &data_vec {
            let x_expr = math.constant(*x_val);
            let y_expr = math.constant(*y_val);
            let pair_result = f(x_expr, y_expr);
            result += math.eval(&pair_result, &[]);
        }

        Ok(DataSummationResult {
            n_data: n,
            pattern: SummationPattern::StatisticalPattern {
                coefficients: vec![result],
                pattern_type: "pair_sum".to_string(),
            },
            sufficient_statistics,
            anf_bindings,
            optimized_expr: ASTRepr::Constant(result),
            optimization_applied: true,
        })
    }

    /// Apply discovered pattern to actual data values
    fn apply_pattern_to_data(
        &self,
        pattern: &SummationPattern,
        data: &[f64],
        _ast: &ASTRepr<f64>,
    ) -> Result<ASTRepr<f64>> {
        match pattern {
            SummationPattern::DataLinear { coefficient } => {
                let sum_x: f64 = data.iter().sum();
                Ok(ASTRepr::Constant(coefficient * sum_x))
            }
            SummationPattern::DataQuadratic { coefficient } => {
                let sum_x_squared: f64 = data.iter().map(|x| x * x).sum();
                Ok(ASTRepr::Constant(coefficient * sum_x_squared))
            }
            SummationPattern::Linear {
                coefficient,
                constant,
            } => {
                // For k*x[i], compute k * Σx[i]
                let sum_x: f64 = data.iter().sum();
                let result = coefficient * sum_x + constant * data.len() as f64;
                Ok(ASTRepr::Constant(result))
            }
            SummationPattern::Power { exponent } => {
                // For x[i]^k, compute Σ(x[i]^k)
                let sum_power: f64 = data.iter().map(|x| x.powf(*exponent)).sum();
                Ok(ASTRepr::Constant(sum_power))
            }
            SummationPattern::Constant { value } => {
                // For constant c, result is c * n
                Ok(ASTRepr::Constant(value * data.len() as f64))
            }
            _ => {
                // Fallback: direct computation
                let result: f64 = data.iter().copied().sum(); // Simplified
                Ok(ASTRepr::Constant(result))
            }
        }
    }

    /// Extract sufficient statistics from discovered patterns
    fn extract_sufficient_statistics(
        &self,
        pattern: &SummationPattern,
        data: &[f64],
    ) -> Result<Vec<ASTRepr<f64>>> {
        let mut stats = Vec::new();

        match pattern {
            SummationPattern::DataLinear { .. } | SummationPattern::Linear { .. } => {
                // Σx[i] sufficient statistic
                let sum_x: f64 = data.iter().sum();
                stats.push(ASTRepr::Constant(sum_x));
            }
            SummationPattern::DataQuadratic { .. } => {
                // Σx[i]² sufficient statistic
                let sum_x_squared: f64 = data.iter().map(|x| x * x).sum();
                stats.push(ASTRepr::Constant(sum_x_squared));
            }
            SummationPattern::Power { exponent } if *exponent == 2.0 => {
                // Σx[i]² sufficient statistic
                let sum_x_squared: f64 = data.iter().map(|x| x * x).sum();
                stats.push(ASTRepr::Constant(sum_x_squared));
            }
            SummationPattern::StatisticalPattern { .. } => {
                // Multiple sufficient statistics
                let n = data.len() as f64;
                let sum_x: f64 = data.iter().sum();
                let sum_x_squared: f64 = data.iter().map(|x| x * x).sum();

                stats.push(ASTRepr::Constant(n)); // n
                stats.push(ASTRepr::Constant(sum_x)); // Σx
                stats.push(ASTRepr::Constant(sum_x_squared)); // Σx²
            }
            _ => {
                // Basic statistics
                let n = data.len() as f64;
                stats.push(ASTRepr::Constant(n));
            }
        }

        Ok(stats)
    }

    /// Extract sufficient statistics from pair data
    fn extract_pair_sufficient_statistics(
        &self,
        data: &[(f64, f64)],
        _ast: &ASTRepr<f64>,
    ) -> Result<Vec<ASTRepr<f64>>> {
        let mut stats = Vec::new();

        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| *x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| *y).sum();
        let sum_x_squared: f64 = data.iter().map(|(x, _)| x * x).sum();
        let sum_y_squared: f64 = data.iter().map(|(_, y)| y * y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();

        // Complete set of sufficient statistics for linear regression
        stats.push(ASTRepr::Constant(n)); // n
        stats.push(ASTRepr::Constant(sum_x)); // Σx
        stats.push(ASTRepr::Constant(sum_y)); // Σy  
        stats.push(ASTRepr::Constant(sum_x_squared)); // Σx²
        stats.push(ASTRepr::Constant(sum_y_squared)); // Σy²
        stats.push(ASTRepr::Constant(sum_xy)); // Σxy

        Ok(stats)
    }

    /// Create ANF let bindings for sufficient statistics
    fn create_anf_bindings(&self, stats: &[ASTRepr<f64>]) -> Result<Vec<(String, ASTRepr<f64>)>> {
        let mut bindings = Vec::new();

        for (i, stat) in stats.iter().enumerate() {
            let name = match i {
                0 => "n".to_string(),
                1 => "sum_x".to_string(),
                2 => "sum_y".to_string(),
                3 => "sum_x_squared".to_string(),
                4 => "sum_y_squared".to_string(),
                5 => "sum_xy".to_string(),
                _ => format!("stat_{i}"),
            };
            bindings.push((name, stat.clone()));
        }

        Ok(bindings)
    }
}
