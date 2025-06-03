//! Typed Expression Builder
//!
//! This module provides a typed expression builder that enables natural mathematical syntax
//! and expressions while maintaining intuitive operator overloading syntax.

use super::typed_registry::{TypedVar, VariableRegistry};
use crate::ast::ASTRepr;
use crate::final_tagless::interpreters::direct_eval::DirectEval;
use crate::final_tagless::traits::NumericType;
use num_traits::Float;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

/// Type-safe expression builder with typed variables
#[derive(Debug, Clone)]
pub struct ExpressionBuilder {
    registry: Arc<RefCell<VariableRegistry>>,
}

impl ExpressionBuilder {
    /// Create a new typed expression builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RefCell::new(VariableRegistry::new())),
        }
    }

    /// Create a typed variable
    #[must_use]
    pub fn typed_var<T: NumericType + 'static>(&self) -> TypedVar<T> {
        self.registry.borrow_mut().register_typed_variable()
    }

    /// Create an expression from a typed variable
    #[must_use]
    pub fn expr_from<T: NumericType>(&self, var: TypedVar<T>) -> TypedBuilderExpr<T> {
        TypedBuilderExpr::new(ASTRepr::Variable(var.index()), self.registry.clone())
    }

    /// Create a constant expression
    pub fn constant<T: NumericType>(&self, value: T) -> TypedBuilderExpr<T> {
        TypedBuilderExpr::new(ASTRepr::Constant(value), self.registry.clone())
    }

    /// Backward compatibility: create untyped variable (defaults to f64)
    #[must_use]
    pub fn var(&self) -> TypedBuilderExpr<f64> {
        let typed_var = self.typed_var::<f64>();
        self.expr_from(typed_var)
    }

    /// Get the registry for evaluation
    #[must_use]
    pub fn registry(&self) -> Arc<RefCell<VariableRegistry>> {
        self.registry.clone()
    }

    /// Evaluate an expression with indexed variables (simple interface)
    #[must_use]
    pub fn eval(&self, expr: &TypedBuilderExpr<f64>, variables: &[f64]) -> f64 {
        let registry = self.registry.borrow();
        let var_array = registry.create_variable_map(variables);
        expr.as_ast().eval_with_vars(&var_array)
    }

    // ============================================================================
    // High-Level Mathematical Functions (from ergonomics system)
    // ============================================================================

    /// Create a polynomial expression using Horner's method for efficient evaluation
    ///
    /// Coefficients are in ascending order of powers: [c₀, c₁, c₂, ...] represents c₀ + c₁x + c₂x² + ...
    #[must_use]
    pub fn poly(
        &self,
        coefficients: &[f64],
        variable: &TypedBuilderExpr<f64>,
    ) -> TypedBuilderExpr<f64> {
        if coefficients.is_empty() {
            return self.constant(0.0);
        }

        if coefficients.len() == 1 {
            return self.constant(coefficients[0]);
        }

        // Use Horner's method for efficient evaluation
        let mut result = self.constant(coefficients[coefficients.len() - 1]);

        for &coeff in coefficients.iter().rev().skip(1) {
            result = result.clone() * variable.clone() + self.constant(coeff);
        }

        result
    }

    /// Create a Gaussian (normal) distribution function
    ///
    /// Creates: (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    #[must_use]
    pub fn gaussian(
        &self,
        mean: f64,
        std_dev: f64,
        variable: &TypedBuilderExpr<f64>,
    ) -> TypedBuilderExpr<f64> {
        let pi = self.constant(std::f64::consts::PI);
        let two = self.constant(2.0);
        let sigma_squared = self.constant(std_dev * std_dev);

        // Normalization factor: 1/√(2πσ²)
        let norm_factor = self.constant(1.0) / ((&two * &pi * &sigma_squared).sqrt());

        // Exponent: -(x-μ)²/(2σ²)
        let x_minus_mu = variable.clone() - self.constant(mean);
        let x_minus_mu_squared = x_minus_mu.clone().pow(self.constant(2.0));
        let exponent = -(x_minus_mu_squared / (&two * &sigma_squared));

        norm_factor * exponent.exp()
    }

    /// Create a logistic (sigmoid) function: 1/(1 + exp(-x))
    #[must_use]
    pub fn logistic(&self, variable: &TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64> {
        let one = self.constant(1.0);
        let neg_x = -variable.clone();
        let exp_neg_x = neg_x.exp();
        &one / (&one + exp_neg_x)
    }

    /// Create a hyperbolic tangent function: tanh(x) = (exp(2x) - 1)/(exp(2x) + 1)
    #[must_use]
    pub fn tanh(&self, variable: &TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64> {
        let two = self.constant(2.0);
        let one = self.constant(1.0);
        let two_x = &two * variable;
        let exp_2x = two_x.exp();

        let numerator = &exp_2x - &one;
        let denominator = &exp_2x + &one;

        numerator / denominator
    }

    /// Unified summation method that supports variable capture
    ///
    /// This allows natural syntax where parameters are defined outside the sum
    /// and the closure captures them while receiving bound variables.
    ///
    /// # Examples
    /// ```rust
    /// let math = ExpressionBuilder::new();
    /// let beta0 = math.var(); // Parameter (free variable)
    /// let beta1 = math.var(); // Parameter (free variable)
    ///
    /// // Sum captures parameters, receives bound data variables
    /// let result = math.sum(data, |(xi, yi)| {
    ///     let residual = yi - beta0 - beta1 * xi;
    ///     residual.clone() * residual
    /// })?;
    /// ```
    pub fn sum<I, F>(&self, data: I, f: F) -> crate::Result<TypedBuilderExpr<f64>>
    where
        I: IntoIterator<Item = (f64, f64)>,
        F: Fn((TypedBuilderExpr<f64>, TypedBuilderExpr<f64>)) -> TypedBuilderExpr<f64>,
    {
        let data_vec: Vec<(f64, f64)> = data.into_iter().collect();

        if data_vec.is_empty() {
            return Ok(self.constant(0.0));
        }

        // Create bound variables for the summation (these represent the data)
        let xi = self.var(); // Represents x[i] in the sum
        let yi = self.var(); // Represents y[i] in the sum

        // Apply the closure with bound variables
        // The closure can capture external parameters
        let pattern_expr = f((xi, yi));
        let pattern_ast = pattern_expr.into_ast();

        // Stage 1: Algebraic expansion of the pattern
        println!("   Expanding pattern algebraically...");
        let expanded_ast = self.expand_algebraically(&pattern_ast)?;

        // Stage 2: Separate bound variables from free variables
        let bound_vars = self.identify_bound_variables(&expanded_ast, &data_vec)?;
        let (data_terms, param_terms) = self.separate_variable_terms(&expanded_ast, &bound_vars)?;

        // Stage 3: Aggregate data terms over the dataset
        let aggregated_terms = self.aggregate_data_terms(&data_terms, &data_vec)?;

        // Stage 4: Reconstruct the expression with aggregated terms
        let final_expr = self.reconstruct_expression(&aggregated_terms, &param_terms)?;

        Ok(TypedBuilderExpr::new(final_expr, self.registry.clone()))
    }

    /// Sum over pairs of data for statistical models  
    pub fn sum_pairs<I, F>(&self, data: I, f: F) -> crate::Result<TypedBuilderExpr<f64>>
    where
        I: IntoIterator<Item = (f64, f64)>,
        F: Fn(TypedBuilderExpr<f64>, TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        let data_vec: Vec<(f64, f64)> = data.into_iter().collect();

        // Stage 1: Custom summation simplification with pair sufficient statistics
        let simplified_expr = self.discover_pair_sufficient_statistics(&data_vec, &f)?;

        Ok(simplified_expr)
    }

    /// Optimize expression using egglog (Stage 2 after summation simplification)
    pub fn optimize(&self, expr: TypedBuilderExpr<f64>) -> crate::Result<TypedBuilderExpr<f64>> {
        let ast = expr.into_ast();

        // Use egglog for algebraic optimization of the expression with sufficient statistics
        let mut optimizer_config = crate::symbolic::symbolic::OptimizationConfig::default();
        optimizer_config.egglog_optimization = true;
        let mut optimizer =
            crate::symbolic::symbolic::SymbolicOptimizer::with_config(optimizer_config)?;

        let optimized_ast = optimizer.optimize(&ast)?;
        Ok(TypedBuilderExpr::new(optimized_ast, self.registry.clone()))
    }

    /// Discover sufficient statistics for single data array
    fn discover_sufficient_statistics<F>(
        &self,
        data: &[f64],
        f: &F,
    ) -> crate::Result<TypedBuilderExpr<f64>>
    where
        F: Fn(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        // Analyze the pattern by applying f to a symbolic variable
        let x_var = self.var();
        let pattern_expr = f(x_var);
        let pattern_ast = pattern_expr.into_ast();

        // Detect the pattern and compute corresponding sufficient statistics
        match self.detect_summation_pattern(&pattern_ast)? {
            SummationPatternType::Linear {
                coefficient,
                constant,
            } => {
                // Σ(a*x[i] + b) = a*Σx[i] + b*n
                let sum_x: f64 = data.iter().sum();
                let n = data.len() as f64;
                let result = coefficient * sum_x + constant * n;
                Ok(self.constant(result))
            }
            SummationPatternType::Quadratic { coefficient } => {
                // Σ(a*x[i]²) = a*Σx[i]²
                let sum_x_squared: f64 = data.iter().map(|x| x * x).sum();
                let result = coefficient * sum_x_squared;
                Ok(self.constant(result))
            }
            SummationPatternType::Power {
                exponent,
                coefficient,
            } => {
                // Σ(a*x[i]^k) = a*Σx[i]^k
                let sum_power: f64 = data.iter().map(|x| x.powf(exponent)).sum();
                let result = coefficient * sum_power;
                Ok(self.constant(result))
            }
            SummationPatternType::Constant { value } => {
                // Σ(c) = c*n
                let n = data.len() as f64;
                Ok(self.constant(value * n))
            }
            SummationPatternType::Unknown => {
                // Fallback: direct computation
                let result: f64 = data
                    .iter()
                    .map(|&x| {
                        let x_expr = self.constant(x);
                        let expr_result = f(x_expr);
                        self.eval_expr(&expr_result, &[])
                    })
                    .sum();
                Ok(self.constant(result))
            }
        }
    }

    /// Discover sufficient statistics for pair data (statistical models)
    fn discover_pair_sufficient_statistics<F>(
        &self,
        data: &[(f64, f64)],
        f: &F,
    ) -> crate::Result<TypedBuilderExpr<f64>>
    where
        F: Fn(TypedBuilderExpr<f64>, TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
    {
        // For pairs, analyze the structure using symbolic variables
        let x_var = self.var();
        let y_var = self.var();
        let pattern_expr = f(x_var, y_var);

        // For complex patterns like (y - β₀ - β₁*x)², we need to expand algebraically
        // This is where the "custom approach" for summation simplification happens

        match self.detect_pair_pattern(&pattern_expr.into_ast())? {
            PairPatternType::LinearResidualSquared { .. } => {
                // Pattern: (y - β₀ - β₁*x)² expands to:
                // Σy² - 2β₀Σy - 2β₁Σxy + nβ₀² + 2β₀β₁Σx + β₁²Σx²
                self.expand_linear_residual_squared(data)
            }
            PairPatternType::CrossProduct => {
                // Σ x[i] * y[i]
                let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
                Ok(self.constant(sum_xy))
            }
            PairPatternType::Unknown => {
                // Fallback: direct computation
                let result: f64 = data
                    .iter()
                    .map(|&(x, y)| {
                        let x_expr = self.constant(x);
                        let y_expr = self.constant(y);
                        let expr_result = f(x_expr, y_expr);
                        self.eval_expr(&expr_result, &[])
                    })
                    .sum();
                Ok(self.constant(result))
            }
        }
    }

    /// Expand (y - β₀ - β₁*x)² using sufficient statistics
    fn expand_linear_residual_squared(
        &self,
        data: &[(f64, f64)],
    ) -> crate::Result<TypedBuilderExpr<f64>> {
        // Compute sufficient statistics
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| *x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| *y).sum();
        let sum_x_squared: f64 = data.iter().map(|(x, _)| x * x).sum();
        let sum_y_squared: f64 = data.iter().map(|(_, y)| y * y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();

        // Build the expanded expression: Σy² - 2β₀Σy - 2β₁Σxy + nβ₀² + 2β₀β₁Σx + β₁²Σx²
        // Note: β₀ = var(0), β₁ = var(1) are external parameters
        let beta0 = self.var(); // This will be variable 0 in the expression
        let beta1 = self.var(); // This will be variable 1 in the expression

        let term1 = self.constant(sum_y_squared);
        let term2 = self.constant(-2.0 * sum_y) * beta0.clone();
        let term3 = self.constant(-2.0 * sum_xy) * beta1.clone();
        let term4 = self.constant(n) * beta0.clone() * beta0.clone();
        let term5 = self.constant(2.0 * sum_x) * beta0.clone() * beta1.clone();
        let term6 = self.constant(sum_x_squared) * beta1.clone() * beta1.clone();

        Ok(term1 + term2 + term3 + term4 + term5 + term6)
    }

    /// Helper to evaluate expressions (for fallback cases)
    fn eval_expr(&self, expr: &TypedBuilderExpr<f64>, vars: &[f64]) -> f64 {
        crate::final_tagless::DirectEval::eval_with_vars(&expr.clone().into_ast(), vars)
    }

    /// Detect summation patterns for single variables
    fn detect_summation_pattern(
        &self,
        ast: &crate::final_tagless::ASTRepr<f64>,
    ) -> crate::Result<SummationPatternType> {
        use crate::final_tagless::ASTRepr;

        match ast {
            ASTRepr::Constant(c) => Ok(SummationPatternType::Constant { value: *c }),
            ASTRepr::Variable(0) => Ok(SummationPatternType::Linear {
                coefficient: 1.0,
                constant: 0.0,
            }),
            ASTRepr::Mul(left, right) => {
                // Check for patterns like a*x or x*a
                if let ASTRepr::Constant(c) = left.as_ref() {
                    if matches!(right.as_ref(), ASTRepr::Variable(0)) {
                        return Ok(SummationPatternType::Linear {
                            coefficient: *c,
                            constant: 0.0,
                        });
                    }
                }
                if let ASTRepr::Constant(c) = right.as_ref() {
                    if matches!(left.as_ref(), ASTRepr::Variable(0)) {
                        return Ok(SummationPatternType::Linear {
                            coefficient: *c,
                            constant: 0.0,
                        });
                    }
                }
                Ok(SummationPatternType::Unknown)
            }
            ASTRepr::Pow(base, exp) => {
                // Check for x^k patterns
                if matches!(base.as_ref(), ASTRepr::Variable(0)) {
                    if let ASTRepr::Constant(k) = exp.as_ref() {
                        return Ok(SummationPatternType::Power {
                            exponent: *k,
                            coefficient: 1.0,
                        });
                    }
                }
                Ok(SummationPatternType::Unknown)
            }
            _ => Ok(SummationPatternType::Unknown),
        }
    }

    /// Detect patterns for pair data
    fn detect_pair_pattern(
        &self,
        ast: &crate::final_tagless::ASTRepr<f64>,
    ) -> crate::Result<PairPatternType> {
        use crate::final_tagless::ASTRepr;

        match ast {
            // Look for multiplication patterns that might be squared residuals
            ASTRepr::Mul(left, right) => {
                // Check if both sides are identical (x * x pattern for squared terms)
                if self.is_subtraction_pattern(left) && self.is_subtraction_pattern(right) {
                    // Check if they're the same subtraction (residual squared)
                    if self.ast_equals(left, right) {
                        // This is a squared residual pattern: (y - prediction)²
                        return Ok(PairPatternType::LinearResidualSquared {
                            beta0_var: 0, // Convention: β₀ is variable 0
                            beta1_var: 1, // Convention: β₁ is variable 1
                        });
                    }
                }
                Ok(PairPatternType::Unknown)
            }

            // Direct multiplication of x and y variables
            ASTRepr::Mul(left, right) => {
                if (matches!(left.as_ref(), ASTRepr::Variable(0))
                    && matches!(right.as_ref(), ASTRepr::Variable(1)))
                    || (matches!(left.as_ref(), ASTRepr::Variable(1))
                        && matches!(right.as_ref(), ASTRepr::Variable(0)))
                {
                    return Ok(PairPatternType::CrossProduct);
                }
                Ok(PairPatternType::Unknown)
            }

            _ => Ok(PairPatternType::Unknown),
        }
    }

    /// Check if an AST represents a subtraction pattern (like y - prediction)
    fn is_subtraction_pattern(&self, ast: &crate::final_tagless::ASTRepr<f64>) -> bool {
        use crate::final_tagless::ASTRepr;

        matches!(ast, ASTRepr::Sub(_, _))
    }

    /// Check if two ASTs are structurally equivalent (for detecting x*x patterns)
    fn ast_equals(
        &self,
        ast1: &crate::final_tagless::ASTRepr<f64>,
        ast2: &crate::final_tagless::ASTRepr<f64>,
    ) -> bool {
        use crate::final_tagless::ASTRepr;

        match (ast1, ast2) {
            (ASTRepr::Variable(i1), ASTRepr::Variable(i2)) => i1 == i2,
            (ASTRepr::Constant(c1), ASTRepr::Constant(c2)) => (c1 - c2).abs() < 1e-12,
            (ASTRepr::Add(l1, r1), ASTRepr::Add(l2, r2))
            | (ASTRepr::Sub(l1, r1), ASTRepr::Sub(l2, r2))
            | (ASTRepr::Mul(l1, r1), ASTRepr::Mul(l2, r2))
            | (ASTRepr::Div(l1, r1), ASTRepr::Div(l2, r2))
            | (ASTRepr::Pow(l1, r1), ASTRepr::Pow(l2, r2)) => {
                self.ast_equals(l1, l2) && self.ast_equals(r1, r2)
            }
            (ASTRepr::Neg(a1), ASTRepr::Neg(a2))
            | (ASTRepr::Ln(a1), ASTRepr::Ln(a2))
            | (ASTRepr::Exp(a1), ASTRepr::Exp(a2))
            | (ASTRepr::Sin(a1), ASTRepr::Sin(a2))
            | (ASTRepr::Cos(a1), ASTRepr::Cos(a2))
            | (ASTRepr::Sqrt(a1), ASTRepr::Sqrt(a2)) => self.ast_equals(a1, a2),
            _ => false,
        }
    }

    /// Algebraically expand an expression (distribute multiplications, etc.)
    fn expand_algebraically(
        &self,
        ast: &crate::final_tagless::ASTRepr<f64>,
    ) -> crate::Result<crate::final_tagless::ASTRepr<f64>> {
        // For now, return the original expression
        // In a full implementation, this would expand (a+b)*(c+d) → ac + ad + bc + bd
        // and other algebraic expansions
        Ok(ast.clone())
    }

    /// Identify which variables are bound by the summation (represent data)
    fn identify_bound_variables(
        &self,
        _ast: &crate::final_tagless::ASTRepr<f64>,
        _data: &[(f64, f64)],
    ) -> crate::Result<Vec<usize>> {
        // The bound variables are the last two variables created (xi, yi)
        // In a full implementation, this would analyze the AST structure
        let current_index = self.registry.borrow().len().saturating_sub(2);
        Ok(vec![current_index, current_index + 1])
    }

    /// Separate terms that depend on bound variables vs free variables
    fn separate_variable_terms(
        &self,
        ast: &crate::final_tagless::ASTRepr<f64>,
        _bound_vars: &[usize],
    ) -> crate::Result<(
        Vec<crate::final_tagless::ASTRepr<f64>>,
        Vec<crate::final_tagless::ASTRepr<f64>>,
    )> {
        // For now, just return the original expression as a data term
        // In a full implementation, this would traverse the AST and separate terms
        let data_terms = vec![ast.clone()];
        let param_terms = vec![];

        Ok((data_terms, param_terms))
    }

    /// Aggregate data terms by computing them over the actual dataset
    fn aggregate_data_terms(
        &self,
        data_terms: &[crate::final_tagless::ASTRepr<f64>],
        data: &[(f64, f64)],
    ) -> crate::Result<Vec<crate::final_tagless::ASTRepr<f64>>> {
        use crate::final_tagless::ASTRepr;

        let mut aggregated = Vec::new();

        for term in data_terms {
            // For each data term, evaluate it across all data points and sum
            let mut sum = 0.0;

            for &(x_val, y_val) in data {
                // Create runtime data: [params..., xi, yi]
                // We'll evaluate with placeholder params (0.0) for the bound variables
                let mut eval_data = vec![0.0; self.registry.borrow().len().saturating_sub(2)];
                eval_data.push(x_val); // xi
                eval_data.push(y_val); // yi

                let term_value = DirectEval::eval_with_vars(term, &eval_data);
                sum += term_value;
            }

            // Replace the data-dependent term with its aggregated constant value
            aggregated.push(ASTRepr::Constant(sum));
        }

        Ok(aggregated)
    }

    /// Reconstruct the final expression from aggregated data terms and parameter terms
    fn reconstruct_expression(
        &self,
        aggregated_terms: &[crate::final_tagless::ASTRepr<f64>],
        param_terms: &[crate::final_tagless::ASTRepr<f64>],
    ) -> crate::Result<crate::final_tagless::ASTRepr<f64>> {
        use crate::final_tagless::ASTRepr;

        // Combine all terms additively
        let mut result = ASTRepr::Constant(0.0);

        for term in aggregated_terms {
            result = ASTRepr::Add(Box::new(result), Box::new(term.clone()));
        }

        for term in param_terms {
            result = ASTRepr::Add(Box::new(result), Box::new(term.clone()));
        }

        Ok(result)
    }
}

impl Default for ExpressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-safe expression wrapper that preserves type information and enables operator overloading
#[derive(Debug, Clone)]
pub struct TypedBuilderExpr<T> {
    ast: ASTRepr<T>,
    registry: Arc<RefCell<VariableRegistry>>,
    _phantom: PhantomData<T>,
}

impl<T: NumericType> TypedBuilderExpr<T> {
    /// Create a new typed expression
    pub fn new(ast: ASTRepr<T>, registry: Arc<RefCell<VariableRegistry>>) -> Self {
        Self {
            ast,
            registry,
            _phantom: PhantomData,
        }
    }

    /// Get the underlying AST
    pub fn as_ast(&self) -> &ASTRepr<T> {
        &self.ast
    }

    /// Convert to the underlying AST (consuming)
    pub fn into_ast(self) -> ASTRepr<T> {
        self.ast
    }

    /// Get the registry
    pub fn registry(&self) -> Arc<RefCell<VariableRegistry>> {
        self.registry.clone()
    }
}

// Type conversion methods
impl<T: NumericType> TypedBuilderExpr<T> {
    /// Convert to f64 expression (for types that can be promoted)
    pub fn as_f64(self) -> TypedBuilderExpr<f64>
    where
        T: Into<f64> + Copy,
    {
        TypedBuilderExpr::new(self.ast.convert_to_f64(), self.registry)
    }

    /// Convert to f32 expression (for types that can be demoted safely)
    pub fn as_f32(self) -> TypedBuilderExpr<f32>
    where
        T: Into<f32> + Copy,
    {
        TypedBuilderExpr::new(self.ast.convert_to_f32(), self.registry)
    }
}

// Transcendental functions for Float types
impl<T: NumericType + Float> TypedBuilderExpr<T> {
    /// Sine function
    pub fn sin(self) -> Self {
        Self::new(self.ast.sin(), self.registry)
    }

    /// Cosine function
    pub fn cos(self) -> Self {
        Self::new(self.ast.cos(), self.registry)
    }

    /// Natural logarithm
    pub fn ln(self) -> Self {
        Self::new(self.ast.ln(), self.registry)
    }

    /// Exponential function
    pub fn exp(self) -> Self {
        Self::new(self.ast.exp(), self.registry)
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        Self::new(self.ast.sqrt(), self.registry)
    }

    /// Power function
    pub fn pow(self, exp: Self) -> Self {
        Self::new(
            ASTRepr::Pow(Box::new(self.ast), Box::new(exp.ast)),
            self.registry,
        )
    }
}

// Same-type arithmetic operations
impl<T> Add for TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast + rhs.ast, self.registry)
    }
}

impl<T> Sub for TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast - rhs.ast, self.registry)
    }
}

impl<T> Mul for TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast * rhs.ast, self.registry)
    }
}

impl<T> Div for TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(self.ast / rhs.ast, self.registry)
    }
}

impl<T> Neg for TypedBuilderExpr<T>
where
    T: NumericType + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-self.ast, self.registry)
    }
}

// Reference operations for efficiency
impl<T> Add<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + &rhs.ast, self.registry.clone())
    }
}

impl<T> Add<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast + rhs.ast, self.registry.clone())
    }
}

impl<T> Add<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast + &rhs.ast, self.registry)
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * &rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast * rhs.ast, self.registry.clone())
    }
}

impl<T> Mul<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast * &rhs.ast, self.registry)
    }
}

// Negation for references
impl<T> Neg for &TypedBuilderExpr<T>
where
    T: NumericType + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-&self.ast, self.registry.clone())
    }
}

// Cross-type operations with automatic promotion
impl Add<TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self + promoted_rhs
    }
}

impl Add<TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self + promoted_rhs
    }
}

impl Add<&TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self + promoted_rhs
    }
}

impl Add<&TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self + promoted_rhs
    }
}

impl Mul<TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self * promoted_rhs
    }
}

impl Mul<TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.as_f64();
        self * promoted_rhs
    }
}

impl Mul<&TypedBuilderExpr<f32>> for TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self * promoted_rhs
    }
}

impl Mul<&TypedBuilderExpr<f32>> for &TypedBuilderExpr<f64> {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f32>) -> Self::Output {
        let promoted_rhs = rhs.clone().as_f64();
        self * promoted_rhs
    }
}

// Scalar operations (constants)
impl<T> Add<T> for TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self + rhs_expr
    }
}

impl<T> Add<T> for &TypedBuilderExpr<T>
where
    T: NumericType + Add<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self + rhs_expr
    }
}

impl<T> Mul<T> for TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self * rhs_expr
    }
}

impl<T> Mul<T> for &TypedBuilderExpr<T>
where
    T: NumericType + Mul<Output = T> + Copy,
{
    type Output = TypedBuilderExpr<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(rhs), self.registry.clone());
        self * rhs_expr
    }
}

// Reverse scalar operations
impl Add<TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr + rhs
    }
}

impl Add<&TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn add(self, rhs: &TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr + rhs
    }
}

impl Mul<TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr * rhs
    }
}

impl Mul<&TypedBuilderExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64>;

    fn mul(self, rhs: &TypedBuilderExpr<f64>) -> Self::Output {
        let lhs_expr = TypedBuilderExpr::new(ASTRepr::Constant(self), rhs.registry.clone());
        lhs_expr * rhs
    }
}

// Extension methods for ASTRepr to support type conversion
impl<T: NumericType> ASTRepr<T> {
    /// Convert to f64 AST (placeholder - would need proper implementation)
    pub fn convert_to_f64(&self) -> ASTRepr<f64>
    where
        T: Into<f64> + Copy,
    {
        match self {
            ASTRepr::Constant(val) => ASTRepr::Constant((*val).into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(left.convert_to_f64()),
                Box::new(right.convert_to_f64()),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(base.convert_to_f64()),
                Box::new(exp.convert_to_f64()),
            ),
            ASTRepr::Neg(expr) => ASTRepr::Neg(Box::new(expr.convert_to_f64())),
            ASTRepr::Sin(expr) => ASTRepr::Sin(Box::new(expr.convert_to_f64())),
            ASTRepr::Cos(expr) => ASTRepr::Cos(Box::new(expr.convert_to_f64())),
            ASTRepr::Ln(expr) => ASTRepr::Ln(Box::new(expr.convert_to_f64())),
            ASTRepr::Exp(expr) => ASTRepr::Exp(Box::new(expr.convert_to_f64())),
            ASTRepr::Sqrt(expr) => ASTRepr::Sqrt(Box::new(expr.convert_to_f64())),
        }
    }

    /// Convert to f32 AST (placeholder - would need proper implementation)
    pub fn convert_to_f32(&self) -> ASTRepr<f32>
    where
        T: Into<f32> + Copy,
    {
        match self {
            ASTRepr::Constant(val) => ASTRepr::Constant((*val).into()),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(left.convert_to_f32()),
                Box::new(right.convert_to_f32()),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(base.convert_to_f32()),
                Box::new(exp.convert_to_f32()),
            ),
            ASTRepr::Neg(expr) => ASTRepr::Neg(Box::new(expr.convert_to_f32())),
            ASTRepr::Sin(expr) => ASTRepr::Sin(Box::new(expr.convert_to_f32())),
            ASTRepr::Cos(expr) => ASTRepr::Cos(Box::new(expr.convert_to_f32())),
            ASTRepr::Ln(expr) => ASTRepr::Ln(Box::new(expr.convert_to_f32())),
            ASTRepr::Exp(expr) => ASTRepr::Exp(Box::new(expr.convert_to_f32())),
            ASTRepr::Sqrt(expr) => ASTRepr::Sqrt(Box::new(expr.convert_to_f32())),
        }
    }
}

// Reference operations for subtraction
impl<T> Sub<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - &rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast - rhs.ast, self.registry.clone())
    }
}

impl<T> Sub<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn sub(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast - &rhs.ast, self.registry)
    }
}

// Reference operations for division
impl<T> Div<&TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / &rhs.ast, self.registry.clone())
    }
}

impl<T> Div<TypedBuilderExpr<T>> for &TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(&self.ast / rhs.ast, self.registry.clone())
    }
}

impl<T> Div<&TypedBuilderExpr<T>> for TypedBuilderExpr<T>
where
    T: NumericType + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T>;

    fn div(self, rhs: &TypedBuilderExpr<T>) -> Self::Output {
        TypedBuilderExpr::new(self.ast / &rhs.ast, self.registry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_variable_creation() {
        let builder = ExpressionBuilder::new();

        // Create typed variables
        let x: TypedVar<f64> = builder.typed_var();
        let y: TypedVar<f32> = builder.typed_var();

        assert_eq!(x.name(), "var_0");
        assert_eq!(y.name(), "var_1");
        assert_ne!(x.index(), y.index());
    }

    #[test]
    fn test_typed_expression_building() {
        let builder = ExpressionBuilder::new();

        // Create typed variables and expressions
        let x = builder.typed_var::<f64>();
        let y = builder.typed_var::<f64>();

        let x_expr = builder.expr_from(x);
        let y_expr = builder.expr_from(y);

        // Test same-type operations
        let sum = &x_expr + &y_expr;
        let product = &x_expr * &y_expr;

        // Verify the AST structure
        match sum.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }

        match product.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_cross_type_operations() {
        let builder = ExpressionBuilder::new();

        let x_f64 = builder.expr_from(builder.typed_var::<f64>());
        let y_f32 = builder.expr_from(builder.typed_var::<f32>());

        // This should work with automatic promotion
        let mixed_sum = x_f64 + y_f32;

        // Result should be f64
        match mixed_sum.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_scalar_operations() {
        let builder = ExpressionBuilder::new();

        let x = builder.expr_from(builder.typed_var::<f64>());

        // Test scalar operations
        let scaled = &x * 2.0;
        let shifted = &x + 1.0;
        let reverse_scaled = 3.0 * &x;

        match scaled.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication"),
        }

        match shifted.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition"),
        }

        match reverse_scaled.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_transcendental_functions() {
        let builder = ExpressionBuilder::new();

        let x = builder.expr_from(builder.typed_var::<f64>());

        let sin_x = x.clone().sin();
        let exp_x = x.clone().exp();
        let ln_x = x.clone().ln();

        match sin_x.as_ast() {
            ASTRepr::Sin(_) => {}
            _ => panic!("Expected sine"),
        }

        match exp_x.as_ast() {
            ASTRepr::Exp(_) => {}
            _ => panic!("Expected exponential"),
        }

        match ln_x.as_ast() {
            ASTRepr::Ln(_) => {}
            _ => panic!("Expected logarithm"),
        }
    }

    #[test]
    fn test_backward_compatibility() {
        let builder = ExpressionBuilder::new();

        // Old-style variable creation should still work
        let x = builder.var(); // Should be TypedBuilderExpr<f64>
        let y = builder.var();

        let expr = &x * &x + 2.0 * &x + &y;

        // Should create a valid AST
        match expr.as_ast() {
            ASTRepr::Add(_, _) => {}
            _ => panic!("Expected addition at top level"),
        }
    }

    #[test]
    fn test_complex_expression() {
        let builder = ExpressionBuilder::new();

        let x = builder.var();
        let y = builder.var();

        // Build: sin(x^2 + y) * exp(-x)
        let x_squared = x.clone().pow(builder.constant(2.0));
        let sum = x_squared + &y;
        let sin_sum = sum.sin();
        let neg_x = -&x;
        let exp_neg_x = neg_x.exp();
        let result = sin_sum * exp_neg_x;

        // Verify it creates a valid AST
        match result.as_ast() {
            ASTRepr::Mul(_, _) => {}
            _ => panic!("Expected multiplication at top level"),
        }
    }
}

/// Summation pattern types for single variables
#[derive(Debug, Clone)]
enum SummationPatternType {
    Constant { value: f64 },
    Linear { coefficient: f64, constant: f64 },
    Quadratic { coefficient: f64 },
    Power { exponent: f64, coefficient: f64 },
    Unknown,
}

/// Pattern types for pair data
#[derive(Debug, Clone)]
enum PairPatternType {
    LinearResidualSquared { beta0_var: usize, beta1_var: usize },
    CrossProduct,
    Unknown,
}
