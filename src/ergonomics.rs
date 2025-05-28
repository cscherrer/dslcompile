//! Ergonomic API for Mathematical Expression Building
//!
//! This module provides a user-friendly, fluent API for building mathematical expressions
//! without requiring deep knowledge of the underlying AST structure or variable management.
//!
//! # Key Features
//!
//! - **Unified Builder**: Single entry point for all expression building
//! - **Automatic Variables**: Smart variable management with intuitive naming
//! - **Fluent Interface**: Method chaining for natural mathematical syntax
//! - **Type Safety**: Compile-time validation of expression structure
//! - **Performance**: Zero-cost abstractions over the core AST
//!
//! # Quick Start
//!
//! ```rust
//! use mathjit::ergonomics::MathBuilder;
//!
//! // Create a builder
//! let mut math = MathBuilder::new();
//!
//! // Build expressions with natural syntax
//! let x = math.var("x");
//! let y = math.var("y");
//! let expr = math.poly(&[1.0, 2.0, 3.0], &x) + math.sin(&y);
//!
//! // Evaluate with named variables
//! let result = math.eval(&expr, &[("x", 2.0), ("y", 1.0)]);
//! ```

use crate::error::{MathJITError, Result};
use crate::final_tagless::{ASTRepr, ExpressionBuilder, VariableRegistry};
use crate::symbolic::SymbolicOptimizer;
use crate::symbolic_ad::SymbolicAD;
use std::collections::HashMap;

/// Unified mathematical expression builder with ergonomic API
///
/// This is the main entry point for building mathematical expressions in an intuitive way.
/// It automatically manages variables, provides common mathematical functions, and offers
/// a fluent interface for expression construction.
#[derive(Debug, Clone)]
pub struct MathBuilder {
    /// Internal expression builder for variable management
    builder: ExpressionBuilder,
    /// Cache of commonly used constants
    constants: HashMap<String, f64>,
    /// Symbolic optimizer for expression simplification
    optimizer: Option<SymbolicOptimizer>,
}

impl MathBuilder {
    /// Create a new mathematical expression builder
    #[must_use]
    pub fn new() -> Self {
        let mut constants = HashMap::new();

        // Pre-populate common mathematical constants
        constants.insert("pi".to_string(), std::f64::consts::PI);
        constants.insert("e".to_string(), std::f64::consts::E);
        constants.insert("tau".to_string(), std::f64::consts::TAU);
        constants.insert("sqrt2".to_string(), std::f64::consts::SQRT_2);
        constants.insert("ln2".to_string(), std::f64::consts::LN_2);
        constants.insert("ln10".to_string(), std::f64::consts::LN_10);

        Self {
            builder: ExpressionBuilder::new(),
            constants,
            optimizer: None,
        }
    }

    /// Create a new builder with symbolic optimization enabled
    pub fn with_optimization() -> Result<Self> {
        let mut builder = Self::new();
        builder.optimizer = Some(SymbolicOptimizer::new()?);
        Ok(builder)
    }

    // ============================================================================
    // Variable and Constant Creation
    // ============================================================================

    /// Create a variable and return its AST representation
    /// Variables are automatically registered and can be referenced by name
    #[must_use]
    pub fn var(&mut self, name: &str) -> ASTRepr<f64> {
        self.builder.var(name)
    }

    /// Create a constant value
    #[must_use]
    pub fn constant(&self, value: f64) -> ASTRepr<f64> {
        self.builder.constant(value)
    }

    /// Get a predefined mathematical constant
    ///
    /// Available constants: pi, e, tau, sqrt2, ln2, ln10
    pub fn math_constant(&self, name: &str) -> Result<ASTRepr<f64>> {
        self.constants
            .get(name)
            .map(|&value| ASTRepr::Constant(value))
            .ok_or_else(|| {
                MathJITError::InvalidInput(format!(
                    "Unknown mathematical constant: {name}. Available: {}",
                    self.constants
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })
    }

    // ============================================================================
    // Basic Arithmetic Operations
    // ============================================================================

    /// Addition operation (prefer using + operator)
    #[must_use]
    pub fn add(&self, left: &ASTRepr<f64>, right: &ASTRepr<f64>) -> ASTRepr<f64> {
        left + right
    }

    /// Subtraction operation (prefer using - operator)
    #[must_use]
    pub fn sub(&self, left: &ASTRepr<f64>, right: &ASTRepr<f64>) -> ASTRepr<f64> {
        left - right
    }

    /// Multiplication operation (prefer using * operator)
    #[must_use]
    pub fn mul(&self, left: &ASTRepr<f64>, right: &ASTRepr<f64>) -> ASTRepr<f64> {
        left * right
    }

    /// Division operation (prefer using / operator)
    #[must_use]
    pub fn div(&self, left: &ASTRepr<f64>, right: &ASTRepr<f64>) -> ASTRepr<f64> {
        left / right
    }

    /// Power operation
    #[must_use]
    pub fn pow(&self, base: &ASTRepr<f64>, exp: &ASTRepr<f64>) -> ASTRepr<f64> {
        base.pow_ref(exp)
    }

    /// Negation operation (prefer using - operator)
    #[must_use]
    pub fn neg(&self, expr: &ASTRepr<f64>) -> ASTRepr<f64> {
        -expr
    }

    // ============================================================================
    // Transcendental Functions
    // ============================================================================

    /// Natural logarithm
    #[must_use]
    pub fn ln(&self, expr: &ASTRepr<f64>) -> ASTRepr<f64> {
        expr.ln_ref()
    }

    /// Exponential function
    #[must_use]
    pub fn exp(&self, expr: &ASTRepr<f64>) -> ASTRepr<f64> {
        expr.exp_ref()
    }

    /// Sine function
    #[must_use]
    pub fn sin(&self, expr: &ASTRepr<f64>) -> ASTRepr<f64> {
        expr.sin_ref()
    }

    /// Cosine function
    #[must_use]
    pub fn cos(&self, expr: &ASTRepr<f64>) -> ASTRepr<f64> {
        expr.cos_ref()
    }

    /// Square root
    #[must_use]
    pub fn sqrt(&self, expr: &ASTRepr<f64>) -> ASTRepr<f64> {
        expr.sqrt_ref()
    }

    // ============================================================================
    // High-Level Mathematical Functions
    // ============================================================================

    /// Create a polynomial expression using Horner's method for efficient evaluation
    ///
    /// Coefficients are in ascending order of powers: [c₀, c₁, c₂, ...] represents c₀ + c₁x + c₂x² + ...
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::ergonomics::MathBuilder;
    ///
    /// let mut math = MathBuilder::new();
    /// let x = math.var("x");
    ///
    /// // Create 1 + 3x + 2x² (coefficients in ascending order of powers)
    /// let poly = math.poly(&[1.0, 3.0, 2.0], &x);
    ///
    /// // For quadratic ax² + bx + c, use: [c, b, a]
    /// let quadratic = math.poly(&[1.0, -3.0, 2.0], &x); // 2x² - 3x + 1
    /// ```
    #[must_use]
    pub fn poly(&self, coefficients: &[f64], variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        if coefficients.is_empty() {
            return ASTRepr::Constant(0.0);
        }

        if coefficients.len() == 1 {
            return ASTRepr::Constant(coefficients[0]);
        }

        // Use Horner's method for efficient evaluation
        let mut result = ASTRepr::Constant(coefficients[coefficients.len() - 1]);

        for &coeff in coefficients.iter().rev().skip(1) {
            result = self.add(&self.mul(&result, variable), &ASTRepr::Constant(coeff));
        }

        result
    }

    /// Create a linear expression: ax + b
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::ergonomics::MathBuilder;
    ///
    /// let mut math = MathBuilder::new();
    /// let x = math.var("x");
    ///
    /// // Create 2x + 3
    /// let linear = math.linear(2.0, 3.0, &x);
    /// ```
    #[must_use]
    pub fn linear(&self, a: f64, b: f64, variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        self.poly(&[b, a], variable)
    }

    /// Create a quadratic expression: ax² + bx + c
    ///
    /// This is a convenience wrapper around `poly` for the common case of quadratic expressions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::ergonomics::MathBuilder;
    ///
    /// let mut math = MathBuilder::new();
    /// let x = math.var("x");
    ///
    /// // Create 2x² - 3x + 1
    /// let quadratic = math.quadratic(2.0, -3.0, 1.0, &x);
    ///
    /// // Equivalent to: math.poly(&[1.0, -3.0, 2.0], &x)
    /// ```
    #[must_use]
    pub fn quadratic(&self, a: f64, b: f64, c: f64, variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        self.poly(&[c, b, a], variable)
    }

    /// Create a Gaussian (normal) distribution function
    ///
    /// Creates: (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    #[must_use]
    pub fn gaussian(&self, mean: f64, std_dev: f64, variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        let pi = self.math_constant("pi").unwrap();
        let two = ASTRepr::Constant(2.0);
        let sigma_squared = ASTRepr::Constant(std_dev * std_dev);

        // Normalization factor: 1/√(2πσ²)
        let norm_factor = self.div(
            &ASTRepr::Constant(1.0),
            &self.sqrt(&self.mul(&self.mul(&two, &pi), &sigma_squared)),
        );

        // Exponent: -(x-μ)²/(2σ²)
        let x_minus_mu = self.sub(variable, &ASTRepr::Constant(mean));
        let x_minus_mu_squared = self.pow(&x_minus_mu, &ASTRepr::Constant(2.0));
        let exponent = self.neg(&self.div(&x_minus_mu_squared, &self.mul(&two, &sigma_squared)));

        self.mul(&norm_factor, &self.exp(&exponent))
    }

    /// Create a logistic (sigmoid) function: 1/(1 + exp(-x))
    #[must_use]
    pub fn logistic(&self, variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        let one = ASTRepr::Constant(1.0);
        let neg_x = self.neg(variable);
        let exp_neg_x = self.exp(&neg_x);
        let denominator = self.add(&one, &exp_neg_x);
        self.div(&one, &denominator)
    }

    /// Create a hyperbolic tangent function: tanh(x) = (exp(2x) - 1)/(exp(2x) + 1)
    #[must_use]
    pub fn tanh(&self, variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        let two = ASTRepr::Constant(2.0);
        let one = ASTRepr::Constant(1.0);
        let two_x = self.mul(&two, variable);
        let exp_2x = self.exp(&two_x);

        let numerator = self.sub(&exp_2x, &one);
        let denominator = self.add(&exp_2x, &one);

        self.div(&numerator, &denominator)
    }

    // ============================================================================
    // Evaluation and Optimization
    // ============================================================================

    /// Evaluate an expression with named variables
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathjit::ergonomics::MathBuilder;
    ///
    /// let mut math = MathBuilder::new();
    /// let x = math.var("x");
    /// let y = math.var("y");
    /// let expr = math.add(&x, &y);
    ///
    /// let result = math.eval(&expr, &[("x", 3.0), ("y", 4.0)]);
    /// assert_eq!(result, 7.0);
    /// ```
    #[must_use]
    pub fn eval(&self, expr: &ASTRepr<f64>, variables: &[(&str, f64)]) -> f64 {
        let named_vars: Vec<(String, f64)> = variables
            .iter()
            .map(|(name, value)| ((*name).to_string(), *value))
            .collect();

        self.builder.eval_with_named_vars(expr, &named_vars)
    }

    /// Optimize an expression using symbolic optimization
    pub fn optimize(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.optimize(expr)
        } else {
            // Return the expression unchanged if no optimizer is available
            Ok(expr.clone())
        }
    }

    /// Compute the derivative of an expression with respect to a variable
    pub fn derivative(&mut self, expr: &ASTRepr<f64>, var_name: &str) -> Result<ASTRepr<f64>> {
        // Get the variable index from our registry
        let var_index = self.builder.get_variable_index(var_name).ok_or_else(|| {
            MathJITError::InvalidInput(format!("Variable {var_name} not found in registry"))
        })?;

        // Configure SymbolicAD with the correct number of variables
        let mut config = crate::symbolic_ad::SymbolicADConfig::default();
        config.num_variables = self.builder.num_variables();

        let mut ad = SymbolicAD::with_config(config)?;
        let result = ad.compute_with_derivatives(expr)?;

        // SymbolicAD uses variable indices as strings in the derivatives map
        let var_index_str = var_index.to_string();
        result
            .first_derivatives
            .get(&var_index_str)
            .cloned()
            .ok_or_else(|| {
                MathJITError::InvalidInput(format!(
                    "Variable index {var_index} not found in derivatives"
                ))
            })
    }

    /// Compute the gradient of an expression (all first derivatives)
    pub fn gradient(&mut self, expr: &ASTRepr<f64>) -> Result<HashMap<String, ASTRepr<f64>>> {
        // Configure SymbolicAD with the correct number of variables
        let mut config = crate::symbolic_ad::SymbolicADConfig::default();
        config.num_variables = self.builder.num_variables();

        let mut ad = SymbolicAD::with_config(config)?;
        let result = ad.compute_with_derivatives(expr)?;

        // Convert from index-based to name-based derivatives
        let mut named_derivatives = HashMap::new();
        for (index_str, derivative) in result.first_derivatives {
            if let Ok(index) = index_str.parse::<usize>() {
                if let Some(var_name) = self.builder.get_variable_name(index) {
                    named_derivatives.insert(var_name.to_string(), derivative);
                }
            }
        }

        Ok(named_derivatives)
    }

    // ============================================================================
    // Utility Functions
    // ============================================================================

    /// Get the number of variables registered in this builder
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.builder.num_variables()
    }

    /// Get all variable names
    #[must_use]
    pub fn variable_names(&self) -> &[String] {
        self.builder.variable_names()
    }

    /// Get the variable registry for advanced usage
    #[must_use]
    pub fn registry(&self) -> &VariableRegistry {
        self.builder.registry()
    }

    /// Clear all variables and start fresh
    pub fn clear(&mut self) {
        self.builder = ExpressionBuilder::new();
    }

    /// Validate that an expression is well-formed
    pub fn validate(&self, expr: &ASTRepr<f64>) -> Result<()> {
        // Check for common issues like division by zero constants, invalid variable indices, etc.
        self.validate_recursive(expr)
    }

    /// Recursive validation helper
    fn validate_recursive(&self, expr: &ASTRepr<f64>) -> Result<()> {
        match expr {
            ASTRepr::Constant(value) => {
                if value.is_nan() || value.is_infinite() {
                    return Err(MathJITError::InvalidInput(format!(
                        "Invalid constant value: {value}"
                    )));
                }
            }
            ASTRepr::Variable(index) => {
                if *index >= self.builder.num_variables() {
                    return Err(MathJITError::InvalidInput(format!(
                        "Variable index {index} is out of bounds (max: {})",
                        self.builder.num_variables()
                    )));
                }
            }
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.validate_recursive(left)?;
                self.validate_recursive(right)?;

                // Special check for division by zero constant
                if matches!(expr, ASTRepr::Div(_, _right)) {
                    if let ASTRepr::Constant(value) = right.as_ref() {
                        if value.abs() < f64::EPSILON {
                            return Err(MathJITError::InvalidInput(
                                "Division by zero constant detected".to_string(),
                            ));
                        }
                    }
                }
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                self.validate_recursive(inner)?;
            }
        }
        Ok(())
    }
}

impl Default for MathBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience Functions for Common Use Cases
// ============================================================================

/// Quick function to create and evaluate a simple expression
///
/// # Examples
///
/// ```rust
/// use mathjit::ergonomics::quick_eval;
///
/// // Evaluate x² + 2x + 1 at x = 3
/// let result = quick_eval("x^2 + 2*x + 1", &[("x", 3.0)]);
/// ```
pub fn quick_eval(_expression: &str, _variables: &[(&str, f64)]) -> Result<f64> {
    // This is a placeholder for a future expression parser
    // For now, we'll return an error suggesting the use of MathBuilder
    Err(MathJITError::InvalidInput(
        "Expression parsing not yet implemented. Please use MathBuilder for now.".to_string(),
    ))
}

/// Create common mathematical functions quickly
pub mod presets {
    use super::{ASTRepr, MathBuilder};

    /// Create a standard normal distribution (mean=0, `std_dev=1`)
    #[must_use]
    pub fn standard_normal(math: &MathBuilder, variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        math.gaussian(0.0, 1.0, variable)
    }

    /// Create a `ReLU` activation function: max(0, x)
    #[must_use]
    pub fn relu(math: &MathBuilder, variable: &ASTRepr<f64>) -> ASTRepr<f64> {
        // For now, we'll approximate with a smooth function
        // ReLU(x) ≈ ln(1 + exp(x)) for large positive x, 0 for negative x
        // This is a placeholder - true ReLU would need piecewise functions
        let exp_x = math.exp(variable);
        let one_plus_exp_x = math.add(&ASTRepr::Constant(1.0), &exp_x);
        math.ln(&one_plus_exp_x)
    }

    /// Create a mean squared error loss function: (`y_pred` - `y_true)²`
    #[must_use]
    pub fn mse_loss(
        math: &MathBuilder,
        y_pred: &ASTRepr<f64>,
        y_true: &ASTRepr<f64>,
    ) -> ASTRepr<f64> {
        let diff = math.sub(y_pred, y_true);
        math.pow(&diff, &ASTRepr::Constant(2.0))
    }

    /// Create a cross-entropy loss function: -`y_true` * `ln(y_pred)`
    #[must_use]
    pub fn cross_entropy_loss(
        math: &MathBuilder,
        y_pred: &ASTRepr<f64>,
        y_true: &ASTRepr<f64>,
    ) -> ASTRepr<f64> {
        let ln_pred = math.ln(y_pred);
        let product = math.mul(y_true, &ln_pred);
        math.neg(&product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let mut math = MathBuilder::new();
        let x = math.var("x");
        let y = math.var("y");

        let expr = math.add(&x, &y);
        let result = math.eval(&expr, &[("x", 3.0), ("y", 4.0)]);

        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_polynomial() {
        let mut math = MathBuilder::new();
        let x = math.var("x");

        // Create 2x² + 3x + 1
        let poly = math.poly(&[1.0, 3.0, 2.0], &x);
        let result = math.eval(&poly, &[("x", 2.0)]);

        // 2(4) + 3(2) + 1 = 8 + 6 + 1 = 15
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_quadratic() {
        let mut math = MathBuilder::new();
        let x = math.var("x");

        // Create 2x² - 3x + 1
        let quad = math.quadratic(2.0, -3.0, 1.0, &x);
        let result = math.eval(&quad, &[("x", 2.0)]);

        // 2(4) - 3(2) + 1 = 8 - 6 + 1 = 3
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_linear() {
        let mut math = MathBuilder::new();
        let x = math.var("x");

        // Create 2x + 3
        let linear = math.linear(2.0, 3.0, &x);
        let result = math.eval(&linear, &[("x", 4.0)]);

        // 2(4) + 3 = 8 + 3 = 11
        assert_eq!(result, 11.0);
    }

    #[test]
    fn test_gaussian() {
        let mut math = MathBuilder::new();
        let x = math.var("x");

        let gaussian = math.gaussian(0.0, 1.0, &x);
        let result = math.eval(&gaussian, &[("x", 0.0)]);

        // At x=0, standard normal should be 1/√(2π) ≈ 0.3989
        assert!((result - 0.3989).abs() < 0.001);
    }

    #[test]
    fn test_logistic() {
        let mut math = MathBuilder::new();
        let x = math.var("x");

        let logistic = math.logistic(&x);
        let result = math.eval(&logistic, &[("x", 0.0)]);

        // logistic(0) = 1/(1+1) = 0.5
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_math_constants() {
        let math = MathBuilder::new();

        let pi = math.math_constant("pi").unwrap();
        let e = math.math_constant("e").unwrap();

        if let ASTRepr::Constant(pi_val) = pi {
            assert!((pi_val - std::f64::consts::PI).abs() < 1e-10);
        }

        if let ASTRepr::Constant(e_val) = e {
            assert!((e_val - std::f64::consts::E).abs() < 1e-10);
        }
    }

    #[test]
    fn test_validation() {
        let math = MathBuilder::new();

        // Valid expression
        let valid = ASTRepr::Constant(42.0);
        assert!(math.validate(&valid).is_ok());

        // Invalid expression (NaN)
        let invalid = ASTRepr::Constant(f64::NAN);
        assert!(math.validate(&invalid).is_err());

        // Division by zero
        let div_zero = math.div(&ASTRepr::Constant(1.0), &ASTRepr::Constant(0.0));
        assert!(math.validate(&div_zero).is_err());
    }
}
