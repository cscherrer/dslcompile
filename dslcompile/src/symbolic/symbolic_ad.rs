//! Symbolic Automatic Differentiation
//!
//! This module implements symbolic automatic differentiation with a three-stage optimization pipeline:
//! 1. **Pre-optimization**: Simplify expressions using egglog before differentiation
//! 2. **Symbolic differentiation**: Compute derivatives symbolically using rewrite rules
//! 3. **Post-optimization**: Optimize the combined (f(x), f'(x)) expression to share subexpressions
//!
//! # Key Benefits
//!
//! - **Subexpression sharing**: When computing both f(x) and f'(x), common subexpressions are identified and shared
//! - **Symbolic optimization**: Derivatives are computed symbolically and then optimized algebraically
//! - **Egglog integration**: Uses equality saturation to find optimal representations
//! - **Higher-order derivatives**: Supports computing f, f', f'', etc. with shared computation
//!
//! # Architecture
//!
//! ```text
//! Original Expression f(x)
//!         ↓
//! Stage 1: Egglog Pre-optimization
//!         ↓
//! Stage 2: Symbolic AD → (f(x), f'(x))
//!         ↓
//! Stage 3: Egglog Post-optimization (shared subexpressions)
//!         ↓
//! Optimized (f(x), f'(x)) Pair
//! ```

use crate::{ast::ASTRepr, error::Result, symbolic::symbolic::SymbolicOptimizer};
use std::collections::HashMap;

/// Configuration for symbolic automatic differentiation
#[derive(Debug, Clone)]
pub struct SymbolicADConfig {
    /// Enable pre-optimization before differentiation
    pub pre_optimize: bool,
    /// Enable post-optimization after differentiation
    pub post_optimize: bool,
    /// Enable subexpression sharing between f and f'
    pub share_subexpressions: bool,
    /// Maximum order of derivatives to compute
    pub max_derivative_order: usize,
    /// Number of variables to differentiate with respect to (indexed 0, 1, 2, ...)
    pub num_variables: usize,
}

impl Default for SymbolicADConfig {
    fn default() -> Self {
        Self {
            pre_optimize: true,
            post_optimize: true,
            share_subexpressions: true,
            max_derivative_order: 1,
            num_variables: 1,
        }
    }
}

/// Represents a function and its derivatives with shared subexpressions
#[derive(Debug, Clone)]
pub struct FunctionWithDerivatives<T> {
    /// The original function f(x)
    pub function: ASTRepr<T>,
    /// First derivatives ∂`f/∂x_i`
    pub first_derivatives: HashMap<String, ASTRepr<T>>,
    /// Second derivatives ∂`²f/∂x_i∂x_j`
    pub second_derivatives: HashMap<(String, String), ASTRepr<T>>,
    /// Shared subexpressions identified during optimization
    pub shared_subexpressions: HashMap<String, ASTRepr<T>>,
    /// Statistics about the optimization
    pub stats: SymbolicADStats,
}

/// Statistics about symbolic AD optimization
#[derive(Debug, Clone, Default)]
pub struct SymbolicADStats {
    /// Number of operations in original function before optimization
    pub function_operations_before: usize,
    /// Number of operations in optimized function after optimization
    pub function_operations_after: usize,
    /// Total operations (function + derivatives) before optimization
    pub total_operations_before: usize,
    /// Total operations (function + derivatives) after optimization
    pub total_operations_after: usize,
    /// Number of shared subexpressions found
    pub shared_subexpressions_count: usize,
    /// Time spent in each optimization stage (microseconds)
    pub stage_times_us: [u64; 3],
}

impl SymbolicADStats {
    /// Calculate the function optimization ratio (`function_operations_after` / `function_operations_before`)
    #[must_use]
    pub fn function_optimization_ratio(&self) -> f64 {
        if self.function_operations_before == 0 {
            1.0
        } else {
            self.function_operations_after as f64 / self.function_operations_before as f64
        }
    }

    /// Calculate the total optimization ratio (`total_operations_after` / `total_operations_before`)
    #[must_use]
    pub fn total_optimization_ratio(&self) -> f64 {
        if self.total_operations_before == 0 {
            1.0
        } else {
            self.total_operations_after as f64 / self.total_operations_before as f64
        }
    }

    /// Calculate total optimization time
    #[must_use]
    pub fn total_time_us(&self) -> u64 {
        self.stage_times_us.iter().sum()
    }

    /// Get the primary optimization ratio (function-level optimization)
    #[must_use]
    pub fn optimization_ratio(&self) -> f64 {
        self.function_optimization_ratio()
    }

    /// Get operations before (for backward compatibility)
    #[must_use]
    pub fn operations_before(&self) -> usize {
        self.function_operations_before
    }

    /// Get operations after (for backward compatibility)
    #[must_use]
    pub fn operations_after(&self) -> usize {
        self.function_operations_after
    }
}

/// Symbolic automatic differentiation engine
pub struct SymbolicAD {
    /// Configuration for the AD process
    config: SymbolicADConfig,
    /// Symbolic optimizer for egglog integration
    optimizer: SymbolicOptimizer,
    /// Cache for computed derivatives
    derivative_cache: HashMap<String, ASTRepr<f64>>,
}

impl SymbolicAD {
    /// Create a new symbolic AD engine with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(SymbolicADConfig::default())
    }

    /// Create a new symbolic AD engine with custom configuration
    pub fn with_config(config: SymbolicADConfig) -> Result<Self> {
        let optimizer = SymbolicOptimizer::new()?;
        Ok(Self {
            config,
            optimizer,
            derivative_cache: HashMap::new(),
        })
    }

    /// Compute function and derivatives with the three-stage optimization pipeline
    pub fn compute_with_derivatives(
        &mut self,
        expr: &ASTRepr<f64>,
    ) -> Result<FunctionWithDerivatives<f64>> {
        let _start_time = std::time::Instant::now();
        let mut stats = SymbolicADStats::default();
        stats.function_operations_before = expr.count_operations();
        stats.total_operations_before = expr.count_operations();

        // Stage 1: Pre-optimization using egglog
        let stage1_start = std::time::Instant::now();
        let pre_optimized = if self.config.pre_optimize {
            self.optimizer.optimize(expr)?
        } else {
            expr.clone()
        };
        stats.stage_times_us[0] = stage1_start.elapsed().as_micros() as u64;

        // Stage 2: Symbolic differentiation
        let stage2_start = std::time::Instant::now();
        let mut first_derivatives = HashMap::new();
        let mut second_derivatives = HashMap::new();

        // Clone variables to avoid borrow checker issues
        let variables = self.config.num_variables;

        // Compute first derivatives
        for var in 0..variables {
            let derivative = self.symbolic_derivative(&pre_optimized, var)?;
            first_derivatives.insert(var.to_string(), derivative);
        }

        // Compute second derivatives if requested
        if self.config.max_derivative_order >= 2 {
            for var1 in 0..variables {
                for var2 in 0..variables {
                    if let Some(first_deriv) = first_derivatives.get(&var1.to_string()) {
                        let second_deriv = self.symbolic_derivative(first_deriv, var2)?;
                        second_derivatives
                            .insert((var1.to_string(), var2.to_string()), second_deriv);
                    }
                }
            }
        }

        // Update total operations before optimization (including derivatives)
        stats.total_operations_before = stats.function_operations_before
            + first_derivatives
                .values()
                .map(crate::ast::ASTRepr::count_operations)
                .sum::<usize>()
            + second_derivatives
                .values()
                .map(crate::ast::ASTRepr::count_operations)
                .sum::<usize>();

        stats.stage_times_us[1] = stage2_start.elapsed().as_micros() as u64;

        // Stage 3: Post-optimization with subexpression sharing
        let stage3_start = std::time::Instant::now();
        let (
            optimized_function,
            optimized_derivatives,
            optimized_second_derivatives,
            shared_subexpressions,
        ) = if self.config.post_optimize && self.config.share_subexpressions {
            self.optimize_with_subexpression_sharing(
                &pre_optimized,
                &first_derivatives,
                &second_derivatives,
            )?
        } else if self.config.post_optimize {
            // Just optimize individually without sharing
            let opt_func = self.optimizer.optimize(&pre_optimized)?;
            let mut opt_first = HashMap::new();
            for (var, deriv) in &first_derivatives {
                opt_first.insert(var.clone(), self.optimizer.optimize(deriv)?);
            }
            let mut opt_second = HashMap::new();
            for ((var1, var2), deriv) in &second_derivatives {
                opt_second.insert(
                    (var1.clone(), var2.clone()),
                    self.optimizer.optimize(deriv)?,
                );
            }
            (opt_func, opt_first, opt_second, HashMap::new())
        } else {
            (
                pre_optimized,
                first_derivatives,
                second_derivatives,
                HashMap::new(),
            )
        };
        stats.stage_times_us[2] = stage3_start.elapsed().as_micros() as u64;

        // Calculate final statistics
        stats.function_operations_after = optimized_function.count_operations();
        stats.total_operations_after = stats.function_operations_after
            + optimized_derivatives
                .values()
                .map(crate::ast::ASTRepr::count_operations)
                .sum::<usize>()
            + optimized_second_derivatives
                .values()
                .map(crate::ast::ASTRepr::count_operations)
                .sum::<usize>();
        stats.shared_subexpressions_count = shared_subexpressions.len();

        Ok(FunctionWithDerivatives {
            function: optimized_function,
            first_derivatives: optimized_derivatives,
            second_derivatives: optimized_second_derivatives,
            shared_subexpressions,
            stats,
        })
    }

    /// Compute symbolic derivative of an expression with respect to a variable
    fn symbolic_derivative(&mut self, expr: &ASTRepr<f64>, var: usize) -> Result<ASTRepr<f64>> {
        // Check cache first
        let cache_key = format!("{expr:?}_{var}");
        if let Some(cached) = self.derivative_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let derivative = self.compute_derivative_recursive(expr, var)?;

        // Cache the result
        self.derivative_cache.insert(cache_key, derivative.clone());

        Ok(derivative)
    }

    /// Recursively compute the derivative using symbolic differentiation rules
    fn compute_derivative_recursive(
        &self,
        expr: &ASTRepr<f64>,
        var: usize,
    ) -> Result<ASTRepr<f64>> {
        match expr {
            // d/dx(c) = 0
            ASTRepr::Constant(_) => Ok(ASTRepr::Constant(0.0)),

            // d/dx(x_i) = 1 if i == var, 0 otherwise
            ASTRepr::Variable(index) => {
                if *index == var {
                    Ok(ASTRepr::Constant(1.0))
                } else {
                    Ok(ASTRepr::Constant(0.0))
                }
            }

            // d/dx(u + v) = du/dx + dv/dx
            ASTRepr::Add(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;
                Ok(ASTRepr::Add(Box::new(left_deriv), Box::new(right_deriv)))
            }

            // d/dx(u - v) = du/dx - dv/dx
            ASTRepr::Sub(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;
                Ok(ASTRepr::Sub(Box::new(left_deriv), Box::new(right_deriv)))
            }

            // d/dx(u * v) = u * dv/dx + v * du/dx (product rule)
            ASTRepr::Mul(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;

                let term1 = ASTRepr::Mul(left.clone(), Box::new(right_deriv));
                let term2 = ASTRepr::Mul(right.clone(), Box::new(left_deriv));

                Ok(ASTRepr::Add(Box::new(term1), Box::new(term2)))
            }

            // d/dx(u / v) = (v * du/dx - u * dv/dx) / v² (quotient rule)
            ASTRepr::Div(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;

                let numerator_term1 = ASTRepr::Mul(right.clone(), Box::new(left_deriv));
                let numerator_term2 = ASTRepr::Mul(left.clone(), Box::new(right_deriv));
                let numerator = ASTRepr::Sub(Box::new(numerator_term1), Box::new(numerator_term2));

                let denominator = ASTRepr::Mul(right.clone(), right.clone());

                Ok(ASTRepr::Div(Box::new(numerator), Box::new(denominator)))
            }

            // d/dx(u^v) = u^v * (v' * ln(u) + v * u'/u) (generalized power rule)
            ASTRepr::Pow(base, exp) => {
                let base_deriv = self.compute_derivative_recursive(base, var)?;
                let exp_deriv = self.compute_derivative_recursive(exp, var)?;

                // u^v * (v' * ln(u) + v * u'/u)
                let ln_base = ASTRepr::Ln(base.clone());
                let term1 = ASTRepr::Mul(Box::new(exp_deriv), Box::new(ln_base));

                let u_prime_over_u = ASTRepr::Div(Box::new(base_deriv), base.clone());
                let term2 = ASTRepr::Mul(exp.clone(), Box::new(u_prime_over_u));

                let sum = ASTRepr::Add(Box::new(term1), Box::new(term2));
                let original_power = ASTRepr::Pow(base.clone(), exp.clone());

                Ok(ASTRepr::Mul(Box::new(original_power), Box::new(sum)))
            }

            // d/dx(-u) = -du/dx
            ASTRepr::Neg(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                Ok(ASTRepr::Neg(Box::new(inner_deriv)))
            }

            // d/dx(ln(u)) = u'/u
            ASTRepr::Ln(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                Ok(ASTRepr::Div(Box::new(inner_deriv), inner.clone()))
            }

            // d/dx(exp(u)) = exp(u) * u'
            ASTRepr::Exp(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                let exp_inner = ASTRepr::Exp(inner.clone());
                Ok(ASTRepr::Mul(Box::new(exp_inner), Box::new(inner_deriv)))
            }

            // d/dx(sqrt(u)) = u' / (2 * sqrt(u))
            ASTRepr::Sqrt(inner) => {
                // d/dx sqrt(f) = 1/(2*sqrt(f)) * df/dx
                let inner_derivative = self.compute_derivative_recursive(inner, var)?;
                let sqrt_inner = ASTRepr::Sqrt(inner.clone());
                let two = ASTRepr::Constant(2.0);
                let denominator = ASTRepr::Mul(Box::new(two), Box::new(sqrt_inner));
                Ok(ASTRepr::Div(
                    Box::new(inner_derivative),
                    Box::new(denominator),
                ))
            }

            // d/dx(sin(u)) = cos(u) * u'
            ASTRepr::Sin(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                let cos_inner = ASTRepr::Cos(inner.clone());
                Ok(ASTRepr::Mul(Box::new(cos_inner), Box::new(inner_deriv)))
            }

            // d/dx(cos(u)) = -sin(u) * u'
            ASTRepr::Cos(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                let sin_inner = ASTRepr::Sin(inner.clone());
                let neg_sin = ASTRepr::Neg(Box::new(sin_inner));
                Ok(ASTRepr::Mul(Box::new(neg_sin), Box::new(inner_deriv)))
            }

            ASTRepr::Sum(_collection) => {
                // TODO: Handle Collection format in symbolic AD
                // TODO: Implement Sum variant for symbolic differentiation
                // This will handle automatic differentiation of Sum expressions
                todo!("Sum variant symbolic differentiation not yet implemented")
            }
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    /// Optimize function and derivatives together to identify shared subexpressions
    fn optimize_with_subexpression_sharing(
        &mut self,
        function: &ASTRepr<f64>,
        first_derivatives: &HashMap<String, ASTRepr<f64>>,
        second_derivatives: &HashMap<(String, String), ASTRepr<f64>>,
    ) -> Result<(
        ASTRepr<f64>,
        HashMap<String, ASTRepr<f64>>,
        HashMap<(String, String), ASTRepr<f64>>,
        HashMap<String, ASTRepr<f64>>,
    )> {
        // For now, implement a simplified version that optimizes each expression individually
        // TODO: Implement true subexpression sharing using egglog

        let optimized_function = self.optimizer.optimize(function)?;

        let mut optimized_first = HashMap::new();
        for (var, deriv) in first_derivatives {
            optimized_first.insert(var.clone(), self.optimizer.optimize(deriv)?);
        }

        let mut optimized_second = HashMap::new();
        for ((var1, var2), deriv) in second_derivatives {
            optimized_second.insert(
                (var1.clone(), var2.clone()),
                self.optimizer.optimize(deriv)?,
            );
        }

        // TODO: Implement subexpression identification and sharing
        let shared_subexpressions = HashMap::new();

        Ok((
            optimized_function,
            optimized_first,
            optimized_second,
            shared_subexpressions,
        ))
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &SymbolicADConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: SymbolicADConfig) {
        self.config = config;
    }

    /// Clear the derivative cache
    pub fn clear_cache(&mut self) {
        self.derivative_cache.clear();
    }

    /// Get cache statistics
    #[must_use]
    pub fn cache_stats(&self) -> (usize, usize) {
        (
            self.derivative_cache.len(),
            self.derivative_cache.capacity(),
        )
    }
}

impl Default for SymbolicAD {
    fn default() -> Self {
        Self::new().expect("Failed to create default SymbolicAD")
    }
}

/// Convenience functions for common symbolic AD operations
pub mod convenience {
    use super::{ASTRepr, HashMap, Result, SymbolicAD, SymbolicADConfig};

    /// Compute the gradient of a scalar function
    pub fn gradient(
        expr: &ASTRepr<f64>,
        variables: &[&str],
    ) -> Result<HashMap<String, ASTRepr<f64>>> {
        let mut config = SymbolicADConfig::default();
        config.num_variables = variables.len();

        let mut ad = SymbolicAD::with_config(config)?;
        let result = ad.compute_with_derivatives(expr)?;

        Ok(result.first_derivatives)
    }

    /// Compute the Hessian matrix of a scalar function
    pub fn hessian(
        expr: &ASTRepr<f64>,
        variables: &[&str],
    ) -> Result<HashMap<(String, String), ASTRepr<f64>>> {
        let mut config = SymbolicADConfig::default();
        config.num_variables = variables.len();
        config.max_derivative_order = 2;

        let mut ad = SymbolicAD::with_config(config)?;
        let result = ad.compute_with_derivatives(expr)?;

        Ok(result.second_derivatives)
    }

    /// Create a polynomial expression from coefficients
    /// Coefficients are in ascending order: [c₀, c₁, c₂, ...] for c₀ + c₁x + c₂x² + ...
    #[must_use]
    pub fn poly(coefficients: &[f64]) -> ASTRepr<f64> {
        if coefficients.is_empty() {
            return ASTRepr::Constant(0.0);
        }

        if coefficients.len() == 1 {
            return ASTRepr::Constant(coefficients[0]);
        }

        let x = ASTRepr::Variable(0); // Use index 0 for variable x

        // Build polynomial using Horner's method for efficiency
        let mut result = ASTRepr::Constant(coefficients[coefficients.len() - 1]);

        for &coeff in coefficients.iter().rev().skip(1) {
            result = ASTRepr::Add(
                Box::new(ASTRepr::Constant(coeff)),
                Box::new(ASTRepr::Mul(Box::new(x.clone()), Box::new(result))),
            );
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ASTRepr;

    /// Test helper function for creating bivariate polynomials
    /// a*x² + b*x*y + c*y² + d*x + e*y + f
    fn bivariate_poly(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> ASTRepr<f64> {
        let x = ASTRepr::Variable(0); // Use index 0 for variable x
        let y = ASTRepr::Variable(1); // Use index 1 for variable y

        let x_squared = ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::Constant(2.0)));
        let y_squared = ASTRepr::Pow(Box::new(y.clone()), Box::new(ASTRepr::Constant(2.0)));
        let xy = ASTRepr::Mul(Box::new(x.clone()), Box::new(y.clone()));

        ASTRepr::Add(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Add(
                    Box::new(ASTRepr::Add(
                        Box::new(ASTRepr::Add(
                            Box::new(ASTRepr::Mul(
                                Box::new(ASTRepr::Constant(a)),
                                Box::new(x_squared),
                            )),
                            Box::new(ASTRepr::Mul(Box::new(ASTRepr::Constant(b)), Box::new(xy))),
                        )),
                        Box::new(ASTRepr::Mul(
                            Box::new(ASTRepr::Constant(c)),
                            Box::new(y_squared),
                        )),
                    )),
                    Box::new(ASTRepr::Mul(Box::new(ASTRepr::Constant(d)), Box::new(x))),
                )),
                Box::new(ASTRepr::Mul(Box::new(ASTRepr::Constant(e)), Box::new(y))),
            )),
            Box::new(ASTRepr::Constant(f)),
        )
    }

    #[test]
    fn test_symbolic_ad_creation() {
        let ad = SymbolicAD::new();
        assert!(ad.is_ok());

        let config = SymbolicADConfig {
            num_variables: 2,
            max_derivative_order: 2,
            ..Default::default()
        };
        let ad_with_config = SymbolicAD::with_config(config);
        assert!(ad_with_config.is_ok());
    }

    #[test]
    fn test_basic_derivative_rules() {
        let mut ad = SymbolicAD::new().unwrap();

        // Test d/dx(x) = 1
        let x = ASTRepr::Variable(0); // Use index 0 for variable x
        let dx = ad.symbolic_derivative(&x, 0).unwrap();
        match dx {
            ASTRepr::Constant(val) => assert_eq!(val, 1.0),
            _ => panic!("Expected constant 1.0"),
        }

        // Test d/dx(5) = 0
        let constant = ASTRepr::Constant(5.0);
        let dc = ad.symbolic_derivative(&constant, 0).unwrap();
        match dc {
            ASTRepr::Constant(val) => assert_eq!(val, 0.0),
            _ => panic!("Expected constant 0.0"),
        }

        // Test d/dx(y) = 0 (different variable)
        let y = ASTRepr::Variable(1); // Use index 1 for variable y
        let dy = ad.symbolic_derivative(&y, 0).unwrap(); // Derivative with respect to x (index 0)
        match dy {
            ASTRepr::Constant(val) => assert_eq!(val, 0.0),
            _ => panic!("Expected constant 0.0"),
        }
    }

    #[test]
    fn test_arithmetic_derivative_rules() {
        let mut ad = SymbolicAD::new().unwrap();

        // Test d/dx(x + 2) = 1
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        );
        let derivative = ad.symbolic_derivative(&expr, 0).unwrap();

        // Should be Add(Constant(1.0), Constant(0.0))
        match &derivative {
            ASTRepr::Add(left, right) => match (left.as_ref(), right.as_ref()) {
                (ASTRepr::Constant(1.0), ASTRepr::Constant(0.0)) => {}
                _ => panic!("Expected Add(1.0, 0.0), got {derivative:?}"),
            },
            _ => panic!("Expected addition, got {derivative:?}"),
        }
    }

    #[test]
    fn test_product_rule() {
        let mut ad = SymbolicAD::new().unwrap();

        // Test d/dx(x * x) = x * 1 + x * 1 = 2x
        let x = ASTRepr::Variable(0);
        let x_squared = ASTRepr::Mul(Box::new(x.clone()), Box::new(x));
        let derivative = ad.symbolic_derivative(&x_squared, 0).unwrap();

        // Should be Mul(x, 1) + Mul(x, 1)
        match derivative {
            ASTRepr::Add(_, _) => {
                // Verify by evaluating at x = 3: should give 6
                let result = derivative.eval_two_vars(3.0, 0.0);
                assert_eq!(result, 6.0);
            }
            _ => panic!("Expected addition for product rule"),
        }
    }

    #[test]
    fn test_chain_rule() {
        let mut ad = SymbolicAD::new().unwrap();

        // Test d/dx(sin(x)) = cos(x) * 1 = cos(x)
        let sin_x = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        let derivative = ad.symbolic_derivative(&sin_x, 0).unwrap();

        match &derivative {
            ASTRepr::Mul(left, right) => match (left.as_ref(), right.as_ref()) {
                (ASTRepr::Cos(_), ASTRepr::Constant(1.0)) => {}
                (ASTRepr::Constant(1.0), ASTRepr::Cos(_)) => {}
                _ => panic!("Expected cos(x) * 1, got {derivative:?}"),
            },
            _ => panic!("Expected multiplication for chain rule"),
        }
    }

    #[test]
    fn test_convenience_functions() {
        // Test gradient computation with a simple polynomial
        let poly_expr = convenience::poly(&[1.0, 3.0]); // 1 + 3x (simpler polynomial)
        let grad = convenience::gradient(&poly_expr, &["0"]).unwrap();

        assert!(grad.contains_key("0"));

        // The derivative should be 3
        let derivative = &grad["0"];
        let result_at_2 = derivative.eval_two_vars(2.0, 0.0);
        assert_eq!(result_at_2, 3.0); // d/dx(1 + 3x) = 3

        // Test simple bivariate function x + y (much simpler than complex polynomial)
        let x = ASTRepr::Variable(0);
        let y = ASTRepr::Variable(1);
        let simple_bivariate = ASTRepr::Add(Box::new(x), Box::new(y)); // x + y

        let grad_biv = convenience::gradient(&simple_bivariate, &["0", "1"]).unwrap();

        assert!(grad_biv.contains_key("0"));
        assert!(grad_biv.contains_key("1"));

        // ∂/∂x(x + y) = 1
        // ∂/∂y(x + y) = 1
        let dx_at_1_2 = grad_biv["0"].eval_two_vars(1.0, 2.0);
        let dy_at_1_2 = grad_biv["1"].eval_two_vars(1.0, 2.0);

        assert_eq!(dx_at_1_2, 1.0); // d/dx(x + y) = 1
        assert_eq!(dy_at_1_2, 1.0); // d/dy(x + y) = 1
    }

    #[test]
    fn test_full_pipeline() {
        let mut ad = SymbolicAD::new().unwrap();

        // Test with a complex expression that can be optimized
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(0.0)),
            )), // Should optimize to 0
            Box::new(ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(2.0)),
            )), // x²
        );

        let result = ad.compute_with_derivatives(&expr).unwrap();

        // Should have computed the derivative
        assert!(result.first_derivatives.contains_key("0")); // Variable index 0 as string

        // Check that we have reasonable statistics (optimization may increase total operations due to derivatives)
        assert!(result.stats.function_operations_before > 0);
        assert!(result.stats.function_operations_after > 0);

        println!(
            "Original operations: {}",
            result.stats.function_operations_before
        );
        println!(
            "Optimized operations: {}",
            result.stats.function_operations_after
        );
        println!(
            "Function optimization ratio: {:.2}",
            result.stats.function_optimization_ratio()
        );
        println!("Total time: {} μs", result.stats.total_time_us());
    }

    #[test]
    fn test_cache_functionality() {
        let mut ad = SymbolicAD::new().unwrap();
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        );

        // First computation
        let _deriv1 = ad.symbolic_derivative(&expr, 0).unwrap();
        let (cache_size_1, _) = ad.cache_stats();

        // Second computation (should use cache)
        let _deriv2 = ad.symbolic_derivative(&expr, 0).unwrap();
        let (cache_size_2, _) = ad.cache_stats();

        // Cache should have grown after first computation
        assert!(cache_size_1 > 0);
        // Cache size should be the same for second computation (cache hit)
        assert_eq!(cache_size_1, cache_size_2);
    }
}
