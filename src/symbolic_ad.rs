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

use crate::error::Result;
use crate::final_tagless::JITRepr;
use crate::symbolic::SymbolicOptimizer;
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
    /// Variables to differentiate with respect to
    pub variables: Vec<String>,
}

impl Default for SymbolicADConfig {
    fn default() -> Self {
        Self {
            pre_optimize: true,
            post_optimize: true,
            share_subexpressions: true,
            max_derivative_order: 1,
            variables: vec!["x".to_string()],
        }
    }
}

/// Represents a function and its derivatives with shared subexpressions
#[derive(Debug, Clone)]
pub struct FunctionWithDerivatives<T> {
    /// The original function f(x)
    pub function: JITRepr<T>,
    /// First derivatives ∂f/∂x_i
    pub first_derivatives: HashMap<String, JITRepr<T>>,
    /// Second derivatives ∂²f/∂x_i∂x_j
    pub second_derivatives: HashMap<(String, String), JITRepr<T>>,
    /// Shared subexpressions identified during optimization
    pub shared_subexpressions: HashMap<String, JITRepr<T>>,
    /// Statistics about the optimization
    pub stats: SymbolicADStats,
}

/// Statistics about symbolic AD optimization
#[derive(Debug, Clone, Default)]
pub struct SymbolicADStats {
    /// Number of operations before optimization
    pub operations_before: usize,
    /// Number of operations after optimization
    pub operations_after: usize,
    /// Number of shared subexpressions found
    pub shared_subexpressions_count: usize,
    /// Time spent in each optimization stage (microseconds)
    pub stage_times_us: [u64; 3],
}

impl SymbolicADStats {
    /// Calculate the optimization ratio (operations_after / operations_before)
    pub fn optimization_ratio(&self) -> f64 {
        if self.operations_before == 0 {
            1.0
        } else {
            self.operations_after as f64 / self.operations_before as f64
        }
    }

    /// Calculate total optimization time
    pub fn total_time_us(&self) -> u64 {
        self.stage_times_us.iter().sum()
    }
}

/// Symbolic automatic differentiation engine
pub struct SymbolicAD {
    /// Configuration for the AD process
    config: SymbolicADConfig,
    /// Symbolic optimizer for egglog integration
    optimizer: SymbolicOptimizer,
    /// Cache for computed derivatives
    derivative_cache: HashMap<String, JITRepr<f64>>,
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
        expr: &JITRepr<f64>,
    ) -> Result<FunctionWithDerivatives<f64>> {
        let start_time = std::time::Instant::now();
        let mut stats = SymbolicADStats::default();
        stats.operations_before = expr.count_operations();

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
        let variables = self.config.variables.clone();

        // Compute first derivatives
        for var in &variables {
            let derivative = self.symbolic_derivative(&pre_optimized, var)?;
            first_derivatives.insert(var.clone(), derivative);
        }

        // Compute second derivatives if requested
        if self.config.max_derivative_order >= 2 {
            for var1 in &variables {
                for var2 in &variables {
                    if let Some(first_deriv) = first_derivatives.get(var1) {
                        let second_deriv = self.symbolic_derivative(first_deriv, var2)?;
                        second_derivatives.insert((var1.clone(), var2.clone()), second_deriv);
                    }
                }
            }
        }
        stats.stage_times_us[1] = stage2_start.elapsed().as_micros() as u64;

        // Stage 3: Post-optimization with subexpression sharing
        let stage3_start = std::time::Instant::now();
        let (optimized_function, optimized_derivatives, optimized_second_derivatives, shared_subexpressions) =
            if self.config.post_optimize && self.config.share_subexpressions {
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
                (pre_optimized, first_derivatives, second_derivatives, HashMap::new())
            };
        stats.stage_times_us[2] = stage3_start.elapsed().as_micros() as u64;

        // Calculate final statistics
        stats.operations_after = optimized_function.count_operations()
            + optimized_derivatives
                .values()
                .map(|d| d.count_operations())
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
    fn symbolic_derivative(&mut self, expr: &JITRepr<f64>, var: &str) -> Result<JITRepr<f64>> {
        // Check cache first
        let cache_key = format!("{:?}_{}", expr, var);
        if let Some(cached) = self.derivative_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let derivative = self.compute_derivative_recursive(expr, var)?;
        
        // Cache the result
        self.derivative_cache.insert(cache_key, derivative.clone());
        
        Ok(derivative)
    }

    /// Recursively compute the derivative using symbolic differentiation rules
    fn compute_derivative_recursive(&self, expr: &JITRepr<f64>, var: &str) -> Result<JITRepr<f64>> {
        match expr {
            // d/dx(c) = 0
            JITRepr::Constant(_) => Ok(JITRepr::Constant(0.0)),
            
            // d/dx(x) = 1, d/dx(y) = 0
            JITRepr::Variable(name) => {
                if name == var {
                    Ok(JITRepr::Constant(1.0))
                } else {
                    Ok(JITRepr::Constant(0.0))
                }
            }
            
            // d/dx(u + v) = du/dx + dv/dx
            JITRepr::Add(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;
                Ok(JITRepr::Add(Box::new(left_deriv), Box::new(right_deriv)))
            }
            
            // d/dx(u - v) = du/dx - dv/dx
            JITRepr::Sub(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;
                Ok(JITRepr::Sub(Box::new(left_deriv), Box::new(right_deriv)))
            }
            
            // d/dx(u * v) = u * dv/dx + v * du/dx (product rule)
            JITRepr::Mul(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;
                
                let term1 = JITRepr::Mul(left.clone(), Box::new(right_deriv));
                let term2 = JITRepr::Mul(right.clone(), Box::new(left_deriv));
                
                Ok(JITRepr::Add(Box::new(term1), Box::new(term2)))
            }
            
            // d/dx(u / v) = (v * du/dx - u * dv/dx) / v² (quotient rule)
            JITRepr::Div(left, right) => {
                let left_deriv = self.compute_derivative_recursive(left, var)?;
                let right_deriv = self.compute_derivative_recursive(right, var)?;
                
                let numerator_term1 = JITRepr::Mul(right.clone(), Box::new(left_deriv));
                let numerator_term2 = JITRepr::Mul(left.clone(), Box::new(right_deriv));
                let numerator = JITRepr::Sub(Box::new(numerator_term1), Box::new(numerator_term2));
                
                let denominator = JITRepr::Mul(right.clone(), right.clone());
                
                Ok(JITRepr::Div(Box::new(numerator), Box::new(denominator)))
            }
            
            // d/dx(u^v) = u^v * (v' * ln(u) + v * u'/u) (generalized power rule)
            JITRepr::Pow(base, exp) => {
                let base_deriv = self.compute_derivative_recursive(base, var)?;
                let exp_deriv = self.compute_derivative_recursive(exp, var)?;
                
                // u^v * (v' * ln(u) + v * u'/u)
                let ln_base = JITRepr::Ln(base.clone());
                let term1 = JITRepr::Mul(Box::new(exp_deriv), Box::new(ln_base));
                
                let u_prime_over_u = JITRepr::Div(Box::new(base_deriv), base.clone());
                let term2 = JITRepr::Mul(exp.clone(), Box::new(u_prime_over_u));
                
                let sum = JITRepr::Add(Box::new(term1), Box::new(term2));
                let original_power = JITRepr::Pow(base.clone(), exp.clone());
                
                Ok(JITRepr::Mul(Box::new(original_power), Box::new(sum)))
            }
            
            // d/dx(-u) = -du/dx
            JITRepr::Neg(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                Ok(JITRepr::Neg(Box::new(inner_deriv)))
            }
            
            // d/dx(ln(u)) = u'/u
            JITRepr::Ln(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                Ok(JITRepr::Div(Box::new(inner_deriv), inner.clone()))
            }
            
            // d/dx(exp(u)) = exp(u) * u'
            JITRepr::Exp(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                let exp_inner = JITRepr::Exp(inner.clone());
                Ok(JITRepr::Mul(Box::new(exp_inner), Box::new(inner_deriv)))
            }
            
            // d/dx(sqrt(u)) = u' / (2 * sqrt(u))
            JITRepr::Sqrt(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                let sqrt_inner = JITRepr::Sqrt(inner.clone());
                let two = JITRepr::Constant(2.0);
                let denominator = JITRepr::Mul(Box::new(two), Box::new(sqrt_inner));
                Ok(JITRepr::Div(Box::new(inner_deriv), Box::new(denominator)))
            }
            
            // d/dx(sin(u)) = cos(u) * u'
            JITRepr::Sin(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                let cos_inner = JITRepr::Cos(inner.clone());
                Ok(JITRepr::Mul(Box::new(cos_inner), Box::new(inner_deriv)))
            }
            
            // d/dx(cos(u)) = -sin(u) * u'
            JITRepr::Cos(inner) => {
                let inner_deriv = self.compute_derivative_recursive(inner, var)?;
                let sin_inner = JITRepr::Sin(inner.clone());
                let neg_sin = JITRepr::Neg(Box::new(sin_inner));
                Ok(JITRepr::Mul(Box::new(neg_sin), Box::new(inner_deriv)))
            }
        }
    }

    /// Optimize function and derivatives together to identify shared subexpressions
    fn optimize_with_subexpression_sharing(
        &mut self,
        function: &JITRepr<f64>,
        first_derivatives: &HashMap<String, JITRepr<f64>>,
        second_derivatives: &HashMap<(String, String), JITRepr<f64>>,
    ) -> Result<(
        JITRepr<f64>,
        HashMap<String, JITRepr<f64>>,
        HashMap<(String, String), JITRepr<f64>>,
        HashMap<String, JITRepr<f64>>,
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
        
        Ok((optimized_function, optimized_first, optimized_second, shared_subexpressions))
    }

    /// Get the current configuration
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
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.derivative_cache.len(), self.derivative_cache.capacity())
    }
}

impl Default for SymbolicAD {
    fn default() -> Self {
        Self::new().expect("Failed to create default SymbolicAD")
    }
}

/// Convenience functions for common symbolic AD operations
pub mod convenience {
    use super::*;
    use crate::final_tagless::{JITEval, JITMathExpr};

    /// Compute the gradient of a scalar function
    pub fn gradient(expr: &JITRepr<f64>, variables: &[&str]) -> Result<HashMap<String, JITRepr<f64>>> {
        let mut config = SymbolicADConfig::default();
        config.variables = variables.iter().map(|s| s.to_string()).collect();
        
        let mut ad = SymbolicAD::with_config(config)?;
        let result = ad.compute_with_derivatives(expr)?;
        
        Ok(result.first_derivatives)
    }

    /// Compute the Hessian matrix of a scalar function
    pub fn hessian(
        expr: &JITRepr<f64>,
        variables: &[&str],
    ) -> Result<HashMap<(String, String), JITRepr<f64>>> {
        let mut config = SymbolicADConfig::default();
        config.variables = variables.iter().map(|s| s.to_string()).collect();
        config.max_derivative_order = 2;
        
        let mut ad = SymbolicAD::with_config(config)?;
        let result = ad.compute_with_derivatives(expr)?;
        
        Ok(result.second_derivatives)
    }

    /// Create a simple quadratic function for testing: ax² + bx + c
    pub fn quadratic(a: f64, b: f64, c: f64) -> JITRepr<f64> {
        let x = JITEval::var("x");
        let x_squared = JITEval::pow(x.clone(), JITEval::constant(2.0));
        
        JITEval::add(
            JITEval::add(
                JITEval::mul(JITEval::constant(a), x_squared),
                JITEval::mul(JITEval::constant(b), x),
            ),
            JITEval::constant(c),
        )
    }

    /// Create a multivariate polynomial for testing: ax² + bxy + cy² + dx + ey + f
    pub fn bivariate_quadratic(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> JITRepr<f64> {
        let x = JITEval::var("x");
        let y = JITEval::var("y");
        
        let x_squared = JITEval::pow(x.clone(), JITEval::constant(2.0));
        let y_squared = JITEval::pow(y.clone(), JITEval::constant(2.0));
        let xy = JITEval::mul(x.clone(), y.clone());
        
        JITEval::add(
            JITEval::add(
                JITEval::add(
                    JITEval::add(
                        JITEval::add(
                            JITEval::mul(JITEval::constant(a), x_squared),
                            JITEval::mul(JITEval::constant(b), xy),
                        ),
                        JITEval::mul(JITEval::constant(c), y_squared),
                    ),
                    JITEval::mul(JITEval::constant(d), x),
                ),
                JITEval::mul(JITEval::constant(e), y),
            ),
            JITEval::constant(f),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{DirectEval, JITEval, JITMathExpr};

    #[test]
    fn test_symbolic_ad_creation() {
        let ad = SymbolicAD::new();
        assert!(ad.is_ok());
        
        let config = SymbolicADConfig {
            variables: vec!["x".to_string(), "y".to_string()],
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
        let x = JITEval::var("x");
        let dx = ad.symbolic_derivative(&x, "x").unwrap();
        match dx {
            JITRepr::Constant(val) => assert_eq!(val, 1.0),
            _ => panic!("Expected constant 1.0"),
        }
        
        // Test d/dx(5) = 0
        let constant = JITEval::constant(5.0);
        let dc = ad.symbolic_derivative(&constant, "x").unwrap();
        match dc {
            JITRepr::Constant(val) => assert_eq!(val, 0.0),
            _ => panic!("Expected constant 0.0"),
        }
        
        // Test d/dx(y) = 0 (different variable)
        let y = JITEval::var("y");
        let dy = ad.symbolic_derivative(&y, "x").unwrap();
        match dy {
            JITRepr::Constant(val) => assert_eq!(val, 0.0),
            _ => panic!("Expected constant 0.0"),
        }
    }

    #[test]
    fn test_arithmetic_derivative_rules() {
        let mut ad = SymbolicAD::new().unwrap();
        
        // Test d/dx(x + 2) = 1
        let expr = JITEval::add(JITEval::var("x"), JITEval::constant(2.0));
        let derivative = ad.symbolic_derivative(&expr, "x").unwrap();
        
        // Should be Add(Constant(1.0), Constant(0.0))
        match &derivative {
            JITRepr::Add(left, right) => {
                match (left.as_ref(), right.as_ref()) {
                    (JITRepr::Constant(1.0), JITRepr::Constant(0.0)) => {},
                    _ => panic!("Expected Add(1.0, 0.0), got {:?}", derivative),
                }
            },
            _ => panic!("Expected addition, got {:?}", derivative),
        }
    }

    #[test]
    fn test_product_rule() {
        let mut ad = SymbolicAD::new().unwrap();
        
        // Test d/dx(x * x) = x * 1 + x * 1 = 2x
        let x = JITEval::var("x");
        let x_squared = JITEval::mul(x.clone(), x);
        let derivative = ad.symbolic_derivative(&x_squared, "x").unwrap();
        
        // Should be Mul(x, 1) + Mul(x, 1)
        match derivative {
            JITRepr::Add(_, _) => {
                // Verify by evaluating at x = 3: should give 6
                let result = DirectEval::eval_two_vars(&derivative, 3.0, 0.0);
                assert_eq!(result, 6.0);
            },
            _ => panic!("Expected addition for product rule"),
        }
    }

    #[test]
    fn test_chain_rule() {
        let mut ad = SymbolicAD::new().unwrap();
        
        // Test d/dx(sin(x)) = cos(x) * 1 = cos(x)
        let sin_x = JITEval::sin(JITEval::var("x"));
        let derivative = ad.symbolic_derivative(&sin_x, "x").unwrap();
        
        match &derivative {
            JITRepr::Mul(left, right) => {
                match (left.as_ref(), right.as_ref()) {
                    (JITRepr::Cos(_), JITRepr::Constant(1.0)) => {},
                    (JITRepr::Constant(1.0), JITRepr::Cos(_)) => {},
                    _ => panic!("Expected cos(x) * 1, got {:?}", derivative),
                }
            },
            _ => panic!("Expected multiplication for chain rule"),
        }
    }

    #[test]
    fn test_convenience_functions() {
        // Test gradient computation
        let quadratic = convenience::quadratic(2.0, 3.0, 1.0); // 2x² + 3x + 1
        let grad = convenience::gradient(&quadratic, &["x"]).unwrap();
        
        assert!(grad.contains_key("x"));
        
        // The derivative should be 4x + 3
        let derivative = &grad["x"];
        let result_at_2 = DirectEval::eval_two_vars(derivative, 2.0, 0.0);
        assert_eq!(result_at_2, 11.0); // 4*2 + 3 = 11
        
        // Test bivariate function
        let bivariate = convenience::bivariate_quadratic(1.0, 2.0, 1.0, 0.0, 0.0, 0.0); // x² + 2xy + y²
        let grad_biv = convenience::gradient(&bivariate, &["x", "y"]).unwrap();
        
        assert!(grad_biv.contains_key("x"));
        assert!(grad_biv.contains_key("y"));
        
        // ∂/∂x(x² + 2xy + y²) = 2x + 2y
        // ∂/∂y(x² + 2xy + y²) = 2x + 2y
        let dx_at_1_2 = DirectEval::eval_two_vars(&grad_biv["x"], 1.0, 2.0);
        let dy_at_1_2 = DirectEval::eval_two_vars(&grad_biv["y"], 1.0, 2.0);
        
        assert_eq!(dx_at_1_2, 6.0); // 2*1 + 2*2 = 6
        assert_eq!(dy_at_1_2, 6.0); // 2*1 + 2*2 = 6
    }

    #[test]
    fn test_full_pipeline() {
        let mut ad = SymbolicAD::new().unwrap();
        
        // Test with a complex expression that can be optimized
        let expr = JITEval::add(
            JITEval::mul(JITEval::var("x"), JITEval::constant(0.0)), // Should optimize to 0
            JITEval::pow(JITEval::var("x"), JITEval::constant(2.0)),  // x²
        );
        
        let result = ad.compute_with_derivatives(&expr).unwrap();
        
        // Should have computed the derivative
        assert!(result.first_derivatives.contains_key("x"));
        
        // Check that we have reasonable statistics (optimization may increase total operations due to derivatives)
        assert!(result.stats.operations_before > 0);
        assert!(result.stats.operations_after > 0);
        
        println!("Original operations: {}", result.stats.operations_before);
        println!("Optimized operations: {}", result.stats.operations_after);
        println!("Optimization ratio: {:.2}", result.stats.optimization_ratio());
        println!("Total time: {} μs", result.stats.total_time_us());
    }

    #[test]
    fn test_cache_functionality() {
        let mut ad = SymbolicAD::new().unwrap();
        
        let expr = JITEval::pow(JITEval::var("x"), JITEval::constant(3.0));
        
        // First computation
        let _deriv1 = ad.symbolic_derivative(&expr, "x").unwrap();
        let (cache_size_1, _) = ad.cache_stats();
        
        // Second computation (should use cache)
        let _deriv2 = ad.symbolic_derivative(&expr, "x").unwrap();
        let (cache_size_2, _) = ad.cache_stats();
        
        assert_eq!(cache_size_1, cache_size_2); // Cache size shouldn't change
        
        // Clear cache
        ad.clear_cache();
        let (cache_size_3, _) = ad.cache_stats();
        assert_eq!(cache_size_3, 0);
    }
} 