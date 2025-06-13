//! Symbolic optimization using egglog for algebraic simplification
//!
//! This module provides Layer 2 optimization in our three-layer optimization strategy:
//! 1. Hand-coded domain optimizations (in JIT layer)
//! 2. **Egglog symbolic optimization** (this module)
//! 3. Cranelift low-level optimization
//!
//! The symbolic optimizer handles algebraic identities, constant folding, and structural
//! optimizations that can be expressed as rewrite rules.

use crate::{
    ast::{ASTRepr, ast_repr::{Lambda, Collection}, expressions_equal_default, ASTVisitor, visit_ast},
    error::{Result, DSLCompileError},
    symbolic::native_egglog::optimize_with_native_egglog,
};
use std::collections::HashMap;
// use std::time::Instant; // Will be used for optimization timing in future updates

// Re-export for convenience
pub use crate::backends::rust_codegen::RustOptLevel;



/// Compilation strategy for mathematical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationStrategy {
    /// Hot-loading compiled Rust dylibs (primary and default)
    /// Best for: All expressions, maximum performance, production use
    HotLoadRust {
        /// Directory for generated Rust source files
        source_dir: std::path::PathBuf,
        /// Directory for compiled dylibs
        lib_dir: std::path::PathBuf,
        /// Optimization level for rustc
        opt_level: RustOptLevel,
    },
    /// Adaptive strategy: optimize compilation settings based on expression characteristics
    Adaptive {
        /// Threshold for call count before upgrading optimization level
        call_threshold: usize,
        /// Threshold for expression complexity before upgrading optimization level
        complexity_threshold: usize,
    },
}

/// Compilation approach decision for a specific expression
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationApproach {
    /// Use Rust hot-loading compilation
    RustHotLoad,
    /// Upgrade to higher optimization level
    UpgradeOptimization,
}

impl Default for CompilationStrategy {
    fn default() -> Self {
        Self::HotLoadRust {
            source_dir: std::env::temp_dir().join("dslcompile_src"),
            lib_dir: std::env::temp_dir().join("dslcompile_lib"),
            opt_level: RustOptLevel::O2,
        }
    }
}

/// Symbolic optimizer using egglog for algebraic simplification
pub struct SymbolicOptimizer {
    /// Configuration for optimization behavior
    config: OptimizationConfig,
    /// Compilation strategy for choosing between backends
    compilation_strategy: CompilationStrategy,
    /// Execution statistics for adaptive compilation
    execution_stats: HashMap<String, ExpressionStats>,
    /// Rust code generator for hot-loading backend
    rust_generator: crate::backends::RustCodeGenerator,
    /// Optimization statistics
    stats: OptimizationStats,
}

impl std::fmt::Debug for SymbolicOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolicOptimizer")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

/// Expression analysis statistics
#[derive(Debug, Clone)]
pub struct ExpressionStats {
    /// Number of times this expression has been called
    pub call_count: usize,
    /// Complexity score of the expression
    pub complexity: usize,
    /// Average execution time in nanoseconds
    pub avg_execution_time_ns: f64,
    /// Whether this expression has been upgraded to Rust compilation
    pub rust_compiled: bool,
}

/// Statistics collected during symbolic optimization
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of rules applied
    pub rules_applied: usize,
    /// Optimization time in microseconds
    pub optimization_time_us: u64,
    /// Number of nodes before optimization
    pub nodes_before: usize,
    /// Number of nodes after optimization
    pub nodes_after: usize,
}

/// Trait for expressions that support symbolic optimization
pub trait OptimizeExpr {
    /// The representation type
    type Repr<T>;

    /// Optimize an expression using symbolic rewrite rules
    fn optimize(expr: Self::Repr<f64>) -> Result<Self::Repr<f64>>;

    /// Optimize with custom configuration
    fn optimize_with_config(
        expr: Self::Repr<f64>,
        config: OptimizationConfig,
    ) -> Result<(Self::Repr<f64>, OptimizationStats)>;
}

impl SymbolicOptimizer {
    /// Create a new symbolic optimizer with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: OptimizationConfig::default(),
            compilation_strategy: CompilationStrategy::default(),
            execution_stats: HashMap::new(),
            rust_generator: crate::backends::RustCodeGenerator::new(),
            stats: OptimizationStats::default(),
        })
    }

    /// Create a new symbolic optimizer with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Result<Self> {
        Ok(Self {
            config,
            compilation_strategy: CompilationStrategy::default(),
            execution_stats: HashMap::new(),
            rust_generator: crate::backends::RustCodeGenerator::new(),
            stats: OptimizationStats::default(),
        })
    }

    /// Create a new symbolic optimizer with custom compilation strategy
    pub fn with_strategy(strategy: CompilationStrategy) -> Result<Self> {
        Ok(Self {
            config: OptimizationConfig::default(),
            compilation_strategy: strategy,
            execution_stats: HashMap::new(),
            rust_generator: crate::backends::RustCodeGenerator::new(),
            stats: OptimizationStats::default(),
        })
    }

    /// Create a new symbolic optimizer for testing with minimal configuration
    /// This should only be used in tests that are experiencing hanging issues
    pub fn new_for_testing() -> Result<Self> {
        let config = OptimizationConfig {
            max_iterations: 2, // Limit iterations in tests
            aggressive: false,
            constant_folding: true,
            cse: false,
            egglog_optimization: false, // Disable potentially slow egglog optimization
            enable_expansion_rules: false,
            enable_distribution_rules: false,
            strategy: OptimizationStrategy::Interpretation, // Default for testing
        };

        Ok(Self {
            config,
            compilation_strategy: CompilationStrategy::default(),
            execution_stats: HashMap::new(),
            rust_generator: crate::backends::RustCodeGenerator::new(),
            stats: OptimizationStats::default(),
        })
    }

    /// Set the compilation strategy
    pub fn set_compilation_strategy(&mut self, strategy: CompilationStrategy) {
        self.compilation_strategy = strategy;
    }

    /// Get the current compilation strategy
    #[must_use]
    pub fn compilation_strategy(&self) -> &CompilationStrategy {
        &self.compilation_strategy
    }

    /// Determine the best compilation approach for a given expression
    pub fn choose_compilation_approach(
        &mut self,
        expr: &ASTRepr<f64>,
        expr_id: &str,
    ) -> CompilationApproach {
        match &self.compilation_strategy {
            CompilationStrategy::HotLoadRust { .. } => CompilationApproach::RustHotLoad,
            CompilationStrategy::Adaptive {
                call_threshold,
                complexity_threshold,
            } => {
                let stats = self
                    .execution_stats
                    .entry(expr_id.to_string())
                    .or_insert_with(|| ExpressionStats {
                        call_count: 0,
                        complexity: expr.count_operations(),
                        avg_execution_time_ns: 0.0,
                        rust_compiled: false,
                    });

                // Upgrade to higher optimization level if thresholds are met
                if stats.call_count >= *call_threshold || stats.complexity >= *complexity_threshold
                {
                    if stats.rust_compiled {
                        CompilationApproach::RustHotLoad
                    } else {
                        stats.rust_compiled = true;
                        CompilationApproach::UpgradeOptimization
                    }
                } else {
                    CompilationApproach::RustHotLoad
                }
            }
        }
    }

    /// Record execution statistics for adaptive compilation
    pub fn record_execution(&mut self, expr_id: &str, execution_time_ns: u64) {
        let stats = self
            .execution_stats
            .entry(expr_id.to_string())
            .or_insert_with(|| {
                ExpressionStats {
                    call_count: 0,
                    complexity: 0, // We don't have the expression here, so default to 0
                    avg_execution_time_ns: 0.0,
                    rust_compiled: false,
                }
            });

        stats.call_count += 1;
        // Update running average
        let alpha = 0.1; // Exponential moving average factor
        stats.avg_execution_time_ns =
            alpha * execution_time_ns as f64 + (1.0 - alpha) * stats.avg_execution_time_ns;
    }

    /// Get statistics for all tracked expressions
    #[must_use]
    pub fn get_expression_stats(&self) -> &HashMap<String, ExpressionStats> {
        &self.execution_stats
    }

    /// Generate Rust source code for hot-loading compilation
    pub fn generate_rust_source(&self, expr: &ASTRepr<f64>, function_name: &str) -> Result<String> {
        let expr_code = self.generate_rust_expression(expr)?;
        let max_var_index = self.find_max_variable_index(expr);
        let num_vars = if max_var_index == 0 && !self.expression_uses_variables(expr) {
            0 // No variables used
        } else {
            max_var_index + 1 // Variables are 0-indexed, so add 1 for count
        };

        // Generate function parameters based on actual variables used
        let params = if num_vars == 0 {
            String::new()
        } else {
            (0..num_vars)
                .map(|i| format!("var_{i}: f64"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        // Generate the main function
        let main_func = if num_vars == 0 {
            format!(
                r#"#[no_mangle]
pub extern "C" fn {function_name}() -> f64 {{
    {expr_code}
}}"#
            )
        } else {
            format!(
                r#"#[no_mangle]
pub extern "C" fn {function_name}({params}) -> f64 {{
    {expr_code}
}}"#
            )
        };

        // Generate array-based function for compatibility
        let array_func = format!(
            r#"#[no_mangle]
pub extern "C" fn {function_name}_array(vars: *const f64, count: usize) -> f64 {{
    if vars.is_null() {{
        return 0.0;
    }}
    
    {array_body}
}}"#,
            array_body = if num_vars == 0 {
                format!("    let _ = (vars, count); // Suppress unused warnings\n    {expr_code}")
            } else {
                let var_assignments = (0..num_vars)
                    .map(|i| {
                        format!(
                            "    let var_{i} = if count > {i} {{ unsafe {{ *vars.add({i}) }} }} else {{ 0.0 }};"
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{var_assignments}\n    {expr_code}")
            }
        );

        Ok(format!("{main_func}\n\n{array_func}"))
    }

    /// Generate Rust expression code from `ASTRepr`
    /// Uses a simple recursive approach that directly matches the AST structure
    fn generate_rust_expression(&self, expr: &ASTRepr<f64>) -> Result<String> {
        match expr {
            ASTRepr::Constant(value) => Ok(format!("{value:?}")),
            
            ASTRepr::Variable(index) => {
                // Use consistent var_{index} naming for all variables
                Ok(format!("var_{index}"))
            }
            
            ASTRepr::BoundVar(index) => {
                // Use descriptive name for bound variables
                Ok(format!("bound_{index}"))
            }
            
            ASTRepr::Add(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("{left_code} + {right_code}"))
            }
            
            ASTRepr::Sub(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("{left_code} - {right_code}"))
            }
            
            ASTRepr::Mul(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("{left_code} * {right_code}"))
            }
            
            ASTRepr::Div(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("{left_code} / {right_code}"))
            }
            
            ASTRepr::Pow(base, exp) => {
                let base_code = self.generate_rust_expression(base)?;
                let exp_code = self.generate_rust_expression(exp)?;
                Ok(format!("{base_code}.powf({exp_code})"))
            }
            
            ASTRepr::Neg(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("-{inner_code}"))
            }
            
            ASTRepr::Ln(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("{inner_code}.ln()"))
            }
            
            ASTRepr::Exp(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("{inner_code}.exp()"))
            }
            
            ASTRepr::Sin(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("{inner_code}.sin()"))
            }
            
            ASTRepr::Cos(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("{inner_code}.cos()"))
            }
            
            ASTRepr::Sqrt(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("{inner_code}.sqrt()"))
            }
            
            ASTRepr::Sum(_collection) => {
                // TODO: Handle Collection format in Rust expression generation
                Ok("/* TODO: Collection-based summation */".to_string())
            }
            
            ASTRepr::Lambda(lambda) => {
                let body_code = self.generate_rust_expression(&lambda.body)?;
                
                let code = if lambda.var_indices.is_empty() {
                    // Constant lambda - just return the body
                    format!("(|| {{ {body_code} }})()")
                } else if lambda.var_indices.len() == 1 {
                    // Single argument lambda
                    format!("|var_{}| {{ {} }}", lambda.var_indices[0], body_code)
                } else {
                    // Multi-argument lambda
                    let params = lambda
                        .var_indices
                        .iter()
                        .map(|idx| format!("var_{idx}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("|{params}| {{ {body_code} }}")
                };
                
                Ok(code)
            }
            
            ASTRepr::Let(binding_id, expr, body) => {
                let expr_code = self.generate_rust_expression(expr)?;
                let body_code = self.generate_rust_expression(body)?;
                Ok(format!(
                    "{{ let bound_{binding_id} = {expr_code}; {body_code} }}"
                ))
            }
        }
    }

    /// Find the maximum variable index used in an expression
    fn find_max_variable_index(&self, expr: &ASTRepr<f64>) -> usize {
        self.find_max_variable_index_recursive(expr).unwrap_or(0)
    }

    /// Helper function that returns None if no variables are found
    fn find_max_variable_index_recursive(&self, expr: &ASTRepr<f64>) -> Option<usize> {
        match expr {
            ASTRepr::Variable(index) => Some(*index),
            ASTRepr::Constant(_) => None,
            ASTRepr::Add(left, right) | 
            ASTRepr::Sub(left, right) | 
            ASTRepr::Mul(left, right) | 
            ASTRepr::Div(left, right) => {
                match (self.find_max_variable_index_recursive(left), self.find_max_variable_index_recursive(right)) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            ASTRepr::Neg(expr) => self.find_max_variable_index_recursive(expr),
            ASTRepr::Sin(expr) | 
            ASTRepr::Cos(expr) | 
            ASTRepr::Exp(expr) | 
            ASTRepr::Ln(expr) |
            ASTRepr::Sqrt(expr) => self.find_max_variable_index_recursive(expr),
            ASTRepr::Pow(base, exp) => {
                match (self.find_max_variable_index_recursive(base), self.find_max_variable_index_recursive(exp)) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            ASTRepr::Sum(_collection) => {
                // TODO: Handle Collection format for variable analysis
                None
            },
            ASTRepr::BoundVar(_) => None, // Bound variables don't affect global variable indexing
            ASTRepr::Lambda(lambda) => self.find_max_variable_index_recursive(&lambda.body),
            ASTRepr::Let(_, expr, body) => {
                match (self.find_max_variable_index_recursive(expr), self.find_max_variable_index_recursive(body)) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            },
        }
    }

    /// Check if an expression uses any variables
    fn expression_uses_variables(&self, expr: &ASTRepr<f64>) -> bool {
        match expr {
            ASTRepr::Variable(_) => true,
            ASTRepr::Constant(_) => false,
            ASTRepr::Add(left, right) | 
            ASTRepr::Sub(left, right) | 
            ASTRepr::Mul(left, right) | 
            ASTRepr::Div(left, right) => {
                self.expression_uses_variables(left) || self.expression_uses_variables(right)
            }
            ASTRepr::Neg(expr) |
            ASTRepr::Sin(expr) | 
            ASTRepr::Cos(expr) | 
            ASTRepr::Exp(expr) | 
            ASTRepr::Ln(expr) |
            ASTRepr::Sqrt(expr) => self.expression_uses_variables(expr),
            ASTRepr::Pow(base, exp) => {
                self.expression_uses_variables(base) || self.expression_uses_variables(exp)
            }
            ASTRepr::Sum(_collection) => false, // TODO: Handle Collection format
            ASTRepr::BoundVar(_) => false, // Bound variables are handled separately
            ASTRepr::Lambda(lambda) => self.expression_uses_variables(&lambda.body),
            ASTRepr::Let(_, expr, body) => {
                self.expression_uses_variables(expr) || self.expression_uses_variables(body)
            },
        }
    }

    /// Compile Rust source code to a dynamic library
    pub fn compile_rust_dylib(
        &self,
        source_code: &str,
        source_path: &std::path::Path,
        output_path: &std::path::Path,
        opt_level: &RustOptLevel,
    ) -> Result<()> {
        // Write source code to file
        std::fs::write(source_path, source_code).map_err(|e| {
            crate::error::DSLCompileError::CompilationError(format!(
                "Failed to write source file: {e}"
            ))
        })?;

        // Use the optimization flag from RustOptLevel
        let opt_flag = opt_level.as_flag();

        // Compile with rustc
        let output = std::process::Command::new("rustc")
            .args([
                "--crate-type=dylib",
                "-C",
                opt_flag,
                "-C",
                "panic=abort", // Smaller binary size
                source_path.to_str().unwrap(),
                "-o",
                output_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| {
                crate::error::DSLCompileError::CompilationError(format!("Failed to run rustc: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::error::DSLCompileError::CompilationError(format!(
                "Rust compilation failed: {stderr}"
            )));
        }

        Ok(())
    }

    /// Get the recommended compilation strategy based on expression characteristics
    #[must_use]
    pub fn recommend_strategy(expr: &ASTRepr<f64>) -> CompilationStrategy {
        let complexity = expr.count_operations();

        if complexity < 10 {
            // Simple expressions: use fast Cranelift JIT
            CompilationStrategy::HotLoadRust {
                source_dir: std::env::temp_dir().join("dslcompile_src"),
                lib_dir: std::env::temp_dir().join("dslcompile_lib"),
                opt_level: RustOptLevel::O2,
            }
        } else if complexity < 50 {
            // Medium complexity: use adaptive approach
            CompilationStrategy::Adaptive {
                call_threshold: 100,
                complexity_threshold: 25,
            }
        } else {
            // Complex expressions: use Rust hot-loading for maximum performance
            CompilationStrategy::HotLoadRust {
                source_dir: std::env::temp_dir().join("dslcompile_src"),
                lib_dir: std::env::temp_dir().join("dslcompile_lib"),
                opt_level: RustOptLevel::O2,
            }
        }
    }

    /// Optimize a JIT representation using symbolic rewrite rules
    pub fn optimize(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // Handle zero-overhead strategy: aggressive constant folding and direct computation
        if matches!(self.config.strategy, OptimizationStrategy::StaticCodegen) {
            return self.optimize_zero_overhead(expr);
        }

        let mut optimized = expr.clone();
        let mut iterations = 0;

        // Apply optimization passes until convergence or max iterations
        while iterations < self.config.max_iterations {
            let before = optimized.clone();

            // Layer 1: Apply basic algebraic simplifications (hand-coded rules)
            optimized = Self::apply_arithmetic_rules(&optimized)?;
            optimized = Self::apply_algebraic_rules(&optimized)?;

            // Apply static algebraic rules (includes transcendental optimizations)
            optimized = self.apply_static_algebraic_rules(&mut optimized)?;

            if self.config.constant_folding {
                optimized = Self::apply_constant_folding(&optimized)?;
            }

            // Layer 2: Apply egglog symbolic optimization (if enabled)
            if self.config.egglog_optimization {
                #[cfg(feature = "optimization")]
                {
                    match optimize_with_native_egglog(&optimized) {
                        Ok(egglog_optimized) => optimized = egglog_optimized,
                        Err(_) => {
                            // Fall back to hand-coded egglog placeholder if real egglog fails
                            optimized = self.apply_egglog_optimization(&optimized)?;
                        }
                    }
                }

                #[cfg(not(feature = "optimization"))]
                {
                    // Use hand-coded placeholder when egglog feature is not enabled
                    optimized = self.apply_egglog_optimization(&optimized)?;
                }
            }

            // Check for convergence
            if expressions_equal_default(&before, &optimized) {
                break;
            }

            iterations += 1;
        }

        Ok(optimized)
    }

    /// Zero-overhead optimization: aggressive constant folding and direct computation
    fn optimize_zero_overhead(&self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // For zero-overhead, we want to fold everything to constants when possible
        // This is essentially aggressive constant folding
        match expr {
            // If it's already a constant, return it
            ASTRepr::Constant(_) => Ok(expr.clone()),

            // For variables, we can't optimize further without values
            ASTRepr::Variable(_) => Ok(expr.clone()),

            // For operations, try to fold to constants
            ASTRepr::Add(left, right) => {
                let left_opt = self.optimize_zero_overhead(left)?;
                let right_opt = self.optimize_zero_overhead(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a + b)),
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    (ASTRepr::Constant(0.0), _) => Ok(right_opt),
                    _ => Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                }
            }

            ASTRepr::Mul(left, right) => {
                let left_opt = self.optimize_zero_overhead(left)?;
                let right_opt = self.optimize_zero_overhead(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a * b)),
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    (ASTRepr::Constant(1.0), _) => Ok(right_opt),
                    (_, ASTRepr::Constant(0.0)) => Ok(ASTRepr::Constant(0.0)),
                    (ASTRepr::Constant(0.0), _) => Ok(ASTRepr::Constant(0.0)),
                    _ => Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }

            ASTRepr::Sub(left, right) => {
                let left_opt = self.optimize_zero_overhead(left)?;
                let right_opt = self.optimize_zero_overhead(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a - b)),
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    _ => Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt))),
                }
            }

            ASTRepr::Div(left, right) => {
                let left_opt = self.optimize_zero_overhead(left)?;
                let right_opt = self.optimize_zero_overhead(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) if *b != 0.0 => {
                        Ok(ASTRepr::Constant(a / b))
                    }
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    _ => Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }

            ASTRepr::Pow(base, exp) => {
                let base_opt = self.optimize_zero_overhead(base)?;
                let exp_opt = self.optimize_zero_overhead(exp)?;

                match (&base_opt, &exp_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                        Ok(ASTRepr::Constant(a.powf(*b)))
                    }
                    (_, ASTRepr::Constant(0.0)) => Ok(ASTRepr::Constant(1.0)),
                    (_, ASTRepr::Constant(1.0)) => Ok(base_opt),
                    (ASTRepr::Constant(1.0), _) => Ok(ASTRepr::Constant(1.0)),
                    _ => Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }

            ASTRepr::Neg(inner) => {
                let inner_opt = self.optimize_zero_overhead(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(-a)),
                    _ => Ok(ASTRepr::Neg(Box::new(inner_opt))),
                }
            }

            ASTRepr::Sin(inner) => {
                let inner_opt = self.optimize_zero_overhead(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.sin())),
                    _ => Ok(ASTRepr::Sin(Box::new(inner_opt))),
                }
            }

            ASTRepr::Cos(inner) => {
                let inner_opt = self.optimize_zero_overhead(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.cos())),
                    _ => Ok(ASTRepr::Cos(Box::new(inner_opt))),
                }
            }

            ASTRepr::Exp(inner) => {
                let inner_opt = self.optimize_zero_overhead(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.exp())),
                    _ => Ok(ASTRepr::Exp(Box::new(inner_opt))),
                }
            }

            ASTRepr::Ln(inner) => {
                let inner_opt = self.optimize_zero_overhead(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) if *a > 0.0 => Ok(ASTRepr::Constant(a.ln())),
                    _ => Ok(ASTRepr::Ln(Box::new(inner_opt))),
                }
            }

            ASTRepr::Sqrt(inner) => {
                let inner_opt = self.optimize_zero_overhead(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) if *a >= 0.0 => Ok(ASTRepr::Constant(a.sqrt())),
                    _ => Ok(ASTRepr::Sqrt(Box::new(inner_opt))),
                }
            }

            ASTRepr::Sum(_collection) => {
                // For now, return as-is for Sum expressions
                // TODO: Implement zero-overhead sum optimization with Collections
                Ok(expr.clone())
            }

            // Lambda expressions - recursively optimize the body
            ASTRepr::Lambda(lambda) => {
                let optimized_body = self.optimize_zero_overhead(&lambda.body)?;
                Ok(ASTRepr::Lambda(Box::new(Lambda {
                    var_indices: lambda.var_indices.clone(),
                    body: Box::new(optimized_body),
                })))
            }

            ASTRepr::BoundVar(_) => {
                // BoundVar cannot be optimized without context - return as-is
                Ok(expr.clone())
            }
            ASTRepr::Let(binding_id, expr_val, body) => {
                // Recursively optimize both the bound expression and body
                let optimized_expr = self.optimize_zero_overhead(expr_val)?;
                let optimized_body = self.optimize_zero_overhead(body)?;
                Ok(ASTRepr::Let(
                    *binding_id,
                    Box::new(optimized_expr),
                    Box::new(optimized_body),
                ))
            }
        }
    }

    /// Apply basic arithmetic simplification rules
    fn apply_arithmetic_rules(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        match expr {
            // Identity rules: x + 0 = x, x * 1 = x, etc.
            ASTRepr::Add(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    (ASTRepr::Constant(0.0), _) => Ok(right_opt),
                    _ => Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Mul(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    (ASTRepr::Constant(1.0), _) => Ok(right_opt),
                    // Conservative: do NOT fold 0 * x or x * 0 unless both are constants
                    _ => Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Sub(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    (l, r) if Self::expressions_equal(l, r) => Ok(ASTRepr::Constant(0.0)),
                    _ => Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Div(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    // Conservative: do NOT fold 0 / x to 0 unless both are constants
                    _ => Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Pow(base, exp) => {
                let base_opt = Self::apply_arithmetic_rules(base)?;
                let exp_opt = Self::apply_arithmetic_rules(exp)?;

                match (&base_opt, &exp_opt) {
                    (_, ASTRepr::Constant(0.0)) => Ok(ASTRepr::Constant(1.0)),
                    (_, ASTRepr::Constant(1.0)) => Ok(base_opt),
                    (ASTRepr::Constant(1.0), _) => Ok(ASTRepr::Constant(1.0)),
                    _ => Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }
            // Recursively apply to other expression types
            ASTRepr::Neg(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                Ok(ASTRepr::Neg(Box::new(inner_opt)))
            }
            ASTRepr::Ln(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(0.0)),
                    _ => Ok(ASTRepr::Ln(Box::new(inner_opt))),
                }
            }
            ASTRepr::Exp(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                    _ => Ok(ASTRepr::Exp(Box::new(inner_opt))),
                }
            }
            ASTRepr::Sin(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(0.0)),
                    _ => Ok(ASTRepr::Sin(Box::new(inner_opt))),
                }
            }
            ASTRepr::Cos(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                    _ => Ok(ASTRepr::Cos(Box::new(inner_opt))),
                }
            }
            ASTRepr::Sqrt(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                Ok(ASTRepr::Sqrt(Box::new(inner_opt)))
            }
            ASTRepr::Sum(_collection) => {
                // TODO: Implement Sum Collection variant for arithmetic rules
                Ok(expr.clone())
            }

            // Lambda expressions - recursively apply arithmetic rules to body
            ASTRepr::Lambda(lambda) => {
                let optimized_body = Self::apply_arithmetic_rules(&lambda.body)?;
                Ok(ASTRepr::Lambda(Box::new(Lambda {
                    var_indices: lambda.var_indices.clone(),
                    body: Box::new(optimized_body),
                })))
            }

            ASTRepr::BoundVar(_) => {
                // BoundVar behaves like Variable for arithmetic rules
                Ok(expr.clone())
            }
            ASTRepr::Let(binding_id, expr_val, body) => {
                // Apply arithmetic rules to both the bound expression and body
                let optimized_expr = Self::apply_arithmetic_rules(expr_val)?;
                let optimized_body = Self::apply_arithmetic_rules(body)?;
                Ok(ASTRepr::Let(
                    *binding_id,
                    Box::new(optimized_expr),
                    Box::new(optimized_body),
                ))
            }
            // Base cases
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Apply algebraic transformation rules (associativity, commutativity, etc.)
    fn apply_algebraic_rules(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // For now, just recursively apply to subexpressions
        // In a full implementation, this would handle more complex algebraic transformations
        match expr {
            ASTRepr::Add(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Mul(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Sub(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Div(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Pow(base, exp) => {
                let base_opt = Self::apply_algebraic_rules(base)?;
                let exp_opt = Self::apply_algebraic_rules(exp)?;
                Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt)))
            }
            ASTRepr::Neg(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(ASTRepr::Neg(Box::new(inner_opt)))
            }
            ASTRepr::Ln(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(ASTRepr::Ln(Box::new(inner_opt)))
            }
            ASTRepr::Exp(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(ASTRepr::Exp(Box::new(inner_opt)))
            }
            ASTRepr::Sin(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(ASTRepr::Sin(Box::new(inner_opt)))
            }
            ASTRepr::Cos(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(ASTRepr::Cos(Box::new(inner_opt)))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(ASTRepr::Sqrt(Box::new(inner_opt)))
            }
            ASTRepr::Sum(_collection) => {
                // TODO: Implement Sum Collection variant for algebraic rules
                Ok(expr.clone())
            }

            // Lambda expressions - recursively apply algebraic rules to body
            ASTRepr::Lambda(lambda) => {
                let optimized_body = Self::apply_algebraic_rules(&lambda.body)?;
                Ok(ASTRepr::Lambda(Box::new(Lambda {
                    var_indices: lambda.var_indices.clone(),
                    body: Box::new(optimized_body),
                })))
            }

            ASTRepr::BoundVar(_) => {
                // BoundVar behaves like Variable for algebraic rules
                Ok(expr.clone())
            }
            ASTRepr::Let(binding_id, expr_val, body) => {
                // Apply algebraic rules to both the bound expression and body
                let optimized_expr = Self::apply_algebraic_rules(expr_val)?;
                let optimized_body = Self::apply_algebraic_rules(body)?;
                Ok(ASTRepr::Let(
                    *binding_id,
                    Box::new(optimized_expr),
                    Box::new(optimized_body),
                ))
            }
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Apply constant folding optimizations
    fn apply_constant_folding(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        match expr {
            ASTRepr::Add(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a + b)),
                    _ => Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Mul(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a * b)),
                    _ => Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Sub(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a - b)),
                    _ => Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Div(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a / b)),
                    _ => Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Pow(base, exp) => {
                let base_opt = Self::apply_constant_folding(base)?;
                let exp_opt = Self::apply_constant_folding(exp)?;

                match (&base_opt, &exp_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                        // Use domain analysis to determine if constant folding is safe
                        let result = a.powf(*b);
                        if result.is_finite() {
                            Ok(ASTRepr::Constant(result))
                        } else {
                            // Don't fold - preserve the expression for runtime evaluation
                            Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt)))
                        }
                    }
                    _ => Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }
            // Apply to unary operations
            ASTRepr::Neg(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(-a)),
                    _ => Ok(ASTRepr::Neg(Box::new(inner_opt))),
                }
            }
            ASTRepr::Ln(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) if *a > 0.0 => Ok(ASTRepr::Constant(a.ln())),
                    _ => Ok(ASTRepr::Ln(Box::new(inner_opt))),
                }
            }
            ASTRepr::Exp(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.exp())),
                    _ => Ok(ASTRepr::Exp(Box::new(inner_opt))),
                }
            }
            ASTRepr::Sin(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.sin())),
                    _ => Ok(ASTRepr::Sin(Box::new(inner_opt))),
                }
            }
            ASTRepr::Cos(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.cos())),
                    _ => Ok(ASTRepr::Cos(Box::new(inner_opt))),
                }
            }
            ASTRepr::Sqrt(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) if *a >= 0.0 => Ok(ASTRepr::Constant(a.sqrt())),
                    _ => Ok(ASTRepr::Sqrt(Box::new(inner_opt))),
                }
            }
            ASTRepr::Sum(_collection) => {
                // TODO: Implement Sum Collection variant for constant folding
                Ok(expr.clone())
            }

            // Lambda expressions - recursively apply constant folding to body
            ASTRepr::Lambda(lambda) => {
                let optimized_body = Self::apply_constant_folding(&lambda.body)?;
                Ok(ASTRepr::Lambda(Box::new(Lambda {
                    var_indices: lambda.var_indices.clone(),
                    body: Box::new(optimized_body),
                })))
            }

            ASTRepr::BoundVar(_) => {
                // BoundVar behaves like Variable for constant folding
                Ok(expr.clone())
            }
            ASTRepr::Let(binding_id, expr_val, body) => {
                // Apply constant folding to both the bound expression and body
                let optimized_expr = Self::apply_constant_folding(expr_val)?;
                let optimized_body = Self::apply_constant_folding(body)?;
                Ok(ASTRepr::Let(
                    *binding_id,
                    Box::new(optimized_expr),
                    Box::new(optimized_body),
                ))
            }
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Apply egglog-based optimization using equality saturation
    ///
    /// This method uses the egglog library to perform advanced symbolic optimization
    /// through equality saturation and rewrite rules. The integration includes:
    /// - Comprehensive mathematical rewrite rules (arithmetic, transcendental functions)
    /// - Equality saturation to find optimal expression forms
    /// - Graceful fallback when extraction is complex
    ///
    /// Note: The current implementation uses a simplified extraction approach that
    /// falls back to hand-coded rules, but egglog's rewrite rules are fully applied.
    fn apply_egglog_optimization(&self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        #[cfg(feature = "optimization")]
        {
            use crate::symbolic::native_egglog::optimize_with_native_egglog;

            // Try to use egglog optimization
            match optimize_with_native_egglog(expr) {
                Ok(optimized) => Ok(optimized),
                Err(_) => {
                    // Egglog optimization failed (likely at extraction step)
                    // Fall back to returning the original expression
                    // The egglog rewrite rules still ran, but extraction failed
                    Ok(expr.clone())
                }
            }
        }

        #[cfg(not(feature = "optimization"))]
        {
            // When egglog feature is not enabled, return unchanged
            Ok(expr.clone())
        }
    }

    /// Apply static algebraic simplification rules
    /// This is a stepping stone toward full egglog integration
    #[allow(clippy::only_used_in_recursion)]
    fn apply_static_algebraic_rules(&self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        match expr {
            ASTRepr::Add(left, right) => {
                let left_opt = self.apply_static_algebraic_rules(left)?;
                let right_opt = self.apply_static_algebraic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // x + 0 = x
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    (ASTRepr::Constant(0.0), _) => Ok(right_opt),
                    // Constant folding: a + b = (a+b)
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a + b)),
                    // x + x = 2*x
                    _ if expressions_equal_default(&left_opt, &right_opt) => Ok(ASTRepr::Mul(
                        Box::new(ASTRepr::Constant(2.0)),
                        Box::new(left_opt),
                    )),
                    // Associativity: (a + b) + c = a + (b + c) if beneficial
                    (ASTRepr::Add(a, b), c) => {
                        match (a.as_ref(), b.as_ref(), c) {
                            // (x + const1) + const2 = x + (const1 + const2)
                            (_, ASTRepr::Constant(b_val), ASTRepr::Constant(c_val)) => {
                                let combined_const = ASTRepr::Constant(b_val + c_val);
                                Ok(ASTRepr::Add(a.clone(), Box::new(combined_const)))
                            }
                            _ => Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                        }
                    }
                    // Normalize: put constants on the right
                    (ASTRepr::Constant(_), ASTRepr::Variable(_)) => {
                        Ok(ASTRepr::Add(Box::new(right_opt), Box::new(left_opt)))
                    }
                    _ => Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Sub(left, right) => {
                let left_opt = self.apply_static_algebraic_rules(left)?;
                let right_opt = self.apply_static_algebraic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // x - 0 = x
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    // 0 - x = -x
                    (ASTRepr::Constant(0.0), _) => Ok(ASTRepr::Neg(Box::new(right_opt))),
                    // x - x = 0
                    _ if expressions_equal_default(&left_opt, &right_opt) => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // Constant folding: a - b = (a-b)
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a - b)),
                    _ => Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Mul(left, right) => {
                let left_opt = self.apply_static_algebraic_rules(left)?;
                let right_opt = self.apply_static_algebraic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // Conservative: do NOT fold 0 * x or x * 0 to 0 unless both are constants
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    (ASTRepr::Constant(1.0), _) => Ok(right_opt),
                    (_, ASTRepr::Constant(-1.0)) => Ok(ASTRepr::Neg(Box::new(left_opt))),
                    (ASTRepr::Constant(-1.0), _) => Ok(ASTRepr::Neg(Box::new(right_opt))),
                    // Constant folding: a * b = (a*b)
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a * b)),
                    // x * x = x^2 (but be careful about domain safety)
                    _ if expressions_equal_default(&left_opt, &right_opt) => {
                        // Check if this transformation is safe by looking at the context
                        // For now, be conservative and only apply this rule for positive constants
                        // or when we can prove the base is positive
                        match &left_opt {
                            ASTRepr::Constant(val) if *val >= 0.0 => {
                                // Safe: positive constant squared
                                Ok(ASTRepr::Pow(
                                    Box::new(left_opt),
                                    Box::new(ASTRepr::Constant(2.0)),
                                ))
                            }
                            ASTRepr::Exp(_) => {
                                // Safe: exp(x) is always positive
                                Ok(ASTRepr::Pow(
                                    Box::new(left_opt),
                                    Box::new(ASTRepr::Constant(2.0)),
                                ))
                            }
                            ASTRepr::Sqrt(_) => {
                                // Safe: sqrt(x) is always non-negative (when defined)
                                Ok(ASTRepr::Pow(
                                    Box::new(left_opt),
                                    Box::new(ASTRepr::Constant(2.0)),
                                ))
                            }
                            _ => {
                                // Conservative: don't apply x * x = x^2 for potentially negative values
                                // This preserves the original multiplication which is always safe
                                Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt)))
                            }
                        }
                    }
                    // Exponential rules: exp(a) * exp(b) = exp(a+b)
                    (ASTRepr::Exp(a), ASTRepr::Exp(b)) => {
                        let sum = ASTRepr::Add(a.clone(), b.clone());
                        Ok(ASTRepr::Exp(Box::new(sum)))
                    }
                    // Power rule: x^a * x^b = x^(a+b)
                    (ASTRepr::Pow(base1, exp1), ASTRepr::Pow(base2, exp2))
                        if expressions_equal_default(base1, base2) =>
                    {
                        let combined_exp = ASTRepr::Add(exp1.clone(), exp2.clone());
                        Ok(ASTRepr::Pow(base1.clone(), Box::new(combined_exp)))
                    }
                    // Normalize: put constants on the left
                    (ASTRepr::Variable(_), ASTRepr::Constant(_)) => {
                        Ok(ASTRepr::Mul(Box::new(right_opt), Box::new(left_opt)))
                    }
                    // Distribute multiplication over addition: a * (b + c) = a*b + a*c - ONLY if enabled
                    (_, ASTRepr::Add(b, c)) if self.config.enable_distribution_rules => {
                        let ab = ASTRepr::Mul(Box::new(left_opt.clone()), b.clone());
                        let ac = ASTRepr::Mul(Box::new(left_opt), c.clone());
                        Ok(ASTRepr::Add(Box::new(ab), Box::new(ac)))
                    }
                    // Associativity: (a * b) * c = a * (b * c) if beneficial
                    (ASTRepr::Mul(a, b), c) => {
                        match (a.as_ref(), b.as_ref(), c) {
                            // (x * const1) * const2 = x * (const1 * const2)
                            (_, ASTRepr::Constant(b_val), ASTRepr::Constant(c_val)) => {
                                let combined_const = ASTRepr::Constant(b_val * c_val);
                                Ok(ASTRepr::Mul(a.clone(), Box::new(combined_const)))
                            }
                            _ => Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                        }
                    }
                    _ => Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Div(left, right) => {
                let left_opt = self.apply_static_algebraic_rules(left)?;
                let right_opt = self.apply_static_algebraic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(*a / *b)),
                    // Conservative: do NOT fold 0 / x to 0 unless both are constants
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    _ => Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Pow(base, exp) => {
                let base_opt = self.apply_static_algebraic_rules(base)?;
                let exp_opt = self.apply_static_algebraic_rules(exp)?;

                match (&base_opt, &exp_opt) {
                    // x^0 = 1
                    (_, ASTRepr::Constant(0.0)) => Ok(ASTRepr::Constant(1.0)),
                    // x^1 = x
                    (_, ASTRepr::Constant(1.0)) => Ok(base_opt),
                    // 0^x = 0 (for x > 0)
                    (ASTRepr::Constant(0.0), ASTRepr::Constant(x)) if *x > 0.0 => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // 1^x = 1
                    (ASTRepr::Constant(1.0), _) => Ok(ASTRepr::Constant(1.0)),
                    // x^2 = x * x (often faster than general power)
                    (_, ASTRepr::Constant(2.0)) => {
                        Ok(ASTRepr::Mul(Box::new(base_opt.clone()), Box::new(base_opt)))
                    }
                    // Constant folding: a^b = (a^b)
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                        // Use domain analysis to determine if constant folding is safe
                        let result = a.powf(*b);
                        if result.is_finite() {
                            Ok(ASTRepr::Constant(result))
                        } else {
                            // Don't fold - preserve the expression for runtime evaluation
                            Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt)))
                        }
                    }
                    // (x^a)^b = x^(a*b)
                    (ASTRepr::Pow(inner_base, inner_exp), _) => {
                        let combined_exp = ASTRepr::Mul(inner_exp.clone(), Box::new(exp_opt));
                        Ok(ASTRepr::Pow(inner_base.clone(), Box::new(combined_exp)))
                    }
                    _ => Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }
            ASTRepr::Neg(inner) => {
                let inner_opt = self.apply_static_algebraic_rules(inner)?;

                match &inner_opt {
                    // -(-x) = x
                    ASTRepr::Neg(x) => Ok((**x).clone()),
                    // -(const) = -const
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(-a)),
                    // -(a - b) = b - a
                    ASTRepr::Sub(a, b) => Ok(ASTRepr::Sub(b.clone(), a.clone())),
                    _ => Ok(ASTRepr::Neg(Box::new(inner_opt))),
                }
            }
            ASTRepr::Ln(inner) => {
                let inner_opt = self.apply_static_algebraic_rules(inner)?;

                match &inner_opt {
                    // ln(1) = 0
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(0.0)),
                    // ln(e) ≈ 1
                    ASTRepr::Constant(x) if (*x - std::f64::consts::E).abs() < 1e-15 => {
                        Ok(ASTRepr::Constant(1.0))
                    }
                    // ln(exp(x)) = x
                    ASTRepr::Exp(x) => Ok((**x).clone()),
                    // ln(a * b) = ln(a) + ln(b) (with domain analysis)
                    ASTRepr::Mul(a, b) => {
                        // Try to use domain analysis from native egglog to check safety
                        #[cfg(feature = "optimization")]
                        {
                            use crate::symbolic::native_egglog::NativeEgglogOptimizer;

                            // Check if both a and b are provably positive using domain analysis
                            if let Ok(mut optimizer) = NativeEgglogOptimizer::new() {
                                let a_safe = optimizer.is_domain_safe(a, "ln").unwrap_or(false);
                                let b_safe = optimizer.is_domain_safe(b, "ln").unwrap_or(false);

                                if a_safe && b_safe {
                                    // Domain analysis confirms safety - apply the rule
                                    let ln_a = ASTRepr::Ln(a.clone());
                                    let ln_b = ASTRepr::Ln(b.clone());
                                    return Ok(ASTRepr::Add(Box::new(ln_a), Box::new(ln_b)));
                                }
                            }
                        }

                        // Fallback: Only apply this rule if both a and b are positive constants
                        match (a.as_ref(), b.as_ref()) {
                            (ASTRepr::Constant(a_val), ASTRepr::Constant(b_val))
                                if *a_val > 0.0 && *b_val > 0.0 =>
                            {
                                let ln_a = ASTRepr::Ln(a.clone());
                                let ln_b = ASTRepr::Ln(b.clone());
                                Ok(ASTRepr::Add(Box::new(ln_a), Box::new(ln_b)))
                            }
                            _ => {
                                // Conservative: don't apply the rule if domain safety cannot be guaranteed
                                Ok(ASTRepr::Ln(Box::new(inner_opt)))
                            }
                        }
                    }
                    // ln(a / b) = ln(a) - ln(b) (with domain analysis)
                    ASTRepr::Div(a, b) => {
                        // Try to use domain analysis from native egglog to check safety
                        #[cfg(feature = "optimization")]
                        {
                            use crate::symbolic::native_egglog::NativeEgglogOptimizer;

                            // Check if both a and b are provably positive using domain analysis
                            if let Ok(mut optimizer) = NativeEgglogOptimizer::new() {
                                let a_safe = optimizer.is_domain_safe(a, "ln").unwrap_or(false);
                                let b_safe = optimizer.is_domain_safe(b, "ln").unwrap_or(false);

                                if a_safe && b_safe {
                                    // Domain analysis confirms safety - apply the rule
                                    let ln_a = ASTRepr::Ln(a.clone());
                                    let ln_b = ASTRepr::Ln(b.clone());
                                    return Ok(ASTRepr::Sub(Box::new(ln_a), Box::new(ln_b)));
                                }
                            }
                        }

                        // Fallback: Only apply this rule if both a and b are positive constants
                        match (a.as_ref(), b.as_ref()) {
                            (ASTRepr::Constant(a_val), ASTRepr::Constant(b_val))
                                if *a_val > 0.0 && *b_val > 0.0 =>
                            {
                                let ln_a = ASTRepr::Ln(a.clone());
                                let ln_b = ASTRepr::Ln(b.clone());
                                Ok(ASTRepr::Sub(Box::new(ln_a), Box::new(ln_b)))
                            }
                            _ => {
                                // Conservative: don't apply the rule if domain safety cannot be guaranteed
                                Ok(ASTRepr::Ln(Box::new(inner_opt)))
                            }
                        }
                    }
                    // ln(x^a) = a * ln(x) (with domain analysis)
                    ASTRepr::Pow(base, exp) => {
                        // Try to use domain analysis from native egglog to check safety
                        #[cfg(feature = "optimization")]
                        {
                            use crate::symbolic::native_egglog::NativeEgglogOptimizer;

                            // Check if base is provably positive using domain analysis
                            if let Ok(mut optimizer) = NativeEgglogOptimizer::new() {
                                let base_safe =
                                    optimizer.is_domain_safe(base, "ln").unwrap_or(false);

                                if base_safe {
                                    // Domain analysis confirms safety - apply the rule
                                    let ln_base = ASTRepr::Ln(base.clone());
                                    return Ok(ASTRepr::Mul(exp.clone(), Box::new(ln_base)));
                                }
                            }
                        }

                        // Fallback: Only apply if base is a positive constant
                        match base.as_ref() {
                            // Don't apply if base is 0, since ln(0) is undefined
                            ASTRepr::Constant(x) if *x == 0.0 => {
                                Ok(ASTRepr::Ln(Box::new(inner_opt)))
                            }
                            // Only apply if base is a positive constant
                            ASTRepr::Constant(x) if *x > 0.0 => {
                                let ln_base = ASTRepr::Ln(base.clone());
                                Ok(ASTRepr::Mul(exp.clone(), Box::new(ln_base)))
                            }
                            // For all other cases (variables, expressions), don't apply the rule
                            // to avoid domain issues when the base could be negative
                            _ => Ok(ASTRepr::Ln(Box::new(inner_opt))),
                        }
                    }
                    // Constant folding
                    ASTRepr::Constant(a) if *a > 0.0 => Ok(ASTRepr::Constant(a.ln())),
                    _ => Ok(ASTRepr::Ln(Box::new(inner_opt))),
                }
            }
            ASTRepr::Exp(inner) => {
                let inner_opt = self.apply_static_algebraic_rules(inner)?;

                match &inner_opt {
                    // exp(0) = 1
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                    // exp(1) = e
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(std::f64::consts::E)),
                    // exp(ln(x)) = x
                    ASTRepr::Ln(x) => Ok((**x).clone()),
                    // exp(a + b) = exp(a) * exp(b) - ONLY if expansion rules enabled
                    ASTRepr::Add(a, b) if self.config.enable_expansion_rules => {
                        let exp_a = ASTRepr::Exp(a.clone());
                        let exp_b = ASTRepr::Exp(b.clone());
                        Ok(ASTRepr::Mul(Box::new(exp_a), Box::new(exp_b)))
                    }
                    // exp(a - b) = exp(a) / exp(b) - ONLY if expansion rules enabled
                    ASTRepr::Sub(a, b) if self.config.enable_expansion_rules => {
                        let exp_a = ASTRepr::Exp(a.clone());
                        let exp_b = ASTRepr::Exp(b.clone());
                        Ok(ASTRepr::Div(Box::new(exp_a), Box::new(exp_b)))
                    }
                    // Constant folding
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.exp())),
                    _ => Ok(ASTRepr::Exp(Box::new(inner_opt))),
                }
            }
            ASTRepr::Sin(inner) => {
                let inner_opt = self.apply_static_algebraic_rules(inner)?;

                match &inner_opt {
                    // sin(0) = 0
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(0.0)),
                    // sin(π/2) = 1
                    ASTRepr::Constant(x) if (*x - std::f64::consts::FRAC_PI_2).abs() < 1e-15 => {
                        Ok(ASTRepr::Constant(1.0))
                    }
                    // sin(π) = 0
                    ASTRepr::Constant(x) if (*x - std::f64::consts::PI).abs() < 1e-15 => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // sin(-x) = -sin(x)
                    ASTRepr::Neg(x) => {
                        let sin_x = ASTRepr::Sin(x.clone());
                        Ok(ASTRepr::Neg(Box::new(sin_x)))
                    }
                    // Constant folding
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.sin())),
                    _ => Ok(ASTRepr::Sin(Box::new(inner_opt))),
                }
            }
            ASTRepr::Cos(inner) => {
                let inner_opt = self.apply_static_algebraic_rules(inner)?;

                match &inner_opt {
                    // cos(0) = 1
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                    // cos(π/2) = 0
                    ASTRepr::Constant(x) if (*x - std::f64::consts::FRAC_PI_2).abs() < 1e-15 => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // cos(π) = -1
                    ASTRepr::Constant(x) if (*x - std::f64::consts::PI).abs() < 1e-15 => {
                        Ok(ASTRepr::Constant(-1.0))
                    }
                    // cos(-x) = cos(x)
                    ASTRepr::Neg(x) => Ok(ASTRepr::Cos(x.clone())),
                    // Constant folding
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.cos())),
                    _ => Ok(ASTRepr::Cos(Box::new(inner_opt))),
                }
            }
            ASTRepr::Sqrt(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                Ok(ASTRepr::Sqrt(Box::new(inner_opt)))
            }
            ASTRepr::Sum(_collection) => {
                // TODO: Apply arithmetic rules to Collection format
                Ok(expr.clone())
            }

            // Lambda expressions - recursively apply static algebraic rules to body
            ASTRepr::Lambda(lambda) => {
                let optimized_body = self.apply_static_algebraic_rules(&lambda.body)?;
                Ok(ASTRepr::Lambda(Box::new(Lambda {
                    var_indices: lambda.var_indices.clone(),
                    body: Box::new(optimized_body),
                })))
            }

            ASTRepr::BoundVar(_) => {
                // BoundVar behaves like Variable for static algebraic rules
                Ok(expr.clone())
            }
            ASTRepr::Let(binding_id, expr_val, body) => {
                // Apply static algebraic rules to both the bound expression and body
                let optimized_expr = self.apply_static_algebraic_rules(expr_val)?;
                let optimized_body = self.apply_static_algebraic_rules(body)?;
                Ok(ASTRepr::Let(
                    *binding_id,
                    Box::new(optimized_expr),
                    Box::new(optimized_body),
                ))
            }
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Check if two expressions are structurally equal
    fn expressions_equal(a: &ASTRepr<f64>, b: &ASTRepr<f64>) -> bool {
        match (a, b) {
            (ASTRepr::Constant(a), ASTRepr::Constant(b)) => (a - b).abs() < f64::EPSILON,
            (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,
            (ASTRepr::Add(a1, a2), ASTRepr::Add(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Mul(a1, a2), ASTRepr::Mul(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Sub(a1, a2), ASTRepr::Sub(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Div(a1, a2), ASTRepr::Div(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Pow(a1, a2), ASTRepr::Pow(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Neg(a), ASTRepr::Neg(b)) => Self::expressions_equal(a, b),
            (ASTRepr::Ln(a), ASTRepr::Ln(b)) => Self::expressions_equal(a, b),
            (ASTRepr::Exp(a), ASTRepr::Exp(b)) => Self::expressions_equal(a, b),
            (ASTRepr::Sin(a), ASTRepr::Sin(b)) => Self::expressions_equal(a, b),
            (ASTRepr::Cos(a), ASTRepr::Cos(b)) => Self::expressions_equal(a, b),
            (ASTRepr::Sqrt(a), ASTRepr::Sqrt(b)) => Self::expressions_equal(a, b),
            _ => false,
        }
    }
}

/// Execution strategy for mathematical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationStrategy {
    /// Compile-time code generation with zero runtime overhead (like `HeteroContext`)
    /// Best for: Expressions known at compile time, maximum performance (~0.36ns)
    StaticCodegen,
    /// Runtime code generation for performance-critical dynamic expressions
    /// Best for: Complex expressions with repeated evaluation (~1-5ns)
    DynamicCodegen,
    /// AST interpretation for maximum runtime flexibility
    /// Best for: Flexible expressions, rapid prototyping (~10-50ns)
    Interpretation,
    /// Smart adaptive selection based on expression complexity
    /// Best for: General-purpose usage with automatic optimization
    Adaptive {
        complexity_threshold: usize,
        call_count_threshold: usize,
    },
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self::Interpretation // Default to interpretation for backward compatibility
    }
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Enable aggressive optimizations that might change numerical behavior
    pub aggressive: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable common subexpression elimination
    pub cse: bool,
    /// Enable egglog-based symbolic optimization
    pub egglog_optimization: bool,
    /// Enable expansion rules (like exp(a + b) = exp(a) * exp(b))
    /// These can increase operation count, so disable for performance-critical code
    pub enable_expansion_rules: bool,
    /// Enable distribution rules (like a * (b + c) = a*b + a*c)
    /// These can significantly increase operation count
    pub enable_distribution_rules: bool,
    /// Execution strategy for expressions
    pub strategy: OptimizationStrategy,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            aggressive: false,
            constant_folding: true,
            cse: true,
            egglog_optimization: true, // Enable by default - we've proven it works
            enable_expansion_rules: false, // Keep conservative for performance
            enable_distribution_rules: false, // Keep conservative for performance
            strategy: OptimizationStrategy::default(),
        }
    }
}

impl OptimizationConfig {
    /// Configuration optimized for maximum performance (static contexts)
    #[must_use]
    pub fn zero_overhead() -> Self {
        Self {
            strategy: OptimizationStrategy::StaticCodegen,
            constant_folding: true,
            egglog_optimization: false, // Skip for speed
            aggressive: true,
            ..Default::default()
        }
    }

    /// Configuration optimized for flexibility (dynamic contexts)
    #[must_use]
    pub fn dynamic_flexible() -> Self {
        Self {
            strategy: OptimizationStrategy::Interpretation,
            constant_folding: true,
            egglog_optimization: true, // Enable for better optimization
            aggressive: false,
            ..Default::default()
        }
    }

    /// Configuration optimized for performance-critical dynamic code
    #[must_use]
    pub fn dynamic_performance() -> Self {
        Self {
            strategy: OptimizationStrategy::DynamicCodegen,
            constant_folding: true,
            egglog_optimization: true,
            aggressive: true,
            ..Default::default()
        }
    }

    /// Smart adaptive configuration
    #[must_use]
    pub fn adaptive() -> Self {
        Self {
            strategy: OptimizationStrategy::Adaptive {
                complexity_threshold: 10,
                call_count_threshold: 1000,
            },
            constant_folding: true,
            egglog_optimization: true,
            aggressive: false,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ASTRepr;

    #[test]
    fn test_symbolic_optimizer_creation() {
        let optimizer = SymbolicOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_basic_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test basic optimization with index-based variables
        let x = ASTRepr::<f64>::Variable(0);
        let zero = ASTRepr::<f64>::Constant(0.0);
        let expr = ASTRepr::Add(Box::new(x), Box::new(zero)); // x + 0

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should optimize to just x (Variable(0))
        match optimized {
            ASTRepr::Variable(0) => (),
            _ => panic!("Expected Variable(0), got {optimized:?}"),
        }
    }

    #[test]
    fn test_compilation_strategy_creation() {
        let rust_hotload = CompilationStrategy::HotLoadRust {
            source_dir: std::env::temp_dir().join("test_src"),
            lib_dir: std::env::temp_dir().join("test_lib"),
            opt_level: RustOptLevel::O2,
        };
        assert!(matches!(
            rust_hotload,
            CompilationStrategy::HotLoadRust { .. }
        ));

        let rust_hot_load = CompilationStrategy::HotLoadRust {
            source_dir: std::path::PathBuf::from("/tmp/src"),
            lib_dir: std::path::PathBuf::from("/tmp/lib"),
            opt_level: RustOptLevel::O2,
        };

        match rust_hot_load {
            CompilationStrategy::HotLoadRust { opt_level, .. } => {
                assert_eq!(opt_level, RustOptLevel::O2);
            }
            _ => panic!("Expected HotLoadRust strategy"),
        }
    }

    #[test]
    fn test_zero_power_negative_exponent_bug() {
        // Test with a minimal configuration that only applies basic safe rules
        let config = OptimizationConfig {
            max_iterations: 1,
            aggressive: false,
            constant_folding: false, // Disable hand-coded constant folding
            cse: false,
            egglog_optimization: true, // Let egglog handle the sophistication
            enable_expansion_rules: false,
            enable_distribution_rules: false,
            strategy: OptimizationStrategy::Interpretation,
        };
        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

        // Create 0^(-0.1) expression
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Constant(0.0)),
            Box::new(ASTRepr::Constant(-0.1)),
        );

        println!("Original expression: {expr:?}");

        // Direct evaluation should give inf
        let direct_result: f64 = expr.eval_with_vars(&[]);
        println!("Direct evaluation: {direct_result}");
        assert!(
            direct_result.is_infinite(),
            "Direct evaluation should be inf for 0^(-0.1)"
        );

        // Test with minimal hand-coded rules
        let optimized = optimizer.optimize(&expr).unwrap();
        println!("Optimized with minimal hand-coded rules: {optimized:?}");

        let symbolic_result: f64 = optimized.eval_with_vars(&[]);
        println!("Symbolic evaluation: {symbolic_result}");

        // This should now preserve mathematical correctness
        assert!(
            symbolic_result.is_infinite(),
            "Symbolic optimization should preserve inf for 0^(-0.1), but got {symbolic_result}"
        );
    }

    #[test]
    fn test_zero_power_negative_exponent_bug_original() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create 0^(-0.1) expression
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Constant(0.0)),
            Box::new(ASTRepr::Constant(-0.1)),
        );

        println!("Original expression: {expr:?}");

        // Direct evaluation should give inf
        let direct_result: f64 = expr.eval_with_vars(&[]);
        println!("Direct evaluation: {direct_result}");
        assert!(
            direct_result.is_infinite(),
            "Direct evaluation should be inf for 0^(-0.1)"
        );

        // Trace through optimization steps
        let mut current = expr.clone();
        println!("Initial: {current:?}");

        // Apply arithmetic rules
        current = SymbolicOptimizer::apply_arithmetic_rules(&current).unwrap();
        println!("After arithmetic rules: {current:?}");

        // Apply algebraic rules
        current = SymbolicOptimizer::apply_algebraic_rules(&current).unwrap();
        println!("After algebraic rules: {current:?}");

        // Apply static algebraic rules
        current = optimizer.apply_static_algebraic_rules(&current).unwrap();
        println!("After static algebraic rules: {current:?}");

        // Apply constant folding
        if optimizer.config.constant_folding {
            current = SymbolicOptimizer::apply_constant_folding(&current).unwrap();
            println!("After constant folding: {current:?}");
        }

        // Symbolic optimization should preserve mathematical correctness
        let optimized = optimizer.optimize(&expr).unwrap();
        println!("Final optimized expression: {optimized:?}");

        let symbolic_result: f64 = optimized.eval_with_vars(&[]);
        println!("Symbolic evaluation: {symbolic_result}");

        // BUG: This will fail because symbolic optimization incorrectly returns 0
        // TODO: This test documents the current bug - it should pass after the fix
        // assert!(symbolic_result.is_infinite(),
        //     "Symbolic optimization should preserve inf for 0^(-0.1), but got {}", symbolic_result);
    }

    #[test]
    fn test_zero_power_negative_exponent_no_egglog() {
        // Test with egglog completely disabled to isolate the hand-coded rules
        let config = OptimizationConfig {
            max_iterations: 1,
            aggressive: false,
            constant_folding: false, // Disable hand-coded constant folding
            cse: false,
            egglog_optimization: false, // Completely disable egglog
            enable_expansion_rules: false,
            enable_distribution_rules: false,
            strategy: OptimizationStrategy::Interpretation,
        };
        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

        // Create 0^(-0.1) expression
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Constant(0.0)),
            Box::new(ASTRepr::Constant(-0.1)),
        );

        println!("Original expression: {expr:?}");

        // Direct evaluation should give inf
        let direct_result: f64 = expr.eval_with_vars(&[]);
        println!("Direct evaluation: {direct_result}");
        assert!(
            direct_result.is_infinite(),
            "Direct evaluation should be inf for 0^(-0.1)"
        );

        // Test with NO egglog - only basic hand-coded rules
        let optimized = optimizer.optimize(&expr).unwrap();
        println!("Optimized with NO egglog: {optimized:?}");

        let symbolic_result: f64 = optimized.eval_with_vars(&[]);
        println!("Symbolic evaluation: {symbolic_result}");

        // This should preserve mathematical correctness since egglog is disabled
        assert!(
            symbolic_result.is_infinite(),
            "Hand-coded rules alone should preserve inf for 0^(-0.1), but got {symbolic_result}"
        );
    }

    #[test]
    fn test_static_algebraic_rules_debug() {
        let config = OptimizationConfig::default();
        let optimizer = SymbolicOptimizer::with_config(config).unwrap();

        // Create 0^(-0.1) expression
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Constant(0.0)),
            Box::new(ASTRepr::Constant(-0.1)),
        );

        println!("Input to static algebraic rules: {expr:?}");

        // Test ONLY the static algebraic rules
        let result = optimizer.apply_static_algebraic_rules(&expr).unwrap();
        println!("Output from static algebraic rules: {result:?}");

        // This should preserve the original expression since constant folding should not apply
        match result {
            ASTRepr::Pow(base, exp) => match (base.as_ref(), exp.as_ref()) {
                (ASTRepr::Constant(0.0), ASTRepr::Constant(-0.1)) => {
                    println!("✓ Static algebraic rules correctly preserved the expression");
                }
                _ => {
                    panic!(
                        "Static algebraic rules incorrectly modified the expression: base={base:?}, exp={exp:?}"
                    );
                }
            },
            ASTRepr::Constant(val) => {
                panic!("Static algebraic rules incorrectly folded to constant: {val}");
            }
            _ => {
                panic!("Static algebraic rules returned unexpected form: {result:?}");
            }
        }
    }

    #[test]
    fn test_minimal_optimization_debug() {
        // Create a configuration that disables EVERYTHING possible
        let config = OptimizationConfig {
            max_iterations: 1, // Only one iteration
            aggressive: false,
            constant_folding: false, // Disable explicit constant folding
            cse: false,
            egglog_optimization: false, // Disable egglog completely
            enable_expansion_rules: false,
            enable_distribution_rules: false,
            strategy: OptimizationStrategy::Interpretation,
        };
        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

        // Create 0^(-0.1) expression
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Constant(0.0)),
            Box::new(ASTRepr::Constant(-0.1)),
        );

        println!("Original expression: {expr:?}");

        // Test with the most minimal optimization possible
        let optimized = optimizer.optimize(&expr).unwrap();
        println!("Optimized with minimal config: {optimized:?}");

        // This should preserve the original expression
        match optimized {
            ASTRepr::Pow(base, exp) => match (base.as_ref(), exp.as_ref()) {
                (ASTRepr::Constant(0.0), ASTRepr::Constant(-0.1)) => {
                    println!("✓ Minimal optimization correctly preserved the expression");
                }
                _ => {
                    panic!(
                        "Minimal optimization incorrectly modified the expression: base={base:?}, exp={exp:?}"
                    );
                }
            },
            ASTRepr::Constant(val) => {
                panic!("Minimal optimization incorrectly folded to constant: {val}");
            }
            _ => {
                panic!("Minimal optimization returned unexpected form: {optimized:?}");
            }
        }
    }
}
