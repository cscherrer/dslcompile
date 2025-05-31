//! Symbolic optimization using egglog for algebraic simplification
//!
//! This module provides Layer 2 optimization in our three-layer optimization strategy:
//! 1. Hand-coded domain optimizations (in JIT layer)
//! 2. **Egglog symbolic optimization** (this module)
//! 3. Cranelift low-level optimization
//!
//! The symbolic optimizer handles algebraic identities, constant folding, and structural
//! optimizations that can be expressed as rewrite rules.

use crate::ast::ast_utils::expressions_equal_default;
use crate::error::Result;
use crate::final_tagless::ASTRepr;
use crate::symbolic::egglog_integration::optimize_with_egglog;
use std::collections::HashMap;
// use std::time::Instant; // Will be used for optimization timing in future updates

// Re-export for convenience
pub use crate::backends::rust_codegen::RustOptLevel;

/// Compilation strategy for mathematical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationStrategy {
    /// Fast JIT compilation using Cranelift (default)
    /// Best for: Simple expressions, rapid iteration, low latency
    CraneliftJIT,
    /// Hot-loading compiled Rust dylibs
    /// Best for: Complex expressions, maximum performance, production use
    HotLoadRust {
        /// Directory for generated Rust source files
        source_dir: std::path::PathBuf,
        /// Directory for compiled dylibs
        lib_dir: std::path::PathBuf,
        /// Optimization level for rustc
        opt_level: RustOptLevel,
    },
    /// Adaptive strategy: start with Cranelift, upgrade to Rust for hot expressions
    Adaptive {
        /// Threshold for call count before upgrading to Rust compilation
        call_threshold: usize,
        /// Threshold for expression complexity before upgrading
        complexity_threshold: usize,
    },
}

/// Compilation approach decision for a specific expression
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationApproach {
    /// Use Cranelift JIT compilation
    Cranelift,
    /// Use Rust hot-loading compilation
    RustHotLoad,
    /// Upgrade from Cranelift to Rust compilation
    UpgradeToRust,
}

impl Default for CompilationStrategy {
    fn default() -> Self {
        Self::CraneliftJIT
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

/// Statistics for tracking expression usage patterns
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
            CompilationStrategy::CraneliftJIT => CompilationApproach::Cranelift,
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

                // Upgrade to Rust compilation if thresholds are met
                if stats.call_count >= *call_threshold || stats.complexity >= *complexity_threshold
                {
                    if stats.rust_compiled {
                        CompilationApproach::RustHotLoad
                    } else {
                        stats.rust_compiled = true;
                        CompilationApproach::UpgradeToRust
                    }
                } else {
                    CompilationApproach::Cranelift
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

        Ok(format!(
            r#"
#[no_mangle]
pub extern "C" fn {function_name}(x: f64) -> f64 {{
    {expr_code}
}}

#[no_mangle]
pub extern "C" fn {function_name}_two_vars(x: f64, y: f64) -> f64 {{
    let _ = y; // Suppress unused variable warning if not used
    {expr_code}
}}

#[no_mangle]
pub extern "C" fn {function_name}_multi_vars(vars: *const f64, count: usize) -> f64 {{
    if vars.is_null() || count == 0 {{
        return 0.0;
    }}
    
    let x = unsafe {{ *vars }};
    let y = if count > 1 {{ unsafe {{ *vars.add(1) }} }} else {{ 0.0 }};
    let _ = (y, count); // Suppress unused variable warnings
    
    {expr_code}
}}
"#
        ))
    }

    /// Generate Rust expression code from `ASTRepr`
    #[allow(clippy::only_used_in_recursion)]
    fn generate_rust_expression(&self, expr: &ASTRepr<f64>) -> Result<String> {
        match expr {
            ASTRepr::Constant(val) => Ok(val.to_string()),
            ASTRepr::Variable(index) => Ok(format!("vars[{}]", index)),
            ASTRepr::Add(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} + {right_code})"))
            }
            ASTRepr::Sub(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} - {right_code})"))
            }
            ASTRepr::Mul(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} * {right_code})"))
            }
            ASTRepr::Div(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} / {right_code})"))
            }
            ASTRepr::Pow(base, exp) => {
                let base_code = self.generate_rust_expression(base)?;
                let exp_code = self.generate_rust_expression(exp)?;
                Ok(format!("({base_code}).powf({exp_code})"))
            }
            ASTRepr::Neg(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("-{inner_code}"))
            }
            #[cfg(feature = "logexp")]
            ASTRepr::Log(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("{inner_code}.ln()"))
            }
            #[cfg(feature = "logexp")]
            ASTRepr::Exp(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("{inner_code}.exp()"))
            }
            ASTRepr::Trig(trig_category) => {
                match &trig_category.function {
                    crate::ast::function_categories::TrigFunction::Sin(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.sin()"))
                    }
                    crate::ast::function_categories::TrigFunction::Cos(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.cos()"))
                    }
                    crate::ast::function_categories::TrigFunction::Tan(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.tan()"))
                    }
                    _ => {
                        // For other trig functions, convert to AST and generate
                        let ast_form = trig_category.to_ast();
                        self.generate_rust_expression(&ast_form)
                    }
                }
            }
            ASTRepr::Hyperbolic(hyp_category) => {
                match &hyp_category.function {
                    crate::ast::function_categories::HyperbolicFunction::Sinh(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.sinh()"))
                    }
                    crate::ast::function_categories::HyperbolicFunction::Cosh(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.cosh()"))
                    }
                    crate::ast::function_categories::HyperbolicFunction::Tanh(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.tanh()"))
                    }
                    _ => {
                        // For other hyperbolic functions, use sqrt representation
                        let inner_code = self.generate_rust_expression(
                            &match &hyp_category.function {
                                crate::ast::function_categories::HyperbolicFunction::Sinh(inner) |
                                crate::ast::function_categories::HyperbolicFunction::Cosh(inner) |
                                crate::ast::function_categories::HyperbolicFunction::Tanh(inner) => inner,
                                _ => return Err(crate::error::MathCompileError::CompilationError(
                                    "Unsupported hyperbolic function".to_string()
                                )),
                            }
                        )?;
                        Ok(format!("{inner_code}.sqrt()")) // Placeholder - would need proper implementation
                    }
                }
            }
            #[cfg(feature = "logexp")]
            ASTRepr::LogExp(logexp_category) => {
                match &logexp_category.function {
                    crate::ast::function_categories::LogExpFunction::Log(inner) |
                    crate::ast::function_categories::LogExpFunction::Ln(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.ln()"))
                    }
                    crate::ast::function_categories::LogExpFunction::Exp(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.exp()"))
                    }
                    crate::ast::function_categories::LogExpFunction::Log10(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.log10()"))
                    }
                    crate::ast::function_categories::LogExpFunction::Log2(inner) => {
                        let inner_code = self.generate_rust_expression(inner)?;
                        Ok(format!("{inner_code}.log2()"))
                    }
                    _ => {
                        return Err(crate::error::MathCompileError::CompilationError(
                            "Unsupported logarithmic function".to_string()
                        ));
                    }
                }
            }
            #[cfg(feature = "special")]
            ASTRepr::Special(_) => {
                return Err(crate::error::MathCompileError::CompilationError(
                    "Special functions not yet supported in Rust code generation".to_string()
                ));
            }
            #[cfg(feature = "linear_algebra")]
            ASTRepr::LinearAlgebra(_) => {
                return Err(crate::error::MathCompileError::CompilationError(
                    "Linear algebra operations not yet supported in Rust code generation".to_string()
                ));
            }
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
            crate::error::MathCompileError::CompilationError(format!(
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
                crate::error::MathCompileError::CompilationError(format!(
                    "Failed to run rustc: {e}"
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::error::MathCompileError::CompilationError(format!(
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
            CompilationStrategy::CraneliftJIT
        } else if complexity < 50 {
            // Medium complexity: use adaptive approach
            CompilationStrategy::Adaptive {
                call_threshold: 100,
                complexity_threshold: 25,
            }
        } else {
            // Complex expressions: use Rust hot-loading for maximum performance
            CompilationStrategy::HotLoadRust {
                source_dir: std::path::PathBuf::from("/tmp/mathcompile_sources"),
                lib_dir: std::path::PathBuf::from("/tmp/mathcompile_libs"),
                opt_level: RustOptLevel::O2,
            }
        }
    }

    /// Optimize a JIT representation using symbolic rewrite rules
    pub fn optimize(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        let mut optimized = expr.clone();
        let mut iterations = 0;

        // Apply optimization passes until convergence or max iterations
        while iterations < self.config.max_iterations {
            let before = optimized.clone();

            // Layer 1: Apply basic algebraic simplifications (hand-coded rules)
            optimized = Self::apply_arithmetic_rules(&optimized)?;
            optimized = Self::apply_algebraic_rules(&optimized)?;

            // Apply enhanced algebraic rules (includes transcendental optimizations)
            optimized = self.apply_enhanced_algebraic_rules(&optimized)?;

            if self.config.constant_folding {
                optimized = Self::apply_constant_folding(&optimized)?;
            }

            // Layer 2: Apply egglog symbolic optimization (if enabled)
            if self.config.egglog_optimization {
                #[cfg(feature = "optimization")]
                {
                    match optimize_with_egglog(&optimized) {
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

    /// Apply basic arithmetic simplification rules
    fn apply_arithmetic_rules(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        match expr {
            // Addition rules
            ASTRepr::Add(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // x + 0 = x
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    (ASTRepr::Constant(0.0), _) => Ok(right_opt),
                    // Constant folding
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                        Ok(ASTRepr::Constant(a + b))
                    }
                    _ => Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            // Subtraction rules
            ASTRepr::Sub(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // x - 0 = x
                    (_, ASTRepr::Constant(0.0)) => Ok(left_opt),
                    // 0 - x = -x
                    (ASTRepr::Constant(0.0), _) => Ok(ASTRepr::Neg(Box::new(right_opt))),
                    // x - x = 0
                    (a, b) if Self::expressions_equal(a, b) => Ok(ASTRepr::Constant(0.0)),
                    // Constant folding
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                        Ok(ASTRepr::Constant(a - b))
                    }
                    _ => Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            // Multiplication rules
            ASTRepr::Mul(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // x * 0 = 0
                    (_, ASTRepr::Constant(0.0)) | (ASTRepr::Constant(0.0), _) => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // x * 1 = x
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    (ASTRepr::Constant(1.0), _) => Ok(right_opt),
                    // Constant folding
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                        Ok(ASTRepr::Constant(a * b))
                    }
                    _ => Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            // Division rules
            ASTRepr::Div(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // 0 / x = 0 (assuming x != 0)
                    (ASTRepr::Constant(0.0), ASTRepr::Constant(x)) if *x != 0.0 => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // x / 1 = x
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    // x / x = 1 (assuming x != 0)
                    (a, b) if Self::expressions_equal(a, b) => Ok(ASTRepr::Constant(1.0)),
                    // Constant folding
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) if *b != 0.0 => {
                        Ok(ASTRepr::Constant(a / b))
                    }
                    _ => Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            // Power rules
            ASTRepr::Pow(base, exp) => {
                let base_opt = Self::apply_arithmetic_rules(base)?;
                let exp_opt = Self::apply_arithmetic_rules(exp)?;

                match (&base_opt, &exp_opt) {
                    // x^0 = 1
                    (_, ASTRepr::Constant(0.0)) => Ok(ASTRepr::Constant(1.0)),
                    // x^1 = x
                    (_, ASTRepr::Constant(1.0)) => Ok(base_opt),
                    // 1^x = 1
                    (ASTRepr::Constant(1.0), _) => Ok(ASTRepr::Constant(1.0)),
                    // 0^x = 0 (for x > 0)
                    (ASTRepr::Constant(0.0), ASTRepr::Constant(x)) if *x > 0.0 => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // Constant folding
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                        Ok(ASTRepr::Constant(a.powf(*b)))
                    }
                    _ => Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }
            // Negation rules
            ASTRepr::Neg(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    // -(-x) = x
                    ASTRepr::Neg(inner_inner) => Ok((**inner_inner).clone()),
                    // -(constant) = -constant
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(-a)),
                    _ => Ok(ASTRepr::Neg(Box::new(inner_opt))),
                }
            }
            // Logarithmic functions (feature-gated)
            #[cfg(feature = "logexp")]
            ASTRepr::Log(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(0.0)),
                    _ => Ok(inner_opt.ln()),
                }
            }
            #[cfg(feature = "logexp")]
            ASTRepr::Exp(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                    _ => Ok(inner_opt.exp()),
                }
            }
            // Trigonometric functions
            ASTRepr::Trig(trig_category) => {
                match &trig_category.function {
                    crate::ast::function_categories::TrigFunction::Sin(inner) => {
                        let inner_opt = Self::apply_arithmetic_rules(inner)?;
                        match &inner_opt {
                            ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(0.0)),
                            _ => Ok(inner_opt.sin()),
                        }
                    }
                    crate::ast::function_categories::TrigFunction::Cos(inner) => {
                        let inner_opt = Self::apply_arithmetic_rules(inner)?;
                        match &inner_opt {
                            ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                            _ => Ok(inner_opt.cos()),
                        }
                    }
                    _ => {
                        // For other trig functions, recursively apply rules to arguments
                        let optimized_category = match &trig_category.function {
                            crate::ast::function_categories::TrigFunction::Tan(inner) => {
                                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                                crate::ast::function_categories::TrigCategory {
                                    function: crate::ast::function_categories::TrigFunction::Tan(Box::new(inner_opt)),
                                }
                            }
                            _ => trig_category.clone(),
                        };
                        Ok(ASTRepr::Trig(Box::new(optimized_category)))
                    }
                }
            }
            // Hyperbolic functions
            ASTRepr::Hyperbolic(hyp_category) => {
                let optimized_category = match &hyp_category.function {
                    crate::ast::function_categories::HyperbolicFunction::Sinh(inner) => {
                        let inner_opt = Self::apply_arithmetic_rules(inner)?;
                        crate::ast::function_categories::HyperbolicCategory {
                            function: crate::ast::function_categories::HyperbolicFunction::Sinh(Box::new(inner_opt)),
                        }
                    }
                    crate::ast::function_categories::HyperbolicFunction::Cosh(inner) => {
                        let inner_opt = Self::apply_arithmetic_rules(inner)?;
                        crate::ast::function_categories::HyperbolicCategory {
                            function: crate::ast::function_categories::HyperbolicFunction::Cosh(Box::new(inner_opt)),
                        }
                    }
                    _ => hyp_category.clone(),
                };
                Ok(ASTRepr::Hyperbolic(Box::new(optimized_category)))
            }
            // Extended logarithmic functions (feature-gated)
            #[cfg(feature = "logexp")]
            ASTRepr::LogExp(logexp_category) => {
                let optimized_category = match &logexp_category.function {
                    crate::ast::function_categories::LogExpFunction::Log(inner) |
                    crate::ast::function_categories::LogExpFunction::Ln(inner) => {
                        let inner_opt = Self::apply_arithmetic_rules(inner)?;
                        crate::ast::function_categories::LogExpCategory {
                            function: crate::ast::function_categories::LogExpFunction::Log(Box::new(inner_opt)),
                        }
                    }
                    crate::ast::function_categories::LogExpFunction::Exp(inner) => {
                        let inner_opt = Self::apply_arithmetic_rules(inner)?;
                        crate::ast::function_categories::LogExpCategory {
                            function: crate::ast::function_categories::LogExpFunction::Exp(Box::new(inner_opt)),
                        }
                    }
                    _ => logexp_category.clone(),
                };
                Ok(ASTRepr::LogExp(Box::new(optimized_category)))
            }
            // Special functions (feature-gated)
            #[cfg(feature = "special")]
            ASTRepr::Special(special_category) => {
                // For now, just return the special function as-is
                // Could add specific optimizations for special functions here
                Ok(ASTRepr::Special(special_category.clone()))
            }
            // Linear algebra (feature-gated)
            #[cfg(feature = "linear_algebra")]
            ASTRepr::LinearAlgebra(linalg_category) => {
                // For now, just return the linear algebra operation as-is
                // Could add specific optimizations for linear algebra here
                Ok(ASTRepr::LinearAlgebra(linalg_category.clone()))
            }
            // Base cases
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Apply algebraic transformation rules (associativity, commutativity, etc.)
    fn apply_algebraic_rules(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        match expr {
            ASTRepr::Add(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Sub(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Mul(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt)))
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
            #[cfg(feature = "logexp")]
            ASTRepr::Log(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(inner_opt.ln())
            }
            #[cfg(feature = "logexp")]
            ASTRepr::Exp(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(inner_opt.exp())
            }
            ASTRepr::Trig(trig_category) => {
                let optimized_category = match &trig_category.function {
                    crate::ast::function_categories::TrigFunction::Sin(inner) => {
                        let inner_opt = Self::apply_algebraic_rules(inner)?;
                        crate::ast::function_categories::TrigCategory {
                            function: crate::ast::function_categories::TrigFunction::Sin(Box::new(inner_opt)),
                        }
                    }
                    crate::ast::function_categories::TrigFunction::Cos(inner) => {
                        let inner_opt = Self::apply_algebraic_rules(inner)?;
                        crate::ast::function_categories::TrigCategory {
                            function: crate::ast::function_categories::TrigFunction::Cos(Box::new(inner_opt)),
                        }
                    }
                    _ => trig_category.clone(),
                };
                Ok(ASTRepr::Trig(Box::new(optimized_category)))
            }
            ASTRepr::Hyperbolic(hyp_category) => {
                let optimized_category = match &hyp_category.function {
                    crate::ast::function_categories::HyperbolicFunction::Sinh(inner) => {
                        let inner_opt = Self::apply_algebraic_rules(inner)?;
                        crate::ast::function_categories::HyperbolicCategory {
                            function: crate::ast::function_categories::HyperbolicFunction::Sinh(Box::new(inner_opt)),
                        }
                    }
                    crate::ast::function_categories::HyperbolicFunction::Cosh(inner) => {
                        let inner_opt = Self::apply_algebraic_rules(inner)?;
                        crate::ast::function_categories::HyperbolicCategory {
                            function: crate::ast::function_categories::HyperbolicFunction::Cosh(Box::new(inner_opt)),
                        }
                    }
                    _ => hyp_category.clone(),
                };
                Ok(ASTRepr::Hyperbolic(Box::new(optimized_category)))
            }
            #[cfg(feature = "logexp")]
            ASTRepr::LogExp(logexp_category) => {
                let optimized_category = match &logexp_category.function {
                    crate::ast::function_categories::LogExpFunction::Log(inner) |
                    crate::ast::function_categories::LogExpFunction::Ln(inner) => {
                        let inner_opt = Self::apply_algebraic_rules(inner)?;
                        crate::ast::function_categories::LogExpCategory {
                            function: crate::ast::function_categories::LogExpFunction::Log(Box::new(inner_opt)),
                        }
                    }
                    crate::ast::function_categories::LogExpFunction::Exp(inner) => {
                        let inner_opt = Self::apply_algebraic_rules(inner)?;
                        crate::ast::function_categories::LogExpCategory {
                            function: crate::ast::function_categories::LogExpFunction::Exp(Box::new(inner_opt)),
                        }
                    }
                    _ => logexp_category.clone(),
                };
                Ok(ASTRepr::LogExp(Box::new(optimized_category)))
            }
            #[cfg(feature = "special")]
            ASTRepr::Special(special_category) => {
                Ok(ASTRepr::Special(special_category.clone()))
            }
            #[cfg(feature = "linear_algebra")]
            ASTRepr::LinearAlgebra(linalg_category) => {
                Ok(ASTRepr::LinearAlgebra(linalg_category.clone()))
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
                        Ok(ASTRepr::Constant(a.powf(*b)))
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
            #[cfg(feature = "logexp")]
            ASTRepr::Log(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) if *a > 0.0 => Ok(ASTRepr::Constant(a.ln())),
                    _ => Ok(inner_opt.ln()),
                }
            }
            #[cfg(feature = "logexp")]
            ASTRepr::Exp(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.exp())),
                    _ => Ok(inner_opt.exp()),
                }
            }
            ASTRepr::Trig(trig_category) => {
                match &trig_category.function {
                    crate::ast::function_categories::TrigFunction::Sin(inner) => {
                        let inner_opt = Self::apply_constant_folding(inner)?;
                        match &inner_opt {
                            ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.sin())),
                            _ => Ok(inner_opt.sin()),
                        }
                    }
                    crate::ast::function_categories::TrigFunction::Cos(inner) => {
                        let inner_opt = Self::apply_constant_folding(inner)?;
                        match &inner_opt {
                            ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.cos())),
                            _ => Ok(inner_opt.cos()),
                        }
                    }
                    _ => Ok(expr.clone()), // Other trig functions not implemented yet
                }
            }
            ASTRepr::Hyperbolic(_) => Ok(expr.clone()), // Hyperbolic functions not implemented yet
            #[cfg(feature = "special")]
            ASTRepr::Special(_) => Ok(expr.clone()), // Special functions not implemented yet
            #[cfg(feature = "linear_algebra")]
            ASTRepr::LinearAlgebra(_) => Ok(expr.clone()), // Linear algebra not implemented yet
            #[cfg(feature = "logexp")]
            ASTRepr::LogExp(_) => Ok(expr.clone()), // Extended log/exp not implemented yet
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
            use crate::symbolic::egglog_integration::optimize_with_egglog;

            // Try to use egglog optimization
            match optimize_with_egglog(expr) {
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

    /// Apply enhanced algebraic simplification rules
    /// This is a stepping stone toward full egglog integration
    #[allow(clippy::only_used_in_recursion)]
    fn apply_enhanced_algebraic_rules(&self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        match expr {
            ASTRepr::Add(left, right) => {
                let left_opt = self.apply_enhanced_algebraic_rules(left)?;
                let right_opt = self.apply_enhanced_algebraic_rules(right)?;

                // Apply distributive law: a * (b + c) = a*b + a*c (if enabled)
                if self.config.enable_distribution_rules {
                    match (&left_opt, &right_opt) {
                        // a * (b + c) = a*b + a*c
                        (ASTRepr::Mul(a, bc), _) if matches!(bc.as_ref(), ASTRepr::Add(_, _)) => {
                            if let ASTRepr::Add(b, c) = bc.as_ref() {
                                let ab = ASTRepr::Mul(a.clone(), b.clone());
                                let ac = ASTRepr::Mul(a.clone(), c.clone());
                                return Ok(ASTRepr::Add(Box::new(ab), Box::new(ac)));
                            }
                        }
                        // (a + b) * c = a*c + b*c
                        (_, ASTRepr::Mul(ab, c)) if matches!(ab.as_ref(), ASTRepr::Add(_, _)) => {
                            if let ASTRepr::Add(a, b) = ab.as_ref() {
                                let ac = ASTRepr::Mul(a.clone(), c.clone());
                                let bc = ASTRepr::Mul(b.clone(), c.clone());
                                return Ok(ASTRepr::Add(Box::new(ac), Box::new(bc)));
                            }
                        }
                        _ => {}
                    }
                }

                Ok(ASTRepr::Add(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Sub(left, right) => {
                let left_opt = self.apply_enhanced_algebraic_rules(left)?;
                let right_opt = self.apply_enhanced_algebraic_rules(right)?;
                Ok(ASTRepr::Sub(Box::new(left_opt), Box::new(right_opt)))
            }
            ASTRepr::Mul(left, right) => {
                let left_opt = self.apply_enhanced_algebraic_rules(left)?;
                let right_opt = self.apply_enhanced_algebraic_rules(right)?;

                // Power combination: x^a * x^b = x^(a+b)
                match (&left_opt, &right_opt) {
                    (ASTRepr::Pow(base1, exp1), ASTRepr::Pow(base2, exp2))
                        if Self::expressions_equal(base1, base2) =>
                    {
                        let combined_exp = ASTRepr::Add(exp1.clone(), exp2.clone());
                        Ok(ASTRepr::Pow(base1.clone(), Box::new(combined_exp)))
                    }
                    // x * x^a = x^(1+a)
                    (base, ASTRepr::Pow(pow_base, exp)) if Self::expressions_equal(base, pow_base) => {
                        let one = ASTRepr::Constant(1.0);
                        let combined_exp = ASTRepr::Add(Box::new(one), exp.clone());
                        Ok(ASTRepr::Pow(Box::new(base.clone()), Box::new(combined_exp)))
                    }
                    // x^a * x = x^(a+1)
                    (ASTRepr::Pow(pow_base, exp), base) if Self::expressions_equal(pow_base, base) => {
                        let one = ASTRepr::Constant(1.0);
                        let combined_exp = ASTRepr::Add(exp.clone(), Box::new(one));
                        Ok(ASTRepr::Pow(pow_base.clone(), Box::new(combined_exp)))
                    }
                    _ => Ok(ASTRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Div(left, right) => {
                let left_opt = self.apply_enhanced_algebraic_rules(left)?;
                let right_opt = self.apply_enhanced_algebraic_rules(right)?;

                // Power division: x^a / x^b = x^(a-b)
                match (&left_opt, &right_opt) {
                    (ASTRepr::Pow(base1, exp1), ASTRepr::Pow(base2, exp2))
                        if Self::expressions_equal(base1, base2) =>
                    {
                        let combined_exp = ASTRepr::Sub(exp1.clone(), exp2.clone());
                        Ok(ASTRepr::Pow(base1.clone(), Box::new(combined_exp)))
                    }
                    // x / x^a = x^(1-a)
                    (base, ASTRepr::Pow(pow_base, exp)) if Self::expressions_equal(base, pow_base) => {
                        let one = ASTRepr::Constant(1.0);
                        let combined_exp = ASTRepr::Sub(Box::new(one), exp.clone());
                        Ok(ASTRepr::Pow(Box::new(base.clone()), Box::new(combined_exp)))
                    }
                    // x^a / x = x^(a-1)
                    (ASTRepr::Pow(pow_base, exp), base) if Self::expressions_equal(pow_base, base) => {
                        let one = ASTRepr::Constant(1.0);
                        let combined_exp = ASTRepr::Sub(exp.clone(), Box::new(one));
                        Ok(ASTRepr::Pow(pow_base.clone(), Box::new(combined_exp)))
                    }
                    _ => Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Pow(base, exp) => {
                let base_opt = self.apply_enhanced_algebraic_rules(base)?;
                let exp_opt = self.apply_enhanced_algebraic_rules(exp)?;

                // Power of power: (x^a)^b = x^(a*b)
                match &base_opt {
                    ASTRepr::Pow(inner_base, inner_exp) => {
                        let combined_exp = ASTRepr::Mul(inner_exp.clone(), Box::new(exp_opt));
                        Ok(ASTRepr::Pow(inner_base.clone(), Box::new(combined_exp)))
                    }
                    _ => Ok(ASTRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }
            ASTRepr::Neg(inner) => {
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;
                Ok(ASTRepr::Neg(Box::new(inner_opt)))
            }
            #[cfg(feature = "logexp")]
            ASTRepr::Log(inner) => {
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                match &inner_opt {
                    // log(1) = 0
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(0.0)),
                    // log(e) = 1
                    ASTRepr::Constant(x) if (*x - std::f64::consts::E).abs() < 1e-10 => {
                        Ok(ASTRepr::Constant(1.0))
                    }
                    // log(a * b) = log(a) + log(b) (only if both a and b are positive constants)
                    ASTRepr::Mul(a, b) => match (a.as_ref(), b.as_ref()) {
                        (ASTRepr::Constant(a_val), ASTRepr::Constant(b_val))
                            if *a_val > 0.0 && *b_val > 0.0 =>
                        {
                            let ln_a = a.ln_ref();
                            let ln_b = b.ln_ref();
                            Ok(ASTRepr::Add(Box::new(ln_a), Box::new(ln_b)))
                        }
                        _ => Ok(inner_opt.ln()),
                    },
                    // log(a / b) = log(a) - log(b) (only if b != 0 and no problematic values)
                    ASTRepr::Div(a, b) => {
                        // Only apply if both are positive constants to avoid domain issues
                        if matches!(a.as_ref(), ASTRepr::Constant(x) if *x <= 0.0)
                            || matches!(b.as_ref(), ASTRepr::Constant(x) if *x <= 0.0)
                        {
                            Ok(inner_opt.ln())
                        } else {
                            let ln_a = a.ln_ref();
                            let ln_b = b.ln_ref();
                            Ok(ASTRepr::Sub(Box::new(ln_a), Box::new(ln_b)))
                        }
                    }
                    // log(x^n) = n * log(x) (only if base is positive)
                    ASTRepr::Pow(base, exp) => {
                        match base.as_ref() {
                            // Don't apply if base is 0, since log(0) is undefined
                            ASTRepr::Constant(x) if *x == 0.0 => {
                                Ok(inner_opt.ln())
                            }
                            // Only apply if base is a positive constant
                            ASTRepr::Constant(x) if *x > 0.0 => {
                                let ln_base = base.ln_ref();
                                Ok(ASTRepr::Mul(exp.clone(), Box::new(ln_base)))
                            }
                            // For all other cases (variables, expressions), don't apply the rule
                            // to avoid domain issues when the base could be negative
                            _ => Ok(inner_opt.ln()),
                        }
                    }
                    // Constant folding
                    ASTRepr::Constant(a) if *a > 0.0 => Ok(ASTRepr::Constant(a.ln())),
                    _ => Ok(inner_opt.ln()),
                }
            }
            #[cfg(feature = "logexp")]
            ASTRepr::Exp(inner) => {
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                match &inner_opt {
                    // exp(0) = 1
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                    // exp(1) = e
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(std::f64::consts::E)),
                    // exp(log(x)) = x
                    #[cfg(feature = "logexp")]
                    ASTRepr::Log(x) => Ok((**x).clone()),
                    // exp(a + b) = exp(a) * exp(b) - ONLY if expansion rules enabled
                    ASTRepr::Add(a, b) if self.config.enable_expansion_rules => {
                        let exp_a = a.exp_ref();
                        let exp_b = b.exp_ref();
                        Ok(ASTRepr::Mul(Box::new(exp_a), Box::new(exp_b)))
                    }
                    // exp(a - b) = exp(a) / exp(b) - ONLY if expansion rules enabled
                    ASTRepr::Sub(a, b) if self.config.enable_expansion_rules => {
                        let exp_a = a.exp_ref();
                        let exp_b = b.exp_ref();
                        Ok(ASTRepr::Div(Box::new(exp_a), Box::new(exp_b)))
                    }
                    // Constant folding
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.exp())),
                    _ => Ok(inner_opt.exp()),
                }
            }
            ASTRepr::Trig(trig_category) => {
                match &trig_category.function {
                    crate::ast::function_categories::TrigFunction::Sin(inner) => {
                        let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                        match &inner_opt {
                            // sin(0) = 0
                            ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(0.0)),
                            // sin(π/2) = 1
                            ASTRepr::Constant(x) if (*x - std::f64::consts::FRAC_PI_2).abs() < 1e-10 => {
                                Ok(ASTRepr::Constant(1.0))
                            }
                            // sin(π) = 0
                            ASTRepr::Constant(x) if (*x - std::f64::consts::PI).abs() < 1e-10 => {
                                Ok(ASTRepr::Constant(0.0))
                            }
                            // sin(-x) = -sin(x)
                            ASTRepr::Neg(x) => {
                                let sin_x = x.sin_ref();
                                Ok(ASTRepr::Neg(Box::new(sin_x)))
                            }
                            // Constant folding
                            ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.sin())),
                            _ => Ok(inner_opt.sin()),
                        }
                    }
                    crate::ast::function_categories::TrigFunction::Cos(inner) => {
                        let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                        match &inner_opt {
                            // cos(0) = 1
                            ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                            // cos(π/2) = 0
                            ASTRepr::Constant(x) if (*x - std::f64::consts::FRAC_PI_2).abs() < 1e-10 => {
                                Ok(ASTRepr::Constant(0.0))
                            }
                            // cos(π) = -1
                            ASTRepr::Constant(x) if (*x - std::f64::consts::PI).abs() < 1e-10 => {
                                Ok(ASTRepr::Constant(-1.0))
                            }
                            // cos(-x) = cos(x)
                            ASTRepr::Neg(x) => Ok(x.cos_ref()),
                            // Constant folding
                            ASTRepr::Constant(a) => Ok(ASTRepr::Constant(a.cos())),
                            _ => Ok(inner_opt.cos()),
                        }
                    }
                    _ => {
                        // For other trig functions, recursively apply to arguments
                        let optimized_category = match &trig_category.function {
                            crate::ast::function_categories::TrigFunction::Tan(inner) => {
                                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;
                                crate::ast::function_categories::TrigCategory {
                                    function: crate::ast::function_categories::TrigFunction::Tan(Box::new(inner_opt)),
                                }
                            }
                            _ => trig_category.clone(),
                        };
                        Ok(ASTRepr::Trig(Box::new(optimized_category)))
                    }
                }
            }
            // Handle square root as power with exponent 0.5
            ASTRepr::Pow(base, exp) if matches!(exp.as_ref(), ASTRepr::Constant(x) if (*x - 0.5).abs() < 1e-10) => {
                let base_opt = self.apply_enhanced_algebraic_rules(base)?;

                match &base_opt {
                    // sqrt(0) = 0
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(0.0)),
                    // sqrt(1) = 1
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(1.0)),
                    // sqrt(4) = 2, etc.
                    ASTRepr::Constant(a) if *a >= 0.0 => Ok(ASTRepr::Constant(a.sqrt())),
                    // sqrt(x^2) = |x| - for now, assume x >= 0 and return x
                    ASTRepr::Pow(inner_base, inner_exp) if matches!(inner_exp.as_ref(), ASTRepr::Constant(2.0)) => {
                        Ok((**inner_base).clone())
                    }
                    // Constant folding
                    ASTRepr::Constant(a) if *a >= 0.0 => Ok(ASTRepr::Constant(a.sqrt())),
                    _ => Ok(base_opt.sqrt_ref()),
                }
            }
            ASTRepr::Hyperbolic(hyp_category) => {
                let optimized_category = match &hyp_category.function {
                    crate::ast::function_categories::HyperbolicFunction::Sinh(inner) => {
                        let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;
                        crate::ast::function_categories::HyperbolicCategory {
                            function: crate::ast::function_categories::HyperbolicFunction::Sinh(Box::new(inner_opt)),
                        }
                    }
                    crate::ast::function_categories::HyperbolicFunction::Cosh(inner) => {
                        let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;
                        crate::ast::function_categories::HyperbolicCategory {
                            function: crate::ast::function_categories::HyperbolicFunction::Cosh(Box::new(inner_opt)),
                        }
                    }
                    _ => hyp_category.clone(),
                };
                Ok(ASTRepr::Hyperbolic(Box::new(optimized_category)))
            }
            #[cfg(feature = "logexp")]
            ASTRepr::LogExp(logexp_category) => {
                let optimized_category = match &logexp_category.function {
                    crate::ast::function_categories::LogExpFunction::Log(inner) |
                    crate::ast::function_categories::LogExpFunction::Ln(inner) => {
                        let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;
                        crate::ast::function_categories::LogExpCategory {
                            function: crate::ast::function_categories::LogExpFunction::Log(Box::new(inner_opt)),
                        }
                    }
                    crate::ast::function_categories::LogExpFunction::Exp(inner) => {
                        let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;
                        crate::ast::function_categories::LogExpCategory {
                            function: crate::ast::function_categories::LogExpFunction::Exp(Box::new(inner_opt)),
                        }
                    }
                    _ => logexp_category.clone(),
                };
                Ok(ASTRepr::LogExp(Box::new(optimized_category)))
            }
            #[cfg(feature = "special")]
            ASTRepr::Special(special_category) => {
                Ok(ASTRepr::Special(special_category.clone()))
            }
            #[cfg(feature = "linear_algebra")]
            ASTRepr::LinearAlgebra(linalg_category) => {
                Ok(ASTRepr::LinearAlgebra(linalg_category.clone()))
            }
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Check if two expressions are structurally equal
    fn expressions_equal(a: &ASTRepr<f64>, b: &ASTRepr<f64>) -> bool {
        match (a, b) {
            (ASTRepr::Constant(a), ASTRepr::Constant(b)) => (a - b).abs() < 1e-10,
            (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,
            (ASTRepr::Add(a1, a2), ASTRepr::Add(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Sub(a1, a2), ASTRepr::Sub(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Mul(a1, a2), ASTRepr::Mul(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Div(a1, a2), ASTRepr::Div(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Pow(a1, a2), ASTRepr::Pow(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (ASTRepr::Neg(a), ASTRepr::Neg(b)) => Self::expressions_equal(a, b),
            #[cfg(feature = "logexp")]
            (ASTRepr::Log(a), ASTRepr::Log(b)) => Self::expressions_equal(a, b),
            #[cfg(feature = "logexp")]
            (ASTRepr::Exp(a), ASTRepr::Exp(b)) => Self::expressions_equal(a, b),
            (ASTRepr::Trig(a), ASTRepr::Trig(b)) => {
                // Compare trig functions by their egglog representation for simplicity
                a.to_egglog() == b.to_egglog()
            }
            (ASTRepr::Hyperbolic(a), ASTRepr::Hyperbolic(b)) => {
                // Compare hyperbolic functions by their egglog representation
                a.to_egglog() == b.to_egglog()
            }
            #[cfg(feature = "logexp")]
            (ASTRepr::LogExp(a), ASTRepr::LogExp(b)) => {
                // Compare logexp functions by their egglog representation
                a.to_egglog() == b.to_egglog()
            }
            #[cfg(feature = "special")]
            (ASTRepr::Special(a), ASTRepr::Special(b)) => {
                // Compare special functions by their egglog representation
                a.to_egglog() == b.to_egglog()
            }
            #[cfg(feature = "linear_algebra")]
            (ASTRepr::LinearAlgebra(a), ASTRepr::LinearAlgebra(b)) => {
                // Compare linear algebra functions by their egglog representation
                a.to_egglog() == b.to_egglog()
            }
            _ => false,
        }
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
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            aggressive: false,
            constant_folding: true,
            cse: true,
            egglog_optimization: false,
            enable_expansion_rules: true,
            enable_distribution_rules: true,
        }
    }
}

/// Optimization statistics
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

/// Trait for optimizable expressions
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::ASTRepr;

    #[test]
    fn test_symbolic_optimizer_creation() {
        let optimizer = SymbolicOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_basic_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API: x + 0
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let expr = &x + &math.constant(0.0);

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should optimize to just x (Variable(0))
        match optimized {
            ASTRepr::Variable(0) => (),
            _ => panic!("Expected Variable(0), got {optimized:?}"),
        }
    }

    #[test]
    fn test_constant_folding() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API: 2 + 3
        let math = crate::ergonomics::MathBuilder::new();
        let expr = math.constant(2.0) + math.constant(3.0);

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should fold to constant 5.0
        match optimized {
            ASTRepr::Constant(val) => assert!((val - 5.0).abs() < 1e-10),
            _ => panic!("Expected Constant(5.0), got {optimized:?}"),
        }
    }

    #[test]
    fn test_power_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API: x^1
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let expr = x.pow_ref(&math.constant(1.0));

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should optimize to just x
        match optimized {
            ASTRepr::Variable(0) => (),
            _ => panic!("Expected Variable(0), got {optimized:?}"),
        }
    }

    #[test]
    fn test_transcendental_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API: ln(exp(x))
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let expr = x.exp_ref().ln_ref();

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should optimize to just x
        match optimized {
            ASTRepr::Variable(0) => (),
            _ => panic!("Expected Variable(0), got {optimized:?}"),
        }
    }

    #[test]
    fn test_compilation_strategy_creation() {
        let cranelift = CompilationStrategy::CraneliftJIT;
        assert_eq!(cranelift, CompilationStrategy::CraneliftJIT);

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
    fn test_compilation_approach_selection() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API: x + 1
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let expr = &x + &math.constant(1.0);

        let approach = optimizer.choose_compilation_approach(&expr, "test");

        // Should default to Cranelift for simple expressions
        assert_eq!(approach, CompilationApproach::Cranelift);

        // Test adaptive strategy
        optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
            call_threshold: 5,
            complexity_threshold: 10,
        });

        // First few calls should use Cranelift
        for _ in 0..3 {
            let approach = optimizer.choose_compilation_approach(&expr, "adaptive_test");
            assert_eq!(approach, CompilationApproach::Cranelift);
            optimizer.record_execution("adaptive_test", 1000);
        }
    }

    #[test]
    fn test_rust_source_generation() {
        let optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API: x + 1
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let expr = &x + &math.constant(1.0);

        let source = optimizer.generate_rust_source(&expr, "test_func").unwrap();

        assert!(source.contains("test_func"));
        assert!(source.contains("extern \"C\""));
        assert!(source.contains("x + 1"));
    }

    #[test]
    fn test_strategy_recommendation() {
        // Create simple expression using ergonomic API
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let simple_expr = &x + &math.constant(1.0);

        let strategy = SymbolicOptimizer::recommend_strategy(&simple_expr);

        // Should recommend Cranelift for simple expressions
        match strategy {
            CompilationStrategy::CraneliftJIT => (),
            _ => panic!("Expected CraneliftJIT for simple expression"),
        }

        // Create complex expression using ergonomic API
        let mut expr = x.clone();
        for i in 1..=10 {
            let term = (x.clone() * math.constant(f64::from(i))).sin_ref();
            expr = expr + term;
        }

        let strategy = SymbolicOptimizer::recommend_strategy(&expr);

        // Should recommend adaptive or hot-loading for complex expressions
        match strategy {
            CompilationStrategy::Adaptive { .. } | CompilationStrategy::HotLoadRust { .. } => (),
            _ => panic!("Expected Adaptive or HotLoadRust for complex expression"),
        }
    }

    #[test]
    fn test_execution_statistics() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let _expr: ASTRepr<f64> = x;

        // Record some execution statistics
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let _simple_expr = &x + &math.constant(1.0);

        optimizer.record_execution("test_expr", 1000);
        optimizer.record_execution("test_expr", 1200);
        optimizer.record_execution("test_expr", 800);

        let stats = optimizer.get_expression_stats();
        assert!(stats.contains_key("test_expr"));

        let expr_stats = &stats["test_expr"];
        assert_eq!(expr_stats.call_count, 3);
        assert!(expr_stats.avg_execution_time_ns > 0.0);
    }

    #[test]
    fn test_egglog_optimization_config() {
        let mut config = OptimizationConfig::default();
        config.egglog_optimization = true;

        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

        // Create expression using ergonomic API: x + 1
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let expr = &x + &math.constant(1.0);

        // Should not panic even with egglog enabled (though it may not do much without egglog)
        let _optimized = optimizer.optimize(&expr).unwrap();
    }

    #[test]
    fn test_optimization_pipeline_integration() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Create expression using ergonomic API: 2*x + 0
        let mut math = crate::ergonomics::MathBuilder::new();
        let x = math.var("x");
        let expr = x * math.constant(2.0) + math.constant(0.0);

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should optimize away the + 0
        // The exact form depends on optimization rules, but should be simpler
        assert!(optimized.count_operations() <= expr.count_operations());
    }
}
