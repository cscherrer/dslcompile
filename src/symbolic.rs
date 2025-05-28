//! Symbolic optimization using egglog for algebraic simplification
//!
//! This module provides Layer 2 optimization in our three-layer optimization strategy:
//! 1. Hand-coded domain optimizations (in JIT layer)
//! 2. **Egglog symbolic optimization** (this module)
//! 3. Cranelift low-level optimization
//!
//! The symbolic optimizer handles algebraic identities, constant folding, and structural
//! optimizations that can be expressed as rewrite rules.

use crate::ast_utils::expressions_equal_default;
use crate::error::Result;
use crate::final_tagless::ASTRepr;
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

/// Symbolic optimizer for expression simplification
///
/// This is a simplified implementation that will be enhanced with egglog integration.
/// For now, it implements basic algebraic simplifications directly.
#[derive(Debug, Clone)]
pub struct SymbolicOptimizer {
    /// Configuration for optimization behavior
    config: OptimizationConfig,
    /// Compilation strategy
    compilation_strategy: CompilationStrategy,
    /// Statistics for adaptive compilation
    expression_stats: HashMap<String, ExpressionStats>,
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
            expression_stats: HashMap::new(),
        })
    }

    /// Create a new symbolic optimizer with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Result<Self> {
        Ok(Self {
            config,
            compilation_strategy: CompilationStrategy::default(),
            expression_stats: HashMap::new(),
        })
    }

    /// Create a new symbolic optimizer with custom compilation strategy
    pub fn with_strategy(strategy: CompilationStrategy) -> Result<Self> {
        Ok(Self {
            config: OptimizationConfig::default(),
            compilation_strategy: strategy,
            expression_stats: HashMap::new(),
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
                    .expression_stats
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
            .expression_stats
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
        &self.expression_stats
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
            ASTRepr::Constant(value) => Ok(value.to_string()),
            ASTRepr::Variable(index) => {
                // Map variable indices to function parameters
                match *index {
                    0 => Ok("x".to_string()),
                    1 => Ok("y".to_string()),
                    _ => Ok("x".to_string()), // Default to x for unknown indices
                }
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
            crate::error::MathJITError::CompilationError(format!(
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
                crate::error::MathJITError::CompilationError(format!("Failed to run rustc: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::error::MathJITError::CompilationError(format!(
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
                source_dir: std::path::PathBuf::from("/tmp/mathjit_sources"),
                lib_dir: std::path::PathBuf::from("/tmp/mathjit_libs"),
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
                    match crate::egglog_integration::optimize_with_egglog(&optimized) {
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
                    (_, ASTRepr::Constant(0.0)) => Ok(ASTRepr::Constant(0.0)),
                    (ASTRepr::Constant(0.0), _) => Ok(ASTRepr::Constant(0.0)),
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
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) if *b != 0.0 => {
                        Ok(ASTRepr::Constant(a / b))
                    }
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
            use crate::egglog_integration::optimize_with_egglog;

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
                let left_opt = self.apply_enhanced_algebraic_rules(left)?;
                let right_opt = self.apply_enhanced_algebraic_rules(right)?;

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
                let left_opt = self.apply_enhanced_algebraic_rules(left)?;
                let right_opt = self.apply_enhanced_algebraic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // x * 0 = 0
                    (_, ASTRepr::Constant(0.0)) | (ASTRepr::Constant(0.0), _) => {
                        Ok(ASTRepr::Constant(0.0))
                    }
                    // x * 1 = x
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    (ASTRepr::Constant(1.0), _) => Ok(right_opt),
                    // x * -1 = -x
                    (_, ASTRepr::Constant(-1.0)) => Ok(ASTRepr::Neg(Box::new(left_opt))),
                    (ASTRepr::Constant(-1.0), _) => Ok(ASTRepr::Neg(Box::new(right_opt))),
                    // Constant folding: a * b = (a*b)
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a * b)),
                    // x * x = x^2
                    _ if expressions_equal_default(&left_opt, &right_opt) => Ok(ASTRepr::Pow(
                        Box::new(left_opt),
                        Box::new(ASTRepr::Constant(2.0)),
                    )),
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
                    // Distribute multiplication over addition: a * (b + c) = a*b + a*c
                    (_, ASTRepr::Add(b, c)) => {
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
                let left_opt = self.apply_enhanced_algebraic_rules(left)?;
                let right_opt = self.apply_enhanced_algebraic_rules(right)?;

                match (&left_opt, &right_opt) {
                    // 0 / x = 0 (assuming x ≠ 0)
                    (ASTRepr::Constant(0.0), _) => Ok(ASTRepr::Constant(0.0)),
                    // x / 1 = x
                    (_, ASTRepr::Constant(1.0)) => Ok(left_opt),
                    // x / x = 1 (assuming x ≠ 0)
                    _ if expressions_equal_default(&left_opt, &right_opt) => {
                        Ok(ASTRepr::Constant(1.0))
                    }
                    // Constant folding: a / b = (a/b)
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) if *b != 0.0 => {
                        Ok(ASTRepr::Constant(a / b))
                    }
                    _ => Ok(ASTRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            ASTRepr::Pow(base, exp) => {
                let base_opt = self.apply_enhanced_algebraic_rules(base)?;
                let exp_opt = self.apply_enhanced_algebraic_rules(exp)?;

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
                        Ok(ASTRepr::Constant(a.powf(*b)))
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
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                match &inner_opt {
                    // -(-x) = x
                    ASTRepr::Neg(x) => Ok((**x).clone()),
                    // -(0) = 0
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(0.0)),
                    // -(const) = -const
                    ASTRepr::Constant(a) => Ok(ASTRepr::Constant(-a)),
                    // -(a + b) = -a - b
                    ASTRepr::Add(a, b) => {
                        let neg_a = ASTRepr::Neg(a.clone());
                        let neg_b = ASTRepr::Neg(b.clone());
                        Ok(ASTRepr::Sub(Box::new(neg_a), Box::new(neg_b)))
                    }
                    // -(a - b) = b - a
                    ASTRepr::Sub(a, b) => Ok(ASTRepr::Sub(b.clone(), a.clone())),
                    _ => Ok(ASTRepr::Neg(Box::new(inner_opt))),
                }
            }
            ASTRepr::Ln(inner) => {
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                match &inner_opt {
                    // ln(1) = 0
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(0.0)),
                    // ln(e) ≈ 1
                    ASTRepr::Constant(x) if (*x - std::f64::consts::E).abs() < 1e-15 => {
                        Ok(ASTRepr::Constant(1.0))
                    }
                    // ln(exp(x)) = x
                    ASTRepr::Exp(x) => Ok((**x).clone()),
                    // ln(a * b) = ln(a) + ln(b)
                    ASTRepr::Mul(a, b) => {
                        let ln_a = ASTRepr::Ln(a.clone());
                        let ln_b = ASTRepr::Ln(b.clone());
                        Ok(ASTRepr::Add(Box::new(ln_a), Box::new(ln_b)))
                    }
                    // ln(a / b) = ln(a) - ln(b)
                    ASTRepr::Div(a, b) => {
                        let ln_a = ASTRepr::Ln(a.clone());
                        let ln_b = ASTRepr::Ln(b.clone());
                        Ok(ASTRepr::Sub(Box::new(ln_a), Box::new(ln_b)))
                    }
                    // ln(x^a) = a * ln(x)
                    ASTRepr::Pow(base, exp) => {
                        let ln_base = ASTRepr::Ln(base.clone());
                        Ok(ASTRepr::Mul(exp.clone(), Box::new(ln_base)))
                    }
                    // Constant folding
                    ASTRepr::Constant(a) if *a > 0.0 => Ok(ASTRepr::Constant(a.ln())),
                    _ => Ok(ASTRepr::Ln(Box::new(inner_opt))),
                }
            }
            ASTRepr::Exp(inner) => {
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                match &inner_opt {
                    // exp(0) = 1
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(1.0)),
                    // exp(1) = e
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(std::f64::consts::E)),
                    // exp(ln(x)) = x
                    ASTRepr::Ln(x) => Ok((**x).clone()),
                    // exp(a + b) = exp(a) * exp(b)
                    ASTRepr::Add(a, b) => {
                        let exp_a = ASTRepr::Exp(a.clone());
                        let exp_b = ASTRepr::Exp(b.clone());
                        Ok(ASTRepr::Mul(Box::new(exp_a), Box::new(exp_b)))
                    }
                    // exp(a - b) = exp(a) / exp(b)
                    ASTRepr::Sub(a, b) => {
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
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

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
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

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
                let inner_opt = self.apply_enhanced_algebraic_rules(inner)?;

                match &inner_opt {
                    // sqrt(0) = 0
                    ASTRepr::Constant(0.0) => Ok(ASTRepr::Constant(0.0)),
                    // sqrt(1) = 1
                    ASTRepr::Constant(1.0) => Ok(ASTRepr::Constant(1.0)),
                    // sqrt(x^2) = |x| ≈ x for positive domains
                    ASTRepr::Pow(base, exp) if matches!(exp.as_ref(), ASTRepr::Constant(2.0)) => {
                        Ok((**base).clone())
                    }
                    // sqrt(x * x) = |x| ≈ x for positive domains
                    ASTRepr::Mul(a, b) if Self::expressions_equal(a, b) => Ok((**a).clone()),
                    // Constant folding
                    ASTRepr::Constant(a) if *a >= 0.0 => Ok(ASTRepr::Constant(a.sqrt())),
                    _ => Ok(ASTRepr::Sqrt(Box::new(inner_opt))),
                }
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
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            aggressive: false,
            constant_folding: true,
            cse: true,
            egglog_optimization: false,
        }
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
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
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    fn test_symbolic_optimizer_creation() {
        let optimizer = SymbolicOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_arithmetic_identity_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test x + 0 = x
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(0.0));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            ASTRepr::Variable(index) => assert_eq!(index, 0),
            _ => panic!("Expected simplified variable"),
        }
    }

    #[test]
    fn test_constant_folding() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test 2 + 3 = 5
        let expr = ASTEval::add(ASTEval::constant(2.0), ASTEval::constant(3.0));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            ASTRepr::Constant(value) => assert!((value - 5.0).abs() < 1e-10),
            _ => panic!("Expected constant folding"),
        }
    }

    #[test]
    fn test_power_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test x^1 = x
        let expr = ASTEval::pow(ASTEval::var(0), ASTEval::constant(1.0));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            ASTRepr::Variable(index) => assert_eq!(index, 0),
            _ => panic!("Expected simplified variable"),
        }
    }

    #[test]
    fn test_transcendental_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test ln(exp(x)) = x
        let expr = ASTEval::ln(ASTEval::exp(ASTEval::var(0)));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            ASTRepr::Variable(index) => assert_eq!(index, 0),
            _ => panic!("Expected simplified variable"),
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
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));

        // Test Cranelift strategy
        let approach = optimizer.choose_compilation_approach(&expr, "test_expr");
        assert_eq!(approach, CompilationApproach::Cranelift);

        // Test adaptive strategy
        optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
            call_threshold: 100,
            complexity_threshold: 50,
        });

        let approach = optimizer.choose_compilation_approach(&expr, "test_expr2");
        assert_eq!(approach, CompilationApproach::Cranelift); // Should start with Cranelift

        // Simulate many calls to trigger upgrade
        for _ in 0..150 {
            optimizer.record_execution("test_expr2", 1000);
        }

        let approach = optimizer.choose_compilation_approach(&expr, "test_expr2");
        assert_eq!(approach, CompilationApproach::UpgradeToRust);
    }

    #[test]
    fn test_rust_source_generation() {
        let optimizer = SymbolicOptimizer::new().unwrap();

        // Test simple expression: x + 1
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        let source = optimizer.generate_rust_source(&expr, "test_func").unwrap();

        assert!(source.contains("#[no_mangle]"));
        assert!(source.contains("test_func"));
        assert!(source.contains("x + 1"));
    }

    #[test]
    fn test_strategy_recommendation() {
        // Simple expression should use Cranelift
        let simple_expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        let strategy = SymbolicOptimizer::recommend_strategy(&simple_expr);
        assert_eq!(strategy, CompilationStrategy::CraneliftJIT);

        // Complex expression should use adaptive or Rust hot-loading
        let complex_expr = {
            let mut expr = ASTEval::var(0);
            // Create a complex expression with many operations
            for i in 1..60 {
                expr = ASTEval::add(
                    expr,
                    ASTEval::sin(ASTEval::mul(
                        ASTEval::var(0),
                        ASTEval::constant(f64::from(i)),
                    )),
                );
            }
            expr
        };

        let strategy = SymbolicOptimizer::recommend_strategy(&complex_expr);
        match strategy {
            CompilationStrategy::Adaptive { .. } => {
                // Expected for complex expressions
            }
            CompilationStrategy::HotLoadRust { .. } => {
                // Also acceptable for very complex expressions
            }
            _ => panic!("Expected adaptive or Rust hot-loading for complex expression"),
        }
    }

    #[test]
    fn test_execution_statistics() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let _expr: ASTRepr<f64> = ASTEval::var(0);

        // Initialize the expression stats by calling choose_compilation_approach first
        let simple_expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        optimizer.choose_compilation_approach(&simple_expr, "test_expr");

        // Record some executions
        optimizer.record_execution("test_expr", 1000);
        optimizer.record_execution("test_expr", 1500);
        optimizer.record_execution("test_expr", 800);

        let stats = optimizer.get_expression_stats();
        let test_stats = stats.get("test_expr").unwrap();

        assert_eq!(test_stats.call_count, 3);
        assert!(test_stats.avg_execution_time_ns > 0.0);
    }

    #[test]
    fn test_egglog_optimization_config() {
        let config = OptimizationConfig {
            max_iterations: 5,
            aggressive: true,
            constant_folding: true,
            cse: true,
            egglog_optimization: true,
        };

        // Test optimizer with egglog enabled
        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));

        // This should work even with egglog enabled (currently a no-op)
        let optimized = optimizer.optimize(&expr).unwrap();
        assert!(
            matches!(optimized, ASTRepr::Add(_, _)) || matches!(optimized, ASTRepr::Variable(_))
        );
    }

    #[test]
    fn test_optimization_pipeline_integration() {
        let config = OptimizationConfig {
            max_iterations: 10,
            aggressive: true,
            constant_folding: true,
            cse: false, // Keep false for predictable testing
            egglog_optimization: false,
        };

        let mut optimizer = SymbolicOptimizer::with_config(config).unwrap();

        let expr = ASTEval::add(
            ASTEval::mul(ASTEval::var(0), ASTEval::constant(2.0)),
            ASTEval::constant(0.0),
        );

        let optimized = optimizer.optimize(&expr).unwrap();

        // Should simplify to 2*x or x*2
        match optimized {
            ASTRepr::Mul(_, _) => {
                // Expected: multiplication with constant 2 and variable x
            }
            _ => {
                // Could also be further simplified depending on rules applied
                println!("Optimization result: {optimized:?}");
            }
        }
    }
}
